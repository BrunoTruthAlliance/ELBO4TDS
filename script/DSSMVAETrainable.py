import pickle
import os
import torch
import logging
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.modules import loss
from tqdm import *
from model.DSSMVAE import DSSMVAE
from util.utils import get_optimizer, save_checkpoint
from loss import get_loss
from metric import get_metric
# from get_model import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau


def elbo_ssl_fix_var(augs_out, reduce='mean'):
    loss_recon = 0.
    loss_prior = 0.
    all_z = []
    for i in range(len(augs_out)):
        dnn_input_aug_i, z_i, recon_i, _, _ = augs_out[i]
        loss_recon_i = nn.MSELoss(reduction='sum')(recon_i, dnn_input_aug_i)
        loss_recon = loss_recon + loss_recon_i
        all_z.append(z_i)
    loss_prior = PriorLoss()(all_z)

    loss_elbo = (loss_recon + loss_prior) / len(augs_out)
    # print('elbo ing!!!')

    if reduce == 'mean':
        bs = augs_out[0][0].shape[0]
        loss_elbo = loss_elbo / bs

    return loss_elbo


class PriorLoss:

    def __call__(self, all_z):
        # Stack the tensors along a new dimension
        z = torch.stack(all_z, dim=1)  # [bs, num_augs, dim]

        # Compute the mean along the specified axis
        z_mean_aug = z.mean(dim=1, keepdim=True)  # [bs, 1, dim]

        # Calculate the loss
        loss = torch.sum((z - z_mean_aug) ** 2)
        return loss


class DSSMVAETrainable:
    def __init__(self, user_feature=None, item_feature=None, train_dataloader=None, val_dataloader=None, test_dataloader=None, output=1, similarity='dot',
                 user_dnn_size=(64, 32), item_dnn_size=(64, 32), loss_func='bceloss', model='dssm', activation='LeakyReLU', device='cpu',
                 dropout=0.2, l2_normalization=False, use_senet=False, model_path=None, dimensions=16, test_interval=4, dim_z=256, fix_var=True):

        self.loss_func = loss_func
        self.use_senet = use_senet
        self.model_path = model_path
        self.device = device
        print(self.device)
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.output = output
        self.test_interval = test_interval
        self.dim_z = dim_z
        self.fix_var = fix_var

        if train_dataloader is None:
            self.model = torch.load(model_path+'.pt', map_location=torch.device(device))
        else:
            self.train_dataloader = train_dataloader
            self.model = DSSMVAE(user_feature, item_feature, user_dnn_size=user_dnn_size, item_dnn_size=item_dnn_size, output=self.output, similarity=similarity,
                                 dropout=dropout, activation=activation, use_senet=self.use_senet, dimensions=dimensions, l2_normalization=l2_normalization, loss=loss_func, dim_z=dim_z, fix_var=fix_var)

            # self.model = get_model(model, user_feature, item_feature, user_dnn_size=user_dnn_size, item_dnn_size=item_dnn_size, output=self.output, similarity=similarity,
            #                        dropout=dropout, activation=activation, use_senet=self.use_senet, dimensions=dimensions, l2_normalization=l2_normalization, loss=loss_func)
        self.model = self.model.to(device=self.device)

    def set_lr(self, optimizer, decay_factor):
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * decay_factor

    def train(self, epochs=10, optimizer='Adam', lr=1e-5, lr_decay_rate=0, lr_decay_step=0, task_indices=[0], num_augs=8, p=0.2, use_reparam=True, alpha=0.01, perturb_type='bucket_dropout'):
        optimizer = get_optimizer(optimizer, self.model.parameters(), lr)

        if lr_decay_rate == 0 and lr_decay_step == 0:
            is_lr_decay = False
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=lr_decay_step, cooldown=0)
            is_lr_decay = True

        val_auc = 0
        best_val_auc = 0
        early_stop = 0
        for epoch in range(epochs):
            epoch = epoch + 1
            self.model.train()

            total_loss = 0

            for data in self.train_dataloader:
                # x contains a batch_size of users,
                #   and [batch] is equal to the sum of the number of interactions between each user
                # x: [batch, feature_num]
                # y: [batch, task]
                # sum(group_list) = batch
                x, y, group_list, user_embeddings, item_embeddings = data

                x, y = x.to(self.device), y.to(self.device)
                user_embeddings = user_embeddings.to(self.device) if user_embeddings is not None else user_embeddings
                item_embeddings = item_embeddings.to(self.device) if item_embeddings is not None else item_embeddings

                # y: [batch, task]
                y = y[:, task_indices]

                # y_pre: [batch, task]
                # user_emb: [batch, task, user_dnn_size[-1]]
                # item_emb: [batch, task, item_dnn_size[-1]]
                y_pre, user_emb, item_emb, augs_user_out, augs_item_out = self.model(x, user_embeddings, item_embeddings, group=group_list, num_augs=num_augs, p=p, use_reparam=use_reparam, perturb_type=perturb_type)

                loss_sl = get_loss(loss=self.loss_func, true=y, pred=y_pre, group=group_list, user_embedding=user_emb, item_embedding=item_emb)

                loss_ssl_user = elbo_ssl_fix_var(augs_user_out, reduce='mean')
                loss_ssl_item = elbo_ssl_fix_var(augs_item_out, reduce='mean')
                loss_ssl = (loss_ssl_user + loss_ssl_item) / 2

                loss = loss_sl + alpha * loss_ssl

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            total_loss /= len(self.train_dataloader)
            print("epoch:{}, train loss:{:.5}   ".format(epoch, total_loss))
            print(alpha, loss_ssl)

            if (epoch % self.test_interval == 0):
                if self.output == 1 and len(task_indices) == 1:
                    metrics_val = self.eval_validate(task_indices)
                    print(f'validate epoch: {epoch}, {metrics_val}')
                    metrics_test = self.eval_test(task_indices)
                    print(f'test epoch: {epoch}, {metrics_test}')
                    # val auc as model selection metric
                    val_auc = metrics_val['auc']
                    if is_lr_decay:
                        scheduler.step(val_auc)
                    is_best = val_auc > best_val_auc

                    logging.info(f'epoch: {epoch}, is_bset: {is_best}, val: {metrics_val}, test: {metrics_test}')

                    if is_best:
                        save_checkpoint(self.model, optimizer, epoch, f'{self.model_path}.pt')
                        best_val_auc = val_auc
                else:
                    metrics_val = self.eval_val_multi_tasks(task_indices)
                    print(f'validate epoch: {epoch}, {metrics_val}')
                    metrics_test = self.eval_test_multi_tasks(task_indices)
                    print(f'test epoch: {epoch}, {metrics_test}')
                    # val auc as model selection metric
                    val_auc = metrics_val['task_1']['auc']
                    if is_lr_decay:
                        scheduler.step(val_auc)
                    is_best = val_auc > best_val_auc

                    logging.info(f'epoch: {epoch}, is_bset: {is_best}, val: {metrics_val}, test: {metrics_test}')

                    if is_best:
                        save_checkpoint(self.model, optimizer, epoch, f'{self.model_path}.pt')
                        best_val_auc = val_auc

        print("----------finish train----------")
        return best_val_auc

    def eval_validate(self, task_indices=[0]):
        self.model.eval()
        labels, pre_pro = list(), list()
        metrics = None
        with torch.no_grad():
            for data in self.val_dataloader:
                x, y, group_list, user_embeddings, item_embeddings = data
                x, y = x.to(self.device), y.to(self.device)
                user_embeddings = user_embeddings.to(
                    self.device) if user_embeddings is not None else user_embeddings
                item_embeddings = item_embeddings.to(
                    self.device) if item_embeddings is not None else item_embeddings
                y = y[:, task_indices]
                y_pre, user_emb, item_emb = self.model.predict(x, user_embeddings, item_embeddings, group=group_list)
                pre_pro.extend(y_pre[:, -1].cpu().detach().numpy().squeeze())
                labels.extend(y.cpu().detach().numpy().squeeze())
                metrics = get_metric(metrics=metrics, true=y, pred=y_pre, group=group_list, ats=[5])
        metrics = {key: torch.mean(values).item()
                   for key, values in metrics.items()}
        metrics['auc'] = roc_auc_score(np.array(labels), np.array(pre_pro))
        return metrics

    def eval_test(self, task_indices=[0]):
        self.model.eval()
        labels, pre_pro = list(), list()
        metrics = None
        with torch.no_grad():
            for data in self.test_dataloader:
                x, y, group_list, user_embeddings, item_embeddings = data
                x, y = x.to(self.device), y.to(self.device)
                user_embeddings = user_embeddings.to(
                    self.device) if user_embeddings is not None else user_embeddings
                item_embeddings = item_embeddings.to(
                    self.device) if item_embeddings is not None else item_embeddings
                y = y[:, task_indices]
                y_pre, user_emb, item_emb = self.model.predict(x, user_embeddings, item_embeddings, group=group_list)
                pre_pro.extend(y_pre[:, -1].cpu().detach().numpy().squeeze())
                labels.extend(y.cpu().detach().numpy().squeeze())
                metrics = get_metric(metrics=metrics, true=y, pred=y_pre, group=group_list, ats=[5])
        metrics = {key: torch.mean(values).item()
                   for key, values in metrics.items()}
        metrics['auc'] = roc_auc_score(np.array(labels), np.array(pre_pro))
        return metrics

    def eval_test_multi_tasks(self, task_indices):
        self.model.eval()
        num_tasks = len(task_indices)
        all_metrics = [None for _ in range(num_tasks)]
        all_labels = [list() for _ in range(num_tasks)]
        all_pre_pro = [list() for _ in range(num_tasks)]
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                x, y, group_list, user_embeddings, item_embeddings = data
                x, y = x.to(self.device), y.to(self.device)
                user_embeddings = user_embeddings.to(
                    self.device) if user_embeddings is not None else user_embeddings
                item_embeddings = item_embeddings.to(
                    self.device) if item_embeddings is not None else item_embeddings
                y = y[:, task_indices]
                # print(y.shape, )

                y_pre, _, _ = self.model.predict(x, user_embeddings, item_embeddings, group=group_list)

                y_multi = torch.unbind(y, dim=1)
                y_pre_multi = torch.unbind(y_pre, dim=1)

                # print(f'y_pre_multi: {len(y_pre_multi)}: {y_pre_multi[0]}, {y_pre_multi[1]}')
                assert len(y_pre_multi) == num_tasks and len(y_multi) == num_tasks

                for i in range(num_tasks):
                    all_pre_pro[i].extend(y_pre_multi[i].cpu().detach().numpy().squeeze())
                    all_labels[i].extend(y_multi[i].cpu().detach().numpy().squeeze())
                    all_metrics[i] = get_metric(
                        metrics=all_metrics[i], true=y_multi[i], pred=y_pre_multi[i], group=group_list, ats=[5])


                # print(f'all_pre_pro: {(all_pre_pro[0][:10])}: {(all_pre_pro[1][:10])}')
                # print(f'all_pre_pro: {(all_labels[0][:10])}: {(all_labels[1][:10])}')

        for i in range(num_tasks):
            all_metrics[i] = {key: torch.mean(values).item()
                              for key, values in all_metrics[i].items()}
            all_metrics[i]['auc'] = roc_auc_score(np.array(all_labels[i]), np.array(all_pre_pro[i]))

        return_metrics = {}
        for i in range(num_tasks):
            return_metrics[f'task_{i}'] = all_metrics[i]

        return return_metrics

    def eval_val_multi_tasks(self, task_indices):
        self.model.eval()
        num_tasks = len(task_indices)
        all_metrics = [None for _ in range(num_tasks)]
        all_labels = [list() for _ in range(num_tasks)]
        all_pre_pro = [list() for _ in range(num_tasks)]
        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                x, y, group_list, user_embeddings, item_embeddings = data
                x, y = x.to(self.device), y.to(self.device)
                user_embeddings = user_embeddings.to(
                    self.device) if user_embeddings is not None else user_embeddings
                item_embeddings = item_embeddings.to(
                    self.device) if item_embeddings is not None else item_embeddings
                y = y[:, task_indices]
                # print(y.shape, )

                y_pre, _, _ = self.model.predict(x, user_embeddings, item_embeddings, group=group_list)

                y_multi = torch.unbind(y, dim=1)
                y_pre_multi = torch.unbind(y_pre, dim=1)

                # print(f'y_pre_multi: {len(y_pre_multi)}: {y_pre_multi[0]}, {y_pre_multi[1]}')
                assert len(y_pre_multi) == num_tasks and len(y_multi) == num_tasks

                for i in range(num_tasks):
                    all_pre_pro[i].extend(y_pre_multi[i].cpu().detach().numpy().squeeze())
                    all_labels[i].extend(y_multi[i].cpu().detach().numpy().squeeze())
                    all_metrics[i] = get_metric(
                        metrics=all_metrics[i], true=y_multi[i], pred=y_pre_multi[i], group=group_list, ats=[5])


                # print(f'all_pre_pro: {(all_pre_pro[0][:10])}: {(all_pre_pro[1][:10])}')
                # print(f'all_pre_pro: {(all_labels[0][:10])}: {(all_labels[1][:10])}')

        for i in range(num_tasks):
            all_metrics[i] = {key: torch.mean(values).item()
                              for key, values in all_metrics[i].items()}
            all_metrics[i]['auc'] = roc_auc_score(np.array(all_labels[i]), np.array(all_pre_pro[i]))

        return_metrics = {}
        for i in range(num_tasks):
            return_metrics[f'task_{i}'] = all_metrics[i]

        return return_metrics