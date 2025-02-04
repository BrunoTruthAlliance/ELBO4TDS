import argparse
import logging
import os
import random
import torch
import json
import numpy as np
from script.DSSMVAETrainable import DSSMVAETrainable
from data_process.data_loader import data_loader
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--S1', type=str, default='[1]')  # depreciated
    # parser.add_argument('--S2', type=str, default='[1]')  # depreciated
    # parser.add_argument('--m', type=str, default='[0.039]')  # depreciated
    parser.add_argument('--data_dir', type=str,
                        default='data_process/kuairand')
    # 模型名字
    parser.add_argument('--model', type=str, default='dssm')
    # 加载数据进程数量
    parser.add_argument('--data_loader_worker', type=int, default=4)
    # 几轮测试一次
    parser.add_argument('--test_interval', type=int, default=1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--similarity', type=str, default='dot')
    parser.add_argument('--activation', type=str, default='LeakyReLU')
    parser.add_argument('--use_senet', type=bool, default=False)
    parser.add_argument('--l2_normalization', type=str, default='false')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr_decay_rate', type=float, default=0)
    parser.add_argument('--lr_decay_step', type=int, default=0)
    parser.add_argument('--loss', type=str, default='bce_with_logit_loss')

    parser.add_argument('--output', type=int, default=1)
    parser.add_argument('--task_indices', type=str, default='[1]')

    parser.add_argument('--dimension', type=int, default=16)
    # sequential model中交互序列的限长
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)

    parser.add_argument('--mlp_layer', type=str, default='(128, 64, 32)')


    # self-supervised learning
    parser.add_argument('--num_augs', type=int, default=4)
    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument('--use_reparam', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--fix_var', type=int, default=1)
    parser.add_argument('--perturb_type', type=str, default='bucket_dropout')

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


def main():
    train_dataloader, val_dataloader, test_dataloader, user_feature, item_feature, label_feature = data_loader(
        opt.data_dir, batch_size=opt.batch_size, data_loader_worker=opt.data_loader_worker)

    print("----------data load finish----------")
    logging.info(json.dumps(vars(opt)))
    print(json.dumps(vars(opt)))

    save_model_path = os.path.join(opt.data_dir, opt.model)

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    else:
        print("Directory already exists.")

    save_model = os.path.join(save_model_path, f'{opt.model}_{opt.loss}_{opt.lr}_{opt.lr_decay_rate}_{opt.lr_decay_step}_dimz{opt.dim_z}_fixvar{opt.fix_var}_num_augs{opt.num_augs}_p{opt.p}_repa{opt.use_reparam}_alpha{opt.alpha}_{datetime.now().timestamp()}')

    trainable = None

    if opt.model.lower() == 'dssmvae':
        trainable = DSSMVAETrainable(user_feature=user_feature, item_feature=item_feature,
                                     train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                                     user_dnn_size=tuple(eval(opt.mlp_layer)), item_dnn_size=tuple(eval(opt.mlp_layer)),
                                     similarity=opt.similarity, output=opt.output, loss_func=opt.loss, dropout=opt.dropout,
                                     activation='LeakyReLU', use_senet=opt.use_senet, test_interval=opt.test_interval,
                                     dimensions=opt.dimension, model_path=save_model, device=opt.device, model=opt.model,
                                     l2_normalization=opt.l2_normalization, dim_z=opt.dim_z, fix_var=opt.fix_var)


    task_indices = eval(opt.task_indices)
    assert len(task_indices) == opt.output, 'len(task_indices) != opt.output'
    if opt.model.lower() == 'dssmvae':
        trainable.train(epochs=opt.epochs, optimizer='Adam', lr=opt.lr, lr_decay_rate=opt.lr_decay_rate, lr_decay_step=opt.lr_decay_step, task_indices=task_indices, num_augs=opt.num_augs, p=opt.p, use_reparam=opt.use_reparam, alpha=opt.alpha, perturb_type=opt.perturb_type)


if __name__ == '__main__':
    opt = parse_opt()
    mp.set_start_method('spawn')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if opt.output > 1:
        file_handler = logging.FileHandler(os.path.join(opt.data_dir, 'res_multi_task', f'{opt.model}_{opt.loss}_{opt.lr}_{opt.lr_decay_rate}_{opt.lr_decay_step}_{opt.S1}_{opt.mlp_layer}_{opt.smoothing_factor}_dimz{opt.dim_z}_fixvar{opt.fix_var}_num_augs{opt.num_augs}_p{opt.p}_repa{opt.use_reparam}_alpha{opt.alpha}_{opt.perturb_type}.log'), mode='a')
    else:
        file_handler = logging.FileHandler(os.path.join(opt.data_dir, f'{opt.model}_{opt.loss}_{opt.lr}_{opt.lr_decay_rate}_{opt.lr_decay_step}_{opt.S1}_{opt.mlp_layer}_{opt.smoothing_factor}_dimz{opt.dim_z}_fixvar{opt.fix_var}_num_augs{opt.num_augs}_p{opt.p}_repa{opt.use_reparam}_alpha{opt.alpha}_{opt.perturb_type}.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(InfoFilter())
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    set_seed(opt.seed)
    main()
