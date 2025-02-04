from model.EmbeddingModule import EmbeddingModule
from util.utils import get_activation, embedding_list_to_slate, slate_to_list
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F

sys.path.append('../')


class DSSMVAE(nn.Module):
    def __init__(self, user_datatypes, item_datatypes, user_dnn_size=(256, 128), l2_normalization=False,
                 similarity='dot',
                 item_dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False, dimensions=16, output=1,
                 loss='becloss', dim_z=256, fix_var=True):
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.user_datatypes = user_datatypes
        self.item_datatypes = item_datatypes
        self.l2_normalization = l2_normalization
        self.similarity = None
        self.loss = loss

        self.user_tower = TowerVAE(self.user_datatypes,
                                   self.user_dnn_size,
                                   self.dropout,
                                   activation=activation,
                                   use_senet=use_senet,
                                   dimensions=dimensions,
                                   output=output, dim_z=dim_z, fix_var=fix_var)
        self.item_tower = TowerVAE(self.item_datatypes,
                                   self.item_dnn_size,
                                   self.dropout,
                                   activation=activation,
                                   use_senet=use_senet,
                                   dimensions=dimensions,
                                   output=output, dim_z=dim_z, fix_var=fix_var)

    def forward(self, data, user_embeddings, item_embeddings, group=None, num_augs=8, p=0.2, use_reparam=True,
                perturb_type='dropout'):
        user_out, user_dnn_input = self.user_tower(data, user_embeddings)
        item_out, item_dnn_input = self.item_tower(data, item_embeddings)

        # augment the dnn_input for self-supervised task
        augs_user_out = []
        augs_item_out = []
        for _ in range(num_augs):
            if perturb_type == 'dropout':
                user_dnn_input_aug = F.dropout(user_dnn_input, p=p)
                item_dnn_input_aug = F.dropout(item_dnn_input, p=p)
            else:
                user_dnn_input_aug = self.user_tower.get_embedding_perturbed(data, user_embeddings, p, perturb_type)
                item_dnn_input_aug = self.item_tower.get_embedding_perturbed(data, item_embeddings, p, perturb_type)

            user_z, user_recon, user_z_mu, user_z_logvar = self.user_tower.vae_forward(user_dnn_input_aug, use_reparam)
            item_z, item_recon, item_z_mu, item_z_logvar = self.item_tower.vae_forward(item_dnn_input_aug, use_reparam)

            augs_user_out.append((user_dnn_input_aug, user_z, user_recon, user_z_mu, user_z_logvar))
            augs_item_out.append((item_dnn_input_aug, item_z, item_recon, item_z_mu, item_z_logvar))

        similarities, user_out, item_out = self.get_similarity(user_out, item_out, group)

        return similarities, user_out, item_out, augs_user_out, augs_item_out

    def predict(self, data, user_embeddings, item_embeddings, group=None, apply_sigmoid=True):
        user_out, _ = self.user_tower(data, user_embeddings)
        item_out, _ = self.item_tower(data, item_embeddings)
        similarities, user_out, item_out = self.get_similarity(user_out, item_out, group=group)

        if apply_sigmoid:
            similarities = torch.sigmoid(similarities)

        return similarities, user_out, item_out

    def get_similarity(self, user_out, item_out, group=None):
        if self.loss == 'jrc':
            user_out = user_out.view(user_out.shape[0], user_out.shape[1] * 2, user_out.shape[2] // 2)
            item_out = item_out.view(item_out.shape[0], item_out.shape[1] * 2, item_out.shape[2] // 2)
        if self.l2_normalization.lower() == 'true':
            user_out = user_out / \
                       torch.norm(user_out, p=2, dim=-1, keepdim=True)
            item_out = item_out / \
                       torch.norm(item_out, p=2, dim=-1, keepdim=True)

        similarities = (user_out * item_out).sum(dim=-1)

        return similarities, user_out, item_out


class TowerVAE(nn.Module):
    def __init__(self, datatypes, dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False, dimensions=16,
                 output=1, dim_z=256, fix_var=True):
        super().__init__()

        self.fix_var = fix_var

        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(
            datatypes, use_senet, dimensions=dimensions)

        dim_emb = self.embeddings.dim + self.embeddings.embedding_dim
        if self.fix_var:
            self.fc_mu = nn.Linear(dim_emb, dim_z)
        else:
            self.fc_mu = nn.Linear(dim_emb, dim_z)
            self.fc_logvar = nn.Linear(dim_emb, dim_z)

        self.decoder = nn.Linear(dim_z, dim_emb)

        for _ in range(output):
            input_dims = dim_z
            layers = []
            for dim in dnn_size:
                layers.append(nn.Linear(input_dims, dim))
                layers.append(nn.Dropout(dropout))
                layers.append(get_activation(activation))
                input_dims = dim
            self.dnns.append(nn.Sequential(*layers))

    def get_embedding(self, x, pretrain_embeddings):
        dnn_input = self.embeddings(x)

        if pretrain_embeddings is not None:
            dnn_input = torch.cat((dnn_input, pretrain_embeddings), dim=1)

        return dnn_input

    def get_embedding_perturbed(self, x, pretrain_embeddings, p, perturb_type):
        dnn_input = self.embeddings.forward_perturbed(x, p, perturb_type)

        if pretrain_embeddings is not None:
            dnn_input = torch.cat((dnn_input, pretrain_embeddings), dim=1)

        return dnn_input

    def encode(self, x):
        if self.fix_var:
            z_mu = self.fc_mu(x)
            z_logvar = torch.zeros_like(z_mu)
        else:
            z_mu = self.fc_mu(x)
            z_logvar = self.fc_logvar(x)
        return z_mu, z_logvar

    def decode(self, z):
        recon = self.decoder(z)
        return recon

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pretrain_embeddings):
        dnn_input = self.get_embedding(x, pretrain_embeddings)

        # get z_mu
        z_mu = self.fc_mu(dnn_input)

        # downstream prediction task
        results = []
        for dnn in self.dnns:
            results.append(dnn(z_mu))  # using the z_mu for predictive task
        results = torch.stack(results, dim=1)
        return results, dnn_input

    def vae_forward(self, dnn_input, use_reparam=True):
        z_mu, z_logvar = self.encode(dnn_input)
        if use_reparam:
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = z_mu

        recon = self.decode(z)

        return z, recon, z_mu, z_logvar



