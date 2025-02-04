from model.EmbeddingModule import EmbeddingModule
from util.utils import get_activation, embedding_list_to_slate, slate_to_list
import sys
import torch.nn as nn
import torch
sys.path.append('../')


class DSSM(nn.Module):
    def __init__(self, user_datatypes, item_datatypes, user_dnn_size=(256, 128), l2_normalization=False, similarity='dot',
                 item_dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False, dimensions=16, output=1, loss='becloss'):
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.user_datatypes = user_datatypes
        self.item_datatypes = item_datatypes
        self.l2_normalization = l2_normalization
        self.similarity = None
        self.loss = loss

        # Tower can be reused
        self.user_tower = Tower(self.user_datatypes,
                                self.user_dnn_size,
                                self.dropout,
                                activation=activation,
                                use_senet=use_senet,
                                dimensions=dimensions,
                                output=output)
        self.item_tower = Tower(self.item_datatypes,
                                self.item_dnn_size,
                                self.dropout,
                                activation=activation,
                                use_senet=use_senet,
                                dimensions=dimensions,
                                output=output)

    def forward(self, data, user_embeddings, item_embeddings, group=None):
        user_embeddings = self.user_tower(data, user_embeddings)
        item_embeddings = self.item_tower(data, item_embeddings)

        if self.l2_normalization.lower() == 'true':
            user_embeddings = user_embeddings / \
                torch.norm(user_embeddings, p=2, dim=-1, keepdim=True)
            item_embeddings = item_embeddings / \
                torch.norm(item_embeddings, p=2, dim=-1, keepdim=True)

        similarities = (user_embeddings * item_embeddings).sum(dim=-1)

        return similarities, user_embeddings, item_embeddings

    def predict(self, data, user_embeddings, item_embeddings, group=None):
        similarities, user_embeddings, item_embeddings = self(data, user_embeddings, item_embeddings, group=group)

        return torch.sigmoid(similarities), user_embeddings, item_embeddings


class Tower(nn.Module):
    def __init__(self, datatypes, dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False, dimensions=16, output=1):
        super().__init__()
        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(
            datatypes, use_senet, dimensions=dimensions)

        for _ in range(output):
            input_dims = self.embeddings.dim + self.embeddings.embedding_dim
            layers = []
            for dim in dnn_size:
                layers.append(nn.Linear(input_dims, dim))
                layers.append(nn.Dropout(dropout))
                layers.append(get_activation(activation))
                input_dims = dim
            self.dnns.append(nn.Sequential(*layers))

    def forward(self, x, pretrain_embeddings):
        dnn_input = self.embeddings(x)

        if pretrain_embeddings is not None:
            dnn_input = torch.cat((dnn_input, pretrain_embeddings), dim=1)

        results = []
        # 处理多任务
        for dnn in self.dnns:
            results.append(dnn(dnn_input))
        results = torch.stack(results, dim=1)
        return results

    def run_dnn(self, dnn, dnn_input):
        return dnn(dnn_input)
