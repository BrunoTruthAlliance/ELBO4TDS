
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

import numpy as np


class EmbeddingModule(nn.Module):
    def __init__(self, datatypes, use_se_net, dimensions=16):
        super().__init__()
        self.dim = 0
        self.embedding_dim = 0
        self.embs = nn.ModuleList()
        self.datatypes = datatypes
        self.use_se_net = use_se_net
        self.num = 0
        for datatype in datatypes:
            if datatype['type'] == 'SparseEncoder' or datatype['type'] == 'BucketSparseEncoder':
                self.embs.append(nn.Embedding(datatype['length'], dimensions))
                self.dim += dimensions
                self.num += 1
            if datatype['type'] == 'MultiSparseEncoder':
                self.embs.append(nn.EmbeddingBag(
                    datatype['length'], dimensions, mode='mean', padding_idx=0))
                self.dim += dimensions
                self.num += 1
            elif datatype['type'] == 'DenseEncoder':
                self.embs.append(nn.Embedding(
                    len(datatype['length']), dimensions))
                self.dim += dimensions
                self.num += 1
            elif datatype['type'] == 'text':
                self.embedding_dim += 300
                # self.dense_num += 1
        # if self.use_se_net != False:
        #     if self.use_se_net == 'LightSE':
        #         self.se_net = LightSE(self.num)
        #     else:
        #         self.se_net = SENet(self.num)

    def run_emb(self, emb, input):
        return emb(input)

    def forward(self, x):
        emb_output = []
        se_net_input = []
        for index in range(len(self.datatypes)):
            datatype = self.datatypes[index]
            if datatype['type'] == 'MultiSparseEncoder':
                vec = self.embs[index](
                    x[:, datatype['index']: datatype['index'] + datatype['size']].int())
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
                emb_output.append(vec)
            elif datatype['type'] == 'SparseEncoder':
                vec = self.embs[index](x[:, datatype['index']].int())
                emb_output.append(vec)
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
            elif datatype['type'] == 'DenseEncoder':
                thersholds = torch.tensor(datatype['length']).to(x.device)
                input_data = torch.bucketize(
                    x[:, datatype['index']], thersholds)
                vec = self.embs[index](input_data)
                emb_output.append(vec)
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))

        if len(se_net_input) != 0 and self.use_se_net != False:
            se_net_output = self.se_net(torch.cat(se_net_input, dim=1))
            for i in range(self.num):
                emb_output[i] = emb_output[i] * se_net_output[-1, i:i + 1]

        output = torch.cat(emb_output, dim=1)

        return output.float()

    def forward_perturbed(self, x, p, perturb_type):
        emb_output = []
        se_net_input = []
        for index in range(len(self.datatypes)):
            datatype = self.datatypes[index]

            # Create a Bernoulli distribution
            mask = np.random.binomial(n=1, p=p, size=(x.shape[0], 1))
            mask = torch.tensor(mask).float().to(x.device)
            # print(f'mask shape: {mask.shape}: {mask}')

            if datatype['type'] == 'MultiSparseEncoder':
                vec = self.embs[index](
                    x[:, datatype['index']: datatype['index'] + datatype['size']].int())
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
                emb_output.append(vec)
            elif datatype['type'] == 'SparseEncoder':
                vec = self.embs[index](x[:, datatype['index']].int())

                if 'dropout' in perturb_type:
                    vec = (1 - mask) * vec + mask * F.dropout(vec, p=p)

                emb_output.append(vec)
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))

            elif datatype['type'] == 'DenseEncoder':
                thersholds = torch.tensor(datatype['length']).to(x.device)
                input_data = torch.bucketize(x[:, datatype['index']], thersholds)

                if 'bucket' in perturb_type:
                    # print('bucket!!!')
                    eps = torch.randint(low=-2, high=3, size=input_data.shape).float().to(x.device)
                    eps = mask.squeeze() * eps
                    input_data = torch.clamp(input_data + eps, min=0, max=thersholds.shape[0] - 1).long()

                vec = self.embs[index](input_data)
                emb_output.append(vec)
                if self.use_se_net != False:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))

        if len(se_net_input) != 0 and self.use_se_net != False:
            se_net_output = self.se_net(torch.cat(se_net_input, dim=1))
            for i in range(self.num):
                emb_output[i] = emb_output[i] * se_net_output[-1, i:i + 1]

        output = torch.cat(emb_output, dim=1)

        return output.float()
