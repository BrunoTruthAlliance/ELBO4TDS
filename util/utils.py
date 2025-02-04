import datetime
import json
import os
import pickle
import sys
import torch
import random
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn import preprocessing
from torch import nn
from tqdm import tqdm

sys.path.append('../')


def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        else:
            return getattr(nn, activation)()
    else:
        return


def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError(
            "optimizer={} is not supported.".format(optimizer))
    return optimizer


def list_to_slate(true, pred, groups):
    # 将形状为(len)或为(len,1)的tensor转化为(usernum,max_len)
    # groups是一个列表,每个值表示user交互的item的长度
    max_len = max(groups)
    pred = pred.squeeze()
    true = true.squeeze()
    grouped_true = torch.full((len(groups), max_len), -1.0)
    grouped_pred = torch.full((len(groups), max_len), -1.0)

    start_idx = 0

    for i, length in enumerate(groups):
        end_idx = start_idx + length
        grouped_true[i, :length] = true[start_idx:end_idx].view(-1)
        grouped_pred[i, :length] = pred[start_idx:end_idx].view(-1)
        start_idx = end_idx

    return grouped_true, grouped_pred


def embedding_list_to_slate(user_embedding, item_embedding, groups):
    # u,i:[batch*itemnum,dim]
    # return u[batch,dim] i[batch,itemnum,dim]
    max_itemnum = max(groups)
    batch = len(groups)
    dim = user_embedding.shape[1]
    new_user_embedding = torch.zeros(batch, dim).to(user_embedding.device)
    new_item_embedding = torch.zeros(
        batch, max_itemnum, dim).to(item_embedding.device)

    start_idx = 0
    for user_id in range(batch):
        end_idx = start_idx + groups[user_id]
        new_user_embedding[user_id] = user_embedding[start_idx]
        new_item_embedding[user_id, :groups[user_id]
                           ] = item_embedding[start_idx:end_idx]
        start_idx = end_idx
    return new_user_embedding, new_item_embedding


def slate_to_list(pred, group):
    embeddings_list = []
    for id in range(len(group)):
        itemnum = group[id]

        actual_item_embedding = pred[id, :itemnum]

        embeddings_list.append(actual_item_embedding)

    restored_item_embedding = torch.cat(embeddings_list, dim=0)
    return restored_item_embedding


def sample_train(user_id, user_list, maxlen):
    """_summary_

    Args:
        user_id (list): _description_
        user_list (list): _description_
        maxlen (int): _description_

    Returns:
        tuple: _description_
    """
    # seq 输入序列 每个user选取user_list[0: -1]，并将长度补足为maxlen
    # pos 正样本序列 每个user选取label为1的样本，并将长度补足为maxlen
    # neg 负样本序列 每个user选取label为0的样本，并将长度补足为maxlen
    # 长度不够的在前面补0
    # print(user_list)
    # sys.exit()

    seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    nxt = user_list[-1]
    idx = maxlen - 1
    pos_idx = idx
    neg_idx = idx

    for i in reversed(user_list[:-1]):
        seq[idx] = i[0]
        if nxt[1] == 1:
            pos[pos_idx] = nxt[0]
            pos_idx -= 1
        else:
            neg[neg_idx] = nxt[0]
            neg_idx -= 1
        nxt = i
        idx -= 1
        if idx == -1:
            break

    return (user_id, seq, pos, neg)


def sample_test(user_id, user_list, maxlen):
    """_summary_

    Args:
        user_id (list): _description_
        user_list (list): _description_
        maxlen (int): _description_

    Returns:
        tuple: _description_
    """
    seq = np.zeros([maxlen], dtype=np.int32)
    item_list = []
    idx = maxlen - 1

    for i in reversed(user_list):
        if idx >= 0:
            seq[idx] = i[0]
            idx -= 1
        item_list.append(i)

    return (user_id, seq, item_list)


def data_partition(feature, label, maxlen, user_id, item_id, u_idx, i_idx, lab_idx, train=True):
    assert user_id != None and item_id != None, "SASRecTrainable feature name error 1"
    assert u_idx != -1 and i_idx != -1, "SASRecTrainable feature name error 2"

    group_data = defaultdict(list)
    for uid, iid, lab in zip(*(feature[:, u_idx], feature[:, i_idx], label[:, lab_idx])):
        group_data[uid.item()].append((iid.item(), lab.item()))

    one_batch = []
    for user in group_data:
        if len(group_data[user]) < 3:
            continue

        if train == True:
            one_batch.append(sample_train(user, group_data[user], maxlen))
        else:
            one_batch.append(sample_test(user, group_data[user], maxlen))

    return zip(*one_batch)


def mask_train(item_num, user_id, user_list, maxlen, rng, mask_prob):
    seq = user_list
    tokens = []
    labels = []
    rating_pos = []
    rating = []
    mask_token = item_num + 1
    for item in seq:
        prob = rng.random()
        if prob < mask_prob:
            prob /= mask_prob

            if prob < 0.8:
                tokens.append(mask_token)
            elif prob < 0.9:
                tokens.append(rng.randint(1, item_num))
            else:
                tokens.append(item[0])

            labels.append(item[0])
        else:
            tokens.append(item[0])
            labels.append(0)

        rating_pos.append(item[0])
        rating.append(item[1])

    tokens = tokens[-maxlen:]
    labels = labels[-maxlen:]

    mask_len = maxlen - len(tokens)

    tokens = [0] * mask_len + tokens
    labels = [0] * mask_len + labels

    tokens = np.array(tokens, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    rating_pos = np.array(rating_pos, dtype=np.int32)
    labels_rat = np.array([-1] * (item_num + 1), dtype=np.int32)
    labels_rat[rating_pos] = rating

    return (tokens, labels, labels_rat, user_list)


def mask_test(item_num, user_id, user_list, maxlen, mask_prob):
    seq = []
    answer = []
    negs = []
    mask_token = item_num + 1

    if user_list[-1][1] == 1:
        answer.append(user_list[-1][0])
    else:
        negs.append(user_list[-1][0])

    for item in user_list[:-1]:
        seq.append(item[0])
        if item[1] == 1:
            answer.append(item[0])
        else:
            negs.append(item[0])

    candidates = answer + negs
    labels = [1] * len(answer) + [0] * len(negs)

    seq = seq[:-1] + [mask_token]
    seq = seq[-maxlen:]
    padding_len = maxlen - len(seq)
    seq = [0] * padding_len + seq
    return (seq, candidates, labels)


def data_mask(item_num, feature, label, maxlen, user_id, item_id, u_idx, i_idx, lab_idx, mask_prob=0.2,
              random_seed=42, train=True):
    assert user_id != None and item_id != None, "BERT4RecTrainable feature name error 1"
    assert u_idx != -1 and i_idx != -1, "BERT4RecTrainable feature name error 2"

    rng = random.Random(random_seed)

    group_data = defaultdict(list)
    for uid, iid, lab in zip(*(feature[:, u_idx], feature[:, i_idx], label[:, lab_idx])):
        group_data[uid.item()].append((iid.item(), lab.item()))

    one_batch = []
    for user in group_data:
        if len(group_data[user]) < 3:
            continue

        if train == True:
            one_batch.append(mask_train(
                item_num, user, group_data[user], maxlen, rng, mask_prob))
        else:
            one_batch.append(mask_test(
                item_num, user, group_data[user], maxlen, mask_prob))

    return zip(*one_batch)


def binary_label_smoothing(label, smoothing_factor=0.2):
    smooth_label = label * (1 - smoothing_factor) + 0.5 * smoothing_factor
    return smooth_label


def binary_label_smoothing_with_p(label, smoothing_factor=0.2, p=0.2):
    smooth_indicator = torch.empty_like(label).bernoulli_(p)
    smooth_label = smooth_indicator * (label * (1 - smoothing_factor) + 0.5 * smoothing_factor) + (1 - smooth_indicator) * label
    return smooth_label


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, path)