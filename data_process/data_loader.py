from datasets import concatenate_datasets, load_dataset
from os import path
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import partial

CACHE_DIR = 'catch'


def custom_collate_fn(batch, model):
    '''
    输出内容:
    result_tensor:[items,features]为整个读取到的数据(包括label等)
    labels_tensor:[items,task]
    list(sizes):每个user交互过的item长度,方便以后对一个user的交互历史进行处理
    user_embeddings_tensor, item_embeddings_tensor:amazon_reviews数据集的文本特征提前编码为embedding,其他数据集这两个值为None
    '''
    result_tensors, labels, user_text_list, item_text_list, sizes = zip(
        *batch)
    result_tensor = torch.cat(result_tensors, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    user_embeddings_tensor = None
    item_embeddings_tensor = None
    if user_text_list[0] is not None:
        text_list = []
        for sublist in user_text_list:
            text_list.extend(sublist)
        user_embeddings_tensor = torch.tensor(model.encode(text_list))
    if item_text_list[0] is not None:
        text_list = []
        for sublist in item_text_list:
            text_list.extend(sublist)
        item_embeddings_tensor = torch.tensor(model.encode(text_list))

    return result_tensor, labels_tensor, list(sizes), user_embeddings_tensor, item_embeddings_tensor


def get_data_feature(dir):
    with open(path.join(dir, 'feature.json'), 'r') as file:
        feature_data = json.load(file)
    item_feature = [
        entry for entry in feature_data if entry["belongs"] == "item"]
    user_feature = [
        entry for entry in feature_data if entry["belongs"] == "user"]
    label_feature = [
        entry for entry in feature_data if entry["belongs"] == "label"]
    return user_feature, item_feature, label_feature, feature_data


class CustomDataset(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 读取到数据一行是user和他所有交互过的item的记录,每个item中间用\x01隔开,该函数将这一行数据转化为多行,每行是一个user和item交互记录
        example = self.dataset[idx]

        idx = []
        user_text_list = None
        item_text_list = None
        rows = [value.split('\x01') for value in example.values()]
        # length是这个user交互过多少个item
        length = len(rows[0])
        for item in self.feature_data:
            # 当item是text时,需要对文本文件进行处理
            if item['type'] == 'text':
                if item['belongs'] == 'item':
                    item_text_list = rows[item['index']]
                elif item['belongs'] == 'user':
                    user_text_list = rows[item['index']]
                rows[item['index']] = [0] * length
            if item['belongs'] == 'label':
                idx.append(item['index'])

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        return result_tensor, result_tensor[:, idx], user_text_list, item_text_list, length


def data_loader(data_dir, batch_size, data_loader_worker):
    model = None
    # 该模型主要为了处理amazon reviews的文本数据
    if data_dir.split('/')[1] == 'amazon_reviews':
        print('use average_word_embeddings_glove')
        model = SentenceTransformer(
            model_name_or_path='sentence-transformers--average_word_embeddings_glove', local_files_only=True)
        model = model.to('cpu')

    dataset = load_dataset('parquet', data_files={
        'train': path.join(data_dir, 'train', '*', '*.parquet'),
        'val': path.join(data_dir, 'val', '*', '*.parquet'),
        'test': path.join(data_dir, 'test', '*', '*.parquet')
    }, cache_dir=CACHE_DIR)

    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    user_feature, item_feature, label_feature, feature_data = get_data_feature(
        data_dir)

    train_dataset = CustomDataset(train_dataset, feature_data)
    val_dataset = CustomDataset(val_dataset, feature_data)
    test_dataset = CustomDataset(test_dataset, feature_data)
    collate_fn_with_args = partial(
        custom_collate_fn, model=model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    return train_dataloader, val_dataloader, test_dataloader, user_feature, item_feature, label_feature


def custom_collate_fn_irm(batch, model):
    '''
    输出内容:
    result_tensor:[items,features]为整个读取到的数据(包括label等)
    labels_tensor:[items,task]
    list(sizes):每个user交互过的item长度,方便以后对一个user的交互历史进行处理
    user_embeddings_tensor, item_embeddings_tensor:amazon_reviews数据集的文本特征提前编码为embedding,其他数据集这两个值为None
    '''
    result_tensors, labels, user_text_list, item_text_list, sizes, envs = zip(*batch)
    result_tensor = torch.cat(result_tensors, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    envs_tensor = torch.cat(envs, dim=0)
    user_embeddings_tensor = None
    item_embeddings_tensor = None
    if user_text_list[0] is not None:
        text_list = []
        for sublist in user_text_list:
            text_list.extend(sublist)
        user_embeddings_tensor = torch.tensor(model.encode(text_list))
    if item_text_list[0] is not None:
        text_list = []
        for sublist in item_text_list:
            text_list.extend(sublist)
        item_embeddings_tensor = torch.tensor(model.encode(text_list))

    return result_tensor, labels_tensor, list(sizes), user_embeddings_tensor, item_embeddings_tensor, envs_tensor


class CustomDatasetEnv(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 读取到数据一行是user和他所有交互过的item的记录,每个item中间用\x01隔开,该函数将这一行数据转化为多行,每行是一个user和item交互记录
        example = self.dataset[idx]

        idx = []
        user_text_list = None
        item_text_list = None
        rows = [value.split('\x01') for k, value in example.items() if k != 'env']
        # length是这个user交互过多少个item
        length = len(rows[0])
        for item in self.feature_data:
            # 当item是text时,需要对文本文件进行处理
            if item['type'] == 'text':
                if item['belongs'] == 'item':
                    item_text_list = rows[item['index']]
                elif item['belongs'] == 'user':
                    user_text_list = rows[item['index']]
                rows[item['index']] = [0] * length
            if item['belongs'] == 'label':
                idx.append(item['index'])

        envs = torch.tensor([example['env']] * length)

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        return result_tensor, result_tensor[:, idx], user_text_list, item_text_list, length, envs


def data_loader_env(data_dir, batch_size, data_loader_worker):
    model = None
    # # 该模型主要为了处理amazon reviews的文本数据
    # if data_dir.split('/')[1] == 'amazon_reviews':
    #     print('use average_word_embeddings_glove')
    #     model = SentenceTransformer(
    #         model_name_or_path='sentence-transformers--average_word_embeddings_glove', local_files_only=True)
    #     model = model.to('cpu')

    dataset = load_dataset('parquet', data_files={
        'train_0': path.join(data_dir, 'train', 'time_0', '*.parquet'),
        'train_1': path.join(data_dir, 'train', 'time_1', '*.parquet'),
        'train_2': path.join(data_dir, 'train', 'time_2', '*.parquet'),
        'train_3': path.join(data_dir, 'train', 'time_3', '*.parquet'),
        'train_4': path.join(data_dir, 'train', 'time_4', '*.parquet'),
        'train_5': path.join(data_dir, 'train', 'time_5', '*.parquet'),
        'train_6': path.join(data_dir, 'train', 'time_6', '*.parquet'),
        'train_7': path.join(data_dir, 'train', 'time_7', '*.parquet'),
        'val': path.join(data_dir, 'val', '*', '*.parquet'),
        'test': path.join(data_dir, 'test', '*', '*.parquet')
    }, cache_dir=CACHE_DIR)

    train_dataset_0 = dataset['train_0']
    train_dataset_1 = dataset['train_1']
    train_dataset_2 = dataset['train_2']
    train_dataset_3 = dataset['train_3']
    train_dataset_4 = dataset['train_4']
    train_dataset_5 = dataset['train_5']
    train_dataset_6 = dataset['train_6']
    train_dataset_7 = dataset['train_7']

    train_dataset_env_0 = concatenate_datasets([train_dataset_0, train_dataset_1])
    train_dataset_env_1 = concatenate_datasets([train_dataset_2, train_dataset_3])
    train_dataset_env_2 = concatenate_datasets([train_dataset_4, train_dataset_5])
    train_dataset_env_3 = concatenate_datasets([train_dataset_6, train_dataset_7])

    train_dataset_env_0 = train_dataset_env_0.add_column('env', [0] * len(train_dataset_env_0))
    train_dataset_env_1 = train_dataset_env_1.add_column('env', [1] * len(train_dataset_env_1))
    train_dataset_env_2 = train_dataset_env_2.add_column('env', [2] * len(train_dataset_env_2))
    train_dataset_env_3 = train_dataset_env_3.add_column('env', [3] * len(train_dataset_env_3))

    train_dataset = concatenate_datasets(
        [train_dataset_env_0, train_dataset_env_1, train_dataset_env_2, train_dataset_env_3])
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    user_feature, item_feature, label_feature, feature_data = get_data_feature(
        data_dir)

    train_dataset = CustomDatasetEnv(train_dataset, feature_data)
    val_dataset = CustomDataset(val_dataset, feature_data)
    test_dataset = CustomDataset(test_dataset, feature_data)

    collate_fn_with_args_irm = partial(custom_collate_fn_irm, model=model)

    collate_fn_with_args = partial(custom_collate_fn, model=model)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args_irm,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    return train_dataloader, val_dataloader, test_dataloader, user_feature, item_feature, label_feature