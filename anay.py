# !/usr/bin/python
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel, BertConfig, AutoModel, AutoConfig, AutoTokenizer
from transformers import AdamW
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
import math


# 设置参数及文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
max_seq_length = 128  # 输入文本最大长度
batch_size = 1  # 训练时每个batch中的样本数
file_name = 'baseline'  # 指定输出文件的名字
model_name_or_path = './pretrain_models/ernie_gram'  # 预训练模型权重载入路径




class InputExample(object):
    def __init__(self, s1, s2, label=None):
        self.s1 = s1
        self.s2 = s2
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label

                 ):
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    # 将文本输入样例，转换为数字特征，用于模型计算
    features = []
    for example_index, example in enumerate(examples):

        s1 = tokenizer.tokenize(example.s1)
        s2 = tokenizer.tokenize(example.s2)
        _truncate_seq_pair(s1, s2, max_seq_length)

        choices_features = []

        tokens = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
        segment_ids = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids) + 3
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                choices_features=choices_features,
                label=label
            )
        )
    return features


def select_field(features, field):
    return [
        feature.choices_features[field] for feature in features
    ]


class NeuralNet(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_class)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):
        output = self.bert(input_ids, token_type_ids=segment_ids,
                                                                attention_mask=input_mask)

        last_hidden = output.last_hidden_state
        all_hidden_states = output.hidden_states
        # last_hidden, pooler_output, all_hidden_states = self.bert(input_ids, token_type_ids=segment_ids,
        #                                                           attention_mask=input_mask)

        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        f = torch.mean(last_hidden, 1)
        feature = torch.cat((feature, f), 1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)


def similarity(s1,s2, model, tokenizer):
    test_examples = [InputExample(s1=s1, s2=s2, label= None)]
    test_features = convert_examples_to_features(
    test_examples, tokenizer, max_seq_length, True)
    test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
    test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
    test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
    test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = tuple(t.cuda() for t in batch)
            x_ids, x_mask, x_sids = batch
            with autocast():
                y_pred = model(x_ids, x_mask, x_sids).detach()
                sim = float(F.softmax(y_pred, dim=1).cpu().numpy()[0][1])
                return sim


#s1 = '今天会不会下大雨哦'
#s2 = '今天的天气很好哦'
#
#print(similarity(s1,s2))