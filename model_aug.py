import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import layers
import random
import os
import math

from torch.nn import Parameter
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
import torch.nn.init as init


from utils_aug import *
from loss_update import *



class GCL(nn.Module):
    def __init__(self, device, nlayers, n_input, n_hid, n_output, droprate, sparse, batch_size, enable_bias):
        super(GCL, self).__init__()

        #基于GCN学习节点表示
        self.embed1 = GCN(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        self.embed2 = GCN(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        self.device = device

    def RINCE(self, x1, a1, x2, a2, criterion, labels, train_mask):#加了train_mask
        emb1 = self.embed1(x1, a1)
        emb2 = self.embed2(x2, a2)

        loss = criterion(emb1[train_mask], labels[train_mask], train_mask)

        return emb1, emb2, loss


    def get_emb1(self, x1, a1):
        emb1 = self.embed1(x1, a1)
        return emb1

    def get_emb2(self, x, a):
        emb2 = self.embed2(x, a)
        return emb2

class GCN(nn.Module):
    def __init__(self, nlayers, n_input, n_hid, n_output, droprate, enable_bias):
        super(GCN, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = nlayers
        if nlayers >= 2:
            self.graph_convs.append(layers.GraphConv(in_features=n_input, out_features=n_hid, bias=enable_bias))
            for k in range(1, nlayers-1):
                self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
            self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_output, bias=enable_bias))
        else:
            self.graph_convs.append(layers.GraphConv(in_features=n_input, out_features=n_output, bias=enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x, filter):
        if self.K >= 2:
            for k in range(self.K-1):
                x = self.graph_convs[k](x, filter)
                x = self.relu(x)
                x = self.dropout(x)
            x = self.graph_convs[-1](x, filter)
        else:
            x = self.graph_convs[0](x, filter)

        return x


class GCNEnc(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # 定义两个不同形状的权重矩阵
        self.weight1 = nn.Parameter(torch.Tensor(in_features, hidden_features))
        self.weight2 = nn.Parameter(torch.Tensor(hidden_features, out_features))
        self.bias1 = torch.nn.Parameter(torch.Tensor(hidden_features))
        self.bias2 = torch.nn.Parameter(torch.Tensor(out_features))
        self.relu = nn.ReLU()


        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.uniform_(self.bias1, -1e-2, 1e-2)
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        init.uniform_(self.bias2, -1e-2, 1e-2)

    def forward(self, x, adj):
        x = x.to(torch.float32)
        adj = adj.to(torch.float32)
        self.weight1 = self.weight1.to(torch.float32)
        out = adj @ (x @ self.weight1) + self.bias1  # shape: [N, hidden]
        out = self.relu(out)
        out = adj @ (out @ self.weight2) + self.bias2  # shape: [N, out]
        return out

class GCN_Classifier(nn.Module):
    def __init__(self, ninput, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.mlp = nn.Linear(ninput, nclass)
        self.dropout = dropout
        self.norm = nn.LayerNorm(nclass)  # 添加 LayerNorm

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        x = self.norm(x)  # 归一化 logits
        return x