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


from utils_aug import *
from loss import *



class GCL(nn.Module):
    def __init__(self, device, nlayers, n_input, n_hid, n_output, droprate, sparse, batch_size, enable_bias):
    #def __init__(self, device, args, feature, n_input, n_hid, n_output, batch_size):
        super(GCL, self).__init__()

        #基于GCN学习节点表示
        self.embed1 = GCN(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        self.embed2 = GCN(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        self.linear = nn.Linear(n_hid, n_output)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.batch_size = batch_size
        self.device = device

        nn.init.xavier_normal_(self.linear.weight.data)

    def RINCE(self, x1, a1, x2, a2, criterion, labels):
        emb1 = self.embed1(x1, a1)
        emb2 = self.embed2(x2, a2)

        loss = criterion(emb1, emb2, labels)

        return loss


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

class GCN_Classifier(nn.Module):
    def __init__(self, ninput, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.mlp = nn.Linear(ninput, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x