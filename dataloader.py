import time

import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

topk = 4

#通过余弦相似构建相似图和不相似图返回带权邻接矩阵
def construt_graph(features, labels, nnodes, nclasses):
    dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1)
    labels = labels.view(-1, 1)
    data = torch.cat((features, labels), dim=1)
    data = data.double()
    similarity_matrix = torch.zeros((nclasses, nclasses))
    for i in range(0, nclasses):
        x = data[data[:, -1] == i]
        mean_v = torch.mean(x[:, :-1], dim=0, keepdim=True)
        indices = (labels == i).nonzero().squeeze()
        for j in range(0, nclasses):
            x2 = data[data[:, -1] == j]
            sim = cosine_similarity(x, x2)

            sim = torch.tensor(sim)
            avg = torch.mean(sim)
            similarity_matrix[i - 1][j - 1] = avg
            similarity_matrix[j - 1][i - 1] = avg
        data[indices, :-1] = mean_v
    nan_indices = torch.isnan(data)
    # 将 NaN 值替换为 0
    data[nan_indices] = 0
    l_proto = data[:, :-1]

    similarity_matrix.fill_diagonal_(0)
    sorted_sim, sorted_sim_ind = torch.sort(similarity_matrix, dim=1, descending=False)
    class_sims_idx = {}
    for idx in range(nclasses):
        class_sims_idx[idx] = {}
        class_sims_idx[idx]['sim_class_idx2indices'] = sorted_sim_ind[idx].clone().detach().type(
            torch.long)
        class_sims_idx[idx]['sim_class_val'] = sorted_sim[idx]
        class_sims_idx[idx]['sim_class_val'][0] = 1
    weights = torch.zeros((nnodes, nnodes))
    idx_hm, idx_ht = [], []
    k1 = 3
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], -(k1 + 1))[-(k1 + 1):]
        idx_hm.append(idx)
    counter_hm = 0
    edges_hm = 0
    for i, v in enumerate(idx_hm):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights[i][Nv] = dist[i][Nv]
                if weights[Nv][i] == 0:
                    edges_hm += 1
                if weights[Nv][i] == 0 and labels[Nv] != labels[i]:
                    counter_hm += 1
    return weights, class_sims_idx, l_proto

def get_train_test_indices(y, train_ratio=0.7, random_state=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
    return train_idx, test_idx

def load_data():


    data = np.loadtxt('./data/github.csv', dtype=float)
    data = np.unique(data, axis=0)
    label = data[:, -1]
    feature = data[:, :-1]
    label = torch.tensor(label-1)

    feature = torch.tensor(feature)
    [nnodes, n_feat] = feature.shape
    n_class = len(torch.unique(label))

    weights, class_sims_idx, l_proto = construt_graph(feature, label, nnodes, n_class)
    # DGA-2000:
    idx_train = np.load('./data/github/train_mask.npy')
    idx_test = np.load('./data/github/test_mask.npy')
    # DGA-200:
    # idx_train = np.load('./data/ieee/train_mask.npy')
    # idx_test = np.load('./data/ieee/test_mask.npy')

    return feature, weights, label, n_feat, n_class, class_sims_idx, l_proto, idx_train,  idx_test
