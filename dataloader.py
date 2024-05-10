import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch.nn.functional as F

topk = 3

#通过余弦相似构建相似图和不相似图返回带权邻接矩阵
def construt_graph(features, labels, nnodes):

    #dist = cosine_similarity(features)
    #计算节点间相似性
    dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1)
    weights = torch.zeros((nnodes, nnodes))
    idx_hm, idx_ht = [], []
    #获取前K个相似节点下标
    k1 = 3
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], -(k1 + 1))[-(k1 + 1):]
        idx_hm.append(idx)

    counter_hm = 0
    edges_hm = 0
    #计算相似图的异质性并给邻接矩阵赋值
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

    return weights


#获取 验证集和测试集
def get_split(features, labels):
    n = len(torch.unique(labels))
    labels = labels.unsqueeze(1)
    num_samples = features.shape[0]
    data = torch.cat((features, labels), dim=1)
    trains, tests = [], []
    dx_train, dx_test, dy_train, dy_test = [], [], [], []
    for j in range(10):
        indices_test, data_test = [], []
        for i in range(0, n):
            temp = data[data[:, -1] == i]
            X = temp[:, :-1]
            y = temp[:, -1]
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            # 当设置shuffle=False时，会出现训练集和测试集重复数据，甚至测试集中重复数据次数更多
            X_train, X_test, y_train, y_test = train_test_split(X[indices], y[indices], test_size=0.3, random_state=42)
            data_test.append(X_test)
            dy_train.append(y_train.unsqueeze(0))
            dx_train.append(X_train.unsqueeze(0))
            dy_test.append(y_test.unsqueeze(0))
            dx_test.append(X_test.unsqueeze(0))

        data_test = torch.cat(data_test, dim=0)
        for j in range(len(data_test)):
            for i in range(num_samples):
                if i not in indices_test:
                    if all(features[i].eq(data_test[j])):
                        indices_test.append(i)
                        break

        indices_train = torch.zeros(num_samples, dtype=torch.bool)  # torch.Size([n])
        indices_train.fill_(True)
        indices_train[indices_test] = False
        indices_train = np.where(indices_train)[0]
        indices_test = np.array(indices_test)

        indices_train = torch.tensor(indices_train)
        indices_test = torch.tensor(indices_test)

        trains.append(indices_train.unsqueeze(1))
        tests.append(indices_test.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    test_mask_all = torch.cat(tests, 1)

    X_train_mask = torch.cat(dx_train, 1)
    y_train_mask = torch.cat(dy_train, 1)
    X_test_mask = torch.cat(dx_test, 1)
    y_test_mask = torch.cat(dy_test, 1)
    np.save('./data/ieee/X_train_mask.npy', X_train_mask)
    np.save('./data/ieee/y_train_mask.npy', y_train_mask)
    np.save('./data/ieee/X_test_mask.npy', X_test_mask)
    np.save('./data/ieee/y_test_mask.npy', y_test_mask)

    np.save('./data/ieee/train_mask.npy', train_mask_all)
    np.save('./data/ieee/test_mask.npy', test_mask_all)


    return train_mask_all, test_mask_all

def load_data():

    # data = np.loadtxt('./data/ieee.csv', dtype=float)
    data = np.loadtxt('./data/github.csv', dtype=float)
    data = np.unique(data, axis=0)

    feature = data[:, :-1]
    label = data[:, -1]
    label = torch.tensor(label-1)

    feature = torch.tensor(feature)
    [nnodes, n_feat] = feature.shape
    n_class = len(torch.unique(label))
    weights = construt_graph(feature, label, nnodes)

    # idx_train, idx_test = get_split(feature, label)
    idx_train = np.load('./data/github/train_mask.npy')
    idx_test = np.load('./data/github/test_mask.npy')
    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)



    return feature, weights, label, idx_train,  idx_test, n_feat, n_class
