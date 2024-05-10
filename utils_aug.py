import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

plt.rcParams['axes.unicode_minus'] = False

EOS = 1e-10


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def get_adj(dir_adj, gso_type):
    if sp.issparse(dir_adj):
        id = sp.identity(dir_adj.shape[0], format='csc')
        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = sp.diags(row_sum_inv, format='csc')
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

    else:
        id = np.identity(dir_adj.shape[0])
        # Symmetrizing an adjacency matrix
        adj = np.maximum(dir_adj, dir_adj.T)
        # adj = 0.5 * (dir_adj + dir_adj.T)

        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = np.sum(adj, axis=1)
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

    return gso

def convert_sp_mat_to_sp_tensor(X, device):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    g = g.coalesce().to(device)
    return g


#转换数据格式
def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

#计算准确性
def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=1)

    if labels.max() > 1:
        auc = roc_auc_score(labels, F.softmax(output, dim=-1).detach(), average='macro',
                            multi_class='ovr')
    else:
        auc = roc_auc_score(labels, F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    return f1, auc, precision

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def save_checkpoint(model, model_path):
    torch.save(model.state_dict(), model_path)

def perturb_graph(features, adj, fea_mask_rate, edge_dropout_rate, args):

    # 删除节点特征
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * fea_mask_rate), replace=False)
    mask[:, samples] = 1
    # if torch.cuda.is_available():
    #     mask = mask.cuda()
    features_1 = features * (1 - mask)

    # 删除边
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()

    # Generate mask for dropout
    edge_mask = (torch.rand_like(values) > edge_dropout_rate).float()

    # Apply dropout to non-zero elements
    values = values * edge_mask

    # Create new sparse tensor
    adj_1 = torch.sparse.FloatTensor(indices, values, adj.size())


    return features_1, adj_1