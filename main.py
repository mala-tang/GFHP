import argparse
import os

import random
import time

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics


from dataloader import *
from model_aug import *
from utils_tsne import *

from loss_update import *



def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_cl(cl_model, optimizer_cl, baseline_classifier, optimizer_baseline, x1, x2, adj_1, adj_2, criterion, idx_train, label): #cur_split,
    cl_model.train()
    baseline_classifier.eval()

    emb1, emb2, cl_loss = cl_model.RINCE(x1, adj_1, x2, adj_2, criterion, label, idx_train)
    pred1 = baseline_classifier(emb1)
    pred2 = baseline_classifier(emb2)
    kl_loss = F.kl_div(
        F.log_softmax(pred1, dim=1),
        F.softmax(pred2, dim=1),
        reduction='batchmean'
    )
    class_loss = F.cross_entropy(pred1[idx_train], label[idx_train].long())


    loss = class_loss + 0.6*class_loss + 0.4*kl_loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(cl_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(baseline_classifier.parameters(), max_norm=1.0)
    optimizer_cl.step()
    optimizer_baseline.step()


    optimizer_cl.zero_grad()
    optimizer_baseline.zero_grad()
    return cl_loss.item()


def main(args):
    torch.cuda.empty_cache()
    train_epoch_begin_time = time.perf_counter()
    feature, weights, label, n_feat, n_class, class_sims_idx, l_proto, train_mask,  test_mask = load_data()

    if torch.cuda.is_available():
        device = torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    weights = sp.coo_matrix(weights)
    adj = get_adj(weights, 'sym_renorm_adj')
    if sp.issparse(feature):
        feature = cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = feature.to(device)
    adj = cnv_sparse_mat_to_coo_tensor(adj, device)
    x1, adj_1 = perturb_graph(feature, adj, 0.1, 0.1, args)
    x2, adj_2 = perturb_graph(feature, adj, 0.1, 0.1, args)
    label = label.to(device)
    results, f1_result, auc_result, pre_result = [], [], [], []
    cl_model = GCL(device, nlayers=args.nlayers, n_input=n_feat, n_hid=int(args.n_hid), n_output=int(args.n_hid / 2),
                   droprate=args.droprate, sparse=args.sparse, batch_size=args.cl_batch_size,
                   enable_bias=args.enable_bias).to(device)
    criterion = ContrastiveRanking(args, feature, label, len(torch.unique(label)), n_feat, int(args.n_hid / 2),
                                   class_sims_idx, l_proto).to(device)
    optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_clas, weight_decay=args.w_decay)
    baseline_classifier = GCN_Classifier(ninput=int(args.n_hid / 2),
                                         nclass=len(torch.unique(label)), dropout=args.droprate).to(device)
    optimizer_baseline = torch.optim.Adam(baseline_classifier.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)

    for trial in range(args.ntrials):
        cur_split = 0 if (train_mask.shape[1] == 1) else (trial % train_mask.shape[1])
        idx_train = train_mask[:, cur_split]
        idx_test = test_mask[:, cur_split]
        best_acc, best_f1, best_auc, best_pre = 0, 0, 0, 0
        for epoch in range(1, args.epochs + 1):
            train_cl(cl_model, optimizer_cl, baseline_classifier, optimizer_baseline, x1, x2, adj_1, adj_2, criterion, idx_train, label)
            embed = cl_model.get_emb1(x1, adj_1)
            with torch.no_grad():
                output = baseline_classifier(embed)
                loss_test = F.cross_entropy(output[idx_test],
                                            label[idx_test].long())
                f1_test, auc_test, precision_test, cm, prec, f1, acc = calc_accuracy(output[idx_test], label[idx_test].long())

            print(
                '[TEST] Epoch:{:04d} | Main loss:{:.4f} | f1_test:{:.2f}| auc_test:{:.2f}| pre_test:{:.2f}'.format(epoch,
                                                                                                   loss_test.item(),
                                                                                                   f1_test, auc_test,precision_test))
            if f1_test > best_f1:
                best_f1 = f1_test
                best_auc = auc_test
                best_pre = precision_test

        f1_result.append(best_f1)
        auc_result.append(best_auc)
        pre_result.append(best_pre)

    train_epoch_end_time = time.perf_counter()
    train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
    f1_result = np.array(f1_result, dtype=np.float32)
    auc_result = np.array(auc_result, dtype=np.float32)
    print('\n[FINAL RESULT] Dataset:{} | Run:{} | f1:{:.2f}+-{:.2f} | AUC:{:.2f}+-{:.2f} | Precision:{:.2f}+-{:.2f}'
          .format(args.dataset, args.ntrials, np.mean(f1_result), np.std(f1_result), np.mean(auc_result),
                  np.std(auc_result)
                  , np.mean(pre_result), np.std(pre_result)))
    print(f'time:{train_epoch_time_duration}')

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ESSENTIAL

    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-eval_freq', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr_gcl', type=float, default=0.01)
    parser.add_argument('-lr_clas', type=float, default=0.001)
    parser.add_argument('-w_decay', type=float, default=0.000)
    parser.add_argument('-droprate', type=float, default=0.005)
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-dataset', type=str, default='DGA-200')
    parser.add_argument('--enable_bias', type=bool, default=False, help='default as False')
    parser.add_argument('--A_split', type=bool, default=False, help='default as False')


    # GCN Module - Hyper-param
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-n_hid', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)

    # stuff for ranking
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.2, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--m', default=0.99, type=float, help='momentum update to use in contrastive learning')
    parser.add_argument('--do_sum_in_log', type=str2bool, default='True')

    parser.add_argument('--similarity_threshold', default=0.1, type=float, help='')
    parser.add_argument('--n_sim_classes', default=6, type=int, help='')
    parser.add_argument('--use_dynamic_tau', type=str2bool, default='True', help='')
    parser.add_argument('--use_supercategories', type=str2bool, default='False', help='')
    parser.add_argument('--use_same_and_similar_class', type=str2bool, default='False', help='')
    parser.add_argument('--one_loss_per_rank', type=str2bool, default='False')
    parser.add_argument('--mixed_out_in', type=str2bool, default='False')
    parser.add_argument('--roberta_threshold', type=str, default=None,
                        help='one of 05_None; 05_04; 04_None; 06_None; roberta_superclass20; roberta_superclass_40')
    parser.add_argument('--roberta_float_threshold', type=float, nargs='+', default=None, help='')

    parser.add_argument('--exp_name', type=str, default=None, help='set experiment name manually')
    parser.add_argument('--mixed_out_in_log', type=str2bool, default='False', help='')
    parser.add_argument('--out_in_log', type=str2bool, default='False', help='')

    args = parser.parse_args()
    set_env(42)
    print(args)
    main(args)
