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
from utils_aug import *

from loss import *



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


def train_cl(cl_model, optimizer_cl, baseline_classifier, optimizer_baseline, x1, x2, adj_1, adj_2, criterion, idx_train, cur_split, label):
    cl_model.train()
    baseline_classifier.eval()

    optimizer_cl.zero_grad()
    #optimizer_baseline.zero_grad()


    emb1 = cl_model.get_emb1(x1, adj_1)
    emb2 = cl_model.get_emb2(x2, adj_2)
    pred1 = baseline_classifier(emb1)
    pred2 = baseline_classifier(emb2)
    log_pred1 = F.softmax(pred1, dim=1)
    log_pred2 = F.softmax(pred2, dim=1)
    class_loss = F.cross_entropy(pred1[idx_train[:, cur_split]], label[idx_train[:,cur_split]].long())
    kl_loss = F.kl_div(log_pred1/5, log_pred2/5, reduction='batchmean')
    cl_loss = cl_model.RINCE(x1, adj_1, x2, adj_2, criterion, label)
    loss = 0.04 * cl_loss + class_loss + 0.06 * kl_loss
    #print(loss)

    loss.backward(retain_graph=True)#retain_graph=True
    optimizer_cl.step()
    #optimizer_baseline.step()

    return cl_loss.item()#, loss.item()#, main_loss.item(), total_loss.item()

def train_GCNModel(baseline_classifier, optimizer_baseline, optimizer_cl, cl_model, feature, label, idx_train, adj_1, cur_split):
    baseline_classifier.train()
    cl_model.eval()

    emb = cl_model.get_emb1(feature, adj_1)

    output = baseline_classifier(emb)
    loss_train = F.cross_entropy(output[idx_train[:, cur_split]], label[idx_train[:,cur_split]].long())

    baseline_classifier.zero_grad()
    #optimizer_cl.zero_grad()
    loss_train.backward()
    optimizer_baseline.step()
    #optimizer_cl.step()
    return loss_train

def main(args):
    torch.cuda.empty_cache()
    feature, weights, label, idx_train,  idx_test, n_feat, n_class = load_data()

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
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    results, f1_result, auc_result, pre_result = [], [], [], []

    for trial in range(args.ntrials):
        set_env(trial)
        cl_model = GCL(device, nlayers=args.nlayers, n_input=n_feat, n_hid=int(args.n_hid), n_output=int(args.n_hid/2),
                       droprate=args.droprate, sparse=args.sparse, batch_size=args.cl_batch_size,
                       enable_bias=args.enable_bias).to(device)
        criterion = ContrastiveRanking(args, feature, label, len(torch.unique(label)), n_feat, int(args.n_hid/2)).to(device)
        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_clas, weight_decay=args.w_decay)
        baseline_classifier = GCN_Classifier(ninput=int(args.n_hid/2),
                                             # 因为encoder1和encoder2的输出是64(args.embedding_dim = 64)
                                             nclass=len(torch.unique(label)), dropout=args.droprate).to(device)
        optimizer_baseline = torch.optim.Adam(baseline_classifier.parameters(),  # GCN分类器
                                              lr=args.lr_gcl, weight_decay=args.w_decay)

        cur_split = 0 if (idx_train.shape[1] == 1) else (trial % idx_train.shape[1])
        best_acc, best_f1, best_auc, best_pre = 0, 0, 0, 0

        for epoch in range(1, args.epochs + 1):

            train_time_list = []
            train_epoch_begin_time = time.perf_counter()
            #, class_loss
            cl_loss = train_cl(cl_model, optimizer_cl, baseline_classifier, optimizer_baseline, x1, x2, adj_1, adj_2, criterion, idx_train, cur_split, label)
            class_loss = train_GCNModel(baseline_classifier, optimizer_baseline, optimizer_cl, cl_model, feature, label, idx_train,
                                        adj_1, cur_split)
            embed = cl_model.get_emb1(feature, adj_1)
            output = baseline_classifier(embed)
            f1_train, auc_train = calc_accuracy(output[idx_train[:, cur_split]],
                                                                            label[idx_train[:, cur_split]].long())

            train_epoch_end_time = time.perf_counter()
            train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
            train_time_list.append(train_epoch_time_duration)

            print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | Class loss:{:.4f} | f1_train:{:.2f}| auc_train:{:.2f} | Training duration: {:.6f}".\
                  format(epoch, cl_loss, class_loss, f1_train, auc_train, train_epoch_time_duration))

            if epoch % args.eval_freq == 0 and epoch > 100:
                cl_model.eval()
                baseline_classifier.eval()
                embed = cl_model.get_emb1(feature, adj_1)
                with torch.no_grad():
                    output = baseline_classifier(embed)
                    loss_test = F.cross_entropy(output[idx_test[:, cur_split]],
                                                label[idx_test[:, cur_split]].long())
                    f1_test, auc_test = calc_accuracy(output[idx_test[:, cur_split]], label[idx_test[:, cur_split]].long())

                print(
                    '[TEST] Epoch:{:04d} | Main loss:{:.4f} | f1_test:{:.2f}| auc_test:{:.2f}|'.format(epoch, loss_test.item(), f1_test, auc_test))
                if f1_test > best_f1:
                    best_f1 = f1_test
                    best_auc = auc_test
                    best_split = cur_split

        f1_result.append(best_f1)
        auc_result.append(best_auc)

    f1_result = np.array(f1_result, dtype=np.float32)
    auc_result = np.array(auc_result, dtype=np.float32)
    print('\n[FINAL RESULT] Dataset:{} | Run:{} | f1:{:.2f}+-{:.2f} | AUC:{:.2f}+-{:.2f}'
            .format(args.dataset, args.ntrials, np.mean(f1_result), np.std(f1_result), np.mean(auc_result), np.std(auc_result)))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ESSENTIAL

    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-eval_freq', type=int, default=50)
    parser.add_argument('-epochs', type=int, default=500)#400
    parser.add_argument('-lr_gcl', type=float, default=0.01)
    parser.add_argument('-lr_clas', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.)
    parser.add_argument('-droprate', type=float, default=0.005)
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-dataset', type=str, default='DGA-200')
    parser.add_argument('--enable_bias', type=bool, default=False, help='default as False')
    parser.add_argument('--A_split', type=bool, default=False, help='default as False')


    # GCN Module - Hyper-param
    parser.add_argument('-nlayers', type=int, default=4)
    parser.add_argument('-n_hid', type=int, default=256)#128
    parser.add_argument('-cl_batch_size', type=int, default=0)

    # stuff for ranking
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.2, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--m', default=0.99, type=float, help='momentum update to use in contrastive learning')
    parser.add_argument('--do_sum_in_log', type=str2bool, default='True')

    parser.add_argument('--similarity_threshold', default=0.01, type=float, help='')
    parser.add_argument('--n_sim_classes', default=7, type=int, help='')
    parser.add_argument('--use_dynamic_tau', type=str2bool, default='True', help='')
    parser.add_argument('--use_supercategories', type=str2bool, default='False', help='')
    parser.add_argument('--use_same_and_similar_class', type=str2bool, default='True', help='')
    parser.add_argument('--one_loss_per_rank', type=str2bool, default='False')
    parser.add_argument('--mixed_out_in', type=str2bool, default='False')
    parser.add_argument('--roberta_threshold', type=str, default=None,
                        help='one of 05_None; 05_04; 04_None; 06_None; roberta_superclass20; roberta_superclass_40')
    parser.add_argument('--roberta_float_threshold', type=float, nargs='+', default=None, help='')

    parser.add_argument('--exp_name', type=str, default=None, help='set experiment name manually')
    parser.add_argument('--mixed_out_in_log', type=str2bool, default='False', help='')
    parser.add_argument('--out_in_log', type=str2bool, default='False', help='')

    args = parser.parse_args()

    print(args)
    main(args)
