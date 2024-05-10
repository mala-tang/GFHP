from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from sklearn.metrics.pairwise import cosine_similarity


class ContrastiveRanking(nn.Module):
    def __init__(self, opt, features, labels, nclasses, n_fea, n_hid):
        super().__init__()
        self.m = opt.m
        self.do_sum_in_log = opt.do_sum_in_log
        self.feature_size = 7

        self.min_tau = opt.min_tau
        self.max_tau = opt.max_tau
        self.similarity_threshold = opt.similarity_threshold
        self.n_sim_classes = opt.n_sim_classes
        self.use_dynamic_tau = opt.use_dynamic_tau
        self.use_all_ranked_classes_above_threshold = self.similarity_threshold > 0
        self.use_same_and_similar_class = opt.use_same_and_similar_class
        self.one_loss_per_rank = opt.one_loss_per_rank
        self.mixed_out_in = opt.mixed_out_in

        self.ranking_classes(features, labels, nclasses, n_fea, n_hid)

        self.criterion = ContrastiveRankingLoss()

        self.randomK = 5

    def ranking_classes(self, features, labels, nclasses, n_fea, n_hid):
        linear_layer = nn.Linear(n_fea, n_hid).double()
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
        self.l_proto = data[:, :-1]
        self.l_proto = linear_layer(self.l_proto)

        #按层次关系分层
        # DGA-200
        sorted_sim_l, sorted_sim_ind_l = torch.sort(similarity_matrix[:, :4], dim=1, descending=True)
        sorted_sim_r, sorted_sim_ind_r = torch.sort(similarity_matrix[:, 4:], dim=1, descending=True)
        sorted_sim_ind_r = sorted_sim_ind_r + 4
        # 不同层次关系中故障类别相似度排序
        sorted_sim, sorted_sim_ind = torch.zeros((nclasses, nclasses)), torch.zeros((nclasses, nclasses))
        sorted_sim[:4, :] = torch.cat((sorted_sim_l[:4, :], torch.zeros((4,3), dtype=torch.float32)), dim=1)
        sorted_sim[4:, :] = torch.cat((sorted_sim_r[4:, :], torch.zeros((3,4), dtype=torch.float32)), dim=1)
        sorted_sim_ind[:4, :] = torch.cat((sorted_sim_ind_l[:4, :], sorted_sim_ind_r[:4, :].flip(0)), dim=1)
        sorted_sim_ind[4:, :] = torch.cat((sorted_sim_ind_l[4:, :], sorted_sim_ind_r[4:, :].flip(0)), dim=1)


        self.class_sims_idx = {}
        for idx in range(nclasses):
            self.class_sims_idx[idx] = {}
            self.class_sims_idx[idx]['sim_class_idx2indices'] = sorted_sim_ind[idx].clone().detach().type(
                torch.long)
            self.class_sims_idx[idx]['sim_class_val'] = sorted_sim[idx]
            self.class_sims_idx[idx]['sim_class_val'][0] = 1

    def forward(self, anchor, pos, labels):
        # compute scores
        l_pos, l_class_pos, l_neg, masks, below_threshold, dynamic_taus = self.compute_InfoNCE_classSimilarity(
            anchor=anchor, pos=pos, labels=labels)
        l_pos = l_pos.cuda()
        l_class_pos = l_class_pos.cuda()
        l_neg = l_neg.cuda()
        #initially l_neg and l_class pos are identical
        res = {}
        for i, mask in enumerate(masks):
            if (self.use_same_and_similar_class and not i == 0):
                mask = masks[-1]
                for j in range(len(masks)-1):
                    mask = mask | masks[j]

                l_neg[mask] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()

                l_class_pos_cur[~mask] = -float("inf")

            elif self.use_all_ranked_classes_above_threshold or (self.use_same_and_similar_class and i == 0):
                l_neg[mask] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")

            else:
                l_neg[mask] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")
            taus = dynamic_taus[i].view(-1, 1)

            if i == 0:
                l_pos = l_pos.cuda()
                l_class_pos_cur = l_class_pos_cur.cuda()
                l_class_pos_cur = torch.cat([l_pos, l_class_pos_cur], dim=1)

            if self.mixed_out_in and i == 0:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)
            elif self.do_sum_in_log and not(self.mixed_out_in and i ==0):
                loss = self.sum_in_log(l_class_pos_cur, l_neg, taus)
            else:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)

            result = {'score': None,
                      'target': None,
                      'loss': loss}
            res['class_similarity_ranking_class' + str(i)] = result

            if (self.use_same_and_similar_class and not i == 0):
                break


        return self.criterion(res, labels)

    def sum_in_log(self, l_pos, l_neg, tau):
        logits = torch.cat([l_pos, l_neg], dim=1) / (tau)
        logits = F.softmax(logits, dim=1)
        sum_pos = logits[:, 0:l_pos.shape[1]].sum(1)
        sum_pos = sum_pos[sum_pos > 1e-7]
        if len(sum_pos) > 0:
            loss = - torch.log(sum_pos).mean()
        else:
            loss = torch.tensor([0.0]).cuda()
        return loss

    def get_similar_labels(self, labels):
        # in this case use top n classes
        labels = labels.cpu().numpy()

        sim_class_labels = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.long)
        sim_class_sims = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.float)
        sim_leq_thresh = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.bool)
        for i, label in enumerate(labels):
            sim_class_labels[i, :] = self.class_sims_idx[label]['sim_class_idx2indices'] + 1
            sim_class_sims[i, :] = self.class_sims_idx[label]['sim_class_val']
            sim_leq_thresh[i, :] = self.class_sims_idx[label]['sim_class_val'] >= self.similarity_threshold
        # remove columns in which no sample has a similarity  qual to or larger than the selected threshold
        at_least_one_leq_thrsh = torch.sum(sim_leq_thresh, dim=0) > 0
        sim_class_labels = sim_class_labels[:, at_least_one_leq_thrsh]
        sim_leq_thresh = sim_leq_thresh[:, at_least_one_leq_thrsh]

        sim_class_labels = sim_class_labels[:, :self.n_sim_classes]
        sim_class_sims = sim_class_sims[:, :self.n_sim_classes]

        # negate sim_leq_thresh to get a mask that can be applied to set all values below thresh to -inf
        sim_leq_thresh = ~sim_leq_thresh[:, :self.n_sim_classes]
        return sim_class_labels, sim_leq_thresh, sim_class_sims

    def compute_InfoNCE_classSimilarity(self, anchor, pos, labels):
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)
        similar_labels, below_threshold, class_sims = self.get_similar_labels(labels)
        masks = []
        threshold_masks = []
        dynamic_taus = []
        for i in range(similar_labels.shape[1]):
            similar_labels = similar_labels.cpu()
            mask = (labels[:, None] == similar_labels[None, :, i]).transpose(0, 1).cuda()
            masks.append(mask)
            if self.use_all_ranked_classes_above_threshold:
                threshold_masks.append(below_threshold[None, :, i].transpose(0, 1).repeat(1, mask.shape[1]))
            dynamic_taus.append(self.get_dynamic_tau(class_sims[:, i]))

        if self.one_loss_per_rank:
            similarity_scores = reversed(class_sims.unique(sorted=True))
            similarity_scores = similarity_scores[similarity_scores > -1]
            new_masks = []
            new_taus = []
            for s in similarity_scores:
                new_taus.append(self.get_dynamic_tau(torch.ones_like(dynamic_taus[0]) * s))
                mask_all_siblings = torch.zeros_like(masks[0], dtype=torch.bool)
                for i in range(similar_labels.shape[1]):
                    same_score = class_sims[:, i] == s
                    if any(same_score):
                        mask_all_siblings[same_score] = mask_all_siblings[same_score] | masks[i][same_score]
                mask_all_siblings = mask_all_siblings.cuda()
                new_masks.append(mask_all_siblings)
            masks = new_masks
            dynamic_taus = new_taus

        l_class_pos = F.cosine_similarity(anchor.double().unsqueeze(1), self.l_proto.double().unsqueeze(0), dim=-1)
        l_neg = l_class_pos.clone()


        return l_pos, l_class_pos, l_neg, masks, threshold_masks, dynamic_taus

    def get_dynamic_tau(self, similarities):
        dissimilarities = 1 - similarities
        d_taus = self.min_tau + (dissimilarities - 0) / (1 - 0) * (self.max_tau - self.min_tau)
        return d_taus

    def visualize_layers(self, writer_train, epoch):
        self.backbone_q.module.visualize_layers(writer_train, epoch)


class ContrastiveRankingLoss:
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loss = 0.0
        for key, val in outputs.items():
            if 'loss' in val:
                loss = loss + val['loss']
            else:
                loss = loss + self.cross_entropy(val['score'], val['target'])
        loss = loss / float(len(outputs))
        return loss