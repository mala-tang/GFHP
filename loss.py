from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from sklearn.metrics.pairwise import cosine_similarity


class ContrastiveRanking(nn.Module):
    def __init__(self, opt, features, labels, nclasses, n_fea, n_hid, class_sims_idx, l_proto):
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

        self.class_sims_idx = class_sims_idx
        labels = labels.clone().detach().cuda()

        self.similar_labels = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.long)
        self.class_sims = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.float)
        self.below_threshold = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).cuda().type(torch.bool)
        for i, label in enumerate(labels):
            self.similar_labels[i, :] = self.class_sims_idx[int(label)]['sim_class_idx2indices'] + 1
            self.class_sims[i, :] = self.class_sims_idx[int(label)]['sim_class_val']
            self.below_threshold[i, :] = self.class_sims_idx[int(label)]['sim_class_val'] >= self.similarity_threshold
        # remove columns in which no sample has a similarity  qual to or larger than the selected threshold
        at_least_one_leq_thrsh = torch.sum(self.below_threshold, dim=0) > 0
        self.similar_labels = self.similar_labels[:, at_least_one_leq_thrsh]
        self.below_threshold = self.below_threshold[:, at_least_one_leq_thrsh]

        self.similar_labels = self.similar_labels[:, :self.n_sim_classes]
        self.class_sims = self.class_sims[:, :self.n_sim_classes]

        # negate sim_leq_thresh to get a mask that can be applied to set all values below thresh to -inf
        self.below_threshold = ~self.below_threshold[:, :self.n_sim_classes]

        self.masks = []
        self.threshold_masks = []
        self.dynamic_taus = []
        for i in range(self.similar_labels.shape[1]):

            mask = (labels[:, None] == self.similar_labels[None, :, i]).transpose(0, 1).cuda()
            self.masks.append(mask)
            if self.use_all_ranked_classes_above_threshold:
                self.threshold_masks.append(self.below_threshold[None, :, i].transpose(0, 1).repeat(1, mask.shape[1]))
            self.dynamic_taus.append(self.get_dynamic_tau(self.class_sims[:, i]))

        if self.one_loss_per_rank:
            similarity_scores = reversed(self.class_sims.unique(sorted=True))
            similarity_scores = similarity_scores[similarity_scores > -1]
            new_masks = []
            new_taus = []
            for s in similarity_scores:
                new_taus.append(self.get_dynamic_tau(torch.ones_like(self.dynamic_taus[0]) * s))
                mask_all_siblings = torch.zeros_like(self.masks[0], dtype=torch.bool)
                for i in range(self.similar_labels.shape[1]):
                    same_score = self.class_sims[:, i] == s
                    if any(same_score):
                        mask_all_siblings[same_score] = mask_all_siblings[same_score] | self.masks[i][same_score]
                mask_all_siblings = mask_all_siblings.cuda()
                new_masks.append(mask_all_siblings)
            self.masks = new_masks
            self.dynamic_taus = new_taus

        self.linear_layer = nn.Linear(n_fea, n_hid).double()
        self.l_proto = self.linear_layer(l_proto)

        self.criterion = ContrastiveRankingLoss()

        self.randomK = 5



    def forward(self, anchor, labels, train_mask):
        # compute scores
        l_pos, l_class_pos, l_neg = self.compute_InfoNCE_classSimilarity(
            anchor=anchor, train_mask=train_mask)
        l_pos = l_pos.cuda()
        l_class_pos = l_class_pos.cuda()
        l_neg = l_neg.cuda()
        #initially l_neg and l_class pos are identical
        res = {}
        # similar_labels = self.similar_labels[train_mask]
        # below_threshold = self.below_threshold[train_mask]
        # class_sims = self.class_sims
        masks = self.masks
        dynamic_taus = self.dynamic_taus
        for i, mask in enumerate(masks):
            mask = mask[train_mask, train_mask]
            if (self.use_same_and_similar_class and not i == 0):
                mask = masks[-1]
                mask = mask[train_mask, train_mask]
                for j in range(len(masks)-1):
                    mask = mask | masks[j][train_mask, train_mask]

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
            taus = taus[train_mask]

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

    def compute_InfoNCE_classSimilarity(self, anchor,  train_mask):
        anchor = anchor.to(torch.float32)
        pos = self.l_proto[train_mask]
        pos = pos.to(torch.float32)
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)

        l_class_pos = anchor @ pos.T
        l_neg = l_class_pos.clone()

        return l_pos, l_class_pos, l_neg

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
