"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/edwardzhou130/PolarSeg/blob/master/network/lovasz_losses.py
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import evaluate
from .evaluate import CLASSES_NUSCENES
from .evaluate import CLASSES_KITTI

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1.0, ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1.0, ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            # The ignored label is sometimes among predicted classes
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    # mean accross images if per_image
    ious = [mean(iou) for iou in zip(*ious)]
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel
              (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------
class DownstreamLoss(nn.Module):
    """
    Custom which is the sum of a lovasz loss and a crossentropy.
    Main class to instantiate in the code.
    """

    def __init__(self, weights=None, ignore_index=None, device="cpu"):
        super(DownstreamLoss, self).__init__()
        self.ignore_index = ignore_index
        if weights is None:
            self.crossentropy = torch.nn.CrossEntropyLoss()
        else:
            self.crossentropy = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights).float().to(device)
            )

    def forward(self, probas, labels):
        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            probas = probas[valid]
            labels = labels[valid]
        loss1 = self.crossentropy(probas, labels)
        loss2 = lovasz_softmax_flat(probas.softmax(-1), labels)
        return loss1 + loss2


class unknown_aware_infoNCE(nn.Module):
    """
    Custom which is the sum of a lovasz loss and a crossentropy.
    Main class to instantiate in the code.
    """

    def __init__(self, ignore_index=None, config=None):
        super(unknown_aware_infoNCE, self).__init__()
        self.ignore_index = ignore_index
        # self.seen_classes =
        self.unseen_classes = ['motorcycle', 'trailer', 'terrain', 'traffic_cone']
        self.CLASS_LABELS = CLASSES_NUSCENES
        if config['dataset'] == 'kitti':
            self.CLASS_LABELS = CLASSES_KITTI

        self.seen_class_index = list(range(len(self.CLASS_LABELS)))
        for item in self.unseen_classes:
            index = self.CLASS_LABELS.index(item)
            # self.unseen_index.append(index)
            self.seen_class_index.remove(index)

        self.crossentropy = torch.nn.CrossEntropyLoss()

    def pseudo_supervised(self, predictions):
        if predictions.size()[0] == 0: return 0
        predictions = torch.softmax(predictions, dim=1)
        loss = torch.mean(torch.sum(predictions[:, self.seen_class_index], dim=1))
        # loss += torch.mean(1 - torch.sum(predictions[:, self.unseen_index], dim=1))

        return loss

    def forward(self, probas, labels):

        for item in self.unseen_classes:
            index = self.CLASS_LABELS.index(item)
            labels[labels == index] = -200

        seen_index = ((labels != self.ignore_index) & (labels != -200))
        unseen_index = labels == -200

        import pdb
        pdb.set_trace()

        loss1 = self.crossentropy(probas[seen_index], labels[seen_index])
        loss2 = self.pseudo_supervised(probas[unseen_index])
        return loss1 + loss2


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction
              (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of
              size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a
      list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), classes=classes
        )
    return loss


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels,
               or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        # 3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H * W)
    B, C, H, W = probas.size()
    # B * H * W, C = P, C
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


def jaccard_loss(probas, labels, ignore=None, smooth=100, bk_class=None):
    """
    Something wrong with this loss
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction.
              Interpreted as binary (sigmoid) output with outputs of
              size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or
               a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)

    true_1_hot = torch.eye(vprobas.shape[1])[vlabels]

    if bk_class:
        one_hot_assignment = torch.ones_like(vlabels)
        one_hot_assignment[vlabels == bk_class] = 0
        one_hot_assignment = one_hot_assignment.float().unsqueeze(1)
        true_1_hot = true_1_hot * one_hot_assignment

    true_1_hot = true_1_hot.to(vprobas.device)
    intersection = torch.sum(vprobas * true_1_hot)
    cardinality = torch.sum(vprobas + true_1_hot)
    loss = (intersection + smooth / (cardinality - intersection + smooth)).mean()
    return (1 - loss) * smooth


def hinge_jaccard_loss(
    probas, labels, ignore=None, classes="present", hinge=0.1, smooth=100
):
    """
    Multi-class Hinge Jaccard loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction.
              Interpreted as binary (sigmoid) output with outputs of
              size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels,
               or a list of classes to average.
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    C = vprobas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        if c in vlabels:
            c_sample_ind = vlabels == c
            cprobas = vprobas[c_sample_ind, :]
            non_c_ind = np.array([a for a in class_to_sum if a != c])
            class_pred = cprobas[:, c]
            max_non_class_pred = torch.max(cprobas[:, non_c_ind], dim=1)[0]
            TP = (
                torch.sum(torch.clamp(class_pred - max_non_class_pred, max=hinge) + 1.0)
                + smooth
            )
            FN = torch.sum(
                torch.clamp(max_non_class_pred - class_pred, min=-hinge) + hinge
            )

            if (~c_sample_ind).sum() == 0:
                FP = 0
            else:
                nonc_probas = vprobas[~c_sample_ind, :]
                class_pred = nonc_probas[:, c]
                max_non_class_pred = torch.max(nonc_probas[:, non_c_ind], dim=1)[0]
                FP = torch.sum(
                    torch.clamp(class_pred - max_non_class_pred, max=hinge) + 1.0
                )

            losses.append(1 - TP / (TP + FP + FN))

    if len(losses) == 0:
        return 0
    return mean(losses)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(ls, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    ls = iter(ls)
    if ignore_nan:
        ls = ifilterfalse(isnan, ls)
    try:
        n = 1
        acc = next(ls)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(ls, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
