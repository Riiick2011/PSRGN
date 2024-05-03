import torch
from torch.nn import functional as F
import torch.nn as nn

def sigmoid_focal_loss(
    inputs,
    targets,
    reduction = 'none',
    alpha = 0.25,
    gamma = 2.0):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs=inputs.squeeze(1)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def ctr_giou_loss_1d(
    input_offsets,
    target_offsets,
    reduction = 'none',
    eps = 1e-8
):
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)

    # giou
    len_c = lc + rc
    miouk = iouk - ((len_c - unionk) / len_c.clamp(min=eps))

    loss = 1.0 - miouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class OHEMHingeLoss(torch.autograd.Function):
    """
    This class is the core implementation for the completeness loss in paper.
    It compute class-wise hinge loss and performs online hard negative mining (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        n_sample = pred.size()[0]
        assert n_sample == len(labels), "mismatch between sample size and label size"
        losses = torch.zeros(n_sample)
        slopes = torch.zeros(n_sample)
        for i in range(n_sample):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0
        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_num = max(int(group_size * ohem_ratio), 1)
        loss = torch.zeros(1)
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_num].sum()
        ctx.loss_ind = indices[:, :keep_num]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_group = losses.size(0)
        return loss.cuda()

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes

        grad_in = torch.zeros(ctx.shape)
        for group in range(ctx.num_group):
            for idx in ctx.loss_ind[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * grad_output[0].cpu()
        return grad_in.cuda(), None, None, None, None


class CompletenessLoss(torch.nn.Module):
    def __init__(self, ohem_ratio=0.17):
        super(CompletenessLoss, self).__init__()
        self.ohem_ratio = ohem_ratio


        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, labels, sample_split, sample_group_size):
        if sample_split>0 :
            pred_dim = pred.size()[1]
            pred = pred.view(-1, sample_group_size, pred_dim)
            labels = labels.view(-1, sample_group_size)

            pos_group_size = sample_split
            neg_group_size = sample_group_size - sample_split
            pos_prob = pred[:, :sample_split, :].contiguous().view(-1, pred_dim)
            neg_prob = pred[:, sample_split:, :].contiguous().view(-1, pred_dim)
            pos_ls = OHEMHingeLoss.apply(pos_prob, labels[:, :sample_split].contiguous().view(-1), 1,
                                         1.0, pos_group_size)
            neg_ls = OHEMHingeLoss.apply(neg_prob, labels[:, sample_split:].contiguous().view(-1), -1,
                                         self.ohem_ratio, neg_group_size)
            pos_cnt = pos_prob.size(0)
            neg_cnt = max(int(neg_prob.size()[0] * self.ohem_ratio), 1)
            return pos_ls / float(pos_cnt + neg_cnt) + neg_ls / float(pos_cnt + neg_cnt)
        else:
            return torch.tensor([100.0]).cuda()


class ClassWiseRegressionLoss(torch.nn.Module):
    """
    This class implements the location regression loss for each class
    """

    def __init__(self):
        super(ClassWiseRegressionLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, labels, targets):
        indexer = labels.data - 1
        prep = pred[:, indexer, :]
        class_pred = torch.cat((torch.diag(prep[:, :,  0]).view(-1, 1),
                                torch.diag(prep[:, :, 1]).view(-1, 1)),
                               dim=1)
        loss = self.smooth_l1_loss(class_pred.view(-1), targets.view(-1)) * 2
        return loss
