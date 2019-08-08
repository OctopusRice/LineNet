import torch
import torch.nn as nn

from .utils import _tranpose_and_gather_feat


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def _ae_loss(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _ae_line_loss(tag0, tag1, tag2, tag3, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()
    tag2 = tag2.squeeze()
    tag3 = tag3.squeeze()

    tag_mean = (tag0 + tag1 + tag2 + tag3) / 4

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    tag2 = torch.pow(tag2 - tag_mean, 2) / (num + 1e-4)
    tag2 = tag2[mask].sum()
    tag3 = torch.pow(tag3 - tag_mean, 2) / (num + 1e-4)
    tag3 = tag3[mask].sum()
    pull = tag0 + tag1 + tag2 + tag3

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _off_loss(off, gt_off, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_off)

    off = off[mask]
    gt_off = gt_off[mask]

    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss

def _off_line_loss(off, gt_off, mask):
    num = mask.float().sum()

    off = off[mask]
    gt_off = gt_off[mask]

    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss

def _focal_loss_mask(preds, gt, mask):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_mask = mask[pos_inds]
    neg_mask = mask[neg_inds]

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * pos_mask
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights * neg_mask

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class CornerNet_Saccade_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss_mask):
        super(CornerNet_Saccade_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.off_loss    = _off_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_offs  = outs[4]
        br_offs  = outs[5]
        atts     = outs[6]

        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]
        gt_mask     = targets[2]
        gt_tl_off   = targets[3]
        gt_br_off   = targets[4]
        gt_tl_ind   = targets[5]
        gt_br_ind   = targets[6]
        gt_tl_valid = targets[7]
        gt_br_valid = targets[8]
        gt_atts     = targets[9]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, gt_tl_valid)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, gt_br_valid)

        atts = [[_sigmoid(a) for a in att] for att in atts]
        atts = [[att[ind] for att in atts] for ind in range(len(gt_atts))]

        att_loss = 0
        for att, gt_att in zip(atts, gt_atts):
            att_loss += _focal_loss(att, gt_att) / max(len(att), 1)

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags   = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind) for tl_tag in tl_tags]
        br_tags   = [_tranpose_and_gather_feat(br_tag, gt_br_ind) for br_tag in br_tags]
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_mask)
            off_loss += self.off_loss(br_off, gt_br_off, gt_mask)
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + att_loss + pull_loss + push_loss + off_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)

class CornerNet_ifp_Saccade_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss_mask):
        super(CornerNet_ifp_Saccade_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.off_loss    = _off_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_offs  = outs[4]
        br_offs  = outs[5]
        atts     = outs[6]

        gt_tl_heat      = targets[0]
        gt_br_heat      = targets[1]
        gt_tag_mask     = targets[2]
        gt_off_tl_mask  = targets[3]
        gt_off_br_mask  = targets[4]
        gt_tl_off       = targets[5]
        gt_br_off       = targets[6]
        gt_tl_tag_ind   = targets[7]
        gt_br_tag_ind   = targets[8]
        gt_tl_off_ind   = targets[9]
        gt_br_off_ind   = targets[10]
        gt_tl_valid     = targets[11]
        gt_br_valid     = targets[12]
        gt_atts         = targets[13]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, gt_tl_valid)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, gt_br_valid)

        atts = [[_sigmoid(a) for a in att] for att in atts]
        atts = [[att[ind] for att in atts] for ind in range(len(gt_atts))]

        att_loss = 0
        for att, gt_att in zip(atts, gt_atts):
            att_loss += _focal_loss(att, gt_att) / max(len(att), 1)

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags   = [_tranpose_and_gather_feat(tl_tag, gt_tl_tag_ind) for tl_tag in tl_tags]
        br_tags   = [_tranpose_and_gather_feat(br_tag, gt_br_tag_ind) for br_tag in br_tags]
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_tag_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_off_ind) for tl_off in tl_offs]
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_off_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_off_tl_mask)
            off_loss += self.off_loss(br_off, gt_br_off, gt_off_br_mask)
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + att_loss + pull_loss + push_loss + off_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)

class LineNet_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss):
        super(LineNet_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_line_loss
        self.off_loss    = _off_line_loss

    def forward(self, outs, targets):
        t_heats = outs[0]
        l_heats = outs[1]
        b_heats = outs[2]
        r_heats = outs[3]
        t_tags  = outs[4]
        l_tags  = outs[5]
        b_tags  = outs[6]
        r_tags  = outs[7]
        t_offs  = outs[8]
        l_offs  = outs[9]
        b_offs  = outs[10]
        r_offs  = outs[11]

        gt_t_heat = targets[0]
        gt_l_heat = targets[1]
        gt_b_heat = targets[2]
        gt_r_heat = targets[3]
        gt_mask   = targets[4]
        gt_t_off  = targets[5]
        gt_l_off  = targets[6]
        gt_b_off  = targets[7]
        gt_r_off  = targets[8]
        gt_t_ind  = targets[9]
        gt_l_ind  = targets[10]
        gt_b_ind  = targets[11]
        gt_r_ind  = targets[12]

        # focal loss
        focal_loss = 0

        t_heats = [_sigmoid(t) for t in t_heats]
        l_heats = [_sigmoid(l) for l in l_heats]
        b_heats = [_sigmoid(b) for b in b_heats]
        r_heats = [_sigmoid(r) for r in r_heats]

        focal_loss += self.focal_loss(t_heats, gt_t_heat)
        focal_loss += self.focal_loss(l_heats, gt_l_heat)
        focal_loss += self.focal_loss(b_heats, gt_b_heat)
        focal_loss += self.focal_loss(r_heats, gt_r_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0
        t_tags    = [_tranpose_and_gather_feat(t_tag, gt_t_ind) for t_tag in t_tags]
        l_tags    = [_tranpose_and_gather_feat(l_tag, gt_l_ind) for l_tag in l_tags]
        b_tags    = [_tranpose_and_gather_feat(b_tag, gt_b_ind) for b_tag in b_tags]
        r_tags    = [_tranpose_and_gather_feat(r_tag, gt_r_ind) for r_tag in r_tags]
        for t_tag, l_tag, b_tag, r_tag in zip(t_tags, l_tags, b_tags, r_tags):
            pull, push = self.ae_loss(t_tag, l_tag, b_tag, r_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        t_offs   = [_tranpose_and_gather_feat(t_off, gt_t_ind) for t_off in t_offs]
        l_offs   = [_tranpose_and_gather_feat(l_off, gt_l_ind) for l_off in l_offs]
        b_offs   = [_tranpose_and_gather_feat(b_off, gt_b_ind) for b_off in b_offs]
        r_offs   = [_tranpose_and_gather_feat(r_off, gt_r_ind) for r_off in r_offs]
        for t_off, l_off, b_off, r_off in zip(t_offs, l_offs, b_offs, r_offs):
            off_loss += self.off_loss(t_off, gt_t_off, gt_mask)
            off_loss += self.off_loss(l_off, gt_l_off, gt_mask)
            off_loss += self.off_loss(b_off, gt_b_off, gt_mask)
            off_loss += self.off_loss(r_off, gt_r_off, gt_mask)
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + pull_loss + push_loss + off_loss) / max(len(t_heats), 1)
        return loss.unsqueeze(0)

class CornerNet_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss):
        super(CornerNet_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight = off_weight
        self.focal_loss = focal_loss
        self.ae_loss = _ae_loss
        self.off_loss = _off_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags = outs[2]
        br_tags = outs[3]
        tl_offs = outs[4]
        br_offs = outs[5]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_tag_mask = targets[2]
        gt_off_tl_mask = targets[3]
        gt_off_br_mask = targets[4]
        gt_tl_off = targets[5]
        gt_br_off = targets[6]
        gt_tl_tag_ind = targets[7]
        gt_br_tag_ind = targets[8]
        gt_tl_off_ind = targets[9]
        gt_br_off_ind = targets[10]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags = [_tranpose_and_gather_feat(tl_tag, gt_tl_tag_ind) for tl_tag in tl_tags]
        br_tags = [_tranpose_and_gather_feat(br_tag, gt_br_tag_ind) for br_tag in br_tags]
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_tag_mask)# modified by JH_190609
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        tl_offs = [_tranpose_and_gather_feat(tl_off, gt_tl_off_ind) for tl_off in tl_offs]
        br_offs = [_tranpose_and_gather_feat(br_off, gt_br_off_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_off_tl_mask)  # modified by JH_190609
            off_loss += self.off_loss(br_off, gt_br_off, gt_off_br_mask)  # modified by JH_190609
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + pull_loss + push_loss + off_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)