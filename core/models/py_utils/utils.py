import torch
import torch.nn as nn
import config_debug

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _topk_line(scores, K=20):
    batch, cat, length = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / length).long()

    topk_inds = (topk_inds % length).long()
    topk_s   = topk_inds.int().float()
    return topk_scores, topk_inds, topk_clses, topk_s

# def _allk_line(scores):
#     batch, cat, length = scores.size()
#
#     allk_scores = scores.view(batch, -1)
#     allk_inds = torch.arange(cat * length).view(1, cat * length).expand(batch, cat * length)
#
#     allk_clses = (allk_inds / length).int()
#     allk_inds = allk_inds % length
#     allk_s   = allk_inds.int().float()
#     return allk_scores, allk_clses, allk_s

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width  - 1)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections

def _decode_line(
    t_heat, l_heat, b_heat, r_heat, t_tag, l_tag, b_tag, r_tag, t_regr, l_regr, b_regr, r_regr,
    K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    K=50
    batch, cat, height, width = t_heat.size()

    t_heat, t_max_ind = torch.max(t_heat, 3)
    l_heat, l_max_ind = torch.max(l_heat, 2)
    b_heat, b_max_ind = torch.max(b_heat, 3)
    r_heat, r_max_ind = torch.max(r_heat, 2)

    t_heat = torch.sigmoid(t_heat)
    l_heat = torch.sigmoid(l_heat)
    b_heat = torch.sigmoid(b_heat)
    r_heat = torch.sigmoid(r_heat)

    # perform nms on heatmaps
    t_heat = _nms(t_heat, kernel=kernel)
    l_heat = _nms(l_heat, kernel=kernel)
    b_heat = _nms(b_heat, kernel=kernel)
    r_heat = _nms(r_heat, kernel=kernel)

    t_scores, t_inds, t_clses, t_ys = _topk_line(t_heat, K=K)
    l_scores, l_inds, l_clses, l_xs = _topk_line(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys = _topk_line(b_heat, K=K)
    r_scores, r_inds, r_clses, r_xs = _topk_line(r_heat, K=K)

    t_max_ind = t_max_ind.view(batch, -1).gather(1, t_clses * height + t_inds)
    l_max_ind = l_max_ind.view(batch, -1).gather(1, l_clses * width + l_inds)
    b_max_ind = b_max_ind.view(batch, -1).gather(1, b_clses * height + b_inds)
    r_max_ind = r_max_ind.view(batch, -1).gather(1, r_clses * width + r_inds)

    t_inds = (t_ys * width + t_max_ind.float()).long()
    l_inds = (l_xs + width * l_max_ind.float()).long()
    b_inds = (b_ys * width + b_max_ind.float()).long()
    r_inds = (r_xs + width * r_max_ind.float()).long()

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    if no_border:
        t_ys_binds = (t_ys == 0)
        l_xs_binds = (l_xs == 0)
        b_ys_binds = (b_ys == height - 1)
        r_xs_binds = (r_xs == width  - 1)

    if t_regr is not None and l_regr is not None and b_regr is not None and r_regr is not None:
        t_regr = t_regr.view(batch, -1).gather(1, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1)
        l_regr = l_regr.view(batch, -1).gather(1, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1)
        b_regr = b_regr.view(batch, -1).gather(1, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1)
        r_regr = r_regr.view(batch, -1).gather(1, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K)

        t_ys = t_ys + t_regr
        l_xs = l_xs + l_regr
        b_ys = b_ys + b_regr
        r_xs = r_xs + r_regr


    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((t_ys, l_xs, b_ys, r_xs), dim=5)

    t_tag = t_tag.view(batch, -1).gather(1, t_inds)
    t_tag = t_tag.view(batch, K, 1, 1, 1)
    l_tag = l_tag.view(batch, -1).gather(1, l_inds)
    l_tag = l_tag.view(batch, 1, K, 1, 1)
    b_tag = b_tag.view(batch, -1).gather(1, b_inds)
    b_tag = b_tag.view(batch, 1, 1, K, 1)
    r_tag = r_tag.view(batch, -1).gather(1, r_inds)
    r_tag = r_tag.view(batch, 1, 1, 1, K)
    dists  = (torch.abs(t_tag - l_tag) > ae_threshold).int() + (torch.abs(t_tag - b_tag) > ae_threshold).int() + \
             (torch.abs(t_tag - r_tag) > ae_threshold).int() + (torch.abs(l_tag - b_tag) > ae_threshold).int() + \
             (torch.abs(l_tag - r_tag) > ae_threshold).int() + (torch.abs(b_tag - r_tag) > ae_threshold).int()

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores) / 4

    # reject boxes based on classes
    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + (t_clses != r_clses) + (l_clses != b_clses) + \
               (l_clses != r_clses) + (b_clses != r_clses)

    # reject boxes based on distances
    dist_inds = (dists > 0)

    # reject boxes based on widths and heights
    width_inds  = (r_xs < l_xs)
    height_inds = (b_ys < t_ys)

    if no_border:
        scores[t_ys_binds] = -1
        scores[l_xs_binds] = -1
        scores[b_ys_binds] = -1
        scores[r_xs_binds] = -1

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = t_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_scores = t_scores.contiguous().view(batch, -1, 1)
    t_scores = _gather_feat(t_scores, inds).float()
    l_scores = l_scores.contiguous().view(batch, -1, 1)
    l_scores = _gather_feat(l_scores, inds).float()
    b_scores = b_scores.contiguous().view(batch, -1, 1)
    b_scores = _gather_feat(b_scores, inds).float()
    r_scores = r_scores.contiguous().view(batch, -1, 1)
    r_scores = _gather_feat(r_scores, inds).float()

    detections = torch.cat([bboxes, scores, t_scores, l_scores, b_scores, r_scores, clses], dim=2)
    return detections

class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        if config_debug.visualize_jh:
            return conv2, p1_conv1, p2_conv1, pool1, pool2, p_bn1, pool1 + pool2
        else:
            return conv2

class line_pool(nn.Module):
    def __init__(self, dim, pool1):
        super(line_pool, self).__init__()
        self._init_layers(dim, pool1)

    def _init_layers(self, dim, pool1):
        self.p1_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        pool1 = pool1.expand_as(p1_conv1)
        p_conv1 = self.p_conv1(pool1)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        if config_debug.visualize_jh:
            return conv2, p1_conv1, pool1, p_bn1
        else:
            return conv2

class line_pool_tlbr(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(line_pool_tlbr, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)
        self.p3_conv1 = convolution(3, dim, 128)
        self.p4_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool1()
        self.pool4 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 3
        p3_conv1 = self.p3_conv1(x)
        pool3 = self.pool3(p3_conv1)

        # pool 4
        p4_conv1 = self.p4_conv1(x)
        pool4 = self.pool4(p4_conv1)

        # pool 1 + pool 2 + pool 3 + pool 4
        pool1 = pool1.expand_as(p1_conv1)
        pool2 = pool2.expand_as(p1_conv1)
        pool3 = pool3.expand_as(p1_conv1)
        pool4 = pool4.expand_as(p1_conv1)
        p_concat = pool1 + pool2 + pool3 + pool4
        p_conv1 = self.p_conv1(p_concat)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        if config_debug.visualize_jh:
            return conv2, p1_conv1, p2_conv1, p3_conv1, p4_conv1, pool1, pool2, pool3, pool4, p_concat, p_bn1
        else:
            return conv2
