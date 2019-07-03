import torch
import torch.nn as nn
import config_debug

from .py_utils import HorizontalLinePool, VerticalLinePool

from .py_utils.utils import convolution, residual, line_pool_tlbr
from .py_utils.losses import LineNet_Loss
from .py_utils.modules import hg_module, hg, line_net_tlbr

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(line_net_tlbr):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 2
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            hg_module(
                5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)

        tlbr_modules = nn.ModuleList([line_pool_tlbr(256, VerticalLinePool, HorizontalLinePool) for _ in range(stacks)])

        t_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        l_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        b_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        r_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for t_heat, l_heat, b_heat, r_heat in zip(t_heats, l_heats, b_heats, r_heats):
            torch.nn.init.constant_(t_heat[-1].bias, -2.19)
            torch.nn.init.constant_(l_heat[-1].bias, -2.19)
            torch.nn.init.constant_(b_heat[-1].bias, -2.19)
            torch.nn.init.constant_(r_heat[-1].bias, -2.19)

        t_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        l_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        b_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        r_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        t_offs = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        l_offs = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        b_offs = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        r_offs = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        super(model, self).__init__(
            hgs, tlbr_modules, t_heats, l_heats, b_heats, r_heats,
            t_tags, l_tags, b_tags, r_tags, t_offs, l_offs, b_offs, r_offs
        )

        self.loss = LineNet_Loss(pull_weight=1e-1, push_weight=1e-1)
