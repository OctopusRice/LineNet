import torch

from torch import nn
from torch.autograd import Function

import horizontal_line_pool, vertical_line_pool

class HorizontalLinePoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = horizontal_line_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = horizontal_line_pool.backward(input, grad_output)[0]
        return output

class VerticalLinePoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = vertical_line_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = vertical_line_pool.backward(input, grad_output)[0]
        return output


class HorizontalLinePool(nn.Module):
    def forward(self, x):
        return HorizontalLinePoolFunction.apply(x)

class VerticalLinePool(nn.Module):
    def forward(self, x):
        return VerticalLinePoolFunction.apply(x)

