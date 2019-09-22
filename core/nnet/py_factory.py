import os
import torch
import pickle
import importlib
import torch.nn as nn

from ..models.py_utils.data_parallel import DataParallel
import config_debug

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(preds, ys, **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, system_config, model, distributed=False, gpu=None):
        super(NetworkFactory, self).__init__()

        self.system_config = system_config

        self.gpu     = gpu
        self.model   = DummyModule(model)
        self.loss    = model.loss
        self.network = Network(self.model, self.loss)
        
        self.model.cuda()
        
        if distributed:
            from apex.parallel import DistributedDataParallel, convert_syncbn_model
            torch.cuda.set_device(gpu)
            self.network = self.network.cuda(gpu)
            self.network = convert_syncbn_model(self.network)
            self.network = DistributedDataParallel(self.network)
        else:
            self.network = DataParallel(self.network, chunk_sizes=system_config.chunk_sizes)

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_config.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_config.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_config.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def _t_cuda(self, xs):
        if type(xs) is list:
            return [x.cuda(self.gpu, non_blocking=True) for x in xs]
        return xs.cuda(self.gpu, non_blocking=True)

    def train(self, xs, ys, **kwargs):
        xs = [self._t_cuda(x) for x in xs]
        ys = [self._t_cuda(y) for y in ys]

        self.optimizer.zero_grad()
        loss = self.network(xs, ys)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [self._t_cuda(x) for x in xs]
            ys = [self._t_cuda(y) for y in ys]

            loss = self.network(xs, ys)
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [self._t_cuda(x) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))

        names = []
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

            if config_debug.validation:
                for name, param in self.model.named_parameters():
                    names.append(name)
                    if name[7:17] != 'tl_modules' and name[7:17] != 'br_modules' and \
                       name[7:15] != 'tl_heats'   and name[7:15] != 'br_heats'   and \
                       name[7:14] != 'tl_offs'    and name[7:14] != 'br_offs':
                        param.requires_grad = False

        if config_debug.validation:
            with open("./" + config_debug.cfg_file + "_module_names.txt", "w") as f:
                for listitem in names:
                    f.write('%s\n' % listitem)



    def load_params(self, iteration):
        cache_file = self.system_config.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def save_params(self, iteration):
        cache_file = self.system_config.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            opt = self.optimizer.state_dict()

            torch.save({
                'model_state_dict': params,
                'optimizer_state_dict': opt,
            }, f)
