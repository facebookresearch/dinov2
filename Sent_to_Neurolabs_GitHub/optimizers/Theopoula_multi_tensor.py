import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class THEOPOULA(Optimizer):

    def __init__(self, params, lr=1e-1, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, eps=eps, weight_decay=weight_decay)
        super(THEOPOULA, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(THEOPOULA, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params = [p.data for p in group['params']]
            grads = [p.grad for p in group['params']]

            eta, eps, beta, lr, weight_decay = group['eta'], group['eps'], group['beta'], group['lr'], group['weight_decay']

        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads])

        for ((device_params, device_grads), _) in grouped_tensors.values():

            if weight_decay != 0:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)

            boosting = torch._foreach_abs(device_grads)
            torch._foreach_add_(boosting, eps)
            torch._foreach_reciprocal_(boosting)
            torch._foreach_mul_(boosting, math.sqrt(lr))
            torch._foreach_add_(boosting, 1)
            numer = torch._foreach_mul(device_grads, boosting)

            taming = torch._foreach_abs(device_grads)
            torch._foreach_mul_(taming, math.sqrt(lr))
            torch._foreach_add_(taming, 1)


            noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)

            torch._foreach_addcdiv(device_params, numer, taming, -lr)._foreach_add_(noise)

            # if eta > 0:
            #         concat_params = torch.cat([p.view(-1) for p in group['params']])
            #         r = group['r']
            #         total_norm = torch.pow(torch.norm(concat_params), 2 * r)
            #
            #         reg_num = eta * p * total_norm
            #         reg_denom = 1 + math.sqrt(lr) * total_norm
            #         reg = reg_num/reg_denom
            #         p.data.add_(reg)

        return loss

