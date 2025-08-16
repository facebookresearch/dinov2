import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MOMENTUM_THEOPOULA(Optimizer):

    def __init__(self, params, lr=1e-1, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0, momentum=0.9):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, eps=eps, weight_decay=weight_decay, momentum=0.9)
        super(MOMENTUM_THEOPOULA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MOMENTUM_THEOPOULA, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                


                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer']=torch.zeros_like(p.data)
                
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                eta, beta, lr, eps, momentum = group['eta'], group['beta'], group['lr'], group['eps'], group['momentum']

                state['step']+=1

                buf =state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                g_abs = torch.abs(buf)

                #numer = grad * ( 1 + math.sqrt(lr)/(eps + g_abs))
                #denom = 1 + math.sqrt(lr) * g_abs

                numer = torch.mul(buf, 1 + torch.mul(torch.reciprocal(torch.add(g_abs, eps)), math.sqrt(lr)))
                denom = torch.add(torch.mul(g_abs, math.sqrt(lr)), 1)


                #numer = grad * ((eps + g_abs) + math.sqrt(lr))
                #denom = (1 + math.sqrt(lr) * g_abs) * (eps + g_abs)

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)


                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)


                if eta > 0:
                    concat_params = torch.cat([p.view(-1) for p in group['params']])
                    r = group['r']
                    total_norm = torch.pow(torch.norm(concat_params), 2 * r)

                    reg_num = eta * p * total_norm
                    reg_denom = 1 + math.sqrt(lr) * total_norm
                    reg = reg_num/reg_denom
                    p.data.add_(reg)

        return loss

