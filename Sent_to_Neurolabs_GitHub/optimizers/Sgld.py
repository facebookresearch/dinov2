import math
import torch
from torch.optim.optimizer import Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SGLD(Optimizer):

    def __init__(self, params, lr=1e-1, beta=1e14, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)


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
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)


                if len(state) == 0:
                    state['step'] = 0

                beta, lr = group['beta'], group['lr']

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)
                #numer = grad * ( 1 + math.sqrt(lr)/(group['eps'] + torch.abs(grad)))
                #denom = 1 + math.sqrt(lr) * torch.abs(grad)

                p.data.add_(-lr*grad).add_(noise)

        return loss
