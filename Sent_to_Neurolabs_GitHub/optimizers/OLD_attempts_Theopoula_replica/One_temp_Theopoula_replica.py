import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class THEOPOULA(Optimizer):

    def __init__(self, params, lr=1e-2, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0):
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

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)


                if len(state) == 0:
                    state['step'] = 0

                eta, beta, lr, eps = group['eta'], group['beta'], group['lr'], group['eps']

                g_abs = torch.abs(grad)

                #numer = grad * ( 1 + math.sqrt(lr)/(eps + g_abs))
                #denom = 1 + math.sqrt(lr) * g_abs

                numer = torch.mul(grad, 1 + torch.mul(torch.reciprocal(torch.add(g_abs, eps)), math.sqrt(lr)))
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

        return loss.detach() if loss is not None else None

class THEOPOULA_REPLICA(Optimizer):
    def __init__(self, params, lr=1e-2, eta=0, beta=1e14, beta_s=1e12, r=3, eps=1e-8, weight_decay=0):

        # New Add tracking variables
        self.total_attempts=0
        self.accepted_swaps=0
        # New tracking variables for convergence
        self.param_updates = []
        self.grad_norms = []
        self.loss_values = []
        
        
        # Add new tracking variables for detailed swap history
        self.swap_history = {
            'loss1_values': [],
            'loss2_values': [],
            'delta_E': [],
            'acceptance_ratios': [],
            'accepted': [],
            'inverse_temperatures': {'beta': beta, 'beta_s': beta_s}
        }



        # end 

        # New
        #  Convert params iterator to list to prevent it from being consumed
        param_list=list(params)
        # Verify params are not empty
        if len(param_list) == 0:
            raise ValueError("No parameters to optimize.")
        # New

        # New
        defaults = dict(lr=lr, eta=eta, beta=beta, beta_s=beta_s, r=r, eps=eps, weight_decay=weight_decay)
        super(THEOPOULA_REPLICA, self).__init__(param_list, defaults)
        # New

        # Create two THEOPOULA optimizers
        self.theopoula1 = THEOPOULA(param_list, lr=lr, eta=eta, beta=beta, r=r, eps=eps, weight_decay=weight_decay)
        self.theopoula2 = THEOPOULA(param_list, lr=lr, eta=eta, beta=beta_s, r=r, eps=eps, weight_decay=weight_decay)
        self.params = params
        self.lr = lr
        self.eta = eta
        self.beta = beta
        self.beta_s=beta_s
        self.r = r
        self.eps = eps
        self.weight_decay = weight_decay



    def __setstate__(self, state):
        super(THEOPOULA_REPLICA, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss1=None
        loss2=None

        if closure is not None:

            # Track gradient norms before update
            grad_norm = torch.norm(torch.stack([p.grad.norm() 
                                 for group in self.param_groups 
                                 for p in group['params'] if p.grad is not None]))
            self.grad_norms.append(grad_norm.item())



            # First optimizer step with fresh gradients
            self.theopoula1.zero_grad()
            with torch.enable_grad():
                loss1=closure()
                loss1_val=loss1.detach().item() # Get numerical values
                # New
                self.loss_values.append(loss1_val)
            
            # New Store parameter values before update
            old_params = [p.data.clone() for group in self.param_groups for p in group['params']]

            self.theopoula1.step()

            # Calculate parameter updates magnitude
            new_params = [p.data for group in self.param_groups for p in group['params']]
            update_size = torch.norm(torch.stack([torch.norm(new-old) 
                                    for new, old in zip(new_params, old_params)]))
            self.param_updates.append(update_size.item())


            # Recompute gradients with second optimizer
            self.theopoula2.zero_grad()
            with torch.enable_grad():
                loss2=closure()
                loss2_val=loss2.detach().item() # Get numerical values
            self.theopoula2.step()

            device = next(iter(self.theopoula1.param_groups[0]['params'])).device

            # After computing both losses, add detailed logging
            self.swap_history['loss1_values'].append(loss1_val)
            self.swap_history['loss2_values'].append(loss2_val)

            # Metropolis-Hastings criterion for replica exchange
            delta_beta = (self.beta_s - self.beta)  # Difference in inverse temperatures
            delta_E = loss2_val - loss1_val  # Energy (loss) difference

            self.swap_history['delta_E'].append(delta_E)

            # Convert to tensor and calculate acceptance probability
            delta = torch.tensor(delta_beta * delta_E, device=device)
            #acceptance_ratio = torch.exp(delta)
            acceptance_ratio = torch.min(torch.tensor(1.0), torch.exp(-delta))
            #log_acceptance_ratio = torch.min(torch.tensor(0.0), -delta)
            self.swap_history['acceptance_ratios'].append(acceptance_ratio.item())
            
            # Random number for acceptance/rejection
            random_number = torch.rand(1, device=device).item()

            # Track attempts and acceptances
            self.total_attempts+=1
            #log_acceptance_ratio = torch.min(torch.tensor(0.0), delta)
            #accepted = torch.log(torch.rand(1, device=device)) < log_acceptance_ratio
            accepted = random_number < acceptance_ratio
            self.swap_history['accepted'].append(accepted)

            # Accept or reject based on Metropolis criterion
            if accepted:
                self.accepted_swaps +=1
                self._exchange_parameters(self.theopoula1, self.theopoula2)

            # Detailed logging every N steps
            if self.total_attempts % 10 == 0:  # Adjust frequency as needed
                self._log_detailed_stats()

        else:
            # Handle the case where no closure is provided
            self.theopoula1.step()
            self.theopoula2.step()
        # Return the loss from the first optimizer
        return loss1

    def _exchange_parameters(self, optimizer_a, optimizer_b):
        # Exchange parameters between two optimizers
        for param_a, param_b in zip(optimizer_a.param_groups[0]['params'], optimizer_b.param_groups[0]['params']):
            param_a.data, param_b.data = param_b.data.clone(), param_a.data.clone()
    
    # New for the swapping rate
    def get_acceptance_rate(self):
        """Return the current acceptance rate"""
        if self.total_attempts == 0:
            return 0.0
        return self.accepted_swaps / self.total_attempts
    
    def get_convergence_metrics(self):
        """Return metrics related to convergence"""
        return {
            'grad_norms': self.grad_norms,
            'param_updates': self.param_updates,
            'loss_values': self.loss_values,
            'acceptance_rate': self.get_acceptance_rate()
        }
    
    def _log_detailed_stats(self):
        """Log detailed statistics about the replica exchange process"""
        recent_idx = -10  # Look at last 10 attempts
    
        # Calculate recent statistics
        recent_loss1 = self.swap_history['loss1_values'][recent_idx:]
        recent_loss2 = self.swap_history['loss2_values'][recent_idx:]
        recent_delta_E = self.swap_history['delta_E'][recent_idx:]
        recent_acceptance = self.swap_history['acceptance_ratios'][recent_idx:]
    
        print("\nDetailed Replica Exchange Statistics:")
        print(f"Inverse Temperature 1 (beta): {self.swap_history['inverse_temperatures']['beta']:.2e}")
        print(f"Inverse Temperature 2 (beta_s): {self.swap_history['inverse_temperatures']['beta_s']:.2e}")
        print(f"Recent Loss 1 mean ± std: {np.mean(recent_loss1):.4f} ± {np.std(recent_loss1):.4f}")
        print(f"Recent Loss 2 mean ± std: {np.mean(recent_loss2):.4f} ± {np.std(recent_loss2):.4f}")
        print(f"Recent ΔE mean ± std: {np.mean(recent_delta_E):.4f} ± {np.std(recent_delta_E):.4f}")
        print(f"Recent acceptance ratio mean: {np.mean(recent_acceptance):.4f}")
        print(f"Overall acceptance rate: {self.get_acceptance_rate():.4f}")
    
    def get_convergence_metrics(self):
        """Return extended metrics related to convergence and replica exchange"""
        return {'grad_norms': self.grad_norms, 'param_updates': self.param_updates,'loss_values': self.loss_values,'acceptance_rate': self.get_acceptance_rate(),'swap_history': self.swap_history}
