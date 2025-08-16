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
    def __init__(self, params, lr=1e-2, eta=0, betas=None, r=3, eps=1e-8, weight_decay=0):
        """
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            eta: Regularization parameter
            betas: List of inverse temperatures in descending order (highest to lowest)
            r: Parameter for regularization
            eps: Boosting parameter for Theopoula optimizer
            weight_decay: Weight decay factor
        """
        #  Convert params iterator to list to prevent it from being consumed
        param_list=list(params)

        # Verify params are not empty
        if len(param_list) == 0:
            raise ValueError("No parameters to optimize.")

        # Remove or modify the default value assignment
        if betas is None:
            raise ValueError("betas must be provided for THEOPOULA_REPLICA")
    
        # Print debug info
        #print(f"Initializing THEOPOULA_REPLICA with {len(betas)} temperatures")
        #print(f"Temperature ladder: {betas}")
        
        # Verify betas are in descending order
        if not all(betas[i] >= betas[i+1] for i in range(len(betas)-1)):
            raise ValueError("Inverse temperatures (betas) must be in descending order")
        
        # Store parameters
        self.lr = lr
        self.eta = eta
        self.betas = betas
        self.r = r
        self.eps = eps
        self.weight_decay = weight_decay
        
        defaults = dict(lr=lr, eta=eta, betas=betas, r=r, eps=eps, weight_decay=weight_decay)
        super(THEOPOULA_REPLICA, self).__init__(param_list, defaults)

        # Create THEOPOULA optimizers for each temperature
        self.theopoulas = []
        for beta in betas:
            self.theopoulas.append(
                THEOPOULA(param_list, lr=lr, eta=eta, beta=beta, r=r, eps=eps, weight_decay=weight_decay)
            )
        



        # Inizialize tracking variables
        self.total_attempts=0
        self.accepted_swaps=0
        self.param_updates = []
        self.grad_norms = []
        self.loss_values = []
        
        # NEW
        #self.step_counter=0
        # END NEw
        
        # Initialize swap history
        """
        self.swap_history = {
            'loss_values': [[] for _ in range(len(betas))],
            'delta_E': [[] for _ in range(len(betas)-1)],
            'acceptance_ratios': [[] for _ in range(len(betas)-1)],
            'accepted': [[] for _ in range(len(betas)-1)],
            'inverse_temperatures': {f'beta_{i}': beta for i, beta in enumerate(betas)}
        }
        """
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            for opt in self.theopoulas:
                opt.step(None)
            return None
        
        losses = []
        # Track gradient norms
        grad_norm = torch.norm(torch.stack([p.grad.norm() 
                             for group in self.param_groups 
                             for p in group['params'] if p.grad is not None]))
        self.grad_norms.append(grad_norm.item())

        # Step each optimizer and collect losses
        for i, opt in enumerate(self.theopoulas):
            opt.zero_grad()
            with torch.enable_grad():
                loss = closure()
                loss_val = loss.detach().item()
                losses.append(loss_val)
                #self.swap_history['loss_values'][i].append(loss_val)
                
                if i == 0:  # Track first optimizer's loss
                    self.loss_values.append(loss_val)
                    # Track parameter updates for first optimizer
                    old_params = [p.data.clone() for group in self.param_groups for p in group['params']]
            opt.step()

            if i == 0:  # Calculate parameter updates for first optimizer
                new_params = [p.data for group in self.param_groups for p in group['params']]
                update_size = torch.norm(torch.stack([torch.norm(new-old) 
                                        for new, old in zip(new_params, old_params)]))
                self.param_updates.append(update_size.item())
        
        # NEW 
        #self.step_counter +=1
        #if self.step_counter % 10 ==0:
            # Attempt swaps between adjacent pairs
        device = next(iter(self.theopoulas[0].param_groups[0]['params'])).device
        for i in range(len(self.theopoulas)-1):
            self._attempt_swap(
                self.theopoulas[i], 
                self.theopoulas[i+1],
                losses[i],
                losses[i+1],
                self.betas[i],
                self.betas[i+1],
                i,
                device)
        # END NEW
        
        #if self.total_attempts % 20 == 0:   # Changed from 10 before.
        #    self._log_detailed_stats()
        return min(losses)  
    
    def _attempt_swap(self, opt_a, opt_b, loss_a, loss_b, beta_a, beta_b, pair_idx, device):
        #delta_beta = (beta_a-beta_b)
        #delta_E =  (loss_a-loss_b)
    
        #self.swap_history['delta_E'][pair_idx].append(delta_E)
        #self.total_attempts += 1
        if loss_a>loss_b:
            #self.accepted_swaps += 1
            self._exchange_parameters(opt_a, opt_b)


        #delta = torch.tensor(delta_beta * delta_E, device=device)
        #acceptance_ratio = torch.min(torch.tensor(1.0), torch.exp(delta))
        #self.swap_history['acceptance_ratios'][pair_idx].append(acceptance_ratio.item())
    
        #random_number = torch.rand(1, device=device).item()
        #self.total_attempts += 1
    
        #accepted = random_number < acceptance_ratio
        #self.swap_history['accepted'][pair_idx].append(accepted)
    
        #if accepted:
        #    self.accepted_swaps += 1
        #    self._exchange_parameters(opt_a, opt_b)

    def _exchange_parameters(self, optimizer_a, optimizer_b):
        """Exchange parameters between two optimizers"""
        for param_a, param_b in zip(optimizer_a.param_groups[0]['params'], 
                                  optimizer_b.param_groups[0]['params']):
            param_a.data, param_b.data = param_b.data.clone(), param_a.data.clone()
    
    def get_acceptance_rate(self):
        """Return the current acceptance rate for all pairs"""
        if self.total_attempts == 0:
            return [0.0] * (len(self.betas) - 1)
        
        rates = []
        for i in range(len(self.betas) - 1):
            accepted = sum(self.swap_history['accepted'][i])
            attempts = len(self.swap_history['accepted'][i])
            rates.append(accepted / attempts if attempts > 0 else 0.0)
        return rates
    
    def get_convergence_metrics(self):
        """Return extended metrics related to convergence and replica exchange"""
        return {
            'grad_norms': self.grad_norms,
            'param_updates': self.param_updates,
            'loss_values': self.loss_values,
            'acceptance_rates': self.get_acceptance_rate(),
            'swap_history': self.swap_history
        }
    
    def _log_detailed_stats(self):
        """Log detailed statistics about the replica exchange process"""
        recent_idx = -10  # Look at last 10 attempts
        
        print("\nDetailed Replica Exchange Statistics:")
        #print(f"Number of replicas: {len(self.betas)}")
        
        # Print inverse temperatures
        #print("\nInverse Temperatures:")
        #for i, beta in enumerate(self.betas):
        #    print(f"  β_{i}: {beta:.2e}")
        
        # Print recent loss statistics for each replica
        #print("\nLoss Statistics (last 10 steps):")
        #for i in range(len(self.betas)):
        #    recent_losses = self.swap_history['loss_values'][i][recent_idx:]
        #    if recent_losses:
        #        mean_loss = np.mean(recent_losses)
        #        std_loss = np.std(recent_losses)
        #        print(f"  Replica {i}: {mean_loss:.4f} ± {std_loss:.4f}")
        
        # Print swap statistics for each adjacent pair
        print("\nSwap Statistics (last 10 attempts):")
        for i in range(len(self.betas) - 1):
            recent_ratios = self.swap_history['acceptance_ratios'][i][recent_idx:]
            recent_accepted = self.swap_history['accepted'][i][recent_idx:]
            
            if recent_ratios:
                mean_ratio = np.mean(recent_ratios)
                recent_rate = sum(recent_accepted) / len(recent_accepted)
                print(f"\nPair {i}-{i+1}:")
                print(f"  Recent acceptance ratio: {mean_ratio:.4f}")
                print(f"  Recent acceptance rate: {recent_rate:.4f}")
                
                # Warnings for acceptance rates
                if recent_rate > 0.8:
                    print(f"  Warning: Acceptance rate is very high ({recent_rate:.4f})")
                    print("  Consider increasing temperature difference")
                elif recent_rate < 0.2:
                    print(f"  Warning: Acceptance rate is very low ({recent_rate:.4f})")
                    print("  Consider decreasing temperature difference")
        
        # Print overall statistics
        if self.total_attempts > 0:
            overall_rates = self.get_acceptance_rate()
            print("\nOverall Acceptance Rates:")
            for i, rate in enumerate(overall_rates):
                print(f"  Pair {i}-{i+1}: {rate:.4f}")
        else:
            print("\nNo swap attempts yet")
    
    def __setstate__(self, state):
        """Restore optimizer state"""
        super(THEOPOULA_REPLICA, self).__setstate__(state)
        
        # Restore parameters from defaults
        defaults = self.defaults
        self.lr = defaults['lr']
        self.eta = defaults['eta']
        self.betas = defaults['betas']
        self.r = defaults['r']
        self.eps = defaults['eps']
        self.weight_decay = defaults['weight_decay']
        
        # Restore tracking variables if not present
        if not hasattr(self, 'total_attempts'):
            self.total_attempts = 0
        if not hasattr(self, 'accepted_swaps'):
            self.accepted_swaps = 0
        if not hasattr(self, 'param_updates'):
            self.param_updates = []
        if not hasattr(self, 'grad_norms'):
            self.grad_norms = []
        if not hasattr(self, 'loss_values'):
            self.loss_values = []
        if not hasattr(self, 'swap_history'):
            self.swap_history = {
                'loss_values': [[] for _ in range(len(self.betas))],
                'delta_E': [[] for _ in range(len(self.betas)-1)],
                'acceptance_ratios': [[] for _ in range(len(self.betas)-1)],
                'accepted': [[] for _ in range(len(self.betas)-1)],
                'inverse_temperatures': {f'beta_{i}': beta for i, beta in enumerate(self.betas)}
            }