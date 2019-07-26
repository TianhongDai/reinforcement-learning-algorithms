import numpy as np
import torch
from torch.distributions.normal import Normal

# select actions
def select_actions(pi):
    mean, std = pi
    normal_dist = Normal(mean, std)
    return normal_dist.sample().detach().numpy().squeeze()

# evaluate the actions
def eval_actions(pi, actions):
    mean, std = pi
    normal_dist = Normal(mean, std)
    return normal_dist.log_prob(actions).sum(dim=1, keepdim=True)

# conjugated gradient
def conjugated_gradient(fvp, b, update_steps, obs, pi_old, residual_tol=1e-10):
    # the initial solution is zero
    x = torch.zeros(b.size(), dtype=torch.float32)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(update_steps):
        fv_product = fvp(p, obs, pi_old)
        alpha = rdotr / torch.dot(p, fv_product)
        x = x + alpha * p
        r = r - alpha * fv_product
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr 
        p = r + beta * p
        rdotr = new_rdotr
        # if less than residual tot.. break
        if rdotr < residual_tol:
            break
    return x

# line search
def line_search(model, loss_fn, x, full_step, expected_rate, obs, adv, actions, pi_old, max_backtracks=10, accept_ratio=0.1):
    fval = loss_fn(obs, adv, actions, pi_old).data
    for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * full_step
        set_flat_params_to(model, xnew)
        new_fval = loss_fn(obs, adv, actions, pi_old).data
        actual_improve = fval - new_fval
        expected_improve = expected_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def set_flat_params_to(model, flat_params):
    prev_indx = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_indx:prev_indx + flat_size].view(param.size()))
        prev_indx += flat_size
