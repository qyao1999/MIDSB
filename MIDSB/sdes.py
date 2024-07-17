import torch

from .solver import DDIMSolver, HybridSolver


def unsqueeze_xdim(z, xdim):
    """
    Add singleton dimensions to the tensor `z` to match the length of `xdim`.

    Args:
        z (torch.Tensor): The input tensor.
        xdim (tuple): The target dimensions to be unsqueezed.

    Returns:
        torch.Tensor: The unsqueezed tensor.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class DiffusionBridgeSDE():
    def __init__(self, beta=0.1, t_min=3e-2, t_max=1, loss_weight_type=None, device='cpu'):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        self.device = device
        self.beta = torch.tensor(beta).to(self.device)
        self.t_min = t_min
        self.t_max = t_max
        if loss_weight_type is None:
            loss_weight_type = 'constant'
        self.loss_weight_type = loss_weight_type

    def marginal_log_alpha(self, t):
        raise NotImplementedError()

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_alpha(t))

    def marginal_log_sigma(self, t):
        raise NotImplementedError()

    def marginal_sigma(self, t):
        return torch.exp(self.marginal_log_sigma(t))

    def marginal_sigma(self, t):
        return torch.exp(self.marginal_log_sigma(t))

    def marginal_lambda(self, t):
        return self.marginal_log_alpha(t) - self.marginal_log_sigma(t)

    def marginal_logSNR(self, t):
        return 2 * self.marginal_lambda(t)

    def marginal_SNR(self, t):
        return torch.exp(self.marginal_logSNR(t))

    def h(self, s, t):
        return self.marginal_lambda(s) - self.marginal_lambda(t)

    def q_sample(self, t, x0, x1, ot_ode=False, x0_bar=None):
        batch, *xdim = x0.shape

        m = torch.exp(2.0 * self.h(torch.ones_like(t, device=self.device) * self.t_max, t))
        mu_xT, mu_x0 = m * self.marginal_alpha(t) / self.marginal_alpha(torch.ones_like(t, device=self.device) * self.t_max), (1 - m) * self.marginal_alpha(t)
        var = self.marginal_sigma(t) ** 2 * (1 - m)

        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xT = unsqueeze_xdim(mu_xT, xdim)
        var = unsqueeze_xdim(var, xdim)

        if x0_bar is not None:
            x0_hat = unsqueeze_xdim(t ** 2, xdim) * x0_bar + (1 - unsqueeze_xdim(t ** 2, xdim)) * x0
            # if not ot_ode:
            #     x0_hat  += var.sqrt() * torch.randn_like(x0_hat)
            mean = mu_xT * x1 + mu_x0 * x0_hat
        else:
            mean = mu_xT * x1 + mu_x0 * x0

        x_t = mean
        if not ot_ode:
            x_t += var.sqrt() * torch.randn_like(mean)

        if x0_bar is not None:
            return x_t, x0_hat
        else:
            return x_t

    def p_posterior(self, t, s, x, x0, ot_ode=False):
        # x0 = self.marginal_alpha(t) / self.marginal_alpha(s) * x0
        m = torch.exp(2.0 * self.h(s, t))
        mu_xt, mu_x0 = m * self.marginal_alpha(t) / self.marginal_alpha(s), (1 - m) * self.marginal_alpha(t)

        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xt = unsqueeze_xdim(mu_xt, xdim)

        mean = mu_x0 * x0 + mu_xt * x

        xt_prev = mean

        if not ot_ode and t > self.t_min:
            var = self.marginal_sigma(t) ** 2 * (1 - m)
            var = unsqueeze_xdim(var, xdim)
            xt_prev += var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def compute_pred_x0(self, t, xt, net_out, clip_denoise=False):
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)

        batch, *xdim = xt.shape
        alpha_t = unsqueeze_xdim(alpha_t, xdim)
        sigma_t = unsqueeze_xdim(sigma_t, xdim)

        pred_x0 = (xt - sigma_t * net_out) / alpha_t
        return pred_x0

    def compute_label(self, t, x0, xt, x0_hat=None):
        xt = xt.detach()
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)

        batch, *xdim = x0.shape
        alpha_t = unsqueeze_xdim(alpha_t, xdim)
        sigma_t = unsqueeze_xdim(sigma_t, xdim)
        if x0_hat is not None:
            x0_hat = x0_hat.detach()
            label = (xt - x0_hat * alpha_t) / sigma_t
        else:
            label = (xt - x0 * alpha_t) / sigma_t

        return label

    def compute_weight(self, t):
        if self.loss_weight_type == 'constant':
            mse_loss_weight = torch.ones_like(t, device=self.device)
        elif self.loss_weight_type == 'snr':
            mse_loss_weight = torch.exp(self.marginal_logSNR())
        elif self.loss_weight_type.startswith("min_snr_"):
            k = float(self.loss_weight_type.split('min_snr_')[-1])
            snr = torch.exp(self.marginal_logSNR(t))
            mse_loss_weight = torch.stack([snr, k * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        else:
            raise NotImplementedError(f'unsupported weight type {self.loss_weight_type}')
        return mse_loss_weight

    def get_ddim_solver(self, model_fn, num_step=50, ot_ode=False):
        return DDIMSolver(self, model_fn=model_fn, num_step=num_step, ot_ode=ot_ode)


class VP_DiffusionBridgeSDE(DiffusionBridgeSDE):
    def __init__(self, beta=0.1, t_min=3e-2, t_max=1, loss_weight_type=None, device='cpu'):
        super().__init__(beta=beta, t_min=t_min, t_max=t_max, loss_weight_type=loss_weight_type, device=device)

    def marginal_log_alpha(self, t):
        return - 0.5 * t * self.beta

    def marginal_log_sigma(self, t):
        return 0.5 * torch.log(1. - torch.exp(2. * self.marginal_log_alpha(t)))

    def get_hybrid_solver(self, model_fn=None, num_step=50, skip_type='time_uniform', ot_ode=False):
        return HybridSolver(sde=self, model_fn=model_fn, num_step=num_step, skip_type=skip_type, ot_ode=ot_ode, device=self.device)


class VE_DiffusionBridgeSDE(DiffusionBridgeSDE):
    def __init__(self, beta=0.1, t_max=1, loss_weight_type=None, device='cpu'):
        super().__init__(beta=beta, t_max=t_max, loss_weight_type=loss_weight_type, device=device)

    def marginal_log_alpha(self, t):
        return torch.zeros_like(t, device=self.device)

    def marginal_log_sigma(self, t):
        return torch.log(t)
