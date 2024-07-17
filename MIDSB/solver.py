from typing import Callable

import torch


class Solver():
    def sampling(self, x: torch.Tensor):
        raise NotImplementedError()


class DDIMSolver(Solver):
    def __init__(self, sde: object, model_fn:Callable =None, num_step: int =5, ot_ode: bool =False, device: torch.device=None):
        self.sde = sde
        self.model_fn = model_fn
        self.num_step = num_step
        self.ot_ode = ot_ode
        self.device = device if device is not None else sde.device

    def sampling(self, x):
        x = x.to(self.device, non_blocking=True)

        xs = [x]
        pred_x0s = []

        # time_uniform
        timesteps = torch.linspace(self.sde.t_max, self.sde.t_min, self.num_step + 1, device=self.device)

        for i in range(0, self.num_step):
            t, t_prev = timesteps[i], timesteps[i + 1]

            pred_x0 = self.model_fn(x, t)
            x = self.sde.p_posterior(t_prev, t, x, pred_x0, ot_ode=self.ot_ode)
            pred_x0s.append(pred_x0)
            xs.append(x)

        xs = [x.to('cpu', non_blocking=True) for x in xs]
        pred_x0s = [pred_x0.to('cpu', non_blocking=True) for pred_x0 in pred_x0s]
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)


class HybridSolver(Solver):
    def __init__(self, sde: object, model_fn:Callable = None,
                 num_step: int = 5, skip_type: str = 'time_uniform', ot_ode: bool = False,
                 device: torch.device = None):
        super().__init__()
        self.sde = sde
        self.model_fn = model_fn
        self.num_step = num_step
        self.skip_type = skip_type
        self.device = device if device is not None else sde.device
        self.ot_ode = ot_ode
        self.t_min = sde.t_min
        self.t_max = sde.t_max

        # Store SDE-related functions for convenience
        self.marginal_lambda = sde.marginal_lambda
        self.marginal_sigma = sde.marginal_sigma
        self.marginal_alpha = sde.marginal_alpha
        self.h = sde.h

    def inverse_lambda(self, lamb: torch.Tensor) -> torch.Tensor:
        """Invert the lambda function."""
        return (torch.log(1 + torch.exp(2 * lamb)) - 2 * lamb) / self.sde.beta

    def get_time_steps(self, skip_type: str, t_start: float, t_end: float, num_step: int):
        """Generate time steps based on the skip type."""
        if skip_type == 'logSNR':
            lambda_start = self.marginal_lambda(torch.tensor(t_start).to(self.device))
            lambda_end = self.marginal_lambda(torch.tensor(t_end).to(self.device))
            logSNR_steps = torch.linspace(lambda_start.cpu().item(), lambda_end.cpu().item(), num_step + 1).to(self.device)
            return self.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_start, t_end, num_step + 1).to(self.device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_start ** (1. / t_order), t_end ** (1. / t_order), num_step + 1).pow(t_order).to(self.device)
            return t
        else:
            raise ValueError(f"Unsupported skip_type '{skip_type}', must be one of 'logSNR', 'time_uniform', or 'time_quadratic'.")

    def get_orders_and_timesteps(self):
        """Determine the order of solver steps and corresponding time steps."""
        if self.num_step % 2 == 0:
            K = self.num_step // 2
            orders = [2] * K
        else:
            K = self.num_step // 2 + 1
            orders = [2] * (K - 1) + [1]

        if self.skip_type == 'logSNR':
            timesteps = self.get_time_steps(self.skip_type, self.t_max, self.t_min, K)
        else:
            timesteps = self.get_time_steps(self.skip_type, self.t_max, self.t_min, self.num_step)[torch.cumsum(torch.tensor([0] + orders), dim=0).to(self.device)]
        return timesteps, orders


    def first_order_ODE_Solver(self, x, s, t, model_fn):
        # DPM_Solver ++ 1, ODE
        h = self.h(t, s)
        model_s = model_fn(x, s)
        xt = (
                self.marginal_sigma(t) / self.marginal_sigma(s) * x
                - self.marginal_alpha(t) * torch.expm1(- h) * model_s
        )
        return xt, model_s

    def first_order_SDE_Solver(self, x, s, t, model_fn):
        h = self.h(t, s)
        model_s = model_fn(x, s)
        xt = (
                self.marginal_sigma(t) / self.marginal_sigma(s) * torch.exp(- h) * x
                - self.marginal_alpha(t) * torch.expm1(- 2.0 * h) * model_s
        )
        if not self.ot_ode and t > self.t_min:
            xt = xt + self.marginal_sigma(t) * torch.sqrt(- torch.expm1(- 2.0 * h)) * torch.randn_like(xt)
        return xt, model_s

    def second_order_SDE_solver(self, x, s, t, model_fn):
        s1 = s + 0.5 * (t - s)

        h = self.h(t, s)
        r = self.h(s1, s) / h
        x_s1, model_s = self.first_order_ODE_Solver(x, s, s1, model_fn)

        model_s1 = model_fn(x_s1, s1)
        xt = (
                self.marginal_sigma(t) / self.marginal_sigma(s) * torch.exp(- h) * x
                - self.marginal_alpha(t) * torch.expm1(-2 * h) * model_s
                - (0.5 / r) * self.marginal_alpha(t) * torch.expm1(- 2 * h) * (model_s1 - model_s)
        )
        if not self.ot_ode and t > self.t_min:
            xt = xt + self.marginal_sigma(t) * torch.sqrt(- torch.expm1(- 2.0 * h)) * torch.randn_like(xt)

        return xt, model_s

    def sampling(self, x):
        x = x.to(self.device, non_blocking=True)

        xs = [x]
        pred_x0s = []
        timesteps, orders = self.get_orders_and_timesteps()
        for step, order in enumerate(orders):
            s, t = timesteps[step], timesteps[step + 1]
            if order == 2:
                x, x0 = self.second_order_SDE_solver(x, s, t, self.model_fn)
            else:
                x, x0 = self.first_order_SDE_Solver(x, s, t, self.model_fn)
            xs.append(x)
            pred_x0s.append(x0)

        xs = [x.to('cpu', non_blocking=True) for x in xs]
        pred_x0s = [pred_x0.to('cpu', non_blocking=True) for pred_x0 in pred_x0s]

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
