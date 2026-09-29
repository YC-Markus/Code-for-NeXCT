"""Batched adaptation of Muon; see licenses/Muon.txt for upstream attribution.

Groups same-shaped matrices to reduce the number of GPU kernel launches.
"""

from collections import defaultdict

import torch


def _matrix_shape(parameter):
    if parameter.ndim == 4:
        return (parameter.shape[0], parameter.numel() // parameter.shape[0])
    if parameter.ndim == 2:
        return tuple(parameter.shape)
    raise ValueError(
        f"Muon parameters must be 2D or 4D, got {parameter.ndim}D"
    )


def _batched_zeropower_newton_schulz5(gradients, steps=5):
    """Apply the reference Muon Newton-Schulz iteration to a matrix batch."""

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = gradients.bfloat16()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1.0e-7)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    return x


class BatchedSingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Drop-in Muon+Adam that batches equal-shaped Muon parameters.

    S2 has hundreds of small expert tensors but only a few dozen distinct
    matrix shapes. The reference optimizer launches every Newton-Schulz
    matmul separately; this implementation turns each shape family into a
    batched matmul while retaining per-parameter optimizer state.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Every parameter group needs use_muon")
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
            else:
                group["lr"] = group.get("lr", 3.0e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1.0e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})

    @staticmethod
    def _shape_groups(parameters):
        groups = defaultdict(list)
        for parameter in parameters:
            rows, columns = _matrix_shape(parameter)
            key = (
                rows,
                columns,
                parameter.dtype,
                parameter.device,
            )
            groups[key].append(parameter)
        return groups.values()

    @torch.no_grad()
    def _step_muon_group(self, group):
        beta = group["momentum"]
        decay = 1.0 - group["lr"] * group["weight_decay"]
        for parameters in self._shape_groups(group["params"]):
            rows, columns = _matrix_shape(parameters[0])
            gradients = []
            momentum_buffers = []
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                gradients.append(gradient.reshape(rows, columns))
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum_buffers.append(
                    state["momentum_buffer"].reshape(rows, columns)
                )

            gradient_batch = torch.stack(gradients)
            momentum_batch = torch.stack(momentum_buffers)
            momentum_batch.lerp_(gradient_batch, 1.0 - beta)
            update_batch = gradient_batch.lerp_(momentum_batch, beta)
            update_batch = _batched_zeropower_newton_schulz5(
                update_batch, steps=5
            )
            update_batch.mul_(max(1.0, rows / columns) ** 0.5)

            torch._foreach_copy_(
                momentum_buffers, list(momentum_batch.unbind(0))
            )
            torch._foreach_mul_(parameters, decay)
            parameter_updates = [
                update.reshape_as(parameter)
                for update, parameter in zip(
                    update_batch.unbind(0), parameters
                )
            ]
            torch._foreach_add_(
                parameters, parameter_updates, alpha=-group["lr"]
            )

    @torch.no_grad()
    def _step_adam_group(self, group):
        parameters = list(group["params"])
        gradients = []
        exp_avgs = []
        exp_avg_sqs = []
        steps = []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            gradients.append(gradient)
            state = self.state[parameter]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] = 0
            state["step"] += 1
            steps.append(state["step"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

        if not parameters:
            return
        if any(step != steps[0] for step in steps):
            raise RuntimeError(
                "Batched auxiliary Adam requires synchronized step counts"
            )
        beta1, beta2 = group["betas"]
        torch._foreach_lerp_(exp_avgs, gradients, 1.0 - beta1)
        squared_gradients = torch._foreach_mul(gradients, gradients)
        torch._foreach_lerp_(exp_avg_sqs, squared_gradients, 1.0 - beta2)

        step = steps[0]
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        updates = torch._foreach_div(exp_avgs, bias1)
        denominators = torch._foreach_div(exp_avg_sqs, bias2)
        torch._foreach_sqrt_(denominators)
        torch._foreach_add_(denominators, group["eps"])
        torch._foreach_div_(updates, denominators)
        torch._foreach_mul_(
            parameters,
            1.0 - group["lr"] * group["weight_decay"],
        )
        torch._foreach_add_(parameters, updates, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss
