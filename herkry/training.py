"""Checkpoint-compatible optimizer and supervision extracted from the evaluated trainer."""
import torch
import torch.nn.functional as F
from .optim import BatchedSingleDeviceMuonWithAuxAdam

def supervision_weights(stage_loss_ratio):
    ratio = float(stage_loss_ratio)
    return {
        32: ratio**-3,
        64: ratio**-2,
        128: ratio**-1,
        256: 1.0,
    }


def build_optimizer(
    model, learning_rate, logger, mixer_optimizer="adam"
):
    muon_ids = set()
    adam_ids = set()
    for module in model.modules():
        if module.__class__.__name__ == "CondConv2d":
            for parameter in module.expert_weights.parameters():
                if parameter.requires_grad:
                    muon_ids.add(id(parameter))
            if module.expert_biases is not None:
                for parameter in module.expert_biases.parameters():
                    if parameter.requires_grad:
                        adam_ids.add(id(parameter))
            if module.router is not None:
                for parameter in module.router.parameters():
                    if parameter.requires_grad:
                        target = (
                            muon_ids
                            if parameter.ndim >= 2
                            else adam_ids
                        )
                        target.add(id(parameter))
        if module.__class__.__name__ == "KernelSubspaceCondConv2d":
            if module.expert_bias is not None:
                adam_ids.add(id(module.expert_bias))
        if (
            mixer_optimizer == "adam"
            and module.__class__.__name__
            in (
                "TrajectoryContextGaussianBlock",
                "UnrolledTrajectoryContextGaussianBlock",
            )
        ):
            # A 1x24 mixer is a vector regression parameter, not a matrix
            # representation suitable for Muon orthogonalization.
            for parameter in (
                module.linear_feature_to_scalar.parameters()
            ):
                adam_ids.add(id(parameter))
                muon_ids.discard(id(parameter))

    muon_parameters = []
    adam_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in adam_ids:
            adam_parameters.append(parameter)
        elif parameter_id in muon_ids:
            muon_parameters.append(parameter)
        elif parameter.ndim < 2:
            adam_parameters.append(parameter)
        elif parameter.ndim in (2, 4):
            muon_parameters.append(parameter)
        else:
            adam_parameters.append(parameter)
    trainable_count = sum(
        1 for parameter in model.parameters() if parameter.requires_grad
    )
    if len(muon_parameters) + len(adam_parameters) != trainable_count:
        raise RuntimeError("Optimizer parameter partition is incomplete")
    logger.info(
        "optimizer | Muon tensors=%d numel=%d | "
        "Adam tensors=%d numel=%d | 1x1 mixers=%s",
        len(muon_parameters),
        sum(item.numel() for item in muon_parameters),
        len(adam_parameters),
        sum(item.numel() for item in adam_parameters),
        mixer_optimizer,
    )
    return BatchedSingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": 0.002,
                "weight_decay": 0.01,
            },
            {
                "params": adam_parameters,
                "use_muon": False,
                "lr": learning_rate,
                "betas": (0.9, 0.95),
                "weight_decay": 0.01,
            },
        ]
    )


def image_only_loss(
    model, image_outputs, target, stage_loss_ratio=4
):
    total = target.new_tensor(0.0)
    weight_sum = 0.0
    weights = dict(zip(sorted(set(model.output_resolutions)),
                       [float(stage_loss_ratio)**-i for i in (3, 2, 1, 0)]))
    for image, resolution in zip(
        image_outputs, model.output_resolutions
    ):
        weight = weights[resolution]
        total = total + weight * F.l1_loss(image, target)
        weight_sum += weight
    return total / weight_sum


def pre_cgls_image_loss(
    model, aux_outputs, target, stage_loss_ratio=4,
    stage_ratios=(1.0, 1.0, 1.0, 1.0),
):
    """Match direct-loss stage weights without renormalizing skipped cells."""

    total = target.new_tensor(0.0)
    weights = dict(zip(sorted(set(model.output_resolutions)),
                       [float(stage_loss_ratio)**-i for i in (3, 2, 1, 0)]))
    weight_sum = sum(
        weights[resolution] for resolution in model.output_resolutions
    )
    resolutions = sorted(set(model.output_resolutions))
    if len(resolutions) != 4 or len(stage_ratios) != 4:
        raise ValueError("Expected four stage resolutions and auxiliary weights")
    stage_scale = dict(zip(resolutions, stage_ratios))
    for aux, resolution in zip(
        aux_outputs, model.output_resolutions
    ):
        initial_image = aux.get("cgls_initial_image_for_loss")
        if initial_image is not None and stage_scale[resolution] > 0:
            total = total + weights[resolution] * stage_scale[resolution] * F.l1_loss(
                initial_image, target
            )
    return total / weight_sum


def set_stage_cgls_schedule(model, schedule):
    for stage_blocks, iterations in zip(model.stage_blocks, schedule):
        for block in stage_blocks:
            block.cgls_iterations = int(iterations)
