"""Scale-normalized first/second-order 2-D Gaussian mode renderer.

The expensive Gaussian rasterization remains a single multi-channel call to
the production Triton renderer. Local-axis derivative coefficients are
analytically rotated into global image derivative components before rendering.
A single grouped convolution synthesizes all active derivative modes. Its
explicit transpose is used by the multi-mode CGLS operator, so the forward and
adjoint remain an exact discrete pair, including image boundaries.
"""

import torch
import torch.nn.functional as F


MODE_NAMES = ("Gx", "Gy", "Gxx", "Gxy", "Gyy")


def _active_mode_count(order):
    return 1 + 2 * (int(order) >= 1) + 3 * (int(order) >= 2)


def _derivative_kernels(reference, order):
    """Return Dx, Dy[, Dxx, Dxy, Dyy] cross-correlation kernels."""

    values = [
        ((0.0, 0.0, 0.0), (-0.5, 0.0, 0.5), (0.0, 0.0, 0.0)),
        ((0.0, -0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.5, 0.0)),
    ]
    if int(order) >= 2:
        values.extend(
            [
                ((0.0, 0.0, 0.0), (1.0, -2.0, 1.0), (0.0, 0.0, 0.0)),
                ((0.25, 0.0, -0.25), (0.0, 0.0, 0.0), (-0.25, 0.0, 0.25)),
                ((0.0, 1.0, 0.0), (0.0, -2.0, 0.0), (0.0, 1.0, 0.0)),
            ]
        )
    return reference.new_tensor(values).unsqueeze(1)


def combine_global_derivative_components(components, order):
    """Map global derivative-component images to one residual image."""

    component_count = _active_mode_count(order) - 1
    if components.shape[1] != component_count:
        raise ValueError(
            f"Order {order} needs {component_count} derivative components, "
            f"got {components.shape[1]}"
        )
    filtered = filter_global_derivative_components(components, order)
    return filtered.sum(dim=1, keepdim=True)


def filter_global_derivative_components(components, order):
    """Return each filtered global component before fixed summation."""

    component_count = _active_mode_count(order) - 1
    if components.shape[1] != component_count:
        raise ValueError(
            f"Order {order} needs {component_count} derivative components, "
            f"got {components.shape[1]}"
        )
    kernels = _derivative_kernels(components, order)
    return F.conv2d(
        components, kernels, padding=1, groups=component_count
    )


def derivative_component_adjoint(image, order):
    """Exact transpose of :func:`combine_global_derivative_components`."""

    component_count = _active_mode_count(order) - 1
    if image.shape[1] != 1:
        raise ValueError("Derivative adjoint expects a single image channel")
    kernels = _derivative_kernels(image, order)
    repeated = image.expand(-1, component_count, -1, -1).contiguous()
    return F.conv_transpose2d(
        repeated, kernels, padding=1, groups=component_count
    )


def _geometry_factors(geometry, render_size):
    sigma_x = geometry[..., 2:3] * float(render_size)
    sigma_y = geometry[..., 3:4] * float(render_size)
    theta = geometry[..., 4:5]
    return sigma_x, sigma_y, torch.cos(theta), torch.sin(theta)


def local_to_global_modes(geometry, local_modes, order, render_size):
    """Rotate [Gx,Gy,(Gxx,Gxy,Gyy)] coefficients to image axes."""

    order = int(order)
    expected = _active_mode_count(order) - 1
    if local_modes.shape[-1] != expected:
        raise ValueError(
            f"Order {order} expects {expected} local modes, got "
            f"{local_modes.shape[-1]}"
        )
    sigma_x, sigma_y, cosine, sine = _geometry_factors(
        geometry, render_size
    )
    gx, gy = local_modes[..., 0:1], local_modes[..., 1:2]
    wx = gx * sigma_x * cosine - gy * sigma_y * sine
    wy = gx * sigma_x * sine + gy * sigma_y * cosine
    parts = [wx, wy]
    if order == 2:
        gxx, gxy, gyy = (
            local_modes[..., 2:3],
            local_modes[..., 3:4],
            local_modes[..., 4:5],
        )
        sx2 = sigma_x.square()
        sy2 = sigma_y.square()
        sxsy = sigma_x * sigma_y
        c2 = cosine.square()
        s2 = sine.square()
        cs = cosine * sine
        parts.extend(
            [
                gxx * sx2 * c2 - gxy * sxsy * cs + gyy * sy2 * s2,
                2.0 * gxx * sx2 * cs
                + gxy * sxsy * (c2 - s2)
                - 2.0 * gyy * sy2 * cs,
                gxx * sx2 * s2 + gxy * sxsy * cs + gyy * sy2 * c2,
            ]
        )
    return torch.cat(parts, dim=-1).contiguous()


def global_to_local_mode_adjoint(
    geometry, global_adjoint, order, render_size
):
    """Exact coefficient-space transpose of :func:`local_to_global_modes`."""

    order = int(order)
    sigma_x, sigma_y, cosine, sine = _geometry_factors(
        geometry, render_size
    )
    ax, ay = global_adjoint[..., 0:1], global_adjoint[..., 1:2]
    parts = [
        sigma_x * (cosine * ax + sine * ay),
        sigma_y * (-sine * ax + cosine * ay),
    ]
    if order == 2:
        axx, axy, ayy = (
            global_adjoint[..., 2:3],
            global_adjoint[..., 3:4],
            global_adjoint[..., 4:5],
        )
        sx2 = sigma_x.square()
        sy2 = sigma_y.square()
        sxsy = sigma_x * sigma_y
        c2 = cosine.square()
        s2 = sine.square()
        cs = cosine * sine
        parts.extend(
            [
                sx2 * (c2 * axx + 2.0 * cs * axy + s2 * ayy),
                sxsy * (-cs * axx + (c2 - s2) * axy + cs * ayy),
                sy2 * (s2 * axx - 2.0 * cs * axy + c2 * ayy),
            ]
        )
    return torch.cat(parts, dim=-1).contiguous()


def prepare_cgls_render_amplitudes(geometry, coefficients, order, render_size):
    """Pack local [G,Gx,...] coefficients for one cached renderer call."""

    expected = _active_mode_count(order)
    if coefficients.shape[-1] != expected:
        raise ValueError(
            f"Order {order} CGLS expects {expected} modes, got "
            f"{coefficients.shape[-1]}"
        )
    if int(order) == 0:
        return coefficients.contiguous()
    global_modes = local_to_global_modes(
        geometry, coefficients[..., 1:], order, render_size
    )
    return torch.cat([coefficients[..., :1], global_modes], dim=-1)


def unpack_cgls_render_adjoint(
    geometry, global_adjoint, order, render_size
):
    """Transpose packed global renderer coefficients back to local modes."""

    if int(order) == 0:
        return global_adjoint.contiguous()
    local = global_to_local_mode_adjoint(
        geometry, global_adjoint[..., 1:], order, render_size
    )
    return torch.cat([global_adjoint[..., :1], local], dim=-1)


def prepare_derivative_render_amplitudes(
    geometry,
    zero_order_coefficients,
    higher_order_coefficients,
    scalar_coefficients,
    order,
    render_size,
):
    """Rotate local derivative coefficients into global derivative channels.

    ``higher_order_coefficients`` is mode-major with five modes per nonlinear
    feature: Gx, Gy, Gxx, Gxy, Gyy.  Sigma is converted from normalized image
    units to pixels because the discrete derivative operators act per pixel.
    """

    order = int(order)
    if order not in (0, 1, 2):
        raise ValueError(f"Gaussian derivative order must be 0, 1, or 2; got {order}")
    if order == 0:
        return torch.cat([zero_order_coefficients, scalar_coefficients], dim=-1)

    batch, count, _ = zero_order_coefficients.shape
    if higher_order_coefficients.shape[:2] != (batch, count):
        raise ValueError("Higher-order coefficient batch/count mismatch")
    if higher_order_coefficients.shape[-1] % 5 != 0:
        raise ValueError("Higher-order coefficient channels must be 5 * K")
    mode_channels = higher_order_coefficients.shape[-1] // 5
    expected = 5 * mode_channels
    if higher_order_coefficients.shape != (batch, count, expected):
        raise ValueError(
            "Higher-order coefficients must have shape "
            f"{(batch, count, expected)}, got {tuple(higher_order_coefficients.shape)}"
        )
    modes = higher_order_coefficients.reshape(
        batch, count, 5, mode_channels
    )
    gx, gy, gxx, gxy, gyy = modes.unbind(dim=2)

    active_local = torch.cat([gx, gy], dim=-1)
    if order == 2:
        active_local = torch.cat([active_local, gxx, gxy, gyy], dim=-1)
    global_modes = local_to_global_modes(
        geometry, active_local, order, render_size
    )
    parts = [zero_order_coefficients]
    parts.extend(global_modes.split(mode_channels, dim=-1))
    parts.append(scalar_coefficients)
    return torch.cat(parts, dim=-1).contiguous()


def combine_derivative_render(
    rendered,
    nonlinear_channels,
    order,
    mode_channels=1,
    mixed_order_features=False,
):
    """Append a compact derivative residual feature to zero-order features."""

    channels = int(nonlinear_channels)
    mode_channels = int(mode_channels)
    order = int(order)
    expected = (
        channels
        + (2 * (order >= 1) + 3 * (order >= 2)) * mode_channels
        + 1
    )
    if rendered.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} rendered channels for order {order}, "
            f"got {rendered.shape[1]}"
        )
    offset = 0
    zero_order = rendered[:, offset : offset + channels]
    offset += channels
    component_count = _active_mode_count(order) - 1
    component_channels = component_count * mode_channels
    components = rendered[:, offset : offset + component_channels]
    offset += component_channels
    filtered = None
    if component_count == 0:
        derivative_residual = zero_order.new_zeros(
            zero_order.shape[0], mode_channels, *zero_order.shape[2:]
        )
    elif mode_channels != 1:
        # Keep feature groups independent while applying the same mode filters.
        batch, _, height, width = components.shape
        components = components.reshape(
            batch, component_count, mode_channels, height, width
        ).transpose(1, 2).reshape(
            batch * mode_channels, component_count, height, width
        )
        derivative_residual = combine_global_derivative_components(
            components, order
        ).reshape(batch, mode_channels, height, width)
    else:
        filtered = filter_global_derivative_components(components, order)
        derivative_residual = filtered.sum(dim=1, keepdim=True)
    if mixed_order_features:
        if mode_channels != 1:
            raise ValueError("Mixed-order features currently require one mode channel")
        if component_count == 0:
            first_order = derivative_residual
            second_order = derivative_residual
        else:
            first_order = filtered[:, :2].sum(dim=1, keepdim=True)
            second_order = (
                filtered[:, 2:].sum(dim=1, keepdim=True)
                if int(order) >= 2
                else torch.zeros_like(first_order)
            )
        derivative_features = torch.cat(
            [first_order, second_order, first_order * second_order], dim=1
        )
    else:
        derivative_features = derivative_residual
    scalar = rendered[:, offset : offset + 1]
    return torch.cat([zero_order, derivative_features], dim=1), scalar


def coefficient_abs_mean(higher_order_coefficients, nonlinear_channels, order):
    """Return per-sample Gx/Gy/Gxx/Gxy/Gyy coefficient diagnostics."""

    batch, count, _ = higher_order_coefficients.shape
    modes = higher_order_coefficients.reshape(
        batch, count, 5, int(nonlinear_channels)
    )
    values = modes.abs().mean(dim=(1, 3))
    if int(order) < 2:
        values = values.clone()
        values[:, 2:] = 0.0
    if int(order) < 1:
        values = torch.zeros_like(values)
    return values
