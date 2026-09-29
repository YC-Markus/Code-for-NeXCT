"""HerKry: continuous Hermite–Gaussian reconstruction with Krylov memory."""
__version__ = "0.1.0"


def build_model(*args, **kwargs):
    """Lazily import the CUDA implementation (see :mod:`herkry.api`)."""
    from .api import build_model as build
    return build(*args, **kwargs)
