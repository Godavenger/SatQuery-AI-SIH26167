"""Stage 2 specialist heads; neural dependencies load only when requested."""
from .heads import *  # noqa


def __getattr__(name):
    if name in {"AdaptedBackbone", "BandProjector", "GSDConditioner", "SAREncoder",
                "get_backbone", "encode_tile"}:
        from . import backbone
        return getattr(backbone, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
