"""Per-layer state for the LiteAttention (moonmath) attention backend.

The moonmath ``LiteAttention`` kernel is stateful across denoising steps: every
call reads the skip list written by the previous call and emits a refreshed one,
so a K-block that all 384 q-rows of a CTA voted irrelevant is never loaded again.
That list describes one attention layer's score landscape and must not be shared
between layers.

xDiT attention backends are stateless module-level functions, so the instances
live here instead, keyed by the stable per-layer handle that ``USP`` and
``attention`` already accept. Calls that arrive without a handle (cross-attention,
models that do not pass one) run exact dense attention and keep no state.
"""

import weakref
from dataclasses import dataclass

# Key under which USP stashes the per-layer handle in a shallow copy of
# attention_kwargs. The key's PRESENCE is what selects the stateful skip path,
# mirroring how the sparge head balancer publishes its cost sink.
LITE_LAYER_KEY: str = "_lite_attention_layer"

SUPPORTED_ROUND_MODES = ("rtna", "rtne", "rtz")

DEFAULT_THRESHOLD: float = -6.0
# RTZ is the kernel's fastest rounding mode and matches AITER's own default
# bf16 conversion, so switching backends does not change the rounding rule.
DEFAULT_ROUND_MODE: str = "rtz"


@dataclass(frozen=True)
class LiteAttentionConfig:
    threshold: float = DEFAULT_THRESHOLD
    round_mode: str = DEFAULT_ROUND_MODE
    enable_skipping: bool = True


class _Entry:
    __slots__ = ("generation", "config", "lite")

    def __init__(self, generation, config, lite):
        self.generation = generation
        self.config = config
        self.lite = lite


_STATE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_GENERATION: int = 0


def validate_lite_config(config: LiteAttentionConfig) -> None:
    """Raise on knob values the kernel rejects. Called at backend selection so
    a typo fails before the model is loaded rather than mid-denoise."""
    if config.threshold >= 0:
        raise ValueError(
            f"LiteAttention threshold must be negative (log2 units), got {config.threshold}."
        )
    if config.round_mode not in SUPPORTED_ROUND_MODES:
        raise ValueError(
            f"LiteAttention round mode must be one of {SUPPORTED_ROUND_MODES}, "
            f"got {config.round_mode!r}."
        )


def resolve_lite_config() -> LiteAttentionConfig:
    """Read the LiteAttention knobs off the runtime config, falling back to the
    defaults when no runtime state exists (unit tests, direct backend calls)."""
    from xfuser.core.distributed.runtime_state import (
        get_runtime_state,
        runtime_state_is_initialized,
    )

    if not runtime_state_is_initialized():
        return LiteAttentionConfig()
    runtime_config = get_runtime_state().runtime_config
    return LiteAttentionConfig(
        threshold=runtime_config.lite_threshold,
        round_mode=runtime_config.lite_round_mode,
        enable_skipping=not runtime_config.disable_lite_skip,
    )


def reset_lite_attention_state() -> None:
    """Invalidate every layer's learned skip list at a pipeline-run boundary.

    Skipping is monotonic within a run -- a dropped K-block is never
    reconsidered -- so a list carried into an unrelated prompt would keep
    suppressing blocks that the new prompt depends on.
    """
    global _GENERATION
    _GENERATION += 1


def get_lite_attention(layer, config: LiteAttentionConfig):
    """Return this layer's ``LiteAttention``, creating or resetting it as needed.

    The instance itself is cheap; what it owns is a double-buffered int16 skip
    list sized from (B, H, Sq, Skv), which the kernel reallocates on its own when
    the shape signature changes.
    """
    from moonmath_attention import LiteAttention

    entry = _STATE.get(layer)
    if entry is None or entry.config != config:
        lite = LiteAttention(
            threshold=config.threshold,
            enable_skipping=config.enable_skipping,
            round_mode=config.round_mode,
            layout="bhsd",
        )
        _STATE[layer] = _Entry(_GENERATION, config, lite)
        return lite
    if entry.generation != _GENERATION:
        entry.lite.reset_skip_state()
        entry.generation = _GENERATION
    return entry.lite
