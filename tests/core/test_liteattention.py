import sys
import types

import pytest
import torch

from xfuser.core import liteattention
from xfuser.core.distributed import attention_backend
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.liteattention import (
    LITE_LAYER_KEY,
    LiteAttentionConfig,
    get_lite_attention,
    reset_lite_attention_state,
    validate_lite_config,
)

HEAD_DIM = 128


class _Layer:
    """Stand-in for a diffusers attention module used as a per-layer handle."""


class _FakeLiteAttention:
    """Records calls and resets the way moonmath's LiteAttention would."""

    def __init__(self, threshold, enable_skipping, round_mode, layout):
        self.threshold = threshold
        self.enable_skipping = enable_skipping
        self.round_mode = round_mode
        self.layout = layout
        self.calls = 0
        self.resets = 0

    def __call__(self, q, k, v):
        self.calls += 1
        return torch.zeros_like(q)

    def reset_skip_state(self):
        self.resets += 1


@pytest.fixture
def fake_moonmath(monkeypatch):
    """Install an importable moonmath_attention stub and route the backend at it."""
    forward_calls = []

    def forward(q, k, v, layout=None, round_mode=None):
        forward_calls.append(
            {"shape": tuple(q.shape), "layout": layout, "round_mode": round_mode}
        )
        return torch.zeros_like(q)

    module = types.ModuleType("moonmath_attention")
    module.forward = forward
    module.LiteAttention = _FakeLiteAttention
    monkeypatch.setitem(sys.modules, "moonmath_attention", module)
    monkeypatch.setattr(attention_backend, "moonmath_forward", forward, raising=False)
    monkeypatch.setattr(liteattention, "_STATE", type(liteattention._STATE)())
    module.forward_calls = forward_calls
    return module


def _qkv(batch=1, heads=4, seq=384, kv_seq=None, dtype=torch.bfloat16):
    kv_seq = seq if kv_seq is None else kv_seq
    return (
        torch.randn(batch, heads, seq, HEAD_DIM, dtype=dtype),
        torch.randn(batch, heads, kv_seq, HEAD_DIM, dtype=dtype),
        torch.randn(batch, heads, kv_seq, HEAD_DIM, dtype=dtype),
    )


def test_backend_is_registered():
    assert (
        attention_backend.ATTENTION_FUNCTION_REGISTRY[
            AttentionBackendType.LITEATTENTION_ROCM
        ]
        is attention_backend._liteattention_rocm_attn_call
    )


def test_call_without_layer_handle_uses_exact_forward(fake_moonmath):
    query, key, value = _qkv(kv_seq=512)

    output, lse = attention_backend._liteattention_rocm_attn_call(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    assert lse is None
    assert output.shape == query.shape
    # BHSD in, BHSD out: the kernel takes the layout natively, no permute.
    assert fake_moonmath.forward_calls == [
        {"shape": (1, 4, 384, HEAD_DIM), "layout": "bhsd", "round_mode": "rtz"}
    ]


def test_layer_handle_selects_the_skip_kernel(fake_moonmath):
    layer = _Layer()
    query, key, value = _qkv()

    output, lse = attention_backend._liteattention_rocm_attn_call(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        attention_kwargs={LITE_LAYER_KEY: layer},
    )

    assert lse is None
    assert output.shape == query.shape
    assert fake_moonmath.forward_calls == []
    assert get_lite_attention(layer, LiteAttentionConfig()).calls == 1


def test_each_layer_keeps_its_own_skip_state(fake_moonmath):
    config = LiteAttentionConfig()
    first, second = _Layer(), _Layer()

    assert get_lite_attention(first, config) is get_lite_attention(first, config)
    assert get_lite_attention(first, config) is not get_lite_attention(second, config)


def test_state_is_reused_across_steps_and_reset_between_runs(fake_moonmath):
    config = LiteAttentionConfig()
    layer = _Layer()

    lite = get_lite_attention(layer, config)
    assert get_lite_attention(layer, config) is lite
    assert lite.resets == 0

    reset_lite_attention_state()

    assert get_lite_attention(layer, config) is lite
    assert lite.resets == 1
    # Only the first call of the new run pays the reset.
    assert get_lite_attention(layer, config) is lite
    assert lite.resets == 1


def test_changing_knobs_rebuilds_the_kernel_wrapper(fake_moonmath):
    layer = _Layer()

    lite = get_lite_attention(layer, LiteAttentionConfig())
    rebuilt = get_lite_attention(layer, LiteAttentionConfig(threshold=-4.0))

    assert rebuilt is not lite
    assert rebuilt.threshold == -4.0
    assert rebuilt.layout == "bhsd"


def test_disabled_skip_is_forwarded_to_the_kernel(fake_moonmath):
    lite = get_lite_attention(_Layer(), LiteAttentionConfig(enable_skipping=False))

    assert lite.enable_skipping is False


@pytest.fixture
def sdpa_spy(monkeypatch):
    """Count fallbacks without running the real SDPA flash kernel on CPU."""
    calls = []

    def spy(query, key, value, dropout_p, is_causal, attention_kwargs=None):
        calls.append((tuple(query.shape), is_causal, dropout_p))
        return torch.zeros_like(query), None

    monkeypatch.setattr(attention_backend, "_sdpa_flash_attn_call", spy)
    return calls


@pytest.mark.parametrize(
    "overrides, dtype, query_shape, key_shape",
    [
        ({"is_causal": True}, torch.bfloat16, (1, 4, 64, HEAD_DIM), (1, 4, 64, HEAD_DIM)),
        ({"dropout_p": 0.1}, torch.bfloat16, (1, 4, 64, HEAD_DIM), (1, 4, 64, HEAD_DIM)),
        ({}, torch.float32, (1, 4, 64, HEAD_DIM), (1, 4, 64, HEAD_DIM)),
        ({}, torch.bfloat16, (1, 4, 64, 64), (1, 4, 64, 64)),  # head_dim != 128
        ({}, torch.bfloat16, (1, 4, 64, HEAD_DIM), (1, 2, 64, HEAD_DIM)),  # GQA
    ],
)
def test_calls_the_kernel_cannot_serve_fall_back_to_sdpa(
    fake_moonmath, sdpa_spy, overrides, dtype, query_shape, key_shape
):
    query = torch.randn(*query_shape, dtype=dtype)
    key = torch.randn(*key_shape, dtype=dtype)
    value = torch.randn(*key_shape, dtype=dtype)

    call = {"dropout_p": 0.0, "is_causal": False}
    call.update(overrides)
    output, _ = attention_backend._liteattention_rocm_attn_call(
        query, key, value, **call
    )

    assert output.shape == query.shape
    assert len(sdpa_spy) == 1
    assert fake_moonmath.forward_calls == []


def test_supported_call_does_not_fall_back(fake_moonmath, sdpa_spy):
    query, key, value = _qkv(seq=64)

    attention_backend._liteattention_rocm_attn_call(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    assert sdpa_spy == []
    assert len(fake_moonmath.forward_calls) == 1


def test_non_4d_tensors_are_a_hard_error(fake_moonmath):
    query = torch.randn(4, 64, HEAD_DIM, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="4D"):
        attention_backend._liteattention_rocm_attn_call(
            query, query, query, dropout_p=0.0, is_causal=False
        )


def _compat_state(**runtime_overrides):
    from xfuser.config.config import RuntimeConfig
    from xfuser.core.distributed.runtime_state import DiTRuntimeState

    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.runtime_config = RuntimeConfig(**runtime_overrides)
    state.parallel_config = types.SimpleNamespace(ring_degree=1)
    return state


def test_backend_selection_requires_the_kernel_package(monkeypatch):
    from xfuser.core.distributed import runtime_state as runtime_state_module

    state = _compat_state()
    monkeypatch.setitem(runtime_state_module.env_info, "has_moonmath_attention", False)
    with pytest.raises(RuntimeError, match="moonmath_attention"):
        state._check_if_backend_compatible_with_current_configuration(
            AttentionBackendType.LITEATTENTION_ROCM
        )

    monkeypatch.setitem(runtime_state_module.env_info, "has_moonmath_attention", True)
    state._check_if_backend_compatible_with_current_configuration(
        AttentionBackendType.LITEATTENTION_ROCM
    )


def test_backend_selection_validates_the_knobs(monkeypatch):
    from xfuser.core.distributed import runtime_state as runtime_state_module

    monkeypatch.setitem(runtime_state_module.env_info, "has_moonmath_attention", True)

    with pytest.raises(ValueError, match="negative"):
        _compat_state(lite_threshold=0.5)._check_if_backend_compatible_with_current_configuration(
            AttentionBackendType.LITEATTENTION_ROCM
        )
    with pytest.raises(ValueError, match="round mode"):
        _compat_state(lite_round_mode="rtn")._check_if_backend_compatible_with_current_configuration(
            AttentionBackendType.LITEATTENTION_ROCM
        )


def test_backend_rejects_ring_parallelism():
    state = _compat_state()
    state.parallel_config = types.SimpleNamespace(ring_degree=2)

    with pytest.raises(RuntimeError, match="ring parallelism"):
        state._check_if_backend_compatible_with_current_configuration(
            AttentionBackendType.LITEATTENTION_ROCM
        )


def test_validate_lite_config_rejects_bad_knobs():
    validate_lite_config(LiteAttentionConfig())
    with pytest.raises(ValueError, match="negative"):
        validate_lite_config(LiteAttentionConfig(threshold=0.0))
    with pytest.raises(ValueError, match="round mode"):
        validate_lite_config(LiteAttentionConfig(round_mode="rtn"))


def test_usp_publishes_the_layer_handle_only_for_this_backend():
    from xfuser.model_executor.layers.usp import _publish_lite_layer

    layer = _Layer()

    assert _publish_lite_layer(None, None, AttentionBackendType.LITEATTENTION_ROCM) is None
    assert _publish_lite_layer({}, layer, AttentionBackendType.AITER) == {}
    assert _publish_lite_layer(
        {"thw": (1, 2, 3)}, layer, AttentionBackendType.LITEATTENTION_ROCM
    ) == {"thw": (1, 2, 3), LITE_LAYER_KEY: layer}


def test_cli_exposes_the_liteattention_knobs():
    from xfuser.config import FlexibleArgumentParser, xFuserArgs

    parser = FlexibleArgumentParser()
    xFuserArgs.add_runner_args(parser)
    args = parser.parse_args(
        [
            "--model", "dummy",
            "--attention_backend", "liteattention_rocm",
            "--lite_threshold", "-4.0",
            "--lite_round_mode", "rtne",
            "--disable_lite_skip",
        ]
    )

    assert args.lite_threshold == -4.0
    assert args.lite_round_mode == "rtne"
    assert args.disable_lite_skip is True


def test_runtime_config_defaults_match_the_kernel_wrapper_defaults():
    from xfuser.config.config import RuntimeConfig

    defaults = LiteAttentionConfig()
    runtime_config = RuntimeConfig()

    assert runtime_config.lite_threshold == defaults.threshold
    assert runtime_config.lite_round_mode == defaults.round_mode
    assert runtime_config.disable_lite_skip is not defaults.enable_skipping
