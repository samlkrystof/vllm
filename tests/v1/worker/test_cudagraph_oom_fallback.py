# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CUDA graph memory profiling OOM fallback."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.compilation.counter import compilation_counter
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _get_num_kv_blocks_for_cudagraph_profiling,
)
from vllm.v1.worker.gpu_worker import Worker


def _config_with_mamba_cache_mode(mode: str):
    return SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode=mode))


def _mamba_spec(
    *,
    block_size: int = 262144,
    num_speculative_blocks: int = 0,
    mamba_type: str = "mamba2",
) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1024,),),
        dtypes=(torch.float32,),
        page_size_padded=3211264,
        mamba_type=mamba_type,
        num_speculative_blocks=num_speculative_blocks,
    )


def _full_attention_spec(*, block_size: int = 784) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
        page_size_padded=3211264,
    )


def test_cudagraph_profiling_kv_blocks_non_mamba_preserves_capture_size():
    groups = [
        KVCacheGroupSpec(["attn"], _full_attention_spec()),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode("none"),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == 512


def test_cudagraph_profiling_kv_blocks_hybrid_mamba_uses_pages_not_tokens():
    groups = [
        KVCacheGroupSpec(["mamba"], _mamba_spec()),
        KVCacheGroupSpec(["attn"], _full_attention_spec()),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode("none"),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == 1


@pytest.mark.parametrize(
    ("mamba_cache_mode", "expected_blocks"),
    [
        ("none", 4),
        ("align", 5),
    ],
)
def test_cudagraph_profiling_kv_blocks_honor_mamba_state_lower_bound(
    mamba_cache_mode: str,
    expected_blocks: int,
):
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            _mamba_spec(num_speculative_blocks=3),
        ),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode(mamba_cache_mode),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == expected_blocks


def test_cudagraph_profiling_kv_blocks_handles_wrapped_mamba_specs():
    mamba_spec = _mamba_spec(num_speculative_blocks=2)
    groups = [
        KVCacheGroupSpec(
            ["mamba"],
            UniformTypeKVCacheSpecs(
                block_size=mamba_spec.block_size,
                kv_cache_specs={"mamba": mamba_spec},
            ),
        ),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode("align"),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == 4


def test_cudagraph_profiling_kv_blocks_handles_gated_deltanet_specs():
    groups = [
        KVCacheGroupSpec(
            ["gdn"],
            _mamba_spec(
                mamba_type="gdn_attention",
                num_speculative_blocks=2,
            ),
        ),
        KVCacheGroupSpec(["attn"], _full_attention_spec()),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode("align"),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == 4


def test_cudagraph_profiling_kv_blocks_handles_mamba_all_mode():
    groups = [
        KVCacheGroupSpec(["mamba"], _mamba_spec(block_size=262144)),
    ]

    num_blocks = _get_num_kv_blocks_for_cudagraph_profiling(
        _config_with_mamba_cache_mode("all"),
        groups,
        max_capture_size=512,
    )

    assert num_blocks == 1


def test_determine_available_memory_cudagraph_oom_falls_back_to_zero():
    worker = Worker.__new__(Worker)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None,
        gpu_memory_utilization=0.9,
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    )
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker.device = torch.device("cuda")
    worker.init_snapshot = SimpleNamespace(
        free_memory=10 << 30,
        total_memory=16 << 30,
    )
    worker.requested_memory = 8 << 30

    worker.model_runner = MagicMock()
    worker.model_runner.model_memory_usage = 1 << 30
    worker.model_runner.profile_cudagraph_memory.side_effect = torch.OutOfMemoryError(
        "CUDA out of memory"
    )

    profile_result = SimpleNamespace(
        before_profile=SimpleNamespace(torch_peak=100),
        after_profile=SimpleNamespace(free_memory=9 << 30),
        non_torch_increase=200,
        weights_memory=1 << 30,
    )

    @contextmanager
    def fake_memory_profiling(*args, **kwargs):
        yield profile_result

    with (
        patch("vllm.v1.worker.gpu_worker.memory_profiling", fake_memory_profiling),
        patch("vllm.v1.worker.gpu_worker.current_platform.is_rocm", return_value=False),
        patch(
            "torch.accelerator.memory_stats",
            return_value={"allocated_bytes.all.peak": 500},
        ),
        patch("torch.accelerator.synchronize") as mock_sync,
        patch("torch.accelerator.empty_cache") as mock_empty_cache,
    ):
        available_memory = worker.determine_available_memory()

    worker.model_runner.profile_run.assert_called_once()
    worker.model_runner.profile_cudagraph_memory.assert_called_once()
    mock_sync.assert_called_once()
    mock_empty_cache.assert_called_once()
    assert worker.cudagraph_memory_estimate == 0
    assert available_memory == worker.available_kv_cache_memory_bytes


def test_profile_cudagraph_memory_oom_cleans_up_runner_state():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = object()
    runner.cudagraph_dispatcher = SimpleNamespace(
        cudagraph_keys={"decode": {"stale-key"}},
        keys_initialized=True,
    )
    runner._cleanup_profiling_kv_cache = MagicMock()
    runner.maybe_remove_all_loras = MagicMock()
    runner.lora_config = None

    saved_count = compilation_counter.num_cudagraph_captured
    compilation_counter.num_cudagraph_captured = 123

    def raise_oom():
        compilation_counter.num_cudagraph_captured = 456
        raise torch.OutOfMemoryError("CUDA out of memory")

    runner._init_minimal_kv_cache_for_profiling = MagicMock(side_effect=raise_oom)

    try:
        with (
            patch(
                "vllm.v1.worker.gpu_model_runner.set_current_vllm_config",
                return_value=nullcontext(),
            ),
            patch(
                "vllm.v1.worker.gpu_model_runner.set_cudagraph_capturing_enabled"
            ) as mock_set_capture,
            patch(
                "vllm.v1.worker.gpu_model_runner.CUDAGraphWrapper.clear_all_graphs"
            ) as mock_clear_graphs,
            pytest.raises(torch.OutOfMemoryError),
        ):
            runner.profile_cudagraph_memory()

        mock_set_capture.assert_called_once_with(False)
        mock_clear_graphs.assert_called_once()
        assert runner.cudagraph_dispatcher.cudagraph_keys["decode"] == set()
        assert not runner.cudagraph_dispatcher.keys_initialized
        runner._cleanup_profiling_kv_cache.assert_called_once()
        runner.maybe_remove_all_loras.assert_not_called()
        assert compilation_counter.num_cudagraph_captured == 123
    finally:
        compilation_counter.num_cudagraph_captured = saved_count
