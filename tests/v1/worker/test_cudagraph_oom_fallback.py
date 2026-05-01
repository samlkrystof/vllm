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
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


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
