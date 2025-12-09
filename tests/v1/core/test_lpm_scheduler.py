# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LPM (Longest Prefix Match) scheduling policy."""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import (
    LPMRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test


def create_scheduler_with_lpm(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = True,
    num_blocks: int = 10000,
    block_size: int = 16,
) -> Scheduler:
    """Create a scheduler with LPM scheduling policy."""
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="lpm",  # LPM scheduling policy
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32, False)
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_request_with_prompt(
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int = 16,
    arrival_time: float = 0.0,
    block_size: int = 16,
) -> Request:
    """Create a request with specific prompt tokens."""
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=arrival_time,
        block_hasher=block_hasher,
    )


class TestLPMRequestQueue:
    """Tests for LPMRequestQueue class."""

    def test_create_lpm_queue(self):
        """Test that LPM queue can be created."""
        queue = create_request_queue(SchedulingPolicy.LPM)
        assert isinstance(queue, LPMRequestQueue)
        assert len(queue) == 0
        assert not queue

    def test_add_and_pop_requests(self):
        """Test adding and popping requests from LPM queue."""
        queue = LPMRequestQueue()
        
        # Create requests with different prefix lengths (simulated via num_cached_tokens)
        req1 = create_request_with_prompt("req1", [1, 2, 3], arrival_time=1.0)
        req2 = create_request_with_prompt("req2", [1, 2, 3, 4], arrival_time=2.0)
        req3 = create_request_with_prompt("req3", [1, 2], arrival_time=3.0)
        
        # Simulate prefix cache hits
        req1.num_cached_tokens = 10
        req2.num_cached_tokens = 50  # Highest prefix
        req3.num_cached_tokens = 30
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        assert len(queue) == 3
        
        # Should pop in order of highest prefix first
        popped = queue.pop_request()
        assert popped.request_id == "req2"  # Highest prefix (50)
        
        popped = queue.pop_request()
        assert popped.request_id == "req3"  # Second highest (30)
        
        popped = queue.pop_request()
        assert popped.request_id == "req1"  # Lowest (10)
        
        assert len(queue) == 0

    def test_tie_breaking_by_arrival_time(self):
        """Test that requests with same prefix are ordered by arrival time."""
        queue = LPMRequestQueue()
        
        req1 = create_request_with_prompt("req1", [1, 2, 3], arrival_time=3.0)
        req2 = create_request_with_prompt("req2", [1, 2, 3], arrival_time=1.0)  # Earlier
        req3 = create_request_with_prompt("req3", [1, 2, 3], arrival_time=2.0)
        
        # Same prefix length
        req1.num_cached_tokens = 20
        req2.num_cached_tokens = 20
        req3.num_cached_tokens = 20
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Should pop in order of earliest arrival when prefix is same
        assert queue.pop_request().request_id == "req2"  # arrival_time=1.0
        assert queue.pop_request().request_id == "req3"  # arrival_time=2.0
        assert queue.pop_request().request_id == "req1"  # arrival_time=3.0

    def test_remove_request(self):
        """Test removing a specific request from the queue."""
        queue = LPMRequestQueue()
        
        req1 = create_request_with_prompt("req1", [1, 2])
        req2 = create_request_with_prompt("req2", [1, 2, 3])
        req3 = create_request_with_prompt("req3", [1])
        
        req1.num_cached_tokens = 10
        req2.num_cached_tokens = 20
        req3.num_cached_tokens = 5
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Remove the middle priority request
        queue.remove_request(req1)
        
        assert len(queue) == 2
        assert queue.pop_request().request_id == "req2"
        assert queue.pop_request().request_id == "req3"

    def test_peek_request(self):
        """Test peeking at the highest priority request."""
        queue = LPMRequestQueue()
        
        req1 = create_request_with_prompt("req1", [1, 2])
        req2 = create_request_with_prompt("req2", [1, 2, 3])
        
        req1.num_cached_tokens = 10
        req2.num_cached_tokens = 50
        
        queue.add_request(req1)
        queue.add_request(req2)
        
        # Peek should return highest prefix without removing
        peeked = queue.peek_request()
        assert peeked.request_id == "req2"
        assert len(queue) == 2  # Still in queue

    def test_iteration(self):
        """Test iterating over the queue."""
        queue = LPMRequestQueue()
        
        req1 = create_request_with_prompt("req1", [1])
        req2 = create_request_with_prompt("req2", [1, 2])
        req3 = create_request_with_prompt("req3", [1, 2, 3])
        
        req1.num_cached_tokens = 5
        req2.num_cached_tokens = 15
        req3.num_cached_tokens = 25
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Iteration should be in priority order
        request_ids = [r.request_id for r in queue]
        assert request_ids == ["req3", "req2", "req1"]
        
        # Queue should remain unchanged after iteration
        assert len(queue) == 3


class TestLPMScheduler:
    """Tests for LPM scheduling in the Scheduler."""

    def test_lpm_scheduler_creation(self):
        """Test that scheduler with LPM policy can be created."""
        scheduler = create_scheduler_with_lpm()
        assert scheduler.policy == SchedulingPolicy.LPM

    def test_lpm_selects_highest_prefix(self):
        """Test that LPM scheduler selects request with highest prefix cache hit."""
        scheduler = create_scheduler_with_lpm(
            enable_prefix_caching=True,
            max_num_seqs=2,
        )
        
        # Create requests with shared prefix
        # Request 1: [1, 2, 3, 4, 5, 6, 7, 8, ...] (16 tokens) - unique
        # Request 2: [1, 2, 3, 4, 5, 6, 7, 8, ...] (16 tokens) - same as req1, will have cache hit
        
        block_size = 16
        shared_prefix = list(range(1, block_size + 1))  # 16 tokens = 1 block
        
        req1 = create_request_with_prompt(
            "req1", 
            shared_prefix + [100, 101, 102, 103],  # Shared prefix + unique suffix
            arrival_time=1.0,
            block_size=block_size,
        )
        req2 = create_request_with_prompt(
            "req2",
            shared_prefix + [200, 201, 202, 203],  # Same shared prefix + different suffix
            arrival_time=2.0,
            block_size=block_size,
        )
        req3 = create_request_with_prompt(
            "req3",
            [300, 301, 302, 303],  # Completely different - no cache hit
            arrival_time=0.5,  # Earliest arrival
            block_size=block_size,
        )
        
        # Add req1 first and schedule it to populate prefix cache
        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert len(output1.scheduled_new_reqs) == 1
        assert output1.scheduled_new_reqs[0].req_id == "req1"
        
        # Process req1's output
        model_output = ModelRunnerOutput(
            req_ids=["req1"],
            req_id_to_index={"req1": 0},
            sampled_token_ids=[[EOS_TOKEN_ID]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output1, model_output)
        
        # Now add req2 and req3 - req2 should have prefix cache hit
        scheduler.add_request(req3)  # Add first but has no prefix hit
        scheduler.add_request(req2)  # Add second but has prefix cache hit
        
        # LPM should select req2 (higher prefix hit) over req3 (earlier arrival)
        output2 = scheduler.schedule()
        
        # Due to LPM policy, req2 should be scheduled because it has prefix cache hit
        # even though req3 arrived earlier
        scheduled_req_ids = [r.req_id for r in output2.scheduled_new_reqs]
        
        # With LPM, the request with higher prefix cache hit should be scheduled first
        # Since req1 was already processed, req2 should have prefix cache hit
        # while req3 has no prefix cache hit
        assert "req2" in scheduled_req_ids


class TestSchedulingPolicyEnum:
    """Tests for SchedulingPolicy enum."""

    def test_lpm_enum_value(self):
        """Test LPM enum value."""
        assert SchedulingPolicy.LPM.value == "lpm"

    def test_create_queue_for_each_policy(self):
        """Test creating queues for all policies."""
        fcfs_queue = create_request_queue(SchedulingPolicy.FCFS)
        priority_queue = create_request_queue(SchedulingPolicy.PRIORITY)
        lpm_queue = create_request_queue(SchedulingPolicy.LPM)
        
        assert fcfs_queue is not None
        assert priority_queue is not None
        assert lpm_queue is not None
        assert isinstance(lpm_queue, LPMRequestQueue)
