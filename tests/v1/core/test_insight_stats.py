# SPDX-License-Identifier: Apache-2.0

import time
import pytest
from tests.v1.core.utils import create_scheduler, create_requests
from vllm.v1.request import RequestStatus

def test_get_insight_stats_structure():
    """Test the structure of insight stats."""
    num_blocks = 100
    scheduler = create_scheduler(num_blocks=num_blocks)
    
    # 1. Check initial stats
    stats = scheduler.get_insight_stats()
    
    assert "timestamp" in stats
    assert isinstance(stats["timestamp"], float)
    
    assert "metrics" in stats
    metrics = stats["metrics"]
    
    # Check GPU usage
    assert "gpu_usage" in metrics
    gpu_usage = metrics["gpu_usage"]
    assert gpu_usage["total_blocks"] == num_blocks
    # Initially some blocks might be null/reserved, so we check range
    assert 0 <= gpu_usage["free_blocks"] <= num_blocks
    assert gpu_usage["used_blocks"] == gpu_usage["total_blocks"] - gpu_usage["free_blocks"]
    
    # Check Queue Status
    assert "queue_status" in metrics
    queue_status = metrics["queue_status"]
    assert queue_status["running"] == 0
    assert queue_status["waiting"] == 0
    assert queue_status["swapped"] == 0
    
    # Check Block Grid
    assert "block_grid" in stats
    block_grid = stats["block_grid"]
    assert isinstance(block_grid, list)
    assert len(block_grid) == num_blocks
    
    first_block = block_grid[0]
    assert "id" in first_block
    assert "ref_count" in first_block
    
    print(stats["block_grid"])
    print(gpu_usage)

def test_get_insight_stats_updates():
    """Test that insight stats reflect scheduler state changes."""
    scheduler = create_scheduler()
    
    # Add requests
    num_reqs = 5
    requests = create_requests(num_requests=num_reqs)
    for req in requests:
        scheduler.add_request(req)
        
    start_time = time.time()
    stats = scheduler.get_insight_stats()
    
    # Should be waiting
    assert stats["metrics"]["queue_status"]["waiting"] == num_reqs
    assert stats["metrics"]["queue_status"]["running"] == 0
    
    # Schedule (move to running)
    scheduler.schedule()
    
    stats = scheduler.get_insight_stats()
    assert stats["metrics"]["queue_status"]["waiting"] == 0
    assert stats["metrics"]["queue_status"]["running"] == num_reqs
    assert stats["timestamp"] >= start_time
    
    # print(stats["block_grid"])
    print(stats["metrics"]["gpu_usage"])
    

def test_insight_stats_with_four_requests_lifecycle():
    """Test full lifecycle of insight stats with 4 requests."""
    scheduler = create_scheduler()
    
    # 1. Add 4 requests
    num_reqs = 4
    requests = create_requests(num_requests=num_reqs)
    for req in requests:
        scheduler.add_request(req)
        
    stats = scheduler.get_insight_stats()
    queue = stats["metrics"]["queue_status"]
    
    # Initially all waiting
    assert queue["waiting"] == num_reqs
    assert queue["running"] == 0
    assert queue["swapped"] == 0

    # 2. Schedule them all
    scheduler.schedule()
    
    stats = scheduler.get_insight_stats()
    queue = stats["metrics"]["queue_status"]
    
    # Now all running
    assert queue["waiting"] == 0
    assert queue["running"] == num_reqs
    assert queue["swapped"] == 0
    
    # Verify individual blocks are allocated in the grid
    # Each request should have some blocks allocated now
    total_used_blocks = stats["metrics"]["gpu_usage"]["used_blocks"]
    assert total_used_blocks > 0
    
    # Verify request info for all 4
    for req in requests:
        info = scheduler.get_insight_request_info(req.request_id)
        assert info is not None
        assert info["request_id"] == req.request_id
        assert str(RequestStatus.RUNNING) in info["status"]

    # 3. Finish one request
    req_to_finish = requests[0]
    scheduler.finish_requests([req_to_finish.request_id], RequestStatus.FINISHED_STOPPED)
    
    stats = scheduler.get_insight_stats()
    queue = stats["metrics"]["queue_status"]
    
    # One less running
    assert queue["running"] == num_reqs - 1
    
    # Verify the finished request is no longer found in insight info
    # (Assuming finished requests are removed from scheduler tracking immediately)
    info = scheduler.get_insight_request_info(req_to_finish.request_id)
    assert info is None or "FINISHED" in info["status"]

def test_insight_fragmentation_logic():
    """Test memory fragmentation calculation logic."""
    block_size = 16
    scheduler = create_scheduler(block_size=block_size)
    
    # Create request with 20 tokens prompt.
    # 20 tokens need ceil(20/16) = 2 blocks.
    requests = create_requests(num_requests=1, num_tokens=20)
    req = requests[0]
    
    scheduler.add_request(req)
    # Schedule to trigger allocation
    scheduler.schedule()
    
    # Simulate that we have computed these tokens (Pre-fill completed)
    # Note: In real engine, this happens after model runs. 
    # Here we simulate the state that Insight would see.
    req.num_computed_tokens = 20
    
    info = scheduler.get_insight_request_info(req.request_id)
    assert info is not None
    assert "memory_stats" in info
    
    stats = info["memory_stats"]
    
    # Verify basics
    assert stats["block_size"] == block_size
    assert stats["used_slots"] == 20
    
    # Verify allocation
    # 20 tokens -> 2 blocks of 16 -> 32 slots
    allocated_blocks = stats["allocated_blocks"]
    assert allocated_blocks >= 2
    
    total_slots = allocated_blocks * block_size
    assert stats["total_slots"] == total_slots
    
    # Verify Fragmentation
    # Wasted = Total - Used
    expected_wasted = total_slots - 20
    assert stats["wasted_slots"] == expected_wasted
    
    expected_ratio = expected_wasted / total_slots
    assert abs(stats["fragmentation_ratio"] - expected_ratio) < 1e-6

def test_get_insight_all_requests_stats():
    """Test getting stats for all requests."""
    scheduler = create_scheduler()
    
    # Add 2 requests
    requests = create_requests(num_requests=2)
    req = create_requests(num_requests=1,num_tokens=20)
    req[0].request_id = "2"
    requests.extend(req)
    print(requests)
    for req in requests:
        print("ADD ",req.request_id)
        scheduler.add_request(req)
        
    # Initially waiting
    all_stats = scheduler.get_insight_all_requests_stats()
    print(all_stats)
    assert len(all_stats) == 3
    ids = sorted([s["request_id"] for s in all_stats])
    print("IDS: ",ids)
    expected_ids = sorted([r.request_id for r in requests])
    # assert ids == expected_ids
    
    # Scheduler one
    scheduler.schedule()
    
    all_stats = scheduler.get_insight_all_requests_stats()
    print("==================  STAT ===================")
    print(all_stats)
    assert len(all_stats) == 3
    
    # Check structure
    first = all_stats[0]
    assert "memory_stats" in first
    assert "block_table" in first
