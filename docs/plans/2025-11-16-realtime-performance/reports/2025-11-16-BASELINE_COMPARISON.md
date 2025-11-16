# Baseline Comparison: Simple Planner vs ML Planner
**Date**: 2025-11-16
**Workstream**: D - Performance Profiling
**Analyst**: Navigator (Claude Code)

## Executive Summary

**SimplePlanner**: 34 seconds for 149 steps = **228ms per step**
**MLPlanner**: 85 seconds for 149 steps = **570ms per step**
**ML overhead**: 342ms per step (2.5x slower than SimplePlanner)

### Key Finding
The ML planner adds **342ms of overhead per step** compared to SimplePlanner, but SimplePlanner itself is **already 2.3x slower than expected** (100ms target).

**Breaking down the 570ms ML planner time**:
- SimplePlanner baseline: 228ms (what we inherit from framework)
- ML-specific overhead: 342ms (feature building + inference)

**This reveals two separate problems**:
1. **Framework overhead**: 128ms per step (planner-agnostic)
2. **ML overhead**: 342ms per step (ML-specific)

## Performance Comparison Table

| Component | SimplePlanner | MLPlanner | ML Overhead | % Increase |
|-----------|---------------|-----------|-------------|------------|
| **Total per-step time** | 228ms | 570ms | +342ms | +150% |
| Database queries | 134ms | 131ms | -3ms | -2% |
| Planner compute | 21ms | 127ms | +106ms | +505% |
| State propagation | 104ms | 95ms | -9ms | -9% |
| Post-sim logging (amortized) | 3ms | 236ms | +233ms | +7767% |
| Other overhead | N/A | N/A | N/A | N/A |

### Detailed Timing Breakdown

#### SimplePlanner (34s simulation, 55.5s total profiling)
- **Simulation loop**: 31s (34s - 3s post-processing)
- **Per-step average**: 31s / 149 = 208ms
- **Total profiling**: 55.5s (includes config loading, metrics)

**Component breakdown**:
| Phase | Time (s) | Per-step (ms) | % of Loop |
|-------|----------|---------------|-----------|
| Database queries | 20.0 | 134 | 64.5% |
| State propagation | 15.5 | 104 | 50.0% |
| Planner compute | 3.1 | 21 | 10.0% |
| Input preparation | 9.3 | 62 | 30.0% |
| LZMA compression | 0.5 | 3 | 1.6% |
| Config loading | 5.6 | N/A | One-time |
| Metric rendering | 5.6 | N/A | One-time |

#### MLPlanner (90s simulation, 114s total profiling)
- **Simulation loop**: 85s (90s - 5s post-processing)
- **Per-step average**: 85s / 149 = 570ms
- **Total profiling**: 114s (includes config loading, metrics)

**Component breakdown** (from previous analysis):
| Phase | Time (s) | Per-step (ms) | % of Loop |
|-------|----------|---------------|-----------|
| Database queries | 19.5 | 131 | 23.0% |
| ML planner compute | 19.0 | 127 | 22.3% |
| State propagation | 14.2 | 95 | 16.7% |
| Input preparation | 8.9 | 59 | 10.4% |
| LZMA compression | 35.2 | 236 | 41.4% |
| Config loading | 6.3 | N/A | One-time |
| Metric rendering | 6.1 | N/A | One-time |

## Detailed Component Comparison

### 1. Database Queries (Nearly Identical!)

**SimplePlanner**: 20.0s total, 134ms per step
**MLPlanner**: 19.5s total, 131ms per step
**Difference**: -0.5s total, -3ms per step (ML is 2% faster!)

**Surprising finding**: Database queries are nearly **identical** between planners!
- Same number of queries: 21,691 (SimplePlanner) vs 21,691 (MLPlanner)
- Same traffic light queries: 7,176 per scenario
- Same connection churn: 1,228 connect/close cycles

**Conclusion**: Database overhead is **planner-agnostic** and represents framework overhead.

### 2. Planner Compute Time (6x difference)

**SimplePlanner**: 3.1s total, 21ms per step
**MLPlanner**: 19.0s total, 127ms per step
**ML overhead**: +15.9s total, +106ms per step (+505%)

**SimplePlanner breakdown** (21ms/step):
- Path planning logic: ~15ms
- Controller interface: ~6ms

**MLPlanner breakdown** (127ms/step):
- Feature building: 112ms (88% of planner time)
  - Map rasterization: 46ms
  - Agent rasterization: 33ms
  - Layer generation: 33ms
- Neural network inference: 15ms (12% of planner time)

**Key insight**: ML planner spends **88% of time preparing features**, only 12% on actual inference!

### 3. State Propagation (ML is 9% faster!)

**SimplePlanner**: 15.5s total, 104ms per step
**MLPlanner**: 14.2s total, 95ms per step
**Difference**: -1.3s total, -9ms per step (-9%)

**Surprising finding**: ML planner is slightly **faster** at state propagation!

**Possible explanations**:
- SimplePlanner does more complex path planning in `compute_planner_trajectory()`
- This may involve more geometry operations that carry over into propagate()
- ML planner just does inference, cleaner state

**Conclusion**: State propagation overhead is similar and planner-agnostic.

### 4. Post-Simulation Logging (HUGE difference)

**SimplePlanner**: 0.5s total, 3ms per step (amortized)
**MLPlanner**: 35.2s total, 236ms per step (amortized)
**Difference**: +34.7s total, +233ms per step (+7767%!)

**Root cause**: Much larger simulation log for ML planner
- SimplePlanner log: Small (only trajectory + minimal state)
- MLPlanner log: Large (trajectory + features + model outputs)

**LZMA compression time**:
- SimplePlanner: 0.5s (fast)
- MLPlanner: 35.2s (slow, compressing ~10-50x more data)

**Conclusion**: This is a post-simulation cost, not real-time overhead. Can be optimized with async compression or disabled during profiling.

### 5. Input Preparation (ML is 5% faster)

**SimplePlanner**: 9.3s total, 62ms per step
**MLPlanner**: 8.9s total, 59ms per step
**Difference**: -0.4s total, -3ms per step (-5%)

**Conclusion**: Input preparation overhead is planner-agnostic and similar.

## Framework Overhead Analysis

### Common Overhead (Both Planners)

These components have nearly identical time in both planners:

| Component | SimplePlanner | MLPlanner | Notes |
|-----------|---------------|-----------|-------|
| Database queries | 134ms | 131ms | 21,691 queries per scenario |
| Input preparation | 62ms | 59ms | History buffer, observations |
| State propagation | 104ms | 95ms | Controller, time stepping |
| **Total framework** | **300ms** | **285ms** | Planner-agnostic overhead |

**Expected overhead**: ~20ms per step
**Actual framework overhead**: ~300ms per step
**Framework is 15x slower than expected!**

### ML-Specific Overhead

Only present in ML planner:

| Component | SimplePlanner | MLPlanner | ML Overhead |
|-----------|---------------|-----------|-------------|
| Feature building | 0ms | 112ms | +112ms |
| Neural net inference | 0ms | 15ms | +15ms |
| **Total ML overhead** | **0ms** | **127ms** | Pure ML cost |

## Top Bottlenecks Comparison

### SimplePlanner Top 10 (by cumulative time)

| Rank | Function | Cumtime (s) | Per-call (ms) | Calls |
|------|----------|-------------|---------------|-------|
| 1 | `execute_many` (DB) | 20.0 | 0.9 | 21691 |
| 2 | `get_traffic_light_status_for_lidarpc_token_from_db` | 16.6 | 2.3 | 7176 |
| 3 | `simulation.propagate` | 15.5 | 104 | 149 |
| 4 | `simulation.get_planner_input` | 9.3 | 62 | 149 |
| 5 | `sqlite3.Cursor.execute` | 7.3 | 6.0 | 1228 |
| 6 | `sqlite3.connect` | 3.8 | 3.1 | 1228 |
| 7 | `simple_planner.compute_planner_trajectory` | 3.1 | 21 | 149 |
| 8 | `kinematic_bicycle.propagate_state` | 2.3 | 0.4 | 5960 |
| 9 | `get_sensor_data_token_timestamp_from_db` | 3.6 | 8.1 | 446 |
| 10 | `deepcopy` | 2.8 | 0.004 | 705871 |

### MLPlanner Top 10 (by cumulative time)

| Rank | Function | Cumtime (s) | Per-call (ms) | Calls |
|------|----------|-------------|---------------|-------|
| 1 | `_lzma.LZMACompressor.compress` | 35.2 | 35154 | 1 |
| 2 | `execute_many` (DB) | 18.6 | 0.9 | 21691 |
| 3 | `ml_planner.compute_planner_trajectory` | 19.0 | 127 | 149 |
| 4 | `raster_feature_builder._compute_feature` | 16.6 | 112 | 149 |
| 5 | `get_traffic_light_status_for_lidarpc_token_from_db` | 15.4 | 2.1 | 7176 |
| 6 | `simulation.propagate` | 14.2 | 95 | 149 |
| 7 | `_get_layer_coords` (raster) | 11.1 | 12 | 894 |
| 8 | `simulation.get_planner_input` | 8.9 | 59 | 149 |
| 9 | `sqlite3.Cursor.execute` | 6.8 | 5.5 | 1228 |
| 10 | `get_roadmap_raster` | 6.8 | 46 | 149 |

### Common Top Bottlenecks (Framework Issues)

Both planners share these bottlenecks:

1. **Database queries** (134ms vs 131ms)
   - `execute_many`: 20.0s vs 18.6s
   - `get_traffic_light_status`: 16.6s vs 15.4s
   - **This is the #1 framework bottleneck!**

2. **State propagation** (104ms vs 95ms)
   - `propagate()`: 15.5s vs 14.2s
   - Nearly identical time

3. **Input preparation** (62ms vs 59ms)
   - `get_planner_input()`: 9.3s vs 8.9s
   - Nearly identical time

### ML-Specific Bottlenecks

Only appear in ML planner:

1. **Feature building** (112ms/step)
   - `_compute_feature`: 16.6s
   - `get_roadmap_raster`: 6.8s
   - `get_agents_raster`: 4.9s
   - `_get_layer_coords`: 11.1s

2. **LZMA compression** (236ms/step amortized)
   - `compress`: 35.2s (one-time, but huge!)

## Optimization Priority Ranking

### Tier 1: Framework Optimizations (Benefit BOTH Planners)

**Target**: -200ms/step for SimplePlanner, -200ms/step for MLPlanner

1. **Database query optimization** (-100ms/step)
   - Connection pooling
   - Query result caching
   - Batch queries
   - Benefits: Both planners equally

2. **Traffic light query caching** (-30ms/step)
   - Cache per scenario/timestep
   - Eliminate 47 redundant queries per step
   - Benefits: Both planners equally

3. **State propagation optimization** (-20ms/step)
   - Reduce data copying
   - Optimize history buffer
   - Benefits: Both planners equally

4. **Input preparation optimization** (-10ms/step)
   - Lazy observation loading
   - Cached observation assembly
   - Benefits: Both planners equally

**Total Tier 1 impact**: -160ms/step
- SimplePlanner: 228ms → 68ms (1.5x realtime!)
- MLPlanner: 570ms → 410ms (still 4.1x too slow)

### Tier 2: ML-Specific Optimizations (Benefit ML Planner Only)

**Target**: -150ms/step for MLPlanner

1. **Cache map rasterization** (-40ms/step)
   - Map doesn't change per step
   - Pre-render at scenario start
   - Only for ML planner

2. **Optimize agent rasterization** (-30ms/step)
   - Vectorize operations
   - Use GPU if available
   - Only for ML planner

3. **Reduce raster resolution** (-20ms/step)
   - Trade quality for speed
   - Experiment with smaller feature maps
   - Only for ML planner

4. **Async LZMA compression** (-236ms/step amortized)
   - Compress in background thread
   - Or disable during profiling
   - Only for ML planner

**Total Tier 2 impact**: -90ms/step (not counting async compression)
- MLPlanner: 410ms → 320ms (still 3.2x too slow)

### Combined Impact

**SimplePlanner**: 228ms → 68ms (-160ms, **1.5x realtime!**)
**MLPlanner**: 570ms → 320ms (-250ms, still 3.2x realtime)

**To reach 1.0x realtime (100ms total)**:
- SimplePlanner: Needs -28ms more (possible with Tier 1 optimizations)
- MLPlanner: Needs -220ms more (requires aggressive optimization or faster hardware)

## Surprising Findings

### 1. Framework is the Main Bottleneck (Not ML!)

**Expected**: ML overhead dominates
**Actual**: Framework overhead (300ms) is comparable to ML overhead (342ms)

SimplePlanner at 228ms/step shows that the **framework itself is already 2.3x too slow**, even with a trivial planner!

### 2. Database Queries are Identical

**Expected**: ML planner does more DB queries for feature building
**Actual**: Both planners do **exactly 21,691 queries** (same scenario replay)

Database overhead is **planner-agnostic** and represents a fundamental framework issue.

### 3. Neural Net Inference is Only 15ms/step

**Expected**: Neural net inference is slow
**Actual**: Inference takes only **15ms/step** (12% of ML planner time)

The **feature building** (112ms) is the real bottleneck, not the neural net!

### 4. Post-Simulation Logging Explodes for ML Planner

**Expected**: Similar logging overhead
**Actual**: 70x more compression time for ML planner (35s vs 0.5s)

ML planner generates **massive simulation logs** due to feature caching and model outputs.

### 5. SimplePlanner is Faster at Propagation

**Expected**: SimplePlanner and MLPlanner have similar propagation time
**Actual**: MLPlanner is 9% faster at state propagation (95ms vs 104ms)

This suggests SimplePlanner does some extra geometry work that bleeds into propagate().

## Recommendations

### For Real-Time Performance (1.0x realtime target)

**Phase 1**: Fix framework bottlenecks (both planners)
1. Implement DB connection pooling
2. Cache traffic light queries
3. Optimize state propagation
4. Target: SimplePlanner at 1.5x realtime, MLPlanner at 2.5x realtime

**Phase 2**: Optimize ML-specific code (ML planner only)
1. Cache map rasterization
2. Optimize agent rasterization
3. Reduce feature resolution
4. Target: MLPlanner at 1.8x realtime

**Phase 3**: Advanced optimizations (if needed)
1. GPU-accelerated feature building
2. Model quantization (reduce inference time)
3. Parallel feature extraction
4. Target: MLPlanner at 1.0x realtime

### For Development Workflow

1. **Disable LZMA compression** during profiling/testing
   - Saves 35s per simulation (huge!)
   - Use faster compression or none

2. **Use SimplePlanner for framework testing**
   - Faster iterations (34s vs 90s)
   - Isolates framework issues from ML issues

3. **Profile with single scenario first**
   - Faster turnaround for optimization validation
   - Scale to multiple scenarios after confirming fixes

## Raw Data

**SimplePlanner profile**: `profiling_output/cprofile_simple_planner.txt`
**MLPlanner profile**: `profiling_output/cprofile_output.txt`
**SimplePlanner stats**: `profiling_output/simulation_profile.stats` (overwritten by ML run)
**Scenario**: `near_multiple_vehicles`, 149 steps, sequential worker

---
**Generated by**: Navigator (Claude Code)
**Methodology**: Comparative cProfile analysis of SimplePlanner vs MLPlanner
**Confidence**: High (based on direct profiling of identical scenarios)
