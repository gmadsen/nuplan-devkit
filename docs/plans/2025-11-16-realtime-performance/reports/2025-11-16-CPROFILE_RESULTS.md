# cProfile Analysis: nuPlan ML Planner Performance Bottlenecks
**Date**: 2025-11-16
**Workstream**: D - Performance Profiling
**Analyst**: Navigator (Claude Code)

## Executive Summary

**Total simulation time**: 90 seconds for 149 steps = **604ms per step average**
**Expected time**: 100ms per step (80ms planner + 20ms overhead)
**Actual overhead**: **~504ms per step** (5x worse than expected!)

### Critical Finding
The **95ms overhead assumption was drastically wrong**. The actual overhead is ~500ms per step, with major bottlenecks in:
1. **Database queries** (35.8% of time): 21,691 SQLite queries taking 12.7s + 6.8s = 19.5s total
2. **Post-simulation logging** (30.9%): LZMA compression taking 35.2s (one-time cost)
3. **Planner compute** (16.7%): ML planner + feature building taking 19.0s
4. **Map I/O** (3.8%): GeoPackage reads taking 4.4s

## Performance Breakdown

### Overall Timing (114 seconds total profiling time)

| Component | Time (s) | % of Total | Per-Step (ms) | Notes |
|-----------|----------|------------|---------------|-------|
| **Database queries** | 19.5 | 17.1% | 131ms | 21,691 queries! Way too many |
| **LZMA compression** | 35.2 | 30.9% | 236ms | One-time cost at end (amortized) |
| **ML planner compute** | 19.0 | 16.7% | 127ms | Feature building + inference |
| **Map rendering** | 16.6 | 14.5% | 111ms | Raster generation from map data |
| **Map file I/O** | 4.4 | 3.8% | 29ms | GeoPackage reads (22 calls) |
| **Metric computation** | 6.1 | 5.3% | 41ms | Histogram rendering (post-sim) |
| **Config loading** | 6.3 | 5.5% | N/A | One-time initialization |
| **Other** | 7.0 | 6.1% | 47ms | Geometry, transforms, etc. |

**Actual simulation loop time**: ~85 seconds (114s - 6s config - 6s metrics - 35s compression)
**Per-step time**: 85s / 149 steps = **570ms/step**

### Top 20 Hotspots (by cumulative time)

| Rank | Function | Cumtime (s) | % | Per-call (ms) | Calls | Component |
|------|----------|-------------|---|---------------|-------|-----------|
| 1 | `_lzma.LZMACompressor.compress` | 35.2 | 30.9% | 35154 | 1 | Post-sim logging |
| 2 | `execute_many` (query_session) | 18.6 | 16.3% | 0.9 | 21691 | Database |
| 3 | `ml_planner.compute_planner_trajectory` | 19.0 | 16.7% | 127 | 149 | Planner |
| 4 | `raster_feature_builder._compute_feature` | 16.6 | 14.5% | 112 | 149 | Feature building |
| 5 | `get_traffic_light_status_for_lidarpc_token_from_db` | 15.4 | 13.5% | 2.1 | 7176 | Database |
| 6 | `simulation.propagate` | 14.2 | 12.5% | 95 | 149 | State update |
| 7 | `_get_layer_coords` (raster_utils) | 11.1 | 9.7% | 12 | 894 | Raster generation |
| 8 | `simulation.get_planner_input` | 8.9 | 7.8% | 59 | 149 | Input prep |
| 9 | `sqlite3.Cursor.execute` | 6.8 | 6.0% | 5.5 | 1228 | Database |
| 10 | `get_roadmap_raster` | 6.8 | 6.0% | 46 | 149 | Map rasterization |
| 11 | `metric_summary_callback.on_run_simulation_end` | 6.2 | 5.4% | 6196 | 1 | Post-sim metrics |
| 12 | `deepcopy` | 6.1 | 5.3% | 0.0004 | 1535927 | Data copying |
| 13 | `get_agents_raster` | 4.9 | 4.3% | 33 | 149 | Raster generation |
| 14 | `pyogrio.read` (GeoPackage) | 4.4 | 3.8% | 198 | 22 | Map I/O |
| 15 | `torch.Tensor.to` | 5.5 | 4.8% | 12 | 470 | GPU transfers |
| 16 | `sqlite3.connect` | 3.8 | 3.3% | 3.1 | 1228 | Database |
| 17 | `shapely.coords.xy` | 2.9 | 2.5% | 0.2 | 16570 | Geometry |
| 18 | `shapely.intersects` | 0.2 | 0.2% | 0.3 | 894 | Geometry |
| 19 | `pyproj.transform` | 0.6 | 0.5% | 27 | 22 | Coordinate transforms |
| 20 | `translate_longitudinally_and_laterally` | 0.3 | 0.3% | 0.008 | 40060 | Transforms |

### Top 20 Hotspots (by total time, excluding subcalls)

| Rank | Function | Tottime (s) | % | Per-call (ms) | Calls | Component |
|------|----------|-------------|---|---------------|-------|-----------|
| 1 | `_lzma.LZMACompressor.compress` | 35.2 | 30.9% | 35154 | 1 | Post-sim logging |
| 2 | `execute_many` (query_session) | 12.7 | 11.1% | 0.6 | 21691 | Database |
| 3 | `sqlite3.Cursor.execute` | 6.8 | 6.0% | 5.5 | 1228 | Database |
| 4 | `torch.Tensor.to` | 5.5 | 4.8% | 12 | 470 | GPU transfers |
| 5 | `pyogrio.read` (GeoPackage) | 4.4 | 3.8% | 198 | 22 | Map I/O |
| 6 | `sqlite3.connect` | 3.8 | 3.3% | 3.1 | 1228 | Database |
| 7 | `deepcopy` | 2.1 | 1.8% | 0.0001 | 1535927 | Data copying |
| 8 | `shapely.coords.xy` | 1.9 | 1.6% | 0.1 | 16570 | Geometry |
| 9 | `sqlite3.Connection.close` | 0.9 | 0.8% | 0.8 | 1228 | Database |
| 10 | `torch.conv2d` | 0.8 | 0.7% | 0.1 | 7897 | Neural net |
| 11 | `isinstance` | 0.8 | 0.7% | 0.0001 | 5251683 | Type checks |
| 12 | `_reconstruct` (copy) | 0.7 | 0.6% | 0.01 | 164897 | Data copying |
| 13 | `array.append` | 0.7 | 0.6% | 0.0001 | 5892400 | Data structures |
| 14 | `_imp.create_dynamic` | 0.7 | 0.6% | 3.0 | 236 | Import |
| 15 | `marshal.loads` | 0.6 | 0.5% | 0.2 | 3676 | Deserialization |
| 16 | `pyproj.transform` | 0.6 | 0.5% | 26 | 22 | Coordinate transforms |
| 17 | `_deepcopy_dict` | 0.5 | 0.4% | 0.003 | 140356 | Data copying |
| 18 | `implement_array_function` | 0.5 | 0.4% | 0.002 | 229036 | NumPy |
| 19 | `dict.get` | 0.5 | 0.4% | 0.0001 | 3709346 | Dict operations |
| 20 | `compile` | 0.4 | 0.4% | 0.08 | 5542 | Compilation |

## Per-Step Analysis (149 steps)

### Simulation Loop Breakdown (per-step average)

| Phase | Time (ms) | % of Loop | Notes |
|-------|-----------|-----------|-------|
| **get_planner_input()** | 59 | 10.4% | History buffer, observation assembly |
| **planner.compute_trajectory()** | 127 | 22.3% | ML inference + feature building |
| **propagate()** | 95 | 16.7% | State update, controller, time advance |
| **Database queries** | 131 | 23.0% | Far too many queries per step! |
| **Other per-step overhead** | 158 | 27.7% | Callbacks, bookkeeping, data copies |
| **Total per-step** | **570** | 100% | |

### Database Query Analysis (MAJOR BOTTLENECK!)

**Total queries**: 21,691 across 149 steps = **145 queries per step average**

This is **absurdly high**! Each simulation step should only need:
- 1 query for ego state
- 1 query for tracked objects
- 1 query for traffic light status
- Maybe 1-2 queries for map data

Expected: **~5 queries/step**
Actual: **145 queries/step**
**29x more queries than necessary!**

**Breakdown by query type**:
- `get_traffic_light_status_for_lidarpc_token_from_db`: 7176 calls (48 per step!)
- Generic `execute_many`: 21691 calls (145 per step!)
- `sqlite3.Cursor.execute`: 1228 calls (8 per step)
- `sqlite3.connect` + `close`: 1228 pairs (8 connects per step!)

**Critical issues**:
1. Creating new DB connections per query instead of reusing
2. Querying traffic lights 48 times per step (should be cached!)
3. No query result caching or batching

**Time breakdown**:
- Total DB time: 19.5s (131ms per step)
- Connection overhead: 4.7s (3.8s connect + 0.9s close) = 31ms per step
- Query execution: 12.7s execute_many + 6.8s execute = 19.5s = 131ms per step

## Component Deep-Dive

### 1. Database Queries (131ms/step, 23% of loop time)

**Root cause**: Excessive database queries with connection churn

**Evidence**:
- 1228 connection open/close cycles (8 per step!)
- Each connection: 3.1ms open + 0.8ms close = 3.9ms overhead
- 21,691 total queries = 145 queries/step
- Traffic light queries: 7176 calls = 48 per step (should be 1!)

**Optimization potential**: **~100ms/step** (80% reduction possible)
- Use connection pooling: -31ms/step (eliminate 8 connects/step)
- Cache traffic light status: -40ms/step (eliminate 47 redundant queries)
- Batch queries: -30ms/step (reduce round trips)

### 2. Post-Simulation Logging (236ms/step amortized, one-time cost)

**Root cause**: LZMA compression of simulation log

**Evidence**:
- 35.2s spent in LZMA compression (single call at end)
- Amortized: 35.2s / 149 steps = 236ms/step
- This is a one-time post-simulation cost, not per-step

**Optimization potential**: **~200ms/step** (amortized)
- Use faster compression (gzip, zstd): -150ms/step
- Compress asynchronously: -236ms/step (non-blocking)
- Disable compression in profiling: -236ms/step

**Note**: This is NOT part of the real-time loop, so less critical for real-time performance.

### 3. ML Planner Compute (127ms/step, 22% of loop time)

**Root cause**: Feature building dominates planner time

**Breakdown**:
- Feature building: 16.6s / 149 = 112ms/step (88% of planner time)
- Inference: 19.0s - 16.6s = 2.4s / 149 = 16ms/step (12% of planner time)

**Feature building breakdown**:
- Map rasterization: 6.8s (46ms/step) - 41% of feature time
- Agent rasterization: 4.9s (33ms/step) - 29% of feature time
- Layer coords: 11.1s (12ms/step per layer × 894 calls) - 66% of feature time

**Optimization potential**: **~60ms/step** (50% reduction possible)
- Cache rasterized map layers: -30ms/step (map doesn't change per step)
- Optimize agent rasterization: -20ms/step (vectorize operations)
- Reduce raster resolution: -10ms/step (if acceptable)

### 4. Map I/O (29ms/step, 5% of loop time)

**Root cause**: Reading GeoPackage files from disk

**Evidence**:
- 22 calls to `pyogrio.read()` taking 4.4s total
- Average: 198ms per read
- Likely reading map tiles on demand

**Optimization potential**: **~25ms/step**
- Preload map data into memory: -29ms/step (eliminate I/O)
- Cache map queries: -20ms/step

### 5. State Propagation (95ms/step, 17% of loop time)

**Root cause**: Controller updates, history buffer, time stepping

**Evidence**:
- `propagate()` takes 14.2s / 149 = 95ms per step
- Includes controller, observation update, time advance

**Optimization potential**: **~20ms/step** (20% reduction)
- Optimize history buffer appends
- Reduce data copying in state updates

### 6. Data Copying (per-step overhead)

**Evidence**:
- `deepcopy`: 2.1s total, 1.5M calls (10,300 calls/step!)
- Array operations: 0.7s append, 5.9M calls (39,600 calls/step!)

**Optimization potential**: **~30ms/step**
- Use shallow copies where safe
- Reduce defensive copying
- Use views instead of copies

## Surprising Findings

1. **Database query explosion**: 145 queries/step instead of expected ~5
   - This alone accounts for 131ms/step (23% of loop time!)
   - Traffic light status queried 48 times per step (insane!)

2. **Connection churn**: Opening/closing DB connections 8 times per step
   - Should reuse a single connection pool
   - 31ms/step wasted on connection overhead

3. **Post-simulation logging dominates**: 35s LZMA compression
   - This is amortized but still significant
   - Should use async compression or faster algorithm

4. **Feature building is expensive**: 112ms/step for rasterization
   - 88% of ML planner time is feature prep, not inference!
   - Map raster could be cached (doesn't change)

5. **Map I/O is sporadic but slow**: 22 reads at 198ms each
   - Should preload all map data into memory
   - GeoPackage format might not be optimal for random access

## Critical Success Criteria - Updated Assessment

Original assumption: **95ms overhead** to reduce
Actual finding: **~470ms overhead** (excluding 80ms planner + 20ms expected overhead)

### Top 3 Bottlenecks (by impact)

**1. Database Queries (131ms/step, 80% reduction possible)**
   - **Impact**: -100ms/step
   - **Effort**: Medium (refactor to connection pooling + caching)
   - **Quick win**: Cache traffic light status (-40ms/step, Easy)

**2. Feature Building (112ms/step, 50% reduction possible)**
   - **Impact**: -60ms/step
   - **Effort**: Medium (cache map rasters, optimize agent raster)
   - **Quick win**: Cache map rasterization (-30ms/step, Easy)

**3. Map I/O (29ms/step, 90% reduction possible)**
   - **Impact**: -25ms/step
   - **Effort**: Easy (preload map data)
   - **Quick win**: Preload map tiles (-25ms/step, Easy)

### Optimization Roadmap

**Phase 1: Quick Wins (Target: -95ms/step, Easy, 1-2 days)**
1. Cache traffic light status per scenario: -40ms
2. Cache map rasterization: -30ms
3. Preload map data: -25ms

**Phase 2: Medium Effort (Target: -80ms/step, Medium, 3-5 days)**
4. Implement DB connection pooling: -31ms
5. Optimize agent rasterization: -20ms
6. Batch database queries: -30ms

**Phase 3: Major Refactor (Target: -50ms/step, Hard, 1-2 weeks)**
7. Reduce data copying: -30ms
8. Async compression: -20ms (amortized)
9. Optimize state propagation: -20ms

**Total potential**: -225ms/step reduction → **345ms/step final** (still 3.5x realtime)

To reach 1.0x realtime (100ms/step total), we need to reduce by 470ms, which requires:
- All Phase 1-3 optimizations: -225ms
- Reduce planner compute from 80ms to 20ms: -60ms
- Additional unknown optimizations: -185ms

**Realistic target**: 0.5x realtime → 2.0x realtime improvement with Phase 1-2 optimizations.

## Raw Data

**Profile stats file**: `profiling_output/simulation_profile.stats`
**Full output**: `profiling_output/cprofile_output.txt`
**Simulation config**: ML planner, raster model, single scenario (near_multiple_vehicles)
**Steps simulated**: 149
**Total profiling time**: 114 seconds
**Simulation loop time**: ~85 seconds (excluding init + post-processing)

## Next Steps

1. **Instrument simulation loop** with fine-grained timing probes (Task 3)
2. **Verify DB query count** and identify redundant queries
3. **Measure cache hit rate** for map and traffic light queries
4. **Profile simple_planner baseline** to isolate ML-specific overhead
5. **Implement Phase 1 quick wins** and re-profile

---
**Generated by**: Navigator (Claude Code)
**Methodology**: cProfile profiling of single scenario simulation
**Confidence**: High (based on 59M function calls across 114s runtime)
