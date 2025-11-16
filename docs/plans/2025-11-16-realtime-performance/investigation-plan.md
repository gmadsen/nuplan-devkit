# Real-Time Performance Investigation Plan

## Problem Statement

**Expected**: 80ms planner @ 10Hz (100ms period) should run at 1.0x realtime
**Observed**: 80ms planner runs at 0.51x realtime (39s wall-clock for 20s simulation)
**Gap**: 1.95x slower than expected - **WHY?**

## Critical Question

> "An 80ms planner should be able to output at 10Hz in real-time. What are the **exact root cause reasons and design decisions** that prevent this?"

## Investigation Goals

1. **Profile the complete simulation loop** - identify every component's overhead
2. **Measure cumulative time budget** - where does the extra ~100ms per step come from?
3. **Understand architectural bottlenecks** - which design decisions cause slowdowns?
4. **Document the planning architecture** - create comprehensive CLAUDE.md for `nuplan/planning/`
5. **Propose optimization strategies** - how to achieve 1.0x realtime performance

## Time Budget Analysis (Expected vs Actual)

### Expected (1.0x realtime, 100ms total per step)
```
Planner compute:           80ms   (80%)
State propagation:         5ms    (5%)
History buffer update:     3ms    (3%)
Observation processing:    5ms    (5%)
Callbacks (metrics):       5ms    (5%)
Database queries:          2ms    (2%)
──────────────────────────────────
TOTAL:                     100ms  (1.0x realtime)
```

### Actual (0.51x realtime, ~195ms total per step)
```
Unknown breakdown - NEEDS PROFILING
──────────────────────────────────
TOTAL:                     195ms  (0.51x realtime)
Gap to explain:            +95ms  (CRITICAL!)
```

## Investigation Phases

### Phase 1: Detailed Profiling (2-3 hours)
**Goal**: Measure exact time spent in each simulation loop component

#### 1.1 Instrument Simulation Loop
- [ ] Add timing probes to `nuplan/planning/simulation/simulation.py`:
  - `get_planner_input()` duration
  - `planner.compute_trajectory()` duration
  - `propagate()` duration breakdown:
    - History buffer append
    - Ego controller update
    - Observation update
    - Time controller advance
  - Callback execution time (per callback)

#### 1.2 Profile with cProfile
```bash
# Run 1-scenario simulation with detailed profiling
python -m cProfile -o sim_profile.prof \
  nuplan/planning/script/run_simulation.py \
  experiment_name=profile_test \
  planner=simple_planner \
  +simulation=open_loop_boxes \
  scenario_builder=nuplan_mini \
  'scenario_filter.scenario_types=[near_multiple_vehicles]' \
  scenario_filter.num_scenarios_per_type=1 \
  worker=sequential

# Analyze hotspots
python -m pstats sim_profile.prof
> sort cumtime
> stats 50
```

#### 1.3 Line-by-Line Profiling (if needed)
```bash
# Use line_profiler for critical functions
kernprof -l -v nuplan/planning/simulation/simulation.py
```

**Deliverable**: Timing breakdown table showing actual vs expected for each component

---

### Phase 2: Architecture Deep Dive (3-4 hours)
**Goal**: Understand planning module architecture and identify design bottlenecks

#### 2.1 Map the Simulation Loop Call Graph
- [ ] Document call flow from `run_simulation.py` → `Simulation` → callbacks
- [ ] Identify synchronous vs asynchronous operations
- [ ] Map database access patterns (scenario queries, map lookups)
- [ ] Trace observation processing pipeline

#### 2.2 Analyze Critical Path Components

**A. Planner Interface** (`nuplan/planning/simulation/planner/`)
- [ ] How does `SimplePlanner.compute_trajectory()` work?
- [ ] What's the actual vs advertised runtime?
- [ ] Does it block on I/O, computation, or both?

**B. Controller Stack** (`nuplan/planning/simulation/controller/`)
- [ ] `PerfectTrackingController.update_state()` overhead?
- [ ] Does it query database for vehicle dynamics?

**C. Observation Processing** (`nuplan/planning/simulation/observation/`)
- [ ] `TracksObservation.update()` - how expensive?
- [ ] Are observations pre-cached or computed per-step?

**D. History Management** (`nuplan/planning/simulation/history/`)
- [ ] `SimulationHistoryBuffer.append()` - copy vs reference?
- [ ] Does buffer trigger garbage collection?

**E. Scenario Queries** (`nuplan/planning/scenario_builder/`)
- [ ] Are traffic light states queried per-step from SQLite?
- [ ] Do map queries hit database or use in-memory cache?

**F. Callback Overhead** (`nuplan/planning/simulation/callback/`)
- [ ] Which callbacks run on `on_step_end()`?
- [ ] Are metrics computed synchronously per-step or batched?

#### 2.3 Create Comprehensive Planning Architecture Documentation
- [ ] **`nuplan/planning/CLAUDE.md`** - Top-level planning module overview
- [ ] Update simulation CLAUDE.md with profiling results
- [ ] Document design trade-offs (e.g., synchronous callbacks vs async)

**Deliverable**:
- Call graph diagram showing critical path
- Architecture documentation identifying bottlenecks
- Design decision rationale (why synchronous? why per-step callbacks?)

---

### Phase 3: Root Cause Analysis (1-2 hours)
**Goal**: Synthesize profiling + architecture into definitive root cause list

#### 3.1 Categorize Overhead Sources

**Hypothesis 1: Callback Overhead**
- If `TimingCallback` + `MetricCallback` + `SerializationCallback` run per-step:
  - Each callback iterates over history
  - Metrics may compute expensive aggregations
  - Serialization may write to disk per-step
- **Test**: Run simulation with `+callback=[]` (no callbacks)
- **Expected**: If callbacks are the issue, performance → 1.0x realtime

**Hypothesis 2: Database I/O**
- If traffic light states are queried from SQLite per-step (200 queries):
  - 200 × 0.5ms = 100ms overhead
- If map objects are queried per-step instead of cached:
  - Could add 50-100ms per step
- **Test**: Profile `scenario.get_traffic_light_status_at_iteration()`
- **Expected**: Should see SQLite in top 10 hotspots if this is the issue

**Hypothesis 3: Observation Recomputation**
- If `IDMAgents` or `LQRAgents` recompute agent trajectories per-step:
  - Could add 20-50ms per step
- **Test**: Compare `TracksObservation` (simple) vs `IDMAgents` (complex)
- **Expected**: IDMAgents should be slower due to trajectory prediction

**Hypothesis 4: History Buffer Inefficiency**
- If `SimulationHistoryBuffer.append()` does deep copies:
  - 200 steps × (ego state + observations) copies
  - Could add memory allocation overhead
- **Test**: Profile `history_buffer.append()` call
- **Expected**: Should see `copy.deepcopy` or large allocations

**Hypothesis 5: Planner is Actually Slower**
- If `SimplePlanner` advertises 80ms but actually takes 150ms:
  - We've been measuring wall-clock wrong
- **Test**: Add timing probe directly in `SimplePlanner.compute_trajectory()`
- **Expected**: Should see actual planner time per step

#### 3.2 Create Root Cause Summary

**Deliverable**: Ranked list of overhead sources with evidence:
```
1. [XX ms] Callback overhead (TimingCallback, MetricCallback)
   - Evidence: cProfile shows X% time in callbacks
   - Fix: Move metrics to on_simulation_end, disable per-step serialization

2. [XX ms] Database queries (traffic lights, map objects)
   - Evidence: 200 SQLite queries in trace
   - Fix: Cache scenario data in memory during initialization

3. [XX ms] Observation processing (IDMAgents trajectory prediction)
   - Evidence: on_step_end shows X ms in observation.update()
   - Fix: Use simpler TracksObservation or pre-cache predictions
...
```

---

### Phase 4: Optimization Strategy (1-2 hours)
**Goal**: Propose concrete changes to achieve 1.0x realtime performance

#### 4.1 Quick Wins (< 30 min implementation)
- [ ] Disable per-step serialization (`serialize_per_step=False`)
- [ ] Move metric computation to `on_simulation_end`
- [ ] Use `TracksObservation` instead of `IDMAgents` for evaluation runs
- [ ] Cache traffic light states in memory during `on_simulation_start`

#### 4.2 Medium-Effort Optimizations (2-4 hours implementation)
- [ ] Implement async callbacks (don't block simulation loop)
- [ ] Pre-load scenario data into memory (avoid SQLite per-step)
- [ ] Optimize history buffer (use references instead of copies)
- [ ] Profile and optimize SimplePlanner internals

#### 4.3 Long-Term Architectural Changes (days-weeks)
- [ ] Separate simulation thread from visualization thread
- [ ] Implement lock-free message queue for callbacks
- [ ] Move to async/await simulation loop (FastAPI-style)
- [ ] Consider Rust rewrite of critical path (20-100x speedup)

**Deliverable**: Prioritized optimization roadmap with effort estimates

---

### Phase 5: Validation & Documentation (30 min - 1 hour)
**Goal**: Verify optimizations achieve 1.0x realtime and document results

#### 5.1 Benchmark Before/After
```bash
# Baseline (current)
just simulate  # Measure: X.XXx realtime

# After quick wins
just simulate +callback=[]  # Measure: Y.YYx realtime

# After medium optimizations
just simulate +optimized_config  # Measure: Z.ZZx realtime
```

#### 5.2 Document Findings
- [ ] Update `/docs/reports/2025-11-16-realtime-performance-analysis.md`
- [ ] Add profiling results to `nuplan/planning/CLAUDE.md`
- [ ] Create performance tuning guide in docs

**Deliverable**: Performance report with before/after comparison

---

## Success Criteria

✅ **Minimum (Phase 1-3)**: Identify root cause of 0.51x realtime performance
✅ **Target (Phase 4)**: Achieve 0.8-1.0x realtime with optimizations
✅ **Stretch (Phase 5)**: Document performance characteristics for all planners

## Timeline Estimate

| Phase | Duration | Output |
|-------|----------|--------|
| 1. Detailed Profiling | 2-3 hours | Timing breakdown table |
| 2. Architecture Deep Dive | 3-4 hours | Planning CLAUDE.md + call graph |
| 3. Root Cause Analysis | 1-2 hours | Ranked overhead list |
| 4. Optimization Strategy | 1-2 hours | Optimization roadmap |
| 5. Validation | 30-60 min | Performance report |
| **TOTAL** | **8-12 hours** | Complete performance investigation |

## Key Questions to Answer

1. **What is the actual per-step time breakdown?**
   - Planner: X ms, Controller: Y ms, Callbacks: Z ms, etc.

2. **Why do callbacks run synchronously?**
   - Design decision or implementation oversight?

3. **Are database queries cached or per-step?**
   - Traffic lights, map objects, scenario metadata

4. **Can we run at 1.0x realtime without code changes?**
   - Just config changes (disable callbacks, etc.)

5. **What's the theoretical minimum latency?**
   - If we optimize everything, what's the floor?

## Dependencies & Prerequisites

- [x] Streaming visualization working (Phase 1-3 complete)
- [ ] cProfile installed and working
- [ ] line_profiler for fine-grained profiling
- [ ] Understand callback system (already documented)
- [ ] Understand simulation loop (needs deep dive)

## References

- **Existing Documentation**:
  - `nuplan/planning/simulation/CLAUDE.md` - Simulation module overview
  - `nuplan/planning/simulation/callback/CLAUDE.md` - Callback system
  - `docs/reports/2025-11-16-streaming-viz-proof.md` - Current performance baseline

- **Critical Files to Profile**:
  - `nuplan/planning/simulation/simulation.py:142-174` - Main simulation loop
  - `nuplan/planning/simulation/planner/simple_planner.py` - SimplePlanner implementation
  - `nuplan/planning/simulation/callback/metric_callback.py` - Metric computation
  - `nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario.py` - Database queries

- **Tools**:
  - `cProfile` - Python profiler
  - `line_profiler` - Line-by-line profiling
  - `py-spy` - Sampling profiler (alternative)
  - `gprof2dot` - Call graph visualization

---

## Next Steps

1. **Immediate**: Start Phase 1 profiling with cProfile
2. **Parallel**: Begin architecture documentation for `nuplan/planning/`
3. **Blocker**: Need to understand complete simulation loop before optimization

**This investigation is the foundation for all future planning work.**
