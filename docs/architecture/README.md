# nuPlan Simulation Architecture Documentation

This directory contains comprehensive architectural documentation for nuPlan's core simulation system. These documents explain the foundational design patterns, data structures, and execution flow that power closed-loop autonomous vehicle evaluation.

## Quick Navigation

### For Performance Investigation (80ms planner → 0.51x realtime)

**Start here**: [SIMULATION_CORE.md](./SIMULATION_CORE.md) - Understand the main loop structure and timing breakdown

Then investigate in order:
1. **[PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md)** - Planner execution, timing constraints, profiling
2. **[OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md)** - Observation update costs, history buffer performance
3. **[CONTROLLER.md](./CONTROLLER.md)** - Controller timing (typically not a bottleneck)

### For Understanding Complete System

Read in this order:

1. **[SIMULATION_CORE.md](./SIMULATION_CORE.md)** (550 lines)
   - High-level overview of simulation loop
   - Initialization and main step sequence
   - State machine and boundary conditions
   - Threading model
   - **Key for**: Understanding overall architecture

2. **[PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md)** (700 lines)
   - Planner API contract
   - Two-phase initialization (construct → bind to scenario)
   - Timeline constraints (0.1s per timestep)
   - Threading and concurrency
   - **Key for**: Implementing custom planners, performance tuning

3. **[OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md)** (571 lines)
   - Observation types (DetectionsTracks, Sensors, IDMAgents, MLAgents)
   - History buffer as rolling state window
   - Closed-loop coupling (agents respond to ego)
   - Per-timestep data flow
   - **Key for**: Understanding perception inputs to planner

4. **[CONTROLLER.md](./CONTROLLER.md)** (664 lines)
   - Trajectory execution and vehicle dynamics
   - PerfectTracking vs TwoStageController
   - LQR trajectory tracking
   - Kinematic bicycle model with control delays
   - **Key for**: Realistic simulation, understanding actuator limits

## Document Relationships

```
SIMULATION_CORE.md (Integration Hub)
├─ Calls ─→ PLANNER_INTERFACE.md (Planner)
├─ Calls ─→ OBSERVATION_HISTORY.md (Perception)
└─ Calls ─→ CONTROLLER.md (Control)

SIMULATION_CORE.md explains:
  "For more on planner API → see PLANNER_INTERFACE.md"
  "For more on observations → see OBSERVATION_HISTORY.md"
  "For more on control → see CONTROLLER.md"
```

## Total Content

- **4 Architecture Documents**: 2,535 lines of analysis
- **Cross-references** to 13 detailed CLAUDE.md files in the codebase
- **Performance bottleneck analysis** specific to 0.51x realtime problem
- **Threading model** for Ray-based parallelization
- **Common gotchas** and anti-patterns learned from production use

## Key Insights

### Simulation Loop Timing (per 0.1s step)

```
┌─────────────────────────────────────┐
│ get_planner_input()    ~1-2 ms      │ ← Usually negligible
├─────────────────────────────────────┤
│ planner.compute()      ~80 ms       │ ← 🔥 MAIN BOTTLENECK
├─────────────────────────────────────┤
│ propagate()            ~5-10 ms     │ ← Reasonable
│ └─ controller          ~3-5 ms      │
│ └─ observation update  ~2-5 ms      │
├─────────────────────────────────────┤
│ TOTAL                  ~85-100 ms   │ ← At edge of realtime (100ms budget)
└─────────────────────────────────────┘

For 0.51x realtime: Step takes ~200ms
Need to reduce planner from 80ms → ~40ms
```

### Architecture Design Patterns

**Command Pattern**: Simulation.initialize() → get_planner_input() → propagate()
- Clear separation of read vs write operations

**Strategy Pattern**: Pluggable planners, controllers, observations via polymorphism
- Same simulation framework works with different components

**Rolling Window Pattern**: SimulationHistoryBuffer (21 samples = 2.0s @ 0.1s)
- Provides temporal context without unbounded memory

**Two-Phase Initialization**:
- Phase 1: Planner construction (parameters)
- Phase 2: Scenario binding (route, map, goal)
- Enables reuse across scenarios

### Critical Components for Optimization

| Component | Typical Time | Optimization |
|-----------|---|---|
| Planner compute | 80 ms | Profile, reduce feature extraction, optimize inference |
| Observation update | 2-5 ms | Use lightweight types, avoid ML agents if not needed |
| History buffer append | <1 ms | Usually not bottleneck |
| Controller | 3-5 ms | Usually not bottleneck |

## Related Documentation in Codebase

These architecture docs are **synthesis layer** over detailed CLAUDE.md files:

**Simulation Core Layer**:
- `nuplan/planning/simulation/CLAUDE.md` - 8988 lines across 13 submodules
- `nuplan/planning/simulation/runner/CLAUDE.md` - Execution orchestration
- `nuplan/planning/simulation/callback/CLAUDE.md` - Lifecycle hooks

**Component Layer** (linked from architecture docs):
- `nuplan/planning/simulation/planner/CLAUDE.md` - Planning algorithms
- `nuplan/planning/simulation/observation/CLAUDE.md` - Perception systems
- `nuplan/planning/simulation/history/CLAUDE.md` - State tracking
- `nuplan/planning/simulation/controller/CLAUDE.md` - Motion control
- `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory formats
- `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - Time stepping

## How to Use These Docs

### Scenario 1: "My planner runs at 0.51x realtime, help optimize"

1. Read: **SIMULATION_CORE.md** - Get timing breakdown
2. Read: **PLANNER_INTERFACE.md** - Profile your planner with generate_planner_report()
3. Analyze: Which operation in compute_planner_trajectory() takes 80ms?
   - Feature extraction? Map queries? Model inference? All of above?
4. Optimize: See "Optimization Opportunities" section in SIMULATION_CORE.md
5. Measure: Use generate_planner_report() to validate improvements

### Scenario 2: "I want to implement a custom planner"

1. Read: **PLANNER_INTERFACE.md** - Complete API reference
2. Review: "Implementing a Custom Planner" code recipe
3. Understand: Threading model and timing constraints
4. Check: Observation type compatibility

### Scenario 3: "I want to understand closed-loop simulation with agents"

1. Read: **SIMULATION_CORE.md** - Overview of loop structure
2. Read: **OBSERVATION_HISTORY.md** - Focus on IDMAgents and closed-loop coupling
3. Trace: How agent behavior responds to ego trajectory
4. Reference: `nuplan/planning/simulation/observation/CLAUDE.md` for implementation

### Scenario 4: "My simulation isn't realistic enough"

1. Read: **CONTROLLER.md** - Understand realistic control modeling
2. Switch: From PerfectTrackingController → TwoStageController + LQRTracker
3. Tune: Control delays and actuator limits to match your vehicle
4. Validate: Plot control efforts and state errors

## Architecture Evolution

This documentation was created to support:

1. **Phase 1**: Foundational understanding of simulation loop (SIMULATION_CORE.md)
2. **Phase 2**: Component deep-dives (PLANNER_INTERFACE.md, OBSERVATION_HISTORY.md, CONTROLLER.md)
3. **Phase 3**: Optimization targeting (0.51x → 1.0x realtime improvement investigation)

## Notes for Developers

### Adding New Content

When adding optimizations or new features:
1. Update relevant architecture doc (or create new section)
2. Include **Before/After timing** to show impact
3. Document **threading implications** (if any)
4. Add **cross-references** to related components
5. Include **common gotchas** you discovered

### Keeping Docs Fresh

**Update schedule**:
- After major planner/controller/observation changes → Update corresponding doc
- After performance improvements → Update timing tables
- After discovering new gotchas → Add to relevant doc

### Validation

Check docs against actual code:
```bash
# Verify class names and methods exist
grep -r "class AbstractPlanner" nuplan/planning/simulation/
grep -r "def compute_planner_trajectory" nuplan/planning/simulation/

# Verify performance characteristics
# (Run actual simulations to validate timing numbers)
just train-quick
just simulate
```

## Questions Answered

**Q: Why is my 80ms planner only running at 0.51x realtime?**
A: 80ms out of 100ms is 80% utilization. Add overhead, and you're over 100ms per step. See SIMULATION_CORE.md performance section.

**Q: How do observations work with closed-loop agents?**
A: IDMAgents respond to ego trajectory via observation.update_observation(). See OBSERVATION_HISTORY.md "IDMAgents Closed-Loop Coupling".

**Q: Can I parallelize my planner with multi-threading?**
A: Check threading model in PLANNER_INTERFACE.md. Typically yes, but GIL-bound operations won't help.

**Q: What's the difference between initialize() and compute_planner_trajectory()?**
A: Two-phase construction. See PLANNER_INTERFACE.md "Two-Phase Initialization" pattern.

**Q: Why does my history buffer size calculation have a +1?**
A: Off-by-one guard for duration. See SIMULATION_CORE.md "Buffer Size Calculation".

## Contact & Feedback

If these docs are unclear or missing information:
1. Check the referenced CLAUDE.md files in nuplan/planning/simulation/*/CLAUDE.md
2. Review actual implementation in corresponding .py files
3. Add issue to project tracker with improvement suggestions

---

**Generated**: 2025-11-16
**Last Updated**: 2025-11-16
**Scope**: Core simulation architecture investigation for performance optimization

