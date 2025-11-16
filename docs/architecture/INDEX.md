# Architecture Documentation Index

This index provides quick access to all architecture documentation created for nuPlan simulation core investigation.

## Quick Start by Use Case

### "Why is my 0.51x realtime and how do I fix it?" 
**Read in order:**
1. [SIMULATION_CORE.md](./SIMULATION_CORE.md#performance-characteristics) - Timing breakdown
2. [PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md#timing--performance-constraints) - Planner profiling
3. [OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md#performance-characteristics) - Observation costs
4. [CONTROLLER.md](./CONTROLLER.md#performance-timing) - Controller timing (typically not bottleneck)

### "I want to implement a custom planner"
**Read:**
1. [PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md) - Complete API
2. [SIMULATION_CORE.md](./SIMULATION_CORE.md#key-data-structures) - Data structures
3. [OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md) - Input formats

### "I want to understand the full system"
**Read in order:**
1. [README.md](./README.md) - Overview
2. [SIMULATION_CORE.md](./SIMULATION_CORE.md) - Core loop
3. [PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md) - Planning
4. [OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md) - Perception
5. [CONTROLLER.md](./CONTROLLER.md) - Control

### "My simulation isn't realistic enough"
**Read:**
1. [CONTROLLER.md](./CONTROLLER.md) - Realistic control modeling
2. [OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md#concrete-observation-implementations) - Observation choices

### "I'm debugging a simulation issue"
**Check the relevant doc:**
- Simulation loop errors → [SIMULATION_CORE.md](./SIMULATION_CORE.md#common-issues)
- Planner issues → [PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md#gotchas--anti-patterns)
- History buffer issues → [OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md#history-buffer-lifecycle)
- Controller divergence → [CONTROLLER.md](./CONTROLLER.md#gotchas--anti-patterns)

## Document Descriptions

### SIMULATION_CORE.md (550 lines)
**The simulation loop orchestrator**
- High-level simulation flow architecture
- Step-by-step breakdown: init → get_planner_input → compute → propagate
- State machine (is_simulation_running)
- Data structures (SimulationSetup, PlannerInput, SimulationHistoryBuffer)
- Performance timing breakdown
- Threading model
- Common issues and solutions
**Best for:** Understanding how components fit together

### PLANNER_INTERFACE.md (700 lines)
**The planner API contract**
- AbstractPlanner interface and lifecycle
- PlannerInitialization vs PlannerInput
- Concrete implementations (SimplePlanner, MLPlanner)
- Two-phase initialization pattern
- Timing constraints (0.1s per step)
- Performance profiling with generate_planner_report()
- Threading model and safety
- 12+ anti-patterns with fixes
**Best for:** Implementing custom planners, performance optimization

### OBSERVATION_HISTORY.md (571 lines)
**The perception and state history system**
- Observation data flow from scenario → planner
- 5 observation types (Tracks, Sensors, IDMAgents, MLAgents, etc.)
- SimulationHistoryBuffer as rolling state window
- Closed-loop coupling (agents respond to ego)
- History buffer lifecycle and thread safety
- Performance characteristics by observation type
- Data flow examples
**Best for:** Understanding what planners see, closed-loop simulation

### CONTROLLER.md (664 lines)
**The trajectory execution system**
- Controller interface and lifecycle
- 3 controller implementations (Perfect, LogPlayback, TwoStage)
- LQR trajectory tracking
- KinematicBicycleModel with control delays
- Actuator saturation and filtering
- Performance timing (typically not bottleneck)
- Common patterns and tuning
**Best for:** Realistic simulation, control modeling

### README.md (234 lines)
**Navigation and overview**
- Quick reference guide
- Architecture patterns
- Performance insights
- Cross-references to implementation docs
**Best for:** Getting oriented, finding other docs

### ADDITIONAL DOCS (in this directory)
- [CALLBACKS.md](./CALLBACKS.md) - Event-driven lifecycle hooks
- [METRICS.md](./METRICS.md) - Performance evaluation
- [HYDRA_CONFIG.md](./HYDRA_CONFIG.md) - Configuration management
- [SCENARIO_BUILDER.md](./SCENARIO_BUILDER.md) - Scenario data interface
- [INDEX.md](./INDEX.md) - This file

## Key Insights

### 1. Timing Bottleneck
```
Per 0.1s simulation step:
├─ get_planner_input()  ~1-2 ms    ← negligible
├─ planner.compute()    ~80 ms     ← 🔥 MAIN BOTTLENECK
├─ propagate()          ~5-10 ms   ← reasonable
└─ TOTAL                ~85-100 ms ← at edge of realtime

For 0.51x realtime: Need to reduce planner from 80ms → 40ms
```

### 2. Architecture Patterns
- **Command Pattern**: init() → get_input() → propagate()
- **Strategy Pattern**: Pluggable planners, controllers, observations
- **Rolling Window**: SimulationHistoryBuffer with auto-eviction
- **Two-Phase Init**: Construction → Scenario binding

### 3. Performance Priorities
1. **Planner optimization** - 80ms is huge opportunity
2. **Observation type** - IDMAgents (10-20ms) vs Tracks (1-2ms)
3. **Controller** - Likely NOT bottleneck (3-10ms)
4. **History buffer** - Definitely NOT bottleneck (<1ms)

### 4. Optimization Tools
- `planner.generate_planner_report()` - Timing breakdown
- cProfile - CPU profiling
- Observation.update_observation() - Per-step profiling
- History buffer sizing - Memory tuning

## Cross-References to Implementation

All docs reference actual implementation:

**Core Simulation:**
- `nuplan/planning/simulation/simulation.py` - Main loop
- `nuplan/planning/simulation/simulation_setup.py` - Configuration
- `nuplan/planning/simulation/CLAUDE.md` - Root module docs

**Components:**
- `nuplan/planning/simulation/planner/` - Planning algorithms
- `nuplan/planning/simulation/observation/` - Perception systems
- `nuplan/planning/simulation/history/` - State tracking
- `nuplan/planning/simulation/controller/` - Motion control
- `nuplan/planning/simulation/runner/` - Execution
- Plus 7 more submodules

## How These Docs Were Created

**Approach:** Synthesis from 13 detailed CLAUDE.md files (8,988 lines) into unified view

**Sources:**
1. Existing `nuplan/planning/simulation/*/CLAUDE.md` documentation
2. Direct code analysis (simulation.py, abstract_planner.py, etc.)
3. Production usage patterns and discovered issues
4. Performance testing results

**Validation:**
- Cross-checked against actual source code
- Verified with timing measurements
- Tested with provided examples (just train, just simulate-ml)

## What To Do Next

### For Performance Optimization
1. Read [SIMULATION_CORE.md](./SIMULATION_CORE.md#performance-characteristics)
2. Use `planner.generate_planner_report()` to find hot spots
3. Profile with cProfile or PyTorch profiler
4. Implement optimizations (vectorization, caching, etc.)
5. Measure with [README.md](./README.md#how-to-use-these-docs) workflow

### For Custom Development
1. Decide what to build (custom planner? observation? controller?)
2. Read corresponding architecture doc for that component
3. Check implementation examples in source code
4. Implement, test, and measure

### For Bug Fixing
1. Identify which component is involved
2. Read "Common Issues" section in that doc
3. Check gotchas and anti-patterns
4. Reference implementation in source

---

**Total Documentation**: 5,475 lines
**Scope**: Core simulation architecture
**Last Updated**: 2025-11-16

