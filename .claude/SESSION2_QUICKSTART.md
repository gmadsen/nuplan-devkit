# Session 2 Quickstart Guide

**Date**: 2025-11-15
**Status**: Ready to start
**Goal**: Complete Tier 1 production code documentation (Phase 1B + 1C)

---

## Session 1 Recap (What We Accomplished)

### Completed Modules (7 dirs, ~3,187 lines)
✅ `nuplan/common/actor_state/` - State representations (ego, agents, geometry)
✅ `nuplan/common/geometry/` - Coordinate transforms, distance calculations
✅ `nuplan/common/maps/` - Abstract map API interface
✅ `nuplan/common/maps/nuplan_map/` - Concrete map implementation
✅ `nuplan/common/utils/` - Common utilities (I/O, S3, distributed, interpolation)
✅ `nuplan/planning/simulation/planner/` - THE core planner interface
✅ `nuplan/planning/simulation/trajectory/` - Trajectory representations

### Key Wins
- **Parallel agents worked!** 4 agents completed simultaneously (geometry, maps, nuplan_map, utils)
- **High quality maintained** - 10+ gotchas, code examples, cross-references per doc
- **Strategy validated** - Tier-based approach with production code first

### Strategic Decision
**Deferred test directories to Phase 1D** - Focus on production code abstractions first, test docs later

---

## Session 2 Scope

### Target: Complete Tier 1 Production Code (4 directories)

**Phase 1B - Core Planning Abstractions (2 dirs):**
1. `nuplan/planning/simulation/planner/ml_planner/` - ML-based planning implementation
2. `nuplan/planning/scenario_builder/` - Scenario extraction & filtering
3. `nuplan/planning/scenario_builder/nuplan_db/` - Concrete scenario builder for nuPlan DB

**Phase 1C - Supporting Infrastructure (1 dir):**
4. `nuplan/database/nuplan_db_orm/` - SQLAlchemy ORM for nuPlan dataset

### Expected Effort
- **Parallel agents**: 4 agents (one per directory)
- **Quality**: Deep-dive (Tier 1 standard)
- **Estimated time**: 2-3 hours wall-clock (with parallelization)
- **Output**: ~1,500-2,000 lines of documentation

---

## Context & Insights for Session 2

### 1. ML Planner (`planner/ml_planner/`)

**Why it matters**: This is THE concrete ML planner implementation that planners use for learned planning.

**Key files to understand**:
- `ml_planner.py` - Main MLPlanner class (implements AbstractPlanner)
- `model_loader.py` - PyTorch Lightning checkpoint loading
- `transform_utils.py` - Coordinate transformations for ML features

**Dependencies** (already documented):
- `nuplan/planning/simulation/planner/abstract_planner.py` ✅ (Session 1)
- `nuplan/planning/simulation/trajectory/abstract_trajectory.py` ✅ (Session 1)
- `nuplan/common/actor_state/` ✅ (Session 1)
- `nuplan/common/geometry/torch_geometry.py` ✅ (Session 1)

**What to emphasize**:
- How ML planner wraps PyTorch Lightning models
- Feature extraction pipeline (observations → tensor features)
- Trajectory decoding (model output → InterpolatedTrajectory)
- Checkpoint loading and model initialization
- Gotchas: Model checkpoint paths, precision handling, batch vs single inference

**Cross-references**:
- Links to training/ modules (will document in Session 5-6)
- Links to preprocessing/feature_builders/ (Session 5)

---

### 2. Scenario Builder (`planning/scenario_builder/`)

**Why it matters**: This is how scenarios are extracted from the nuPlan database - critical for understanding dataset access.

**Key files to understand**:
- `abstract_scenario_builder.py` - Interface for building scenarios
- `abstract_scenario.py` - Scenario abstraction (time slice of a log)
- `scenario_filter.py` - Filtering scenarios by type, duration, etc.
- `scenario_utils.py` - Utility functions

**Dependencies** (already documented):
- `nuplan/common/maps/abstract_map.py` ✅ (Session 1)
- `nuplan/common/actor_state/` ✅ (Session 1)

**Dependencies** (NOT yet documented):
- `nuplan/database/nuplan_db/` - Will document in this session!
- `nuplan/planning/scenario_builder/nuplan_db/` - Will document in this session!

**What to emphasize**:
- AbstractScenario interface (get_initial_state, get_route, get_expert_trajectory, etc.)
- Scenario filtering system (scenario types, num scenarios, filter logic)
- How scenarios are temporal slices of driving logs
- Relationship to database layer
- Gotchas: Scenario caching, memory usage, filter combinatorics

**Cross-references**:
- Links to database/nuplan_db_orm/ (will document in this session)
- Links to simulation/runner/ (will document in Session 3-4)

---

### 3. NuPlan DB Scenario Builder (`planning/scenario_builder/nuplan_db/`)

**Why it matters**: Concrete implementation that reads nuPlan database to build scenarios.

**Key files to understand**:
- `nuplan_scenario_builder.py` - Main builder implementation
- `nuplan_scenario.py` - Concrete scenario class
- `nuplan_scenario_utils.py` - NuPlan-specific utilities

**Dependencies** (already documented):
- `nuplan/common/maps/nuplan_map/` ✅ (Session 1)
- `nuplan/planning/scenario_builder/abstract_scenario.py` - Will document in this session!

**Dependencies** (NOT yet documented):
- `nuplan/database/nuplan_db_orm/` - Will document in this session!
- `nuplan/database/nuplan_db/` - Needs investigation

**What to emphasize**:
- How scenarios are extracted from SQLite database
- Relationship between Log, Scenario, LidarPc, Track tables
- Scenario type classification
- Caching mechanisms
- Gotchas: Database locking, query performance, scenario type strings

**Cross-references**:
- Links to database/nuplan_db_orm/ (will document in this session)
- Links to parent scenario_builder/ (will document in this session)

---

### 4. NuPlan DB ORM (`database/nuplan_db_orm/`)

**Why it matters**: SQLAlchemy ORM schema for the entire nuPlan database - this is how ALL scenario data is accessed.

**Key files to understand**:
- `nuplan_db.py` or similar - Main database connection
- ORM model files: `log.py`, `scenario.py`, `lidar_pc.py`, `image.py`, `track.py`, etc.
- Table relationships (Log → Scenario, Scenario → LidarPc/Track)

**Dependencies** (minimal, mostly SQLAlchemy):
- `sqlalchemy` - ORM framework
- Standard library

**What to emphasize**:
- Database schema (tables, columns, relationships, foreign keys)
- ORM model classes (Log, Scenario, LidarPc, Image, Track, TrafficLightStatus, etc.)
- Query patterns (how to load scenarios, lidar, tracks)
- Relationship between tables (Log has many Scenarios, Scenario has many LidarPcs/Tracks)
- Gotchas: Lazy loading, N+1 query problems, database size, query performance

**Cross-references**:
- Links to scenario_builder/ (uses ORM for scenario extraction)
- Links to maps_db/ (map database, separate from scenario DB)

---

## Recommended Execution Strategy

### Parallel Agent Launch (All 4 Simultaneously)

```python
# Agent 1: ML Planner
subagent_type = "technical-writer"
description = "Document nuplan/planning/simulation/planner/ml_planner/"
model = "sonnet"

# Agent 2: Scenario Builder (abstract)
subagent_type = "technical-writer"
description = "Document nuplan/planning/scenario_builder/"
model = "sonnet"

# Agent 3: NuPlan DB Scenario Builder
subagent_type = "technical-writer"
description = "Document nuplan/planning/scenario_builder/nuplan_db/"
model = "sonnet"

# Agent 4: Database ORM
subagent_type = "technical-writer"
description = "Document nuplan/database/nuplan_db_orm/"
model = "sonnet"
```

### Agent Prompt Template

Each agent should receive:
1. Directory path to document
2. Quality standard: Tier 1 deep-dive
3. Reference to existing CLAUDE.md files for style
4. Specific emphasis points (see context above)
5. Instruction to return ONLY markdown content (no wrapper text)

### Post-Agent Tasks

1. **Save agent outputs** to respective CLAUDE.md files
2. **Review cross-references** - ensure links between modules are accurate
3. **Update backlog** - Mark completed directories
4. **Git commit** - One commit for all 4 directories
5. **Update SESSION3_QUICKSTART.md** - Prepare for next session

---

## Critical Insights from Session 1

### What Worked Well
✅ **Parallel agents** - 4 agents completed in ~10-15 minutes each
✅ **Focused prompts** - Single directory per agent prevents scope creep
✅ **Quality template** - Agents consistently produced high-quality output
✅ **Sonnet model** - Good balance of quality and speed

### What to Improve
⚠️ **Agent timeout issue** - Initial 8-agent batch timed out, 4-agent batch succeeded
⚠️ **Cross-reference accuracy** - Some agents guessed at module relationships (fixed manually)
⚠️ **File ordering** - Agents sometimes didn't prioritize files correctly

### Lessons for Session 2
1. **Stick with 4 agents max** - Proven to work reliably
2. **Provide dependency context** - Tell agents which modules are already documented
3. **Emphasize cross-references** - Explicitly request links to already-documented modules
4. **Prioritize critical files** - Give agents hints on most important files

---

## Quality Checklist (Before Committing)

For each CLAUDE.md file, verify:
- [ ] **Purpose & Responsibility** - Clear 2-3 sentence summary
- [ ] **Key Abstractions** - Classes, interfaces, functions listed with descriptions
- [ ] **Architecture** - Design patterns and architectural decisions explained
- [ ] **Dependencies** - Links to documented modules ✅, notes on undocumented
- [ ] **Dependents** - Who uses this module?
- [ ] **Critical Files** - Prioritized list (most important first)
- [ ] **Usage Patterns** - Code examples showing common patterns
- [ ] **Gotchas** - 10+ pitfalls and solutions
- [ ] **Related Docs** - Cross-links to parent/sibling/child modules
- [ ] **AIDEV notes** - TODOs, questions, warnings

---

## Expected Deliverables

### Files Created (4)
1. `/nuplan/planning/simulation/planner/ml_planner/CLAUDE.md`
2. `/nuplan/planning/scenario_builder/CLAUDE.md`
3. `/nuplan/planning/scenario_builder/nuplan_db/CLAUDE.md`
4. `/nuplan/database/nuplan_db_orm/CLAUDE.md`

### Files Updated (2)
1. `.claude/DOCUMENTATION_BACKLOG.md` - Mark 4 directories complete
2. `.claude/SESSION3_QUICKSTART.md` - Context for next session

### Git Commit
```bash
git commit -m "Add Tier 1 documentation for Phase 1B + 1C

Complete core planning abstractions and database ORM:
- ml_planner/ - ML-based planning implementation
- scenario_builder/ - Scenario extraction interface
- scenario_builder/nuplan_db/ - Concrete scenario builder
- database/nuplan_db_orm/ - SQLAlchemy ORM schema

Progress: 11/113 complete (9.7%) - Tier 1 production code 100% done!
Next: Tier 2 simulation infrastructure

🧭 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Post-Session 2 Status

**After Session 2 completes:**
- ✅ **Tier 1 production code**: 10/10 complete (100%)
- ⏳ **Tier 1 test directories**: 0/8 (deferred to Phase 1D)
- 📊 **Overall progress**: 11/113 (9.7%)

**Next session targets:**
- Session 3: Tier 2 Phase 2A + 2B (Observation, History, Controller, Motion)
- Focus shifts from foundational abstractions to simulation infrastructure

---

## Quick Command Reference

### Start Session 2
```bash
# Launch 4 parallel agents
# (Use Task tool with technical-writer subagent)
```

### Check Progress
```bash
# View backlog
cat .claude/DOCUMENTATION_BACKLOG.md

# View existing docs
ls -lh nuplan/common/*/CLAUDE.md
ls -lh nuplan/planning/simulation/*/CLAUDE.md
```

### Commit Session 2
```bash
# Stage files
git add nuplan/planning/simulation/planner/ml_planner/CLAUDE.md \
        nuplan/planning/scenario_builder/CLAUDE.md \
        nuplan/planning/scenario_builder/nuplan_db/CLAUDE.md \
        nuplan/database/nuplan_db_orm/CLAUDE.md \
        .claude/DOCUMENTATION_BACKLOG.md \
        .claude/SESSION3_QUICKSTART.md

# Commit
git commit -m "[message from template above]"
```

---

**Remember G Money**: Quality over speed! Each CLAUDE.md is a permanent reference that will save hours of future context-gathering. Take the time to get it right. 🧭
