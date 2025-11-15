# nuPlan CLAUDE.md Documentation Backlog

**Goal**: Create comprehensive CLAUDE.md files for every directory in nuplan-devkit
**Quality Bar**: Deep-dive for Tier 1-2, comprehensive overview for Tier 3-4
**Strategy**: Parallel agents working methodically through backlog

---

## Progress Tracker

**Completed**: 19 / 113 directories (16.8%)
**In Progress**: 0
**Remaining**: 94

**Session 1 Complete**: Phase 1A production code (7 dirs) ✅
**Session 2 Complete**: Phase 1B + 1C production code (4 dirs) ✅
**Session 3 Batch 1 Complete**: Phase 2A + path/occupancy_map (8 dirs) ✅
**Next Session**: Phase 2B remaining + Phase 2C

---

## TIER 1: CRITICAL ABSTRACTIONS (Deep-dive quality)

**Priority**: MUST complete first - everything depends on these

### Phase 1A: Foundation - Common Infrastructure (8 dirs)
- [x] `nuplan/common/actor_state/` ✅ **Session 1**
- [ ] `nuplan/common/actor_state/test/` (DEFERRED to Phase 1D)
- [x] `nuplan/common/geometry/` ✅ **Session 1**
- [ ] `nuplan/common/geometry/test/` (DEFERRED to Phase 1D)
- [x] `nuplan/common/maps/` ✅ **Session 1**
- [ ] `nuplan/common/maps/test/` (DEFERRED to Phase 1D)
- [x] `nuplan/common/maps/nuplan_map/` ✅ **Session 1**
- [ ] `nuplan/common/maps/nuplan_map/test/` (DEFERRED to Phase 1D)

### Phase 1B: Core Planning Abstractions (7 dirs)
- [x] `nuplan/planning/simulation/planner/` ✅ **Session 1**
- [x] `nuplan/planning/simulation/planner/ml_planner/` ✅ **Session 2**
- [ ] `nuplan/planning/simulation/planner/test/` (DEFERRED to Phase 1D)
- [x] `nuplan/planning/simulation/trajectory/` ✅ **Session 1**
- [ ] `nuplan/planning/simulation/trajectory/test/` (DEFERRED to Phase 1D)
- [x] `nuplan/planning/scenario_builder/` ✅ **Session 2**
- [x] `nuplan/planning/scenario_builder/nuplan_db/` ✅ **Session 2**

### Phase 1C: Supporting Infrastructure (3 dirs)
- [x] `nuplan/common/utils/` ✅ **Session 1**
- [ ] `nuplan/common/utils/test_utils/` (DEFERRED to Phase 1D)
- [x] `nuplan/database/nuplan_db_orm/` ✅ **Session 2**

### Phase 1D: Test Directories (8 dirs - DEFERRED)
- [ ] `nuplan/common/actor_state/test/`
- [ ] `nuplan/common/geometry/test/`
- [ ] `nuplan/common/maps/test/`
- [ ] `nuplan/common/maps/nuplan_map/test/`
- [ ] `nuplan/common/utils/test_utils/`
- [ ] `nuplan/planning/simulation/planner/test/`
- [ ] `nuplan/planning/simulation/trajectory/test/`
- [ ] `nuplan/planning/scenario_builder/test/` (also in Phase 4E)

**TIER 1 SUBTOTAL: 18 directories (10 production + 8 test)**

---

## TIER 2: SIMULATION INFRASTRUCTURE (Deep-dive quality)

**Priority**: Core simulation loop components

### Phase 2A: Observation & Input (6 dirs)
- [x] `nuplan/planning/simulation/observation/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/observation/idm/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/observation/test/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/history/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/history/test/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/simulation_time_controller/` ✅ **Session 3 Batch 1**

### Phase 2B: Control & Motion (10 dirs)
- [ ] `nuplan/planning/simulation/controller/`
- [ ] `nuplan/planning/simulation/controller/motion_model/`
- [ ] `nuplan/planning/simulation/controller/tracker/`
- [ ] `nuplan/planning/simulation/controller/tracker/ilqr/`
- [ ] `nuplan/planning/simulation/controller/tracker/lqr/`
- [ ] `nuplan/planning/simulation/controller/test/`
- [x] `nuplan/planning/simulation/path/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/occupancy_map/` ✅ **Session 3 Batch 1**
- [ ] `nuplan/planning/simulation/predictor/`
- [ ] `nuplan/planning/simulation/predictor/test/`

### Phase 2C: Simulation Execution (8 dirs)
- [ ] `nuplan/planning/simulation/callback/`
- [ ] `nuplan/planning/simulation/callback/test/`
- [ ] `nuplan/planning/simulation/main_callback/`
- [ ] `nuplan/planning/simulation/main_callback/test/`
- [ ] `nuplan/planning/simulation/runner/`
- [ ] `nuplan/planning/simulation/runner/test/`
- [ ] `nuplan/planning/simulation/visualization/`
- [ ] `nuplan/planning/simulation/`

**TIER 2 SUBTOTAL: 24 directories**

---

## TIER 3: TRAINING INFRASTRUCTURE (Deep-dive quality)

**Priority**: ML training pipeline

### Phase 3A: Model Architecture (8 dirs)
- [ ] `nuplan/planning/training/modeling/`
- [ ] `nuplan/planning/training/modeling/models/`
- [ ] `nuplan/planning/training/modeling/models/dynamics_layers/`
- [ ] `nuplan/planning/training/modeling/objectives/`
- [ ] `nuplan/planning/training/modeling/objectives/test/`
- [ ] `nuplan/planning/training/modeling/metrics/`
- [ ] `nuplan/planning/training/modeling/test/`
- [ ] `nuplan/planning/training/modeling/torch_module_wrapper/`

### Phase 3B: Data Pipeline (12 dirs)
- [ ] `nuplan/planning/training/preprocessing/`
- [ ] `nuplan/planning/training/preprocessing/feature_builders/`
- [ ] `nuplan/planning/training/preprocessing/feature_builders/test/`
- [ ] `nuplan/planning/training/preprocessing/features/`
- [ ] `nuplan/planning/training/preprocessing/features/test/`
- [ ] `nuplan/planning/training/preprocessing/target_builders/`
- [ ] `nuplan/planning/training/preprocessing/target_builders/test/`
- [ ] `nuplan/planning/training/preprocessing/utils/`
- [ ] `nuplan/planning/training/data_loader/`
- [ ] `nuplan/planning/training/data_loader/test/`
- [ ] `nuplan/planning/training/data_augmentation/`
- [ ] `nuplan/planning/training/data_augmentation/test/`

### Phase 3C: Training Infrastructure (6 dirs)
- [ ] `nuplan/planning/training/callbacks/`
- [ ] `nuplan/planning/training/callbacks/test/`
- [ ] `nuplan/planning/training/experiments/`
- [ ] `nuplan/planning/training/experiments/test/`
- [ ] `nuplan/planning/training/`
- [ ] `nuplan/planning/training/test/`

**TIER 3 SUBTOTAL: 26 directories**

---

## TIER 4: SUPPORTING MODULES (Overview quality)

**Priority**: Important but not critical path

### Phase 4A: Metrics & Evaluation (15 dirs)
- [ ] `nuplan/planning/metrics/`
- [ ] `nuplan/planning/metrics/evaluation_metrics/`
- [ ] `nuplan/planning/metrics/evaluation_metrics/base/`
- [ ] `nuplan/planning/metrics/evaluation_metrics/common/`
- [ ] `nuplan/planning/metrics/evaluation_metrics/common/test/`
- [ ] `nuplan/planning/metrics/evaluation_metrics/scenario_dependent/`
- [ ] `nuplan/planning/metrics/metric_engine/`
- [ ] `nuplan/planning/metrics/metric_engine/test/`
- [ ] `nuplan/planning/metrics/aggregator/`
- [ ] `nuplan/planning/metrics/aggregator/test/`
- [ ] `nuplan/planning/metrics/utils/`
- [ ] `nuplan/planning/metrics/utils/test/`
- [ ] `nuplan/planning/metrics/test/`
- [ ] `nuplan/planning/metrics/metric_result/`
- [ ] `nuplan/planning/metrics/metric_dataframe/`

### Phase 4B: Visualization (nuBoard) (10 dirs)
- [ ] `nuplan/planning/nuboard/`
- [ ] `nuplan/planning/nuboard/base/`
- [ ] `nuplan/planning/nuboard/base/test/`
- [ ] `nuplan/planning/nuboard/tabs/`
- [ ] `nuplan/planning/nuboard/tabs/test/`
- [ ] `nuplan/planning/nuboard/tabs/config/`
- [ ] `nuplan/planning/nuboard/tabs/js_code/`
- [ ] `nuplan/planning/nuboard/utils/`
- [ ] `nuplan/planning/nuboard/utils/test/`
- [ ] `nuplan/planning/nuboard/test/`

### Phase 4C: Database & Data Access (8 dirs)
- [ ] `nuplan/database/`
- [ ] `nuplan/database/nuplan_db/`
- [ ] `nuplan/database/nuplan_db/test/`
- [ ] `nuplan/database/maps_db/`
- [ ] `nuplan/database/utils/`
- [ ] `nuplan/database/utils/boxes/`
- [ ] `nuplan/database/utils/label/`
- [ ] `nuplan/database/common/blob_store/`

### Phase 4D: CLI & Scripts (4 dirs)
- [ ] `nuplan/cli/`
- [ ] `nuplan/planning/script/`
- [ ] `nuplan/planning/script/builders/`
- [ ] `nuplan/planning/script/utils/`

### Phase 4E: Miscellaneous (6 dirs)
- [ ] `nuplan/planning/scenario_builder/test/`
- [ ] `nuplan/planning/scenario_builder/cache/`
- [ ] `nuplan/submission/`
- [ ] `tutorials/`
- [ ] `tutorials/utils/`
- [ ] `docs/`

**TIER 4 SUBTOTAL: 43 directories**

---

## Top-Level Documentation (2 files)

- [ ] `CLAUDE.md` (project root) - Update with documentation index
- [ ] `nuplan/CLAUDE.md` - Top-level package overview

**TOP-LEVEL SUBTOTAL: 2 files**

---

## TOTAL BACKLOG: 113 items

- **Tier 1 (Critical)**: 18 directories (16%)
- **Tier 2 (Simulation)**: 24 directories (21%)
- **Tier 3 (Training)**: 26 directories (23%)
- **Tier 4 (Supporting)**: 43 directories (38%)
- **Top-Level**: 2 files (2%)

---

## Session Plan

**Estimated Sessions**: 5-8 sessions
**Strategy**: Work tier-by-tier, phase-by-phase
**Parallelization**: 4-6 agents per session
**Commit Frequency**: After each phase completion

### Session 1 ✅ COMPLETE (2025-11-15)
- [x] Phase 1A: Production code (7/8 dirs, skipped tests)
- **Delivered**: actor_state, geometry, maps, nuplan_map, utils, planner, trajectory
- **Deferred**: 5 test directories to Phase 1D
- **Commit**: 92d23ea - "Add comprehensive CLAUDE.md documentation for Phase 1A foundations"

### Session 2 ✅ COMPLETE (2025-11-15)
- [x] Phase 1B + 1C: Production code (4/4 dirs)
- **Delivered**: ml_planner (697 lines), scenario_builder (643 lines), scenario_builder/nuplan_db, database/nuplan_db_orm
- **Quality**: Tier 1 deep-dive (10+ gotchas each, code examples, cross-references)
- **Commit**: 114b906 - "Add Tier 1 documentation for Phase 1B + 1C"

### Session 3 Batch 1 ✅ COMPLETE (2025-11-15)
- [x] Phase 2A (all 6 dirs) + path/occupancy_map from Phase 2B (8 total)
- **Delivered**: observation (856 lines), observation/idm (199), observation/test (440), history (460), history/test (402), simulation_time_controller (481), path (538), occupancy_map (572)
- **Total**: 3,748 lines across 8 CLAUDE.md files
- **Strategy**: 8 parallel technical-writer agents (sonnet model)
- **Quality**: Tier 2 deep-dive (10+ gotchas, code examples, cross-references)
- **Commit**: TBD - "Add Session 3 Batch 1 documentation (Phase 2A + path/occupancy_map)"

### Session 3 Batch 2 (NEXT)
- Target: Phase 2B remaining (6 dirs) + Phase 2C partial
- **Remaining Phase 2B**: controller/, controller/motion_model/, controller/tracker/, controller/tracker/ilqr/, controller/tracker/lqr/, controller/test/, predictor/, predictor/test/ (8 dirs)
- **Phase 2C candidates**: callback/, simulation/ (2 dirs core, defer test dirs)
- **Strategy**: Continue 8x parallel agents with improved output handling

### Session 4
- Target: Phase 2C (Simulation execution) - Tier 2 finish

### Session 5
- Target: Phase 3A + 3B (Training models + data)

### Session 6
- Target: Phase 3C (Training infrastructure) - Tier 3 finish

### Session 7
- Target: Phase 4A + 4B (Metrics + nuBoard)

### Session 8
- Target: Phase 4C + 4D + 4E + Top-level - COMPLETE!

---

## Quality Standards

### Deep-Dive (Tier 1-3)
Sections required:
1. Purpose & Responsibility (2-3 sentences)
2. Key Abstractions & Classes (detailed, with inheritance)
3. Architecture & Design Patterns
4. Dependencies (imports)
5. Dependents (who uses this)
6. Critical Files (prioritized list)
7. Common Usage Patterns (with code examples)
8. Gotchas & Pitfalls (10+ items)
9. Test Coverage Notes (if applicable)
10. Related Documentation (cross-links)
11. AIDEV notes

### Overview (Tier 4)
Sections required:
1. Purpose & Responsibility
2. Key Components (concise list)
3. Dependencies & Dependents
4. Critical Files
5. Usage Patterns (brief)
6. Gotchas (5+ items)
7. Related Documentation

---

**Last Updated**: 2025-11-15 (Session 3 Batch 1 complete)
**Current Progress**: 19/113 directories (16.8%)
**Completed Sessions**:
- Session 1: 7 dirs (Phase 1A)
- Session 2: 4 dirs (Phase 1B+1C)
- Session 3 Batch 1: 8 dirs (Phase 2A + path/occupancy_map)
**Next Session**: Session 3 Batch 2 - Phase 2B remaining (8 dirs)
**Test Dirs**: Deferred to Phase 1D (after all production code documented)
**Milestones**:
- ✅ Tier 1 production code 100% complete! (10/10 dirs)
- ✅ Phase 2A 100% complete! (6/6 dirs)
- ⏳ Phase 2B 20% complete (2/10 dirs: path, occupancy_map)
