# nuPlan CLAUDE.md Documentation Backlog

**Goal**: Create comprehensive CLAUDE.md files for every directory in nuplan-devkit
**Quality Bar**: Deep-dive for Tier 1-2, comprehensive overview for Tier 3-4
**Strategy**: Parallel agents working methodically through backlog

---

## Progress Tracker

**Completed**: 36 / 113 directories (31.9%)
**In Progress**: 0
**Remaining**: 77

**Session 1 Complete**: Phase 1A production code (7 dirs) ✅
**Session 2 Complete**: Phase 1B + 1C production code (4 dirs) ✅
**Session 3 Batch 1 Complete**: Phase 2A + path/occupancy_map (8 dirs) ✅
**Session 4 Complete**: Phase 2B (8 dirs) ✅
**Session 5 Batch 1 Complete**: callback/ + metrics + dynamics_layers (3 dirs + 1 enhancement) ✅
**Session 5 Batch 2 Complete**: Phase 2C complete! (6 dirs) ✅
**Next Session**: Phase 3A + 3B (Training Infrastructure)

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
- [x] `nuplan/planning/simulation/controller/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/controller/motion_model/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/controller/tracker/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/controller/tracker/ilqr/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/controller/tracker/lqr/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/controller/test/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/path/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/occupancy_map/` ✅ **Session 3 Batch 1**
- [x] `nuplan/planning/simulation/predictor/` ✅ **Session 4**
- [x] `nuplan/planning/simulation/predictor/test/` ✅ **Session 4**

### Phase 2C: Simulation Execution (8 dirs) - ✅ COMPLETE
- [x] `nuplan/planning/simulation/callback/` ✅ **Session 5 Batch 1**
- [x] `nuplan/planning/simulation/callback/test/` ✅ **Session 5 Batch 2** (pre-existing)
- [x] `nuplan/planning/simulation/main_callback/` ✅ **Session 5 Batch 2**
- [x] `nuplan/planning/simulation/main_callback/test/` ✅ **Session 5 Batch 2** (pre-existing)
- [x] `nuplan/planning/simulation/runner/` ✅ **Session 5 Batch 2**
- [x] `nuplan/planning/simulation/runner/test/` ✅ **Session 5 Batch 2** (pre-existing)
- [x] `nuplan/planning/simulation/visualization/` ✅ **Session 5 Batch 2**
- [x] `nuplan/planning/simulation/` ✅ **Session 5 Batch 2** (enhanced)

**TIER 2 SUBTOTAL: 24 directories**

---

## TIER 3: TRAINING INFRASTRUCTURE (Deep-dive quality)

**Priority**: ML training pipeline

### Phase 3A: Model Architecture (8 dirs)
- [ ] `nuplan/planning/training/modeling/`
- [x] `nuplan/planning/training/modeling/models/` ✅ **Session 5 Batch 1** (enhanced with Urban Driver section)
- [x] `nuplan/planning/training/modeling/models/dynamics_layers/` ✅ **Session 5 Batch 1**
- [ ] `nuplan/planning/training/modeling/objectives/`
- [ ] `nuplan/planning/training/modeling/objectives/test/`
- [x] `nuplan/planning/training/modeling/metrics/` ✅ **Session 5 Batch 1** (enhanced with 5 additions)
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
- **Commit**: 595f9bd - "Add Session 3 Batch 1 documentation (Phase 2A + path/occupancy_map)"

### Session 4 ✅ COMPLETE (2025-11-15)
- [x] Phase 2B complete (8 dirs: controller stack + predictor)
- **Test batch (3 agents)**: controller/ (887 lines), predictor/ (705), motion_model/ (480)
- **Full batch (5 agents)**: tracker/ (940), tracker/ilqr/ (1034), tracker/lqr/ (1253), controller/test/ (780), predictor/test/ (849)
- **Total**: 6,928 lines across 8 CLAUDE.md files
- **Strategy**: Agents write files directly (return summary only) - 67% token savings validated!
- **Quality**: Tier 2 deep-dive, exceeded standards (up to 25 gotchas per file)
- **Innovation**: Direct file-writing pattern validated for production use
- **Commit**: TBD - "Complete Phase 2B documentation (controller + predictor stack)"

### Session 5 Batch 1 ✅ COMPLETE (2025-11-15)
- [x] Mixed phase completion (3 dirs + 1 enhancement)
- **Delivered**: callback/ (998 lines), dynamics_layers/ (1450 lines), models/CLAUDE.md enhanced (+390 lines Urban Driver), metrics/CLAUDE.md enhanced (+601 lines)
- **Total**: 3,439 new/enhanced lines across 3 new + 1 enhanced CLAUDE.md files
- **Strategy**: 4 parallel technical-writer agents (reduced rate)
- **Quality**: Tier 2/3 deep-dive (14-15 gotchas per file, comprehensive examples)
- **Innovation**: Cross-phase efficiency - knocked out high-priority items from multiple phases
- **Commit**: TBD - "Session 5 Batch 1: callback, dynamics_layers, Urban Driver, enhanced metrics"

### Session 5 Batch 2 ✅ COMPLETE (2025-11-15)
- [x] Phase 2C COMPLETE! (6 new dirs + 3 pre-existing test dirs + 1 enhancement)
- **Delivered**: main_callback/ (1376 lines), runner/ (1041 lines), visualization/ (762 lines), simulation/ enhanced (+185 lines)
- **Pre-existing**: callback/test/ (35780 bytes), main_callback/test/ (38673 bytes), runner/test/ (verified existing)
- **Total**: 3,364 new/enhanced lines across 3 new + 1 enhanced + 3 pre-existing CLAUDE.md files
- **Strategy**: 5 parallel technical-writer agents (reduced rate)
- **Quality**: Tier 2 deep-dive (12-16 gotchas per file)
- **Achievement**: ✅ TIER 2 100% COMPLETE! All 24 simulation infrastructure dirs documented
- **Commit**: TBD - "Session 5 Batch 2: Phase 2C complete - simulation execution stack"

### Session 6
- Target: Phase 3A + 3B (Training models + data)

### Session 7
- Target: Phase 3C (Training infrastructure) - Tier 3 finish

### Session 8
- Target: Phase 4A + 4B (Metrics + nuBoard)

### Session 9+
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

**Last Updated**: 2025-11-15 (Session 5 Batch 2 complete)
**Current Progress**: 36/113 directories (31.9%)
**Completed Sessions**:
- Session 1: 7 dirs (Phase 1A)
- Session 2: 4 dirs (Phase 1B+1C)
- Session 3 Batch 1: 8 dirs (Phase 2A + path/occupancy_map)
- Session 4: 8 dirs (Phase 2B complete)
- Session 5 Batch 1: 3 dirs + 1 enhancement (callback, dynamics_layers, models, metrics)
- Session 5 Batch 2: 6 dirs + 1 enhancement (main_callback, runner, visualization, simulation + 3 pre-existing test dirs)
**Next Session**: Session 6 - Phase 3A + 3B (Training Infrastructure)
**Test Dirs**: Deferred to Phase 1D (after all production code documented)
**Milestones**:
- ✅ Tier 1 production code 100% complete! (10/10 dirs)
- ✅ Phase 2A 100% complete! (6/6 dirs)
- ✅ Phase 2B 100% complete! (10/10 dirs)
- ✅ Phase 2C 100% complete! (8/8 dirs)
- ✅ **TIER 2 100% COMPLETE!** (24/24 dirs) 🎉
- ⏳ Phase 3A 38% complete (3/8 dirs)
