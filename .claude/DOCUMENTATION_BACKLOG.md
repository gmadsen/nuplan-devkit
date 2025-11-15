# nuPlan CLAUDE.md Documentation Backlog

**Goal**: Create comprehensive CLAUDE.md files for every directory in nuplan-devkit
**Quality Bar**: Deep-dive for Tier 1-2, comprehensive overview for Tier 3-4
**Strategy**: Parallel agents working methodically through backlog

---

## Progress Tracker

**Completed**: 3 / 105 directories (2.9%)
**In Progress**: 0
**Remaining**: 102

---

## TIER 1: CRITICAL ABSTRACTIONS (Deep-dive quality)

**Priority**: MUST complete first - everything depends on these

### Phase 1A: Foundation - Common Infrastructure (8 dirs)
- [x] `nuplan/common/actor_state/` ✅ **DONE**
- [ ] `nuplan/common/actor_state/test/` (Major test dir)
- [ ] `nuplan/common/geometry/`
- [ ] `nuplan/common/geometry/test/`
- [ ] `nuplan/common/maps/`
- [ ] `nuplan/common/maps/test/`
- [ ] `nuplan/common/maps/nuplan_map/`
- [ ] `nuplan/common/maps/nuplan_map/test/`

### Phase 1B: Core Planning Abstractions (7 dirs)
- [x] `nuplan/planning/simulation/planner/` ✅ **DONE**
- [ ] `nuplan/planning/simulation/planner/ml_planner/`
- [ ] `nuplan/planning/simulation/planner/test/`
- [x] `nuplan/planning/simulation/trajectory/` ✅ **DONE**
- [ ] `nuplan/planning/simulation/trajectory/test/`
- [ ] `nuplan/planning/scenario_builder/`
- [ ] `nuplan/planning/scenario_builder/nuplan_db/`

### Phase 1C: Supporting Infrastructure (3 dirs)
- [ ] `nuplan/common/utils/`
- [ ] `nuplan/common/utils/test_utils/`
- [ ] `nuplan/database/nuplan_db_orm/`

**TIER 1 SUBTOTAL: 18 directories**

---

## TIER 2: SIMULATION INFRASTRUCTURE (Deep-dive quality)

**Priority**: Core simulation loop components

### Phase 2A: Observation & Input (6 dirs)
- [ ] `nuplan/planning/simulation/observation/`
- [ ] `nuplan/planning/simulation/observation/idm/`
- [ ] `nuplan/planning/simulation/observation/test/`
- [ ] `nuplan/planning/simulation/history/`
- [ ] `nuplan/planning/simulation/history/test/`
- [ ] `nuplan/planning/simulation/simulation_time_controller/`

### Phase 2B: Control & Motion (10 dirs)
- [ ] `nuplan/planning/simulation/controller/`
- [ ] `nuplan/planning/simulation/controller/motion_model/`
- [ ] `nuplan/planning/simulation/controller/tracker/`
- [ ] `nuplan/planning/simulation/controller/tracker/ilqr/`
- [ ] `nuplan/planning/simulation/controller/tracker/lqr/`
- [ ] `nuplan/planning/simulation/controller/test/`
- [ ] `nuplan/planning/simulation/path/`
- [ ] `nuplan/planning/simulation/occupancy_map/`
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

### Session 1 (Current)
- [x] Phase 1A: Start (3/8 complete)
- Target: Complete Phase 1A foundation

### Session 2
- Target: Complete Phase 1B + 1C (Tier 1 finish)

### Session 3
- Target: Phase 2A + 2B (Simulation I/O and control)

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

**Last Updated**: 2025-11-15
**Current Session**: 1
**Next Target**: Complete Phase 1A (5 remaining directories)
