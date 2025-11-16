# Active Issues & Tracking

**Status Legend**: ✅ COMPLETE | ⬜ TODO | 🚧 IN PROGRESS | ⬛ BLOCKED | ⚠️ CRITICAL

**Last Updated**: 2025-11-16 (Session VIZ-1)

---

## Quick Status Table

| Issue | Title | Type | Priority | Effort | Status | Session |
|-------|-------|------|----------|--------|--------|---------|
| VIZ-1 | Complete Phase 4 (viz polish) | Feature | Medium | 2-3h | 🚧 50% | VIZ-1/VIZ-2 |
| PERF-1 | Investigate realtime performance bottleneck | Investigation | ⚠️ CRITICAL | 8-12h | ⬜ TODO | VIZ-2+ |
| DOC-1 | Create nuplan/planning/CLAUDE.md | Documentation | High | 3-4h | ⬜ TODO | TBD |
| DOC-2 | Update simulation CLAUDE.md with profiling | Documentation | Medium | 1-2h | ⬜ TODO | After PERF-1 |
| FEAT-1 | Optional replay data saving | Feature | Low | 2-3h | ⬜ DEFERRED | TBD |

**Summary**: 1/5 in progress, 3/5 pending, 1/5 deferred

---

## CRITICAL ISSUES

### ISSUE-PERF-1: Investigate Realtime Performance Bottleneck ⚠️

**Type:** Investigation + Optimization
**Status:** ⬜ TODO (plan ready)
**Priority:** ⚠️ **CRITICAL** (primary project focus)
**Effort:** 8-12 hours (5 phases)
**Dependencies:** Streaming viz working (✅ complete)
**Session:** VIZ-2+ (multi-session)

**Problem:**
An 80ms planner should run at 1.0x realtime (10Hz = 100ms period). Currently runs at 0.51x realtime (195ms total per step). Missing 95ms of overhead.

**Investigation Plan:** `/docs/plans/2025-11-16-REALTIME_PERFORMANCE_INVESTIGATION.md`

**Phases:**
1. ⬜ Detailed Profiling (2-3 hrs) - cProfile + timing probes
2. ⬜ Architecture Deep Dive (3-4 hrs) - Map simulation loop
3. ⬜ Root Cause Analysis (1-2 hrs) - Rank overhead sources
4. ⬜ Optimization Strategy (1-2 hrs) - Quick wins → long-term
5. ⬜ Validation (30-60 min) - Benchmark before/after

**Hypotheses:**
- Callback overhead (metrics/serialization per-step)
- Database I/O (SQLite queries per-step)
- Observation recomputation (IDMAgents)
- History buffer inefficiency (deep copies)
- Planner slower than advertised

**Success Criteria:**
- [ ] Identify root cause of 0.51x realtime
- [ ] Achieve 0.8-1.0x realtime with optimizations
- [ ] Document performance characteristics

**Quote from G Money:**
> "This is the primary aspect of our project and what will drive all future work."

---

## HIGH PRIORITY ISSUES

### ISSUE-DOC-1: Create nuplan/planning/CLAUDE.md

**Type:** Documentation
**Status:** ⬜ TODO
**Priority:** High (needed for PERF-1 Phase 2)
**Effort:** 3-4 hours
**Dependencies:** Performance investigation findings
**Session:** During PERF-1 Phase 2

**Description:**
Create comprehensive top-level planning module documentation covering:
- Package structure (12+ subdirectories)
- Simulation loop architecture
- Key abstractions (planner, controller, observer, history)
- Performance characteristics
- Profiling results (from PERF-1)

**Acceptance Criteria:**
- [ ] Package structure documented
- [ ] Simulation loop call graph created
- [ ] Critical path components identified
- [ ] Performance bottlenecks documented
- [ ] Optimization strategies listed

---

## MEDIUM PRIORITY ISSUES

### ISSUE-VIZ-1: Complete Phase 4 (Integration & Polish)

**Type:** Feature
**Status:** 🚧 IN PROGRESS (50% complete)
**Priority:** Medium
**Effort:** 2-3 hours
**Dependencies:** None
**Session:** VIZ-1 (started), VIZ-2 (finish)

**Description:**
Complete the final phase of streaming visualization system.

**Remaining Tasks:**
- [ ] Performance profiling: Baseline (no streaming) vs streaming
- [ ] Calculate overhead percentage (target < 5%)
- [ ] Edge case: Server crash mid-simulation
- [ ] Edge case: Multiple dashboard clients
- [ ] Edge case: Late-joining client + auto-reconnect
- [ ] User-facing quickstart guide
- [ ] Update design doc with Phase 4 results

**Completed Tasks:**
- [x] Exact message flow proof (302 messages, timing data)
- [x] Proof report created (`/docs/reports/2025-11-16-streaming-viz-proof.md`)

**Acceptance Criteria:**
- [ ] Overhead < 5% vs baseline
- [ ] All edge cases tested and documented
- [ ] Quickstart guide created
- [ ] Design doc updated

**Blocker:** None

---

### ISSUE-DOC-2: Update simulation CLAUDE.md with Profiling Results

**Type:** Documentation
**Status:** ⬜ TODO
**Priority:** Medium
**Effort:** 1-2 hours
**Dependencies:** PERF-1 Phases 1-3 complete
**Session:** After PERF-1 Phase 3

**Description:**
Enhance `nuplan/planning/simulation/CLAUDE.md` with:
- Profiling results (timing breakdown per component)
- Performance characteristics section
- Optimization recommendations
- Known bottlenecks

**Acceptance Criteria:**
- [ ] Timing breakdown table added
- [ ] Performance section created
- [ ] Cross-references to PERF-1 investigation results

---

## LOW PRIORITY / DEFERRED

### ISSUE-FEAT-1: Optional Replay Data Saving

**Type:** Feature
**Status:** ⬜ DEFERRED (nice-to-have)
**Priority:** Low
**Effort:** 2-3 hours
**Dependencies:** None
**Session:** TBD

**Description:**
Add optional feature to save streaming data for offline replay/debugging.

**Rationale for Deferral:**
- Not critical for core functionality
- Can be added later if needed
- Performance investigation is higher priority

**Acceptance Criteria:**
- [ ] Streaming data saved to disk (msgpack/JSON)
- [ ] Replay script created
- [ ] Documentation updated

---

## Completed Issues

### ISSUE-VIZ-0: Build Streaming Visualization System (Phases 1-3)

**Type:** Feature
**Status:** ✅ COMPLETE
**Priority:** High
**Effort:** 6 hours
**Session:** VIZ-1 (2025-11-16)

**Completed:**
- ✅ Phase 1: Design (architecture, component specs)
- ✅ Phase 2: Implementation (server, callback, dashboard)
- ✅ Phase 3: E2E Testing (302 messages transmitted successfully)

**Commit:** `cb6c3a8 Add complete real-time visualization system (Phases 1-3)`

**Key Achievement:**
Fixed architectural mismatch (WebSocket → HTTP POST) and proved E2E functionality with exact timing data.

---

## Issue Templates

### New Issue Template

```markdown
### ISSUE-{PREFIX}-{NUMBER}: {TITLE}

**Type:** {Investigation | Feature | Bug | Documentation | Optimization}
**Status:** {⬜ TODO | 🚧 IN PROGRESS | ✅ COMPLETE | ⬛ BLOCKED | ⬜ DEFERRED}
**Priority:** {⚠️ CRITICAL | High | Medium | Low}
**Effort:** {S (< 1h) | M (1-4h) | L (4-8h) | XL (8-12h) | XXL (12+ h)}
**Dependencies:** {ISSUE-XX, ISSUE-YY, ...}
**Session:** {VIZ-N, PERF-N, DOC-N, ...}

**Description:**
{What needs to be done and why}

**Acceptance Criteria:**
- [ ] {Specific, testable outcome 1}
- [ ] {Specific, testable outcome 2}
- [ ] {Specific, testable outcome 3}

**Blocker:** {None | Description of blocker}
```

---

## Issue Prefixes

- **VIZ-**: Visualization system issues
- **PERF-**: Performance investigation/optimization issues
- **DOC-**: Documentation issues
- **FEAT-**: Feature requests
- **BUG-**: Bug fixes
- **TEST-**: Testing infrastructure

---

**Next Issue ID**: VIZ-2, PERF-2, DOC-3, FEAT-2
