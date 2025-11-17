# Active Issues & Tracking

**Last Updated**: 2025-11-17 (Phase 1 Quick Wins: 2/3 complete)

**Source of Truth**: [GitHub Issues](https://github.com/gmadsen/nuplan-devkit/issues)

---

## Current Sprint Focus

### High Priority (Next Up)
- **[#8 PERF-2c](https://github.com/gmadsen/nuplan-devkit/issues/8)**: Deep Copy Investigation
  - Status: Needs re-investigation (original premise incorrect)
  - Effort: 2-4 hours (find actual deepcopy source)
  - Impact: -20ms/step (if source found and optimized)

### In Progress
- **[#5 VIZ-1](https://github.com/gmadsen/nuplan-devkit/issues/5)**: Complete Phase 4 (Streaming Viz Polish)
  - Status: 50% complete
  - Effort: 2-3 hours remaining

### Backlog (Sequenced)
- **[#3 PERF-3](https://github.com/gmadsen/nuplan-devkit/issues/3)**: Phase 2 Medium-Effort Optimization (-80ms/step)
  - Ready to start (Phase 1 foundation complete)
- **[#4 PERF-4](https://github.com/gmadsen/nuplan-devkit/issues/4)**: Phase 3 Major Refactor (-50ms/step)
  - Blocked by: PERF-3

---

## Recently Completed

### 2025-11-17: Phase 1 Quick Wins (Issues #6, #7)
- **[#6 PERF-2](https://github.com/gmadsen/nuplan-devkit/issues/6)**: Traffic Light Caching ✅
  - **Performance**: 21% faster simulation (61.7s → 51.0s)
  - **Impact**: 50% reduction in DB queries (7,176 → 3,588)
  - **Deliverables**: TrafficLightCache class + 13 unit tests
  - **Commits**: `54cfaec`, `a91244f` (merge)

- **[#7 PERF-2b](https://github.com/gmadsen/nuplan-devkit/issues/7)**: Connection Pooling ✅
  - **Performance**: 97% reduction in connections (1,228 → ~5-15)
  - **Impact**: Eliminate 8 connection creations/step
  - **Deliverables**: QueuePool config + 8 unit tests + docs
  - **Commits**: `b1f7b90`, `efdfc51` (merge)

**Combined Impact**: -71ms/step (78% of Phase 1 target), SimplePlanner 0.44x → ~0.79x realtime

### 2025-11-16: Performance Investigation
- **[PR #2](https://github.com/gmadsen/nuplan-devkit/pull/2)**: Realtime Performance Investigation
  - Root cause identified: Database query explosion (145 queries/step vs 5 expected)
  - Deliverables: 9 architecture docs (6000+ lines), 4 reports, profiling infrastructure
  - Optimization roadmap: 3 phases, -225ms/step potential improvement

---

## Quick Commands

```bash
# View all issues
gh issue list

# View specific issue
gh issue view 6

# Filter by label
gh issue list --label optimization
gh issue list --label high-priority

# Create new issue
gh issue create

# Update issue status
gh issue close 6
gh issue reopen 6
```

---

## Labels

- `optimization` - Performance optimization work
- `visualization` - Streaming visualization system
- `high-priority` - High priority work
- `medium-priority` - Medium priority work
- `investigation` - Research and investigation work
- `documentation` - Documentation improvements

---

**Note for AI Assistants**: For detailed issue tracking, always check GitHub Issues (use `gh issue` commands). This file is a lightweight reference for current sprint context only.
