# Active Issues & Tracking

**Last Updated**: 2025-11-16

**Source of Truth**: [GitHub Issues](https://github.com/gmadsen/nuplan-devkit/issues)

---

## Current Sprint Focus

### High Priority (Next Up)
- **[#6 PERF-2](https://github.com/gmadsen/nuplan-devkit/issues/6)**: Phase 1 Quick Wins Optimization (-95ms/step)
  - Status: Ready to start
  - Effort: 1-2 days
  - Impact: 42% improvement toward realtime

### In Progress
- **[#5 VIZ-1](https://github.com/gmadsen/nuplan-devkit/issues/5)**: Complete Phase 4 (Streaming Viz Polish)
  - Status: 50% complete
  - Effort: 2-3 hours remaining

### Backlog (Sequenced)
- **[#3 PERF-3](https://github.com/gmadsen/nuplan-devkit/issues/3)**: Phase 2 Medium-Effort Optimization (-80ms/step)
  - Blocked by: PERF-2
- **[#4 PERF-4](https://github.com/gmadsen/nuplan-devkit/issues/4)**: Phase 3 Major Refactor (-50ms/step)
  - Blocked by: PERF-3

---

## Recently Completed

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
