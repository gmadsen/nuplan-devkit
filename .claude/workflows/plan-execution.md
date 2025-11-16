# Plan Execution Workflow

This document defines the standard process for executing multi-step plans in the nuPlan project.

## When to Use This Workflow

Use plan-based execution for:
- **Complex investigations** requiring multiple workstreams (architecture + profiling)
- **Multi-phase implementations** with clear milestones
- **Research tasks** that produce documentation and reports
- **Optimization projects** requiring baseline measurements and validation

**Don't use for**:
- Simple bug fixes (just fix it)
- Single-file changes (use normal git workflow)
- Exploratory work without clear deliverables

## Workflow Steps

### 1. Create Plan Document

**Location**: `docs/plans/{YYYY-MM-DD-plan-name}/`

**Structure**:
```
docs/plans/{plan-name}/
├── README.md or investigation-plan.md (the plan itself)
├── reports/                           (all findings go here)
│   ├── {date}-{specific-finding}.md
│   ├── {date}-executive-summary.md
│   └── ...
└── assets/                            (optional: diagrams, screenshots)
```

**Plan document must include**:
- **Objective**: What are we trying to achieve?
- **Critical Questions**: What specific questions need answers?
- **Phases**: Break work into phases with clear deliverables
- **Success Criteria**: How do we know we're done?
- **Effort Estimate**: How long will this take?

### 2. Create Git Branch

**Naming convention**: `plan/{plan-name}`

```bash
# Create branch from main
git checkout main
git pull
git checkout -b plan/{plan-name}

# Plan work happens on this branch
# Commit frequently with clear messages
```

### 3. Execute Plan (Following the Plan Document)

**As work progresses**:
- Create reports in `docs/plans/{plan-name}/reports/`
- Update plan README with progress
- Commit deliverables incrementally
- Update `.claude/SESSION_*.md` tracking documents

**Use parallel agents when possible**:
- Launch multiple agents for independent workstreams
- Synthesize findings at the end
- Cross-validate results

### 4. Definition of Done

Before considering a plan complete, verify:

- [ ] All phases in plan document completed
- [ ] All success criteria met
- [ ] Reports created for key findings
- [ ] Executive summary written (for stakeholders)
- [ ] Validation performed (measurements, tests, etc.)
- [ ] Lessons learned documented

### 5. Create Pull Request

**When plan is complete**:

```bash
# Stage all plan-related files
git add docs/plans/{plan-name}/
git add {other-deliverables}

# Commit with comprehensive message
git commit -m "Complete: {Plan Name}

- {Key finding 1}
- {Key finding 2}
- {Key deliverable 1}
- {Key deliverable 2}

Definition of Done: [link to checklist in plan README]
"

# Create PR
gh pr create \
  --title "Plan: {Plan Name}" \
  --body "$(cat <<'EOF'
## Plan Summary
{1-2 sentence summary}

## Deliverables
- [ ] {Deliverable 1 with link}
- [ ] {Deliverable 2 with link}
- [ ] {Deliverable 3 with link}

## Key Findings
1. {Finding 1}
2. {Finding 2}
3. {Finding 3}

## Evidence
See reports:
- [Executive Summary](docs/plans/{plan-name}/reports/executive-summary.md)
- [Detailed Analysis](docs/plans/{plan-name}/reports/...)

## Definition of Done
- [x] {Success criterion 1}
- [x] {Success criterion 2}
- [x] {Success criterion 3}

## Impact
{Measured results or expected outcomes}

## Validation
{How findings were validated}

## Next Steps
{What should happen after this PR merges}
EOF
)"
```

### 6. Review and Merge

**Review checklist**:
- [ ] All deliverables present and linked
- [ ] Reports are comprehensive and evidence-based
- [ ] Definition of done met
- [ ] Lessons learned captured
- [ ] Next steps clear

**After approval**:
```bash
# Merge to main
gh pr merge --squash --delete-branch

# Update tracking documents on main
git checkout main
git pull
# Update .claude/SESSION_*.md with plan completion
```

## Plan Directory Structure

### Minimal Structure
```
docs/plans/{plan-name}/
├── README.md                          # Plan document
└── reports/
    └── executive-summary.md           # Key findings
```

### Full Structure
```
docs/plans/{plan-name}/
├── README.md                          # Plan overview
├── investigation-plan.md              # Detailed plan (optional if different from README)
├── reports/
│   ├── executive-summary.md           # High-level findings (always include)
│   ├── {component}-analysis.md        # Detailed analyses
│   ├── {date}-findings.md             # Time-stamped findings
│   └── optimization-roadmap.md        # Action items
├── assets/
│   ├── diagrams/
│   ├── screenshots/
│   └── data/
└── profiling_output/                  # Raw data (if applicable)
```

## Report Templates

### Executive Summary Template
```markdown
# Executive Summary: {Plan Name}

**Date**: {YYYY-MM-DD}
**Status**: {In Progress | Complete}
**Branch**: plan/{plan-name}

## TL;DR
{1 paragraph summary of key findings}

## Key Findings
1. {Finding 1 with evidence}
2. {Finding 2 with evidence}
3. {Finding 3 with evidence}

## Recommendations
{Prioritized action items}

## Next Steps
{What happens next}
```

### Detailed Analysis Template
```markdown
# {Component} Analysis

## Objective
{What we're analyzing}

## Methodology
{How we investigated}

## Findings

### Finding 1: {Title}
**Evidence**: {cProfile data, measurements, etc.}
**Impact**: {Quantified impact}
**Recommendation**: {What to do about it}

### Finding 2: {Title}
...

## Conclusion
{Summary and recommendations}
```

## Best Practices

### Do
✅ **Measure before theorizing** - Always profile/test before making assumptions
✅ **Document as you go** - Create reports during investigation, not after
✅ **Use parallel agents** - Launch multiple agents for independent work
✅ **Validate findings** - Cross-check with actual data/measurements
✅ **Create clear deliverables** - Each phase should produce artifacts
✅ **Commit frequently** - Small commits with clear messages

### Don't
❌ **Don't trust assumptions** - Verify with actual measurements
❌ **Don't skip documentation** - Reports are as important as code
❌ **Don't batch everything at the end** - Progressive commits and documentation
❌ **Don't ignore lessons learned** - Capture what went well and what didn't
❌ **Don't merge without PR** - Always use PR for plan completion

## Example Plans

### Investigation Plan (Current)
- **Plan**: `docs/plans/2025-11-16-realtime-performance/`
- **Branch**: `plan/realtime-performance-investigation`
- **Outcome**: Root cause identified (database queries), optimization roadmap created
- **Deliverables**: 9 architecture docs, 4 reports, profiling infrastructure

### Future Plan Types

**Feature Implementation Plan**:
- Design → Implementation → Testing → Documentation → PR

**Optimization Plan**:
- Baseline measurement → Profiling → Optimization → Validation → PR

**Architecture Refactor Plan**:
- Current state analysis → Design proposal → Incremental migration → Validation → PR

## Lessons Learned (From First Plan)

### What Worked Well
- Parallel agent execution (4 agents simultaneously)
- Comprehensive documentation (6000+ lines)
- Evidence-based findings (no speculation)
- Clear optimization roadmap

### What Could Be Improved
- **Measure first, theorize second**: Don't trust calculations, verify with profiling
- **Listen to user hints**: When G Money said "no way this is realtime," that was a signal to measure
- **Validate assumptions early**: Run quick profiling before deep investigation

### Process Improvements
- Created this workflow template
- Standardized directory structure
- Git branch workflow established
- PR template with evidence and DoD

---

**Created**: 2025-11-16 (Session PERF-1)
**Author**: Navigator 🧭
**Purpose**: Standardize plan-based work for complex investigations and implementations
