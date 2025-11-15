# Future: Generic `/document` Slash Command Design

**Status**: Design phase - gathering learnings from nuPlan documentation project
**Goal**: Reusable documentation generation system for any codebase
**Timeline**: Design after nuPlan Sessions 4-8 complete

---

## Vision: Hierarchical Context-Aware Documentation

### End State Characteristics

1. **Optimized recursive context at each layer**
   - Each CLAUDE.md has full context of its children (subtree)
   - Parents can reference children with confidence (✅ markers)
   - Root CLAUDE.md = architectural overview with navigation to details

2. **Depth-first traversal strategy**
   - Document leaf nodes first (no dependencies)
   - Bubble up to parents with complete child context
   - Root documented last with full project understanding

3. **Highly navigable system**
   - Broad architecture at project root
   - Drill down through directory tree for implementation details
   - Cross-references guide navigation

4. **Slash command interface**
   - `/document [path] [options]` - works in any project
   - Automatic discovery, tiering, and agent orchestration

---

## Key Insight: Breadth-First vs Depth-First

### Current Approach (nuPlan Sessions 1-4): Breadth-First by Phase
```
Session 1: Tier 1 Phase 1A (foundations)
Session 2: Tier 1 Phase 1B+1C (planning abstractions)
Session 3: Tier 2 Phase 2A (observation layer)
Session 4: Tier 2 Phase 2B (control layer)
...
```

**Pros**:
- ✅ Natural parallelization (all same tier/importance)
- ✅ Easier manual planning
- ✅ Validates agent scaling early

**Cons**:
- ❌ Parents documented before children
- ❌ Can't use ✅ cross-references to undocumented modules
- ❌ Requires manual phase categorization

### Better Approach: Depth-First Recursive

```
Start at project root:
1. Discover directory tree
2. Identify leaf directories (no subdirs)
3. Document all leaves in parallel (Batch 1)
4. Document their parents with child context (Batch 2)
5. Recurse up to root
6. Root CLAUDE.md references entire tree
```

**Pros**:
- ✅ Children always documented before parents
- ✅ Parents can reference children with ✅
- ✅ Each layer has complete subtree context
- ✅ Automatic traversal (no manual planning)
- ✅ Root becomes perfect architectural overview

**Cons**:
- ⚠️ Harder to parallelize (must wait for children before parents)
- ⚠️ May document low-priority test dirs before high-priority core
- ⚠️ Circular dependencies need special handling

---

## Learnings from nuPlan Documentation (Sessions 1-4)

### Validated Patterns ✅

1. **Agent file-writing pattern** (Session 4 test batch)
   - Agents write files directly with Write tool
   - Return only summary (~1-2K tokens vs ~10K inline)
   - **Result**: 67% token savings, enables 8x parallelization

2. **Quality standards achievable**
   - Tier 1-2: 10+ gotchas, code examples, cross-refs
   - Agents consistently exceed minimums (e.g., 24 gotchas delivered)
   - Sonnet model = optimal quality/speed trade-off

3. **Parallel execution scales - WITH HARD LIMITS** ⚠️
   - ✅ 8 agents: Proven reliable (Sessions 3-4, 100% success)
   - ❌ 12 agents: Complete failure (all timeouts, Session 4)
   - ❌ 16 agents: Complete failure (all timeouts, Session 4)
   - **DISCOVERED LIMIT**: 8-12 concurrent Task tool invocations maximum
   - **SAFE MAXIMUM**: 8 agents per batch (validated)
   - Token budget: ~50K for 8 agents (vs ~150K old way)

4. **Summaries provide value**
   - 3-5 key highlights capture architectural insights
   - Warnings/TODOs surfaced without reading full docs
   - Cross-reference summaries useful for dependency tracking

### Critical Infrastructure Constraint (Session 4 Discovery)

**Task Tool Concurrency Limit Identified via Binary Search**:

| Test | Agents | Result | Token Usage | Evidence |
|------|--------|--------|-------------|----------|
| Sessions 3-4 | 8 | ✅ Success | ~50-80K | Multiple successful runs |
| Session 4 | 12 | ❌ Timeout | ~2K | All agents failed |
| Session 4 | 16 | ❌ Timeout | ~5K | All agents failed |

**Failure Pattern**:
- Uniform "Request timed out" errors across all agents
- Minimal token consumption (agents never execute)
- No partial successes or graceful degradation
- Infrastructure-level rejection at task scheduler

**Root Cause**: Hard concurrency quota between 8-12 simultaneous Task tool invocations

**Practical Implications for `/document` Command**:
- **Maximum safe batch**: 8 agents
- **Recommended default**: 6-8 agents (with safety margin)
- **Large documentation sets**: Must use sequential batching
- **Example**: 24 files = 3 batches of 8 agents each

### Open Questions ❓

1. **Circular dependencies**
   - How to handle A imports B, B imports A?
   - Document both in same batch?
   - Special "dependency cycle" section?

2. ~~**Optimal batch size for depth-first**~~ ✅ **ANSWERED**
   - ~~Process all children of X in one batch?~~
   - ~~What if X has 20 children (too many)?~~
   - **Answer**: Split into batches of ≤8 directories
   - **Solution**: Sequential batching for large directory sets

3. **Bubbling up insights**
   - How do parent docs incorporate child summaries?
   - Should agents read child CLAUDE.md files?
   - Or provide child summaries in parent agent prompt?

4. **Generic directory structure discovery**
   - Works for non-Python projects (Go, Rust, TypeScript)?
   - How to identify "modules" in non-hierarchical structures?
   - File-based vs directory-based documentation?

5. **Auto-tiering**
   - Can we automatically determine Tier 1 vs Tier 4?
   - Heuristics: Import frequency? LOC? Test coverage?
   - Or require manual tier specification?

6. **Cross-project portability**
   - What context is project-specific vs generic?
   - Template sections that work for any codebase?
   - How to customize quality standards per project?

---

## Proposed Command Interface (Draft)

### Basic Usage
```bash
/document [path] [options]

# Examples:
/document .                          # Document entire project (auto-tier)
/document src/core --tier 1          # Deep-dive on core module
/document lib/ --depth-first         # Depth-first traversal
/document . --parallel 8             # Control parallelism
/document api/ --update              # Update existing docs (delta mode)
```

### Options (Proposed)
- `--tier <1-4>` - Quality level (1=deep-dive, 4=overview)
- `--depth-first` - Bottom-up traversal (default: breadth-first by tier)
- `--parallel <N>` - Max concurrent agents (default: 8)
- `--update` - Update existing CLAUDE.md files (delta mode)
- `--dry-run` - Show execution plan without running
- `--exclude <pattern>` - Skip directories (e.g., `--exclude "test/*"`)

### Expected Behavior

1. **Discovery phase**
   - Scan directory tree
   - Identify modules (directories with code files)
   - Build dependency graph (imports, references)

2. **Planning phase**
   - Determine traversal order (depth-first or tier-based)
   - Batch directories for parallel execution
   - Estimate token budget

3. **Execution phase**
   - Launch technical-writer agents in batches
   - Agents write CLAUDE.md files directly
   - Agents return summaries only

4. **Verification phase**
   - Check all files written
   - Validate line counts, section completeness
   - Generate progress report

5. **Root documentation phase**
   - Create/update project root CLAUDE.md
   - Architectural overview with links to subdirs
   - "Start here" navigation guide

---

## Implementation Strategy

### Phase 1: Extract Patterns from nuPlan (Current)
- Complete nuPlan documentation (Sessions 4-8)
- Document all learnings, edge cases, gotchas
- Measure token efficiency, quality, success rates
- Identify reusable vs project-specific patterns

### Phase 2: Design Generic System (Post-nuPlan)
- Define slash command interface
- Design directory discovery algorithm
- Create agent prompt templates
- Handle depth-first traversal logic
- Implement circular dependency resolution

### Phase 3: Build `/document` Command
- Implement as `.claude/commands/document.md`
- Use findings from Phase 1
- Test on small project first (e.g., dotfiles repo)
- Validate cross-project portability

### Phase 4: Apply to nuPlan Validation
- Re-run on undocumented nuPlan sections
- Compare quality vs manual approach
- Iterate based on results

### Phase 5: Generalize & Share
- Package as reusable command template
- Document best practices
- Contribute to Claude Code community?

---

## Success Metrics (How We'll Know It Works)

### Quality Metrics
- [ ] Docs match quality standards (10+ gotchas, examples, cross-refs)
- [ ] Root CLAUDE.md provides clear architectural overview
- [ ] Navigation works (can find details from root in <3 clicks)
- [ ] Zero broken cross-references

### Efficiency Metrics
- [ ] Token usage < 50K per 8-dir batch
- [ ] Wall-clock time < 15 min per batch
- [ ] Agent success rate > 95%

### Portability Metrics
- [ ] Works on 3+ different projects (Python, TypeScript, Go)
- [ ] Minimal manual intervention required
- [ ] Users can run `/document` without understanding internals

### Maintenance Metrics
- [ ] Can update docs when code changes (delta mode)
- [ ] New directories auto-documented when added
- [ ] Stale docs detected and flagged

---

## Related Documentation

- `.claude/DOCUMENTATION_BACKLOG.md` - nuPlan documentation progress tracker
- `.claude/SESSION4_QUICKSTART.md` - Agent file-writing pattern innovation
- `CLAUDE.md` (project root) - nuPlan project context

---

## Notes & Open Thoughts

### Why Depth-First Matters
Consider this structure:
```
nuplan/planning/simulation/
├── planner/            # Needs to reference controller, trajectory
├── controller/         # Needs to reference motion_model, tracker
│   ├── motion_model/   # Leaf - document first
│   └── tracker/
│       ├── lqr/        # Leaf - document first
│       └── ilqr/       # Leaf - document first
└── trajectory/         # Leaf - document first
```

**Depth-first order**:
1. Document: motion_model/, lqr/, ilqr/, trajectory/ (leaves, parallel)
2. Document: controller/, tracker/ (can reference children with ✅)
3. Document: planner/ (can reference controller + trajectory with ✅)
4. Document: simulation/ (root, references entire subtree)

**Result**: Every CLAUDE.md has complete context of its dependencies!

### Bubbling Up Example
When documenting `controller/CLAUDE.md`, agent could:
1. Read `motion_model/CLAUDE.md` (child 1)
2. Read `tracker/CLAUDE.md` (child 2)
3. Extract key gotchas from both
4. Reference them in parent's "Related Documentation" section
5. Summarize child capabilities in parent's "Key Abstractions" section

This creates a **coherent hierarchy** where each level adds value.

---

**Last Updated**: 2025-11-15
**Next Review**: After nuPlan Session 8 (Tier 2 complete)
**Owner**: Navigator 🧭 + G Money
