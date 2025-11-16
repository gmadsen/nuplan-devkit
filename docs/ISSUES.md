# Known Performance Issues

## Scenario Loading Performance

**Symptom**: Initial scenario loading takes 15-30 seconds for small batches (10 scenarios)

**Timeline observation** (from test logs):
```
04:03:24.665 - ScenarioFilter...DONE!
[20 second gap - no logging]
04:03:44.596 - Building metric engines...
```

**Root cause**: Database queries in scenario builder (likely `get_scenarios_from_db()`) are:
- Loading full scenario metadata from SQLite
- Not parallelized
- Possibly running expensive JOINs across tables
- No caching of scenario indices

**Impact**:
- 10 scenarios: ~20 seconds
- 100 scenarios: Estimated ~60-120 seconds
- 1000 scenarios: Estimated ~10-20 minutes
- Makes rapid iteration during development frustrating

**Workarounds**:
1. Use smaller scenario counts for testing (`scenario_filter.num_scenarios_per_type=1`)
2. Cache scenario lists in memory for repeated runs
3. Use pre-built scenario lists from parquet files

**Potential fixes** (not implemented):
1. Add scenario index cache (pickle/msgpack of scenario metadata)
2. Parallelize database queries across scenario types
3. Use lazy loading - only load scenarios as needed, not upfront
4. Pre-compute scenario filter results and save to disk

**Priority**: Medium - annoying but not blocking

**Discovered**: 2025-11-16 during streaming visualization callback testing

---

## Other Known Issues

(Add additional issues as discovered)
