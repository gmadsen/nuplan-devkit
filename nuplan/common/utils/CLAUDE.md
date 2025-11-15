# CLAUDE.md - nuplan/common/utils

## Purpose & Responsibility

**Core utility library providing cross-cutting infrastructure for the entire nuPlan framework.**

This module contains essential utilities used throughout the codebase for:
- **I/O Operations**: Unified file handling for local and S3 storage (async/sync)
- **Cloud Storage**: AWS S3 integration with session management and retry logic
- **Distributed Computing**: Multi-node synchronization and scenario distribution
- **Mathematical Operations**: PyTorch-based numerical utilities (Savitzky-Golay filtering, angle unwrapping)
- **State Representation**: Interpolation interfaces for temporal data
- **Helper Functions**: Retry mechanisms, unique ID generation, incremental tracking

**Design Philosophy**: Location-agnostic I/O (seamlessly handle local paths and S3 URIs), async-first with sync wrappers, and defensive programming with automatic retries for network operations.

---

## Key Abstractions & Classes

### I/O & Storage (`io_utils.py`)

**`NuPath`** - Enhanced `pathlib.Path` that safely handles S3 URI conversions
- Fixes `str(Path("s3://bucket/key"))` → `"s3://bucket/key"` (not `"s3:/bucket/key"`)
- Drop-in replacement for Path when working with mixed local/S3 paths

**Core I/O Functions** (all support local + S3):
- `save_object_as_pickle(path, obj)` / `read_pickle(path)` - Pickle serialization
- `save_text(path, text)` / `read_text(path)` - Text file operations
- `save_buffer(path, bytes)` / `read_binary(path)` - Raw binary I/O
- `path_exists(path, include_directories=True)` - Check existence (local or S3)
- `list_files_in_directory(path)` - List directory contents
- `delete_file(path)` - Remove files
- `safe_path_to_string(path)` - Convert Path/str to properly formatted S3 URI

**Async Variants**: All functions have `*_async` versions for concurrent operations

### S3 Integration (`s3_utils.py`)

**Session Management**:
- `get_async_s3_session()` - Global cached aioboto3 session for async ops
- `_get_sync_session()` - Global cached boto3 session for sync ops
- `get_s3_client()` - Configured S3 client with retry logic
- Supports: AWS SSO, profile-based auth, access key auth, web identity tokens

**S3 Operations** (all with automatic retry on network errors):
- `upload_file_to_s3(local_path, s3_key, bucket)` / `download_file_from_s3(...)`
- `read_binary_file_contents_from_s3(s3_key, bucket)` / `read_text_file_contents_from_s3(...)`
- `delete_file_from_s3(s3_key, bucket)`
- `list_files_in_s3_directory(s3_key, bucket, filter_suffix="")` - Paginated listing
- `check_s3_path_exists(s3_path)` / `check_s3_object_exists(s3_key, bucket)`
- `get_cache_metadata_paths(s3_key, bucket)` - Find metadata CSVs in cache directories

**Path Utilities**:
- `is_s3_path(path)` - Check if path is S3 URI (starts with `s3:/`)
- `split_s3_path(path)` → `(bucket, key)` - Parse S3 URI into components
- `expand_s3_dir(s3_path, filter_suffix="")` - Legacy directory expansion (use `list_files_in_s3_directory`)

**Retry Configuration**:
- Retries on: `ProtocolError`, `SSLError`, `BotoCoreError`, `NoCredentialsError`, `ConnectTimeoutError`
- Retry pattern: 3-7 tries, exponential backoff (1-2s), jitter for S3 path checks

### Distributed Computing (`distributed_scenario_filter.py`)

**`DistributedMode` Enum**:
- `SCENARIO_BASED` - Distribute scenarios evenly across nodes (two-phase: enumerate all, then split)
- `LOG_FILE_BASED` - Distribute log files evenly across nodes (single-phase, faster)
- `SINGLE_NODE` - No distribution (process all scenarios locally)

**`DistributedScenarioFilter`** - Multi-node scenario distribution coordinator
- `__init__(cfg, worker, node_rank, num_nodes, synchronization_path, timeout_seconds=7200, distributed_mode)`
- `get_scenarios()` → `List[AbstractScenario]` - Main entry point, returns node's assigned scenarios

**Workflow**:
1. **LOG_FILE_BASED**: Split DB files → each node processes its chunk → done
2. **SCENARIO_BASED**: Split DB files → enumerate scenarios → sync via barrier → repartition by token → process
3. Uses `FileBackedBarrier` for cross-node synchronization via S3 or shared filesystem

**Dependencies**: Requires `NUM_NODES` and `NODE_RANK` environment variables for multi-node operation

### File-Based Synchronization (`file_backed_barrier.py`)

**`FileBackedBarrier`** - Distributed barrier using filesystem (local or S3)
- `__init__(barrier_directory)` - Create barrier at local or S3 path
- `wait_barrier(activity_id, expected_activity_ids, timeout_s=None, poll_interval_s=1)` - Wait for all activities to complete

**How It Works**:
1. Each process writes a marker file with its `activity_id`
2. Polls directory until all `expected_activity_ids` present
3. Sleep multiplier (20x poll_interval) to ensure all nodes catch up
4. Each process deletes its marker file
5. Poll again until directory empty (all cleaned up)

**Retry Strategy**: S3 operations retry 3x with 0.5s delay, exponential backoff

**Helper**: `distributed_sync(path, timeout_seconds=7200, poll_interval=0.5)` - Convenience wrapper for multi-node sync

### Mathematical Utilities (`torch_math.py`)

**`approximate_derivatives_tensor(y, x, window_length=5, poly_order=2, deriv_order=1)`**
- Approximate dy/dx using Savitzky-Golay filtering
- PyTorch implementation (GPU-compatible)
- **Limitation**: `window_length=3` recommended (others have edge behavior issues)

**`unwrap(angles, dim=-1)`**
- Unwrap angle tensor (like `numpy.unwrap`)
- Changes elements differing by >π to period-complementary values
- Handles multi-dimensional tensors

**`_torch_savgol_filter(y, window_length, poly_order, deriv_order, delta)`**
- Low-level Savitzky-Golay filter (adapted from scipy)
- Uses least-squares polynomial fitting
- **Current restriction**: `window_length=3` only (general solution requires porting `np.polyfit`)

### State Interpolation (`interpolatable_state.py`, `split_state.py`)

**`InterpolatableState`** (ABC) - Interface for temporal state interpolation
- `time_point: TimePoint` - Timestamp for interpolation
- `time_us: int` - Microsecond timestamp
- `to_split_state() → SplitState` - Serialize to interpolatable/fixed components
- `from_split_state(split_state) → InterpolatableState` - Deserialize after interpolation

**`SplitState`** (dataclass) - Decomposed state representation
- `linear_states: List[Any]` - Linearly interpolatable values
- `angular_states: List[float]` - Angular values (2π periodic)
- `fixed_states: List[Any]` - Non-interpolatable constants
- Enables interpolation of complex objects with mixed state types

**Usage Pattern**: Agent states, ego states, and other time-series data implement `InterpolatableState` to enable temporal interpolation while preserving fixed attributes (e.g., agent IDs, vehicle dimensions).

### General Helpers (`helpers.py`)

**`try_n_times(fn, args, kwargs, errors, max_tries, sleep_time=0)`**
- Retry function call up to `max_tries`, catching specified `errors`
- Returns function result or raises last error
- Sleep between attempts

**`keep_trying(fn, args, kwargs, errors, timeout, sleep_time=0.1)`**
- Retry function call until timeout (seconds), catching specified `errors`
- Returns `(result, elapsed_time)` tuple
- Raises `TimeoutError` on timeout

**`get_unique_job_id() → str`**
- SHA256 hash of `NUPLAN_JOB_ID` env var (cluster) or UUID (local)
- **Cached**: Returns same value after first call (use `.cache_clear()` to reset)
- Used for distributed job identification

**`get_unique_incremental_track_id(_: str) → int`**
- Generate monotonically increasing IDs (0, 1, 2, ...)
- **Cached via `@lru_cache`**: Argument ignored but enables cache keying
- Used for assigning unique track IDs to agents

**`static_vars(**kwargs)` decorator**
- Attach static variables to functions (like static vars in C++)
- Usage: `@static_vars(counter=0)` → access via `func.counter`

---

## Architecture & Design Patterns

### 1. **Location-Agnostic I/O Pattern**
All I/O functions transparently handle local paths and S3 URIs:
```python
# Same interface for both
save_pickle(Path("/local/file.pkl"), obj)
save_pickle(Path("s3://bucket/file.pkl"), obj)

# NuPath ensures safe string conversion
path = NuPath("s3://bucket/key")
str(path)  # → "s3://bucket/key" (not "s3:/bucket/key")
```

**Implementation**: Check `is_s3_path()`, then branch to S3 (via boto3) or local (via aiofiles) operations.

### 2. **Async-First with Sync Wrappers**
Core operations are async for performance, with sync wrappers for convenience:
```python
# Async version (parallel operations)
await asyncio.gather(
    read_pickle_async(path1),
    read_pickle_async(path2),
)

# Sync version (blocks until complete)
obj = read_pickle(path)  # Internally: asyncio.run(read_pickle_async(path))
```

### 3. **Global Session Caching**
S3 sessions cached globally to share across forked processes:
```python
G_ASYNC_SESSION = None  # aioboto3.Session
G_SYNC_SESSION = None   # boto3.Session

# Sessions created lazily on first use, reused thereafter
session = get_async_s3_session()  # Returns cached or creates new
```

**Forking-safe**: Global variables persist across `multiprocessing.fork()`.

### 4. **Defensive Retry Pattern**
Network operations wrapped with `@retry` decorator:
```python
@retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
async def download_file_from_s3_async(local_path, s3_key, s3_bucket):
    # Automatically retries on SSL errors, timeouts, credential issues
```

### 5. **Distributed Synchronization via Filesystem**
Multi-node coordination uses marker files (local or S3):
```
barrier_dir/
├── barrier_token_0  # Node 0 finished
├── barrier_token_1  # Node 1 finished
└── barrier_token_2  # Node 2 finished

# All nodes poll until all tokens present → synchronized
```

**Why not Redis/ZooKeeper?** Minimal dependencies, works with existing S3/shared filesystem infrastructure.

### 6. **Split-State Interpolation**
Complex objects with mixed state types decomposed for interpolation:
```python
class AgentState(InterpolatableState):
    def to_split_state(self):
        return SplitState(
            linear_states=[x, y, vx, vy],        # Interpolate linearly
            angular_states=[heading],             # Wrap around 2π
            fixed_states=[agent_id, track_token] # Don't interpolate
        )
```

---

## Dependencies (What We Import)

### External Libraries
- **boto3 / aioboto3**: AWS S3 SDK (sync/async)
- **aiofiles**: Async file I/O
- **asyncio**: Async/await runtime
- **pandas**: CSV handling for token distribution
- **torch**: Mathematical operations (Savitzky-Golay, unwrap)
- **retry**: Retry decorator for network resilience
- **omegaconf**: Hydra config manipulation (in `distributed_scenario_filter`)
- **pathlib**: Path handling
- **tempfile, uuid, hashlib, ssl, urllib3**: Standard library utilities

### Internal nuPlan Dependencies
- `nuplan.common.actor_state.state_representation.TimePoint` (for `InterpolatableState`)
- `nuplan.database.common.blob_store.s3_store.S3Store` (for S3 CSV reading)
- `nuplan.planning.scenario_builder.*` (scenario building, filtering)
- `nuplan.planning.utils.multithreading.worker_pool.WorkerPool` (parallelization)

**Dependency Direction**: `common/utils` is a **foundational layer** - it should not depend on high-level planning modules, but the reverse is occurring with `distributed_scenario_filter.py`. This is a **layering violation** that couples low-level utilities to planning infrastructure.

---

## Dependents (Who Imports Us)

**Ubiquitous usage across the entire codebase**. Key dependents:

### Planning Module (`nuplan/planning/`)
- **Simulation**: `simulation_log.py`, `serialization_callback.py` (save buffers, pickles)
- **Metrics**: `metric_engine.py`, `metric_dataframe.py` (S3 path checks, safe string conversion)
- **Scenario Building**: `scenario_builder/`, `scenario_filter/` (distributed scenario filtering)
- **Training**: Feature caching, checkpoint management (S3 I/O)
- **nuBoard**: `cloud_tab.py`, `nuboard_cloud_utils.py` (S3 client for cloud visualization)

### Database Module (`nuplan/database/`)
- Blob stores, database utilities (S3 operations)

### Common Module (`nuplan/common/`)
- Maps, geometry, actors (helper functions, interpolation)

### Test Infrastructure
- Nearly all test files use utilities for fixtures and mocking

**Usage Patterns**:
1. **Checkpoint Persistence**: Save/load model checkpoints to S3 during training
2. **Feature Caching**: Cache preprocessed features to S3, shared across experiments
3. **Metric Storage**: Save aggregated metrics to S3 for nuBoard visualization
4. **Distributed Training**: Synchronize workers across multi-node training runs
5. **Dataset Access**: Read scenario DBs from S3 in cloud environments

---

## Critical Files (Prioritized)

### Tier 1: Production-Critical Infrastructure

**1. `s3_utils.py` (558 lines)** ⭐⭐⭐
- **Why Critical**: Powers all S3 operations (dataset loading, checkpoints, caching, metrics)
- **Key Functions**: Session management, upload/download with retries, path parsing
- **Gotchas**:
  - Global session caching (can cause auth issues if profile changes mid-run)
  - Retry logic masks transient S3 outages (can lead to long hangs)
  - `expand_s3_dir()` deprecated in favor of `list_files_in_s3_directory_async()`

**2. `io_utils.py` (294 lines)** ⭐⭐⭐
- **Why Critical**: Unified I/O interface used everywhere for local/S3 agnostic operations
- **Key Classes**: `NuPath` (fixes S3 URI string conversion)
- **Gotchas**:
  - Synchronous functions block event loop if called from async context
  - `safe_path_to_string()` essential when passing paths to pandas/other libraries
  - S3 operations use temp files even for pure-memory operations (pickle loads)

**3. `distributed_scenario_filter.py` (274 lines)** ⭐⭐⭐
- **Why Critical**: Enables multi-node simulation/training by partitioning scenarios
- **Modes**: `SCENARIO_BASED` (balanced), `LOG_FILE_BASED` (fast), `SINGLE_NODE`
- **Gotchas**:
  - Requires S3 for multi-node operation (won't work with local paths)
  - `SCENARIO_BASED` mode does 2x scenario enumeration (slow but balanced)
  - Token CSV synchronization can fill S3/disk if not cleaned up
  - **Layering violation**: Imports from `planning` module (should be dependency-free)

### Tier 2: Important Utilities

**4. `file_backed_barrier.py` (231 lines)** ⭐⭐
- **Why Important**: Enables distributed synchronization without external services
- **Use Cases**: Multi-node training sync, cache validation across workers
- **Gotchas**:
  - Sleep multiplier (20x) can cause long waits if nodes have clock skew
  - No automatic cleanup of barrier directories (manual intervention needed)
  - Poll interval too low → S3 rate limiting; too high → slow synchronization

**5. `helpers.py` (119 lines)** ⭐⭐
- **Why Important**: Retry mechanisms used throughout I/O operations
- **Key Functions**: `try_n_times`, `keep_trying`, `get_unique_job_id`, `get_unique_incremental_track_id`
- **Gotchas**:
  - `get_unique_job_id()` caching can cause same ID across separate runs (call `.cache_clear()` explicitly)
  - `get_unique_incremental_track_id()` not thread-safe (uses static variable)

**6. `torch_math.py` (161 lines)** ⭐⭐
- **Why Important**: GPU-compatible numerical operations for trajectory processing
- **Key Functions**: Savitzky-Golay filtering, angle unwrapping
- **Gotchas**:
  - **`window_length=3` restriction** on Savgol filter (edge behavior issues otherwise)
  - `unwrap()` not numerically stable for extreme angle sequences
  - Requires tensors on same device (CPU/GPU) as input

### Tier 3: Interfaces & Small Utilities

**7. `interpolatable_state.py` (53 lines)** ⭐
- Interface for state interpolation (ego/agent states)
- Requires implementation of `to_split_state()` / `from_split_state()`

**8. `split_state.py` (16 lines)** ⭐
- Dataclass for decomposed states (linear/angular/fixed)
- Simple but essential for interpolation framework

---

## Common Usage Patterns

### Pattern 1: S3-Agnostic Checkpoint Saving
```python
from pathlib import Path
from nuplan.common.utils.io_utils import save_object_as_pickle, read_pickle

# Local or S3 - same code!
checkpoint_path = Path(cfg.checkpoint_path)  # Could be "s3://bucket/ckpts/model.pkl"
save_object_as_pickle(checkpoint_path, model.state_dict())

# Later...
state_dict = read_pickle(checkpoint_path)
model.load_state_dict(state_dict)
```

### Pattern 2: Distributed Scenario Processing
```python
from nuplan.common.utils.distributed_scenario_filter import (
    DistributedScenarioFilter, DistributedMode
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

worker = WorkerPool.sequential()  # or .ray_distributed()
node_rank = int(os.environ.get("NODE_RANK", 0))
num_nodes = int(os.environ.get("NUM_NODES", 1))

dist_filter = DistributedScenarioFilter(
    cfg=cfg,
    worker=worker,
    node_rank=node_rank,
    num_nodes=num_nodes,
    synchronization_path="s3://ml-caches/sync/job123",
    distributed_mode=DistributedMode.LOG_FILE_BASED  # Fast mode
)

# Each node gets its assigned scenarios
my_scenarios = dist_filter.get_scenarios()  # Returns subset for this node
```

### Pattern 3: Multi-Node Synchronization
```python
from nuplan.common.utils.file_backed_barrier import distributed_sync

# All nodes wait here until everyone reaches this point
distributed_sync(
    path="s3://ml-caches/barriers/experiment_123",
    timeout_seconds=3600,  # 1 hour timeout
    poll_interval=1.0      # Check every second
)

# Proceed after all nodes synced
print("All nodes ready!")
```

### Pattern 4: Retry with Exponential Backoff
```python
from nuplan.common.utils.helpers import try_n_times, keep_trying

# Retry up to 5 times
result = try_n_times(
    fn=risky_api_call,
    args=[arg1, arg2],
    kwargs={"timeout": 30},
    errors=(ConnectionError, TimeoutError),
    max_tries=5,
    sleep_time=2.0  # 2s between retries
)

# Retry for 60 seconds
result, elapsed = keep_trying(
    fn=check_resource_available,
    args=[resource_id],
    kwargs={},
    errors=(ResourceNotReadyError,),
    timeout=60.0,
    sleep_time=0.5
)
```

### Pattern 5: Async Batch S3 Operations
```python
import asyncio
from nuplan.common.utils.io_utils import save_pickle_async, read_pickle_async

async def save_all_checkpoints(models, checkpoint_dir):
    """Save multiple checkpoints in parallel"""
    tasks = [
        save_pickle_async(checkpoint_dir / f"model_{i}.pkl", model)
        for i, model in enumerate(models)
    ]
    await asyncio.gather(*tasks)

async def load_all_checkpoints(checkpoint_paths):
    """Load multiple checkpoints in parallel"""
    tasks = [read_pickle_async(path) for path in checkpoint_paths]
    return await asyncio.gather(*tasks)

# Usage
asyncio.run(save_all_checkpoints(models, Path("s3://bucket/ckpts")))
```

### Pattern 6: Safe Path Conversion for External Libraries
```python
from pathlib import Path
from nuplan.common.utils.io_utils import safe_path_to_string
import pandas as pd

s3_path = Path("s3://bucket/data/metrics.csv")

# ❌ WRONG - str(Path("s3://...")) → "s3:/..." (missing slash)
df = pd.read_csv(str(s3_path))  # Error: not recognized as S3

# ✅ CORRECT - safe_path_to_string preserves S3 URI
df = pd.read_csv(safe_path_to_string(s3_path))  # Works!
```

### Pattern 7: Angle Unwrapping for Trajectories
```python
import torch
from nuplan.common.utils.torch_math import unwrap

# Heading sequence with wraparound: [359°, 1°] → [359°, 361°]
headings = torch.tensor([6.265, 0.017])  # radians
unwrapped = unwrap(headings)  # [6.265, 6.300] (smooth)

# Works on multi-dimensional tensors
batch_headings = torch.randn(32, 100)  # [batch, time]
unwrapped_batch = unwrap(batch_headings, dim=1)
```

---

## Gotchas & Pitfalls

### S3 Operations

**1. Global Session Caching Can Cause Auth Issues**
```python
# Problem: Session cached with profile A, then profile changes
os.environ["NUPLAN_S3_PROFILE"] = "profile_a"
get_async_s3_session()  # Cached with profile A

os.environ["NUPLAN_S3_PROFILE"] = "profile_b"
get_async_s3_session()  # Still uses profile A (cached!)

# Solution: Force new session
get_async_s3_session(force_new=True)
```

**2. `str(Path("s3://..."))` Breaks S3 URIs**
```python
# ❌ WRONG
path = Path("s3://bucket/key")
pandas.read_csv(str(path))  # → "s3:/bucket/key" (broken!)

# ✅ CORRECT
from nuplan.common.utils.io_utils import safe_path_to_string, NuPath
pandas.read_csv(safe_path_to_string(path))  # → "s3://bucket/key"

# OR use NuPath
path = NuPath("s3://bucket/key")
pandas.read_csv(str(path))  # NuPath.__str__() handles it correctly
```

**3. Retry Logic Can Mask S3 Outages**
```python
# Problem: 3 retries × 7 tries = up to 21 attempts → 60s+ hang
@retry(RETRYABLE_EXCEPTIONS, backoff=2, tries=7, delay=0.5, jitter=(0.5, 3))
async def check_s3_path_exists_async(s3_path):
    # If S3 is down, this blocks for a LONG time
    ...

# Solution: Use shorter timeouts for interactive operations
client = session.client("s3", config=Config(
    connect_timeout=5,  # 5s connection timeout
    read_timeout=10     # 10s read timeout
))
```

**4. S3 Operations Use Temp Files (Memory Overhead)**
```python
# Even pure-memory operations use temp files
await read_binary_file_contents_from_s3_async(s3_key, bucket)
# → Downloads to temp file, reads into memory, deletes temp file

# For large files (GBs), this can:
# 1. Fill /tmp partition
# 2. Slow down due to disk I/O
# 3. Cause permission errors if /tmp is full

# Solution: Set RAY_TMPDIR or TMPDIR to larger partition
export RAY_TMPDIR=/mnt/large_disk/tmp
```

### Distributed Computing

**5. `SCENARIO_BASED` Mode Enumerates Scenarios Twice**
```python
# DistributedMode.SCENARIO_BASED workflow:
# 1. Each node reads its log files → builds scenarios
# 2. All nodes write token CSVs → sync via barrier
# 3. Each node reads ALL token CSVs → repartitions scenarios
# 4. Each node rebuilds scenarios from repartitioned tokens

# For 1M scenarios × 10 nodes → ~10M scenario builds total
# LOG_FILE_BASED skips steps 2-4 (much faster, less balanced)
```

**6. Distributed Sync Requires Manual Cleanup**
```python
# Barrier directories accumulate over time
s3://ml-caches/barriers/
├── job_a1b2c3/  # 2 weeks old
├── job_d4e5f6/  # 1 week old
└── job_g7h8i9/  # Today

# No automatic cleanup! Manual:
aws s3 rm s3://ml-caches/barriers/ --recursive --exclude "*" --include "job_*"
```

**7. Multi-Node Requires S3 or Shared Filesystem**
```python
# ❌ WRONG - Won't work with local paths
DistributedScenarioFilter(
    synchronization_path="/tmp/sync",  # Not visible to other nodes!
    ...
)

# ✅ CORRECT - Use S3 or NFS
DistributedScenarioFilter(
    synchronization_path="s3://ml-caches/sync/job123",
    ...
)
```

### Mathematical Operations

**8. Savitzky-Golay Filter Has Window Size Restriction**
```python
from nuplan.common.utils.torch_math import approximate_derivatives_tensor

# ❌ WRONG - window_length > 3 has edge behavior issues
derivative = approximate_derivatives_tensor(
    y, x, window_length=5  # Unexpected edge values!
)

# ✅ CORRECT - Use window_length=3 (hardcoded edges)
derivative = approximate_derivatives_tensor(
    y, x, window_length=3  # Safe, tested
)

# TODO: Port np.polyfit to remove this restriction (see line 22)
```

**9. `get_unique_incremental_track_id()` Not Thread-Safe**
```python
from nuplan.common.utils.helpers import get_unique_incremental_track_id

# Problem: Static variable incremented without lock
@static_vars(id=-1)
def get_unique_incremental_track_id(_):
    get_unique_incremental_track_id.id += 1  # Race condition!
    return get_unique_incremental_track_id.id

# Solution: Don't call from multiple threads
# OR use thread-local storage / locks (requires refactor)
```

### General I/O

**10. Async Functions Called from Sync Context Block Event Loop**
```python
async def my_async_workflow():
    # ❌ WRONG - Blocks event loop
    result = read_pickle(path)  # Uses asyncio.run() internally!

    # ✅ CORRECT - Use async variants
    result = await read_pickle_async(path)

# Symptom: "RuntimeError: This event loop is already running"
```

**11. `get_unique_job_id()` Caching Persists Across Runs**
```python
from nuplan.common.utils.helpers import get_unique_job_id

job_id_1 = get_unique_job_id()  # Hash of NUPLAN_JOB_ID or UUID
job_id_2 = get_unique_job_id()  # Same value (cached!)

# Problem: If you want unique IDs per experiment within same process
# Solution: Clear cache explicitly
get_unique_job_id.cache_clear()
job_id_new = get_unique_job_id()  # New hash generated
```

---

## Related Documentation

### Internal nuPlan Modules
- **`nuplan/common/CLAUDE.md`** - Overview of common module (actors, maps, geometry)
- **`nuplan/common/actor_state/CLAUDE.md`** - State representations (implements `InterpolatableState`)
- **`nuplan/database/CLAUDE.md`** - Database abstractions and blob stores
- **`nuplan/planning/simulation/CLAUDE.md`** - Simulation loop using these utilities
- **`nuplan/planning/training/CLAUDE.md`** - Training infrastructure (caching, checkpoints)
- **`nuplan/planning/utils/multithreading/CLAUDE.md`** - Worker pools (Ray, sequential)

### External References
- **boto3 Documentation**: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
- **aioboto3 Documentation**: https://aioboto3.readthedocs.io/
- **aiofiles Documentation**: https://github.com/Tinche/aiofiles
- **PyTorch Documentation**: https://pytorch.org/docs/ (for `torch_math` operations)
- **Savitzky-Golay Filter**: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

### AWS Configuration
- **AWS SSO Setup**: https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html
- **AWS Profiles**: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html
- **Web Identity Tokens** (EKS): https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html

---

## AIDEV-NOTES

**AIDEV-NOTE**: `distributed_scenario_filter.py` imports from `nuplan.planning.*` - this is a layering violation! Utilities should not depend on high-level planning modules. Consider refactoring to move scenario distribution logic to `planning/utils/` or creating a dependency-injected interface.

**AIDEV-NOTE**: `get_unique_incremental_track_id()` uses static variable without thread safety. For multi-threaded environments, consider `threading.Lock()` or `multiprocessing.Value()`.

**AIDEV-NOTE**: S3 session caching (global variables) can cause issues when forking processes with different AWS profiles. Monitor `G_ASYNC_SESSION` and `G_SYNC_SESSION` lifecycle carefully in distributed environments.

**AIDEV-TODO**: Port `np.polyfit` to PyTorch to remove `window_length=3` restriction in Savitzky-Golay filter (see `torch_math.py:22-24`).

**AIDEV-TODO**: Deprecate and remove `expand_s3_dir()` in favor of `list_files_in_s3_directory_async()` (warning already present at line 459).

**AIDEV-TODO**: Implement automatic cleanup for barrier directories in distributed sync operations to prevent S3/disk accumulation.
