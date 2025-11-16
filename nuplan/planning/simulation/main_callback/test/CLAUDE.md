# nuplan/planning/simulation/main_callback/test/

**Tier 2 Documentation - Main Callback Test Suite**

## 1. Purpose & Responsibility

This module provides **comprehensive test coverage for the main simulation orchestration callback system** (`AbstractMainCallback`), validating that high-level callbacks correctly execute at process boundaries (simulation run start/end), integrate with external services (S3, metrics aggregation, validation), and handle file I/O, directory creation, and error conditions. Tests use extensive mocking to isolate callback behavior from AWS infrastructure, verify callbacks execute in correct order, and ensure output artifacts (parquet files, validation markers, S3 uploads) are generated correctly.

**Test Philosophy**: Mock all external dependencies (boto3, file I/O, pandas) except the callback under test. Validate main callbacks integrate correctly with multi-simulation orchestration without running full simulation suites.

**AIDEV-NOTE**: `AbstractMainCallback` has only 2 hooks (start/end) vs `AbstractCallback`'s 8 hooks - different abstraction layer!

---

## 2. Scope & Architecture Context

### Callback Hierarchy in nuPlan

```
┌─────────────────────────────────────────────────────────────────┐
│                  Main Simulation Runner                         │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  on_run_simulation_start() ────────────┐              │     │
│  │                                         │              │     │
│  │  for scenario in scenarios:             │              │     │
│  │    run_simulation(scenario) ◄───────────┼─ Per-scenario│     │
│  │      ├─ on_initialization_start()       │  callbacks   │     │
│  │      ├─ on_planner_start()              │  (8 hooks)   │     │
│  │      ├─ on_step_end()                   │              │     │
│  │      └─ on_simulation_end() ◄───────────┘              │     │
│  │                                                         │     │
│  │  on_run_simulation_end() ───────────────┐              │     │
│  └───────────────────────────────────────────────────────┘     │
│                                             │                   │
│                                             ▼                   │
│                      ┌─────────────────────────────────┐        │
│                      │  MultiMainCallback              │        │
│                      │  ┌─────────────────────────┐    │        │
│                      │  │ MetricFileCallback      │◄───┼─ Aggregate│
│                      │  ├─────────────────────────┤    │   metrics │
│                      │  │ MetricAggregatorCallback│    │          │
│                      │  ├─────────────────────────┤    │          │
│                      │  │ MetricSummaryCallback   │◄───┼─ Generate│
│                      │  ├─────────────────────────┤    │   PDFs   │
│                      │  │ ValidationCallback      │◄───┼─ Pass/Fail│
│                      │  ├─────────────────────────┤    │   marker │
│                      │  │ PublisherCallback       │◄───┼─ S3 upload│
│                      │  ├─────────────────────────┤    │          │
│                      │  │ CompletionCallback      │◄───┼─ Competition│
│                      │  └─────────────────────────┘    │   tracking│
│                      └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Difference**:
- **`AbstractCallback`** (tested in `callback/test/`): Per-scenario hooks (step-by-step)
- **`AbstractMainCallback`** (tested here): Per-run hooks (multi-scenario aggregation)

---

## 3. Test Structure & Organization

### Test Files by Callback Type

**test_metric_file_callback.py** (57 lines)
- **Tests**: `MetricFileCallback` - Aggregates per-scenario metric files into consolidated parquet
- **Coverage**: Path handling, directory creation, timing logs
- **Mocks**: None (uses temp directories, but doesn't mock integration logic)
- **Key test cases**:
  - Constructor initialization (`_metric_file_output_path`, `_scenario_metric_paths`)
  - `on_run_simulation_end()` - Integration timing log validation
  - Directory creation with `exist_ok=True, parents=True`
- **Logger validation**: Checks `logger.info` call with timing format `"00:00:00 [HH:MM:SS]"`
- **Mock strategy**: Minimal mocking (real pathlib, temp directories)

**test_metric_aggregator_callback.py** (56 lines)
- **Tests**: `MetricAggregatorCallback` - Runs metric aggregators on collected scenario metrics
- **Coverage**: Constructor, aggregator execution, warning logs for missing files
- **Mocks**: Logger (to validate warning/info calls)
- **Key test cases**:
  - Constructor assigns `_metric_save_path` and `_metric_aggregators` list
  - `on_run_simulation_end()` orchestration:
    1. Loads metric dataframes from parquet files
    2. Calls `aggregator(metric_dataframes)` for each aggregator
    3. Logs warning if no metric files found
    4. Logs info with timing `"00:00:00 [HH:MM:SS]"`
- **Mock aggregator**: Uses `MockAbstractMetricAggregator` from `nuplan.planning.metrics.aggregator.test`
- **Warning scenario**: Tests that missing metrics trigger `"No metric files found for aggregation!"` warning

**test_metric_summary_callback.py** (135 lines)
- **Tests**: `MetricSummaryCallback` - Renders aggregated metrics into PDF summary reports
- **Coverage**: End-to-end integration (dummy metrics → aggregator → PDF rendering)
- **Mocks**: None (full integration test with real MetricsEngine, WeightedAverageMetricAggregator)
- **Key test cases**:
  - **Setup phase**: Creates dummy metric files (ego_acceleration statistics)
  - **Aggregation phase**: Runs `WeightedAverageMetricAggregator`
  - **Rendering phase**: `on_run_simulation_end()` generates summary PDF
  - **Validation**: Checks exactly 1 PDF file exists in output directory
- **Metric generation**:
  - Creates `MetricStatistics` with MAX/MIN/P90 statistics + time series
  - Writes to parquet via `MetricsEngine.write_to_files()`
  - Integrates with `MetricFileCallback` (aggregates scenario files)
- **Complexity**: Most complex test in suite (135 lines, full metric pipeline)

**test_validation_callback.py** (58 lines)
- **Tests**: `ValidationCallback` - Creates pass/fail marker files based on simulation success
- **Coverage**: Runner report parsing, file creation (passed.txt/failed.txt)
- **Mocks**: `_validation_succeeded()` helper function, pandas read_parquet
- **Key test cases**:
  - Constructor initialization (`output_dir`, `_validation_dir_name`)
  - `on_run_simulation_end()` with failure → creates `failed.txt`
  - `on_run_simulation_end()` with success → creates `passed.txt`
  - Helper `_validation_succeeded()` logic:
    - Missing runner_report.parquet → False
    - `succeeded` column has False → False
    - All `succeeded` values True → True
- **File structure**: `output_dir/validation_dir_name/{passed|failed}.txt`
- **Mock strategy**: Patches `_validation_succeeded` to return True/False

**test_publisher_callback.py** (72 lines)
- **Tests**: `PublisherCallback` - Uploads simulation artifacts to S3
- **Coverage**: S3 client interaction, file listing, upload orchestration
- **Mocks**: `boto3`, `list_files()`, `pathlib` (directory iteration)
- **Key test cases**:
  - Constructor builds `_upload_targets` list from config dict
  - `on_run_simulation_end()` calls `s3_client.upload_file()` for each target file
  - Upload config structure: `{"metrics": {"upload": True, "save_path": "...", "remote_path": "..."}}`
  - File discovery via `list_files()` (recursive directory traversal)
  - S3 key construction: `remote_prefix / remote_path / file_path`
- **Validation strategy**: `assert_has_calls()` to verify correct upload parameters
- **Mock complexity**: Patches both `list_files` and `pathlib` for iterator behavior

**test_completion_callback.py** (44 lines)
- **Tests**: `CompletionCallback` - Creates completion markers for competition tracking
- **Coverage**: Environment variable parsing, S3 bucket extraction, completion file creation
- **Mocks**: `os.environ` via `@patch.dict`
- **Key test cases**:
  - Constructor reads `NUPLAN_SERVER_S3_ROOT_URL` and `SCENARIO_FILTER_ID` env vars
  - Bucket extraction: `"my-bucket"` from env var
  - Completion directory: `output_dir/simulation-results/{challenge_name}_{filter_id}/`
  - `on_run_simulation_end()` creates `completed.txt` file
  - Assertion on missing `NUPLAN_SERVER_S3_ROOT_URL` → raises `AssertionError`
- **Environment setup**: Uses `@patch.dict(os.environ, {...})` in `setUp()`
- **Special validation**: Tests that missing env var causes initialization failure

---

## 4. Key Testing Patterns & Strategies

### Pattern 1: Minimal Hook Validation (2 Hooks Only)
Unlike per-scenario callbacks (8 hooks), main callbacks have only 2 hooks:
1. `on_run_simulation_start()` - Called once before all scenarios
2. `on_run_simulation_end()` - Called once after all scenarios

**Example** (from test_metric_file_callback.py):
```python
def test_on_run_simulation_end(self, logger: MagicMock) -> None:
    metric_file_callback = MetricFileCallback(
        metric_file_output_path=self.tmp_dir.name,
        scenario_metric_paths=[self.tmp_dir.name]
    )
    metric_file_callback.on_run_simulation_end()

    # Validate logger called with timing format
    logger.info.assert_has_calls([call('Metric files integration: 00:00:00 [HH:MM:SS]')])
```

**AIDEV-NOTE**: Most tests only validate `on_run_simulation_end()` since `on_run_simulation_start()` is no-op for most callbacks

### Pattern 2: Temporary Directory Management
All file I/O tests use `tempfile.TemporaryDirectory()` with explicit cleanup:

```python
def setUp(self) -> None:
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp_dir.cleanup)  # Ensures cleanup even on failure

def test_something(self):
    path = Path(self.tmp_dir.name) / "output"
    # Use path for test

# tearDown() not needed - addCleanup handles it
```

**Why `addCleanup()` instead of `tearDown()`?**
- Runs even if test fails (exception during test)
- Runs even if `tearDown()` is skipped
- Can register multiple cleanup functions

### Pattern 3: Mock Logger for Timing Validation
Callbacks log execution time in `HH:MM:SS` format - tests validate format:

```python
@patch('nuplan.planning.simulation.main_callback.metric_file_callback.logger')
def test_on_run_simulation_end(self, logger: MagicMock) -> None:
    callback.on_run_simulation_end()

    # Validate timing log (format, not exact value)
    logger.info.assert_has_calls([
        call('Metric files integration: 00:00:00 [HH:MM:SS]')
    ])
```

**AIDEV-NOTE**: Uses `time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))` - tests validate format, not duration

### Pattern 4: Integration Testing for Complex Pipelines
`test_metric_summary_callback.py` uses full integration (no mocks):

```python
def set_up_dummy_metric(self, metric_path, log_name, planner_name, scenario_type, scenario_name):
    # 1. Create MetricStatistics objects
    statistics = [Statistic(...), Statistic(...)]
    result = MetricStatistics(...)

    # 2. Write to parquet via MetricsEngine
    metric_engine = MetricsEngine(main_save_path=metric_path)
    metric_engine.write_to_files(metric_files={...})

    # 3. Aggregate with MetricFileCallback
    metric_file_callback = MetricFileCallback(...)
    metric_file_callback.on_run_simulation_end()

def test_metric_summary_callback_on_simulation_end(self):
    # 4. Run aggregator
    self.weighted_average_metric_aggregator(metric_dataframes=...)

    # 5. Render summary
    self.metric_summary_callback.on_run_simulation_end()

    # 6. Validate PDF exists
    pdf_files = self.metric_summary_output_path.rglob("*.pdf")
    self.assertEqual(len(list(pdf_files)), 1)
```

**Why full integration?**
- MetricSummaryCallback depends on real MetricsEngine output format
- Aggregator behavior is critical (mocking would hide bugs)
- PDF rendering is complex (easier to test end-to-end)

### Pattern 5: Environment Variable Mocking
`test_completion_callback.py` uses `@patch.dict` for env vars:

```python
@patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": "my-bucket"})
@patch.dict(os.environ, {"SCENARIO_FILTER_ID": "1"})
def setUp(self) -> None:
    self.callback = CompletionCallback(output_dir=self.tmp_dir.name, challenge_name='challenge')

def test_initialization(self) -> None:
    self.assertEqual(str(self.callback._bucket), 'my-bucket')
    self.assertEqual(
        str(self.callback._completion_dir),
        '/'.join([self.tmp_dir.name, 'simulation-results/challenge_1'])
    )
```

**AIDEV-NOTE**: `@patch.dict` is decorator AND context manager - can use both ways

### Pattern 6: S3 Client Mocking
`test_publisher_callback.py` mocks boto3 and list_files:

```python
@patch('nuplan.planning.simulation.main_callback.publisher_callback.pathlib')
@patch('nuplan.planning.simulation.main_callback.publisher_callback.list_files')
def test_on_run_simulation_end_push_to_s3(self, mock_files: Mock, mock_pathlib: Mock) -> None:
    # Mock file discovery
    mock_files.return_value = ["a", "b"]

    # Mock path iteration
    fake_path = Mock()
    fake_path.iterdir.return_value = [True]
    mock_pathlib.Path.return_value = fake_path

    self.publisher_callback.on_run_simulation_end()

    # Validate S3 uploads
    expected_calls = [
        call('some/path/to/save/a', 'bucket', 'user/image/path/save/a'),
        call('some/path/to/save/b', 'bucket', 'user/image/path/save/b'),
        # ... more calls
    ]
    self.mock_client.upload_file.assert_has_calls(expected_calls)
```

**Complexity**: Must mock both file discovery AND pathlib iteration

---

## 5. Dependencies & Mocks

### External Dependencies
- **unittest** - Test framework (TestCase, Mock, MagicMock, patch, call)
- **tempfile** - Temporary directories for file I/O tests
- **pathlib** - Path manipulation and validation
- **pandas** - Parquet file I/O, DataFrame aggregation
- **boto3** - AWS S3 client (mocked)
- **os** - Environment variable access

### Internal nuPlan Imports

**Main callback implementations (under test)**:
- `nuplan.planning.simulation.main_callback.metric_file_callback.MetricFileCallback`
- `nuplan.planning.simulation.main_callback.metric_aggregator_callback.MetricAggregatorCallback`
- `nuplan.planning.simulation.main_callback.metric_summary_callback.MetricSummaryCallback`
- `nuplan.planning.simulation.main_callback.validation_callback.ValidationCallback`
- `nuplan.planning.simulation.main_callback.publisher_callback.PublisherCallback`
- `nuplan.planning.simulation.main_callback.completion_callback.CompletionCallback`

**Supporting infrastructure**:
- `nuplan.planning.simulation.main_callback.abstract_main_callback.AbstractMainCallback`
- `nuplan.planning.metrics.metric_engine.MetricsEngine` - Metric computation and file writing
- `nuplan.planning.metrics.metric_dataframe.MetricStatisticsDataFrame` - Parquet serialization
- `nuplan.planning.metrics.metric_file.MetricFile, MetricFileKey` - Metric file structure
- `nuplan.planning.metrics.metric_result.MetricStatistics, Statistic, TimeSeries` - Metric data
- `nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator.WeightedAverageMetricAggregator`

**Test utilities**:
- `nuplan.planning.metrics.aggregator.test.mock_abstract_metric_aggregator.MockAbstractMetricAggregator`

**Mocked (never imported directly in tests)**:
- `boto3.client` - S3 client
- `pandas.read_parquet` - Parquet file reading
- Module-level loggers (`nuplan.planning.simulation.main_callback.*.logger`)

---

## 6. Testing Best Practices Demonstrated

### 1. Separation of Test Concerns
Each test file tests exactly one callback, with focused test cases:

```python
class TestMetricFileCallback(TestCase):
    def test_metric_callback_init(self):
        # Only tests constructor

    def test_on_run_simulation_end(self):
        # Only tests end hook
```

### 2. Consistent Naming Convention
- **Class names**: `TestXxxCallback` (matches callback name)
- **Test methods**: `test_<method_name>` or `test_<behavior>`
- **Variables**: Descriptive (`metric_file_callback` not `mfc`)

### 3. Use of `addCleanup()` for Robust Teardown
```python
def setUp(self) -> None:
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp_dir.cleanup)  # ALWAYS runs
```

**Better than `tearDown()`** because:
- Runs even if test raises exception
- Runs in reverse order (LIFO stack)
- Can register multiple cleanup functions

### 4. Minimal Mocking Principle
Test only mocks what's necessary:
- **test_metric_file_callback.py**: Only logger (real pathlib, pandas)
- **test_publisher_callback.py**: S3 client + file listing (must avoid real AWS)
- **test_metric_summary_callback.py**: NO mocks (full integration)

**AIDEV-NOTE**: Over-mocking makes tests brittle (break on refactor) - mock only external I/O

### 5. Arrange-Act-Assert Clarity
```python
def test_on_run_simulation_end(self, logger: MagicMock) -> None:
    # Arrange
    metric_file_callback = MetricFileCallback(
        metric_file_output_path=self.tmp_dir.name,
        scenario_metric_paths=[self.tmp_dir.name]
    )

    # Act
    metric_file_callback.on_run_simulation_end()

    # Assert
    logger.info.assert_has_calls([call('Metric files integration: 00:00:00 [HH:MM:SS]')])
```

### 6. Test Documentation
Every test has docstring explaining:
- What is being tested
- Expected behavior
- Special conditions (if any)

```python
def test_initialization(self) -> None:
    """Tests that the object is constructed correctly"""
    # ...
```

---

## 7. Key Abstractions Under Test

### AbstractMainCallback Lifecycle
Only 2 hooks (vs AbstractCallback's 8):

```python
class AbstractMainCallback(abc.ABC):
    def on_run_simulation_start(self) -> None:
        """Callback after the simulation function starts."""
        pass

    def on_run_simulation_end(self) -> None:
        """Callback before the simulation function ends."""
        pass
```

**Design rationale**:
- `on_run_simulation_start()`: Setup that spans all scenarios (e.g., open log files)
- `on_run_simulation_end()`: Aggregation across scenarios (e.g., merge parquet files)

**AIDEV-NOTE**: No scenario-specific hooks - use `AbstractCallback` for per-scenario logic

### MultiMainCallback Composition Pattern
```python
class MultiMainCallback(AbstractMainCallback):
    def __init__(self, main_callbacks: List[AbstractMainCallback]):
        self._main_callbacks = main_callbacks

    def on_run_simulation_end(self) -> None:
        for main_callback in self._main_callbacks:
            main_callback.on_run_simulation_end()
```

**Properties**:
- Sequential execution (order matters!)
- No exception isolation (one callback failure crashes all)
- Supports `len()` to count callbacks

**Typical ordering**:
1. `MetricFileCallback` - Aggregate scenario metrics
2. `MetricAggregatorCallback` - Run aggregators on collected metrics
3. `MetricSummaryCallback` - Render aggregated metrics to PDF
4. `ValidationCallback` - Create pass/fail marker
5. `PublisherCallback` - Upload to S3
6. `CompletionCallback` - Mark competition completion

---

## 8. Running the Tests

### Run All Main Callback Tests
```bash
# From repository root
uv run pytest nuplan/planning/simulation/main_callback/test/

# With coverage
uv run pytest --cov=nuplan.planning.simulation.main_callback \
              nuplan/planning/simulation/main_callback/test/

# Verbose output
uv run pytest -v nuplan/planning/simulation/main_callback/test/
```

### Run Specific Test File
```bash
# Metric file callback tests
uv run pytest nuplan/planning/simulation/main_callback/test/test_metric_file_callback.py

# Publisher callback tests
uv run pytest nuplan/planning/simulation/main_callback/test/test_publisher_callback.py

# Integration test (metric summary)
uv run pytest nuplan/planning/simulation/main_callback/test/test_metric_summary_callback.py
```

### Run Specific Test Case
```bash
# Single test method
uv run pytest nuplan/planning/simulation/main_callback/test/test_validation_callback.py::TestValidationCallback::test_on_run_simulation_end

# All tests matching pattern
uv run pytest -k "publisher" nuplan/planning/simulation/main_callback/test/
```

### Debug Test Failures
```bash
# Show print statements and full diffs
uv run pytest -vv -s nuplan/planning/simulation/main_callback/test/test_metric_file_callback.py

# Stop on first failure
uv run pytest -x nuplan/planning/simulation/main_callback/test/

# Enter debugger on failure
uv run pytest --pdb nuplan/planning/simulation/main_callback/test/
```

---

## 9. Common Test Failures & Debugging

### Failure: Logger Assertions Fail
**Symptom**: `AssertionError: Calls not found` in logger.info.assert_has_calls()
**Cause**: Logger not patched at correct module path
**Fix**:
```python
# ❌ Wrong - patches root logger
@patch('logging.logger')

# ✅ Correct - patches callback's module logger
@patch('nuplan.planning.simulation.main_callback.metric_file_callback.logger')
def test_logging(self, logger: MagicMock):
    logger.info.assert_called_with("Expected message")
```

### Failure: Temporary Directory Not Cleaned
**Symptom**: `/tmp` fills up with `pytest-*` directories
**Cause**: Test crashed before `cleanup()` called
**Fix**: Use `addCleanup()` instead of manual `tearDown()`:
```python
def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp_dir.cleanup)  # ALWAYS runs
```

### Failure: Parquet File Not Found
**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: '.../metrics.parquet'`
**Cause**: Test doesn't create metric files before callback runs
**Fix**: Ensure metric files created in setup:
```python
def setUp(self):
    # Create dummy metric file
    self.set_up_dummy_metric(metric_path, ...)
```

### Failure: Mock S3 Client Not Called
**Symptom**: `AssertionError: Expected 'upload_file' to be called`
**Cause**: `list_files()` returns empty list (no files to upload)
**Fix**: Mock `list_files` to return test files:
```python
@patch('nuplan.planning.simulation.main_callback.publisher_callback.list_files')
def test_upload(self, mock_files: Mock):
    mock_files.return_value = ["file1.txt", "file2.txt"]
    # Now upload_file will be called
```

### Failure: Environment Variable Not Found
**Symptom**: `KeyError: 'NUPLAN_SERVER_S3_ROOT_URL'`
**Cause**: Test doesn't mock environment variables
**Fix**: Use `@patch.dict` decorator:
```python
@patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": "s3://bucket"})
def test_completion(self):
    # Env var is mocked
```

### Failure: PDF Not Generated
**Symptom**: `AssertionError: 1 != 0` (expected 1 PDF, got 0)
**Cause**: Metric aggregation failed silently
**Fix**: Check aggregator warnings:
```python
# In test_metric_summary_callback.py
logger.warning.assert_has_calls([
    call('dummy_metric_aggregator: No metric files found for aggregation!')
])
```

---

## 10. Test Coverage Gaps & Future Work

### Current Coverage Strengths
- ✅ Constructor initialization for all callbacks
- ✅ `on_run_simulation_end()` hook execution
- ✅ File I/O (directory creation, file writing)
- ✅ Logger validation (timing logs)
- ✅ S3 upload orchestration (mocked)
- ✅ Environment variable handling
- ✅ Integration test for metric pipeline (full stack)

### Coverage Gaps (Not Tested)
- ⚠️ **`on_run_simulation_start()` hook** - Most tests only validate end hook
- ⚠️ **MultiMainCallback** - No tests for callback composition
- ⚠️ **Error handling** - No tests for callback exceptions during execution
- ⚠️ **S3 upload failures** - PublisherCallback doesn't test network errors
- ⚠️ **Parquet corruption** - No tests for malformed metric files
- ⚠️ **Race conditions** - No tests for concurrent file access (parallel simulations)
- ⚠️ **Disk space exhaustion** - No tests for I/O failures
- ⚠️ **Large file handling** - No tests for 10GB+ metric files
- ⚠️ **Partial aggregation** - No tests for missing scenarios in aggregation
- ⚠️ **PDF rendering failures** - MetricSummaryCallback doesn't test matplotlib errors

### Future Test Additions (Recommended)
1. **Add MultiMainCallback tests** - Validate callback ordering, exception isolation
2. **Add error injection tests** - Simulate I/O failures, S3 errors, corrupted files
3. **Add `on_run_simulation_start()` tests** - Validate start hook execution
4. **Add stress tests** - Large metric files, many scenarios, concurrent access
5. **Add integration tests** - Test callbacks in actual multi-simulation runner

---

## 11. Critical Gotchas & Warnings

### Gotcha 1: Logger Patch Path Must Match Import
**Problem**: Logger patched at wrong module path doesn't intercept callback logs
**Impact**: Tests pass but don't validate logging behavior
**Solution**: Patch at callback module, not root logger
```python
# ❌ Bad - doesn't intercept callback logs
@patch('logging.logger')

# ✅ Good - patches callback's module logger
@patch('nuplan.planning.simulation.main_callback.metric_file_callback.logger')
def test_logging(self, logger: MagicMock):
    callback.on_run_simulation_end()
    logger.info.assert_called()  # Now intercepts logs
```

### Gotcha 2: `addCleanup()` Executes in LIFO Order
**Problem**: Multiple cleanup functions run in reverse registration order
**Impact**: Dependencies cleaned before dependents (crashes)
**Solution**: Register cleanups in reverse dependency order
```python
def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.file_handle = open(f'{self.tmp_dir.name}/data.txt', 'w')

    # ❌ Wrong order - file handle cleaned first
    self.addCleanup(self.tmp_dir.cleanup)
    self.addCleanup(self.file_handle.close)

    # ✅ Correct - file closed before directory deleted
    self.addCleanup(self.file_handle.close)  # Runs FIRST (LIFO)
    self.addCleanup(self.tmp_dir.cleanup)    # Runs SECOND
```

### Gotcha 3: MetricFileCallback Mutates Input Directories
**Problem**: `delete_scenario_metric_files=True` deletes source files during test
**Impact**: Cannot reuse test directories across tests
**Solution**: Use separate temp directories per test
```python
def setUp(self):
    self.tmp_dir = tempfile.TemporaryDirectory()  # Fresh dir per test
    # Don't share directories between tests!
```

### Gotcha 4: PublisherCallback Requires Mock Iterator Setup
**Problem**: Mocking `pathlib.Path.iterdir()` requires return value to be iterable
**Impact**: Test crashes with `TypeError: 'Mock' object is not iterable`
**Solution**: Return list, not bare Mock
```python
# ❌ Wrong - Mock is not iterable
fake_path = Mock()
fake_path.iterdir.return_value = Mock()

# ✅ Correct - return list
fake_path = Mock()
fake_path.iterdir.return_value = [Mock(), Mock()]
```

### Gotcha 5: Environment Variable Mocking Scope
**Problem**: `@patch.dict(os.environ)` only applies during test method execution
**Impact**: Environment changes don't persist to other tests (good!) or setup (bad!)
**Solution**: Apply decorator to `setUp()` if needed during initialization
```python
# ✅ Environment mocked during setUp AND test
@patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": "bucket"})
def setUp(self):
    self.callback = CompletionCallback(...)  # Uses env var

# Test method inherits decorator from setUp
def test_something(self):
    # Env var still mocked here
```

### Gotcha 6: Pandas Parquet Write Requires String Path
**Problem**: `pandas.to_parquet(Path)` may fail with pathlib.Path on some versions
**Impact**: Tests fail with `TypeError: expected str, bytes or os.PathLike`
**Solution**: Convert Path to str explicitly
```python
# ❌ May fail on older pandas
concat_pandas.to_parquet(save_path)

# ✅ Always works
concat_pandas.to_parquet(str(save_path))
```

### Gotcha 7: MetricSummaryCallback Requires Specific Directory Structure
**Problem**: Callback expects metrics in `metric_save_path/*.parquet`
**Impact**: Test fails silently if files in wrong location
**Solution**: Match exact directory structure in test setup
```python
def set_up_dummy_metric(self, metric_path, ...):
    # ✅ Correct structure
    metric_engine = MetricsEngine(main_save_path=metric_path)
    # Writes to: metric_path/ego_acceleration.parquet

    # Pass same path to callback
    callback = MetricSummaryCallback(metric_save_path=str(metric_path))
```

### Gotcha 8: ValidationCallback Silent Failure on Missing Parquet
**Problem**: `_validation_succeeded()` returns False instead of raising on missing file
**Impact**: Test passes when it should fail (no validation happened)
**Solution**: Check logger warnings when validation fails
```python
def test_missing_report(self):
    # Expect FileNotFoundError → caught internally → logs warning
    with patch('nuplan.planning.simulation.main_callback.validation_callback._validation_succeeded') as mock_val:
        mock_val.side_effect = FileNotFoundError()
        # Should log warning and return False
```

### Gotcha 9: S3 Upload Mock Doesn't Validate File Contents
**Problem**: Tests only validate `upload_file()` called, not that file exists
**Impact**: Test passes even if local file missing
**Solution**: Add file existence checks before mocking upload
```python
def test_upload(self):
    # Create actual file
    (Path(self.tmp_dir.name) / "data.txt").write_text("test")

    # Then mock S3 upload
    with patch('boto3.client') as mock_s3:
        callback.on_run_simulation_end()
        # Now we know file existed when upload called
```

### Gotcha 10: Timing Logs Use GMT Time Format
**Problem**: Tests expect timing in `HH:MM:SS` but test runs at exactly 1ms
**Impact**: Test expects `"00:00:00"` but gets `"00:00:00.001"`
**Solution**: Match partial string or use regex
```python
# ❌ Fails if timing not exactly 0
logger.info.assert_called_with('Metric files integration: 00:00:00 [HH:MM:SS]')

# ✅ Matches any HH:MM:SS format
import re
call_args = logger.info.call_args[0][0]
assert re.match(r'Metric files integration: \d{2}:\d{2}:\d{2} \[HH:MM:SS\]', call_args)
```

### Gotcha 11: Metric Aggregator Warnings are Expected
**Problem**: `MockAbstractMetricAggregator` intentionally logs warning about missing files
**Impact**: Tests may fail if not expecting warning
**Solution**: Assert warning is present (not absent)
```python
def test_on_run_simulation_end(self, logger: MagicMock) -> None:
    callback.on_run_simulation_end()

    # ✅ Expect warning for missing files
    logger.warning.assert_has_calls([
        call('dummy_metric_aggregator: No metric files found for aggregation!')
    ])
```

### Gotcha 12: CompletionCallback Requires Specific Env Vars
**Problem**: Missing `NUPLAN_SERVER_S3_ROOT_URL` crashes initialization
**Impact**: Test suite fails if env vars not mocked
**Solution**: Always mock env vars in setUp for CompletionCallback tests
```python
@patch.dict(os.environ, {"NUPLAN_SERVER_S3_ROOT_URL": "bucket"})
@patch.dict(os.environ, {"SCENARIO_FILTER_ID": "1"})
def setUp(self):
    # Now initialization won't crash
    self.callback = CompletionCallback(...)
```

### Gotcha 13: Parquet File Integration Order Matters
**Problem**: MetricFileCallback must run BEFORE MetricSummaryCallback
**Impact**: Summary has no data to render if file callback skipped
**Solution**: Follow correct callback ordering in tests
```python
def setUp(self):
    # 1. Create scenario metrics
    self.set_up_dummy_metric(...)

    # 2. Integrate with MetricFileCallback
    metric_file_callback = MetricFileCallback(...)
    metric_file_callback.on_run_simulation_end()

    # 3. Now MetricSummaryCallback can load integrated files
    self.metric_summary_callback = MetricSummaryCallback(...)
```

### Gotcha 14: Mock S3 Client Must Be Explicitly Assigned
**Problem**: `get_s3_client()` called if `s3_client=None` in constructor
**Impact**: Test tries to connect to real AWS (fails without credentials)
**Solution**: Always pass mock client to constructor
```python
def setUp(self):
    self.mock_client = Mock()  # Don't rely on get_s3_client()
    self.publisher_callback = PublisherCallback(
        uploads={...},
        s3_client=self.mock_client,  # ✅ Explicit mock
        s3_bucket="bucket",
        remote_prefix=["user", "image"]
    )
```

### Gotcha 15: Temporary Directory Cleanup Fails on Windows
**Problem**: Windows locks files in temp directories (can't delete)
**Impact**: `self.tmp_dir.cleanup()` raises `PermissionError`
**Solution**: Explicitly close all file handles before cleanup
```python
def test_something(self):
    file = open(f'{self.tmp_dir.name}/data.txt', 'w')
    file.write("test")
    file.close()  # ✅ Must close before cleanup!

    # Now cleanup won't fail
```

---

## 12. Integration with Main Callback System

### Callback Execution Flow in Simulation Runner

```python
# In nuplan/planning/script/run_simulation.py (simplified)

def run_simulation(cfg: DictConfig) -> None:
    # Build main callbacks
    main_callbacks = build_main_callbacks(cfg)
    multi_callback = MultiMainCallback(main_callbacks)

    # Hook 1: Start
    multi_callback.on_run_simulation_start()

    # Run all scenarios (triggers AbstractCallback hooks internally)
    for scenario in scenarios:
        simulation = build_simulation(scenario, callbacks=per_scenario_callbacks)
        simulation.run()  # Triggers on_step_end(), on_simulation_end(), etc.

    # Hook 2: End (aggregation across scenarios)
    multi_callback.on_run_simulation_end()
```

**Key insight**: Main callbacks run ONCE per multi-scenario run, not per scenario!

### Cross-References

**Upstream Dependencies**:
- **nuplan/planning/metrics/** - MetricsEngine, aggregators, metric files
- **nuplan/planning/simulation/callback/** - Per-scenario callbacks (8 hooks)
- **nuplan/common/utils/s3_utils.py** - S3 client utilities
- **nuplan/common/utils/io_utils.py** - File I/O helpers

**Downstream Consumers**:
- **nuplan/planning/script/run_simulation.py** - Main entry point (calls hooks)
- **nuplan/planning/script/builders/main_callback_builder.py** - Hydra instantiation

**Related Documentation**:
- **callback/test/CLAUDE.md** - Per-scenario callback tests (comparison)
- **metrics/** - Metric computation and aggregation (upstream)
- **Phase 2C**: Main simulation orchestration

---

## 13. Performance Characteristics

### Test Execution Time (Approximate)
- **test_metric_file_callback.py**: ~0.3s (fast, minimal I/O)
- **test_metric_aggregator_callback.py**: ~0.4s (fast, mock aggregator)
- **test_validation_callback.py**: ~0.2s (fast, mock parquet read)
- **test_publisher_callback.py**: ~0.3s (fast, mock S3)
- **test_completion_callback.py**: ~0.2s (fast, env var mocking)
- **test_metric_summary_callback.py**: ~3-8s (slow, real MetricsEngine + PDF rendering)

**Total suite**: ~5-10 seconds (mostly metric_summary integration test)

### Optimization Opportunities
1. **Mock PDF rendering** - Could reduce metric_summary test to <1s
2. **Reduce metric samples** - Currently creates 3 statistics (could use 1)
3. **Parallel test execution** - Tests are independent, could use `pytest-xdist`
4. **Skip slow tests in CI quick mode** - Mark metric_summary as `@pytest.mark.slow`

### Memory Usage
- **Peak memory**: ~200 MB (metric_summary creates full MetricsEngine)
- **Per-test memory**: ~20 MB (pandas DataFrames, temp files)
- **Temporary files**: ~5 MB per test (cleaned up in tearDown)

---

## 14. Summary & Key Takeaways

### What This Test Suite Validates
✅ **Main callback lifecycle** - Both hooks called correctly
✅ **File aggregation** - Metric files merged across scenarios
✅ **S3 upload orchestration** - Correct files uploaded to correct paths
✅ **Validation markers** - Pass/fail files created based on simulation success
✅ **Environment variable handling** - Competition tracking env vars parsed
✅ **Logging statements** - Timing logs in correct format
✅ **Integration pipeline** - Full metric → aggregator → summary → PDF flow

### What Developers Should Know
1. **AbstractMainCallback has only 2 hooks** (vs AbstractCallback's 8)
2. **Main callbacks run ONCE per simulation run** (not per scenario)
3. **Tests use minimal mocking** (only external I/O like S3)
4. **Use `addCleanup()` for robust teardown** (handles exceptions)
5. **Logger patches must match module path** (not root logger)
6. **Integration test validates full pipeline** (no mocks in metric_summary)
7. **Environment variables must be mocked** (for CompletionCallback)

### When to Update Tests
- **New main callback** → Add new test file (follow existing patterns)
- **New hook added to AbstractMainCallback** → Update all callback tests
- **Metric file format changes** → Update test_metric_file_callback.py
- **S3 upload logic changes** → Update test_publisher_callback.py
- **Aggregator interface changes** → Update test_metric_aggregator_callback.py

### Related Documentation
- **Phase 2C**: Main callback system (`nuplan/planning/simulation/main_callback/` parent dir)
- **callback/test/CLAUDE.md**: Per-scenario callback tests (comparison layer)
- **metrics/**: Metric computation and aggregation (upstream dependency)
