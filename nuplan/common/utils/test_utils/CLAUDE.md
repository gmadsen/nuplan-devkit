# CLAUDE.md - nuplan/common/utils/test_utils

## Purpose & Responsibility

**Specialized testing infrastructure for the nuPlan framework.**

This module provides:
- **Test Parameterization**: File-based test parameterization via JSON fixtures (`nuplan_test` decorator)
- **Interface Validation**: Automated checking of abstract interface implementations
- **Function Signature Validation**: Verify function swappability for mocking/dependency injection
- **S3 Mocking**: Test utilities for mocking AWS S3 operations with moto
- **Pytest Integration**: Custom pytest plugin for nuPlan-specific test workflows
- **Patching Utilities**: Enhanced `unittest.mock.patch` with signature validation

**Design Philosophy**: Make testing nuPlan components easy, safe, and declarative. Enable visual regression testing with JSON snapshots, ensure interface contracts are honored, and provide realistic S3 mocking for cloud-dependent code.

---

## Key Abstractions & Classes

### Test Parameterization (`nuplan_test.py`, `plugin.py`, `instances.py`)

**`@nuplan_test(path)` Decorator** - File-based test parameterization
- **Without path**: Hardcoded test (no JSON fixture)
- **With file path**: Load single JSON file as test input
- **With directory path**: Generate test for each `.json` file in directory

**Usage Example**:
```python
from nuplan.common.utils.test_utils.nuplan_test import nuplan_test

@nuplan_test("fixtures/test_scenario.json")
def test_planner_on_scenario(scene):
    # `scene` is a fixture containing parsed JSON data
    planner = MyPlanner()
    result = planner.plan(scene)
    assert result.is_valid()

@nuplan_test("fixtures/")  # Directory of JSON files
def test_planner_on_all_scenarios(scene):
    # Runs once per JSON file in fixtures/
    # Test ID = filename (without .json)
    ...
```

**How It Works**:
1. **Collection Phase** (`pytest_collection_finish` hook):
   - Pytest discovers all `@nuplan_test` marked tests
   - Registers each test in `REGISTRY` with metadata (path, params, type)
2. **Execution Phase** (`scene` fixture):
   - Fixture queries `REGISTRY` for test type (hardcoded/filebased/newable)
   - Loads JSON data if file-based
   - Yields data to test function
3. **Special Test ID** (`<newname>`):
   - Placeholder for creating new JSON fixtures interactively
   - Skipped during test runs

**`Registry` Class** - Global test metadata store
- `add(id, params, absdirpath, relpath)` - Register test during collection
- `get_type(id)` → `"hardcoded" | "filebased" | "newable" | "invalid"`
- `get_data(id)` → Load JSON fixture or return `{}`

**`TestInfo` Class** - Test configuration metadata
- `params: Optional[str]` - JSON filename (without extension)
- `absdirpath: Optional[str]` - Absolute directory path
- `relpath: Optional[str]` - Relative path from test file
- Methods: `is_hardcoded()`, `is_file_based()`, `is_newable()`, `is_invalid()`

### Interface Validation (`interface_validation.py`)

**`assert_class_properly_implements_interface(interface_class_type, derived_class_type)`**

Validates that a class correctly implements an abstract interface with three checks:

1. **Inheritance Check**: `derived_class_type` is subclass of `interface_class_type`
2. **Method Presence Check**: All abstract methods in interface exist in derived class
3. **Signature Compatibility Check**: Each method signature matches (types, defaults, kwdefaults)

**Usage Example**:
```python
from nuplan.common.utils.test_utils.interface_validation import (
    assert_class_properly_implements_interface
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

def test_my_planner_implements_interface():
    assert_class_properly_implements_interface(
        AbstractPlanner,  # Interface
        MyPlanner        # Implementation
    )
    # Raises TypeError if:
    # - MyPlanner doesn't inherit from AbstractPlanner
    # - MyPlanner missing abstract methods (e.g., compute_planner_trajectory)
    # - Method signatures don't match (wrong types, defaults)
```

**Internal Functions**:
- `_assert_derived_is_child_of_base()` - Check subclass relationship
- `_get_public_methods(class_type, only_abstract=False)` - Extract public methods
  - Excludes `_private` and `__magic__` methods
  - Filters by `__isabstractmethod__` if `only_abstract=True`
- `_assert_abstract_methods_present()` - Check all abstract methods implemented

**Error Messages**: Rich, multi-line error messages with missing method names and class details

### Function Signature Validation (`function_validation.py`)

**`assert_functions_swappable(first_func, second_func)`**

Validates that `second_func` can safely replace `first_func` (e.g., in mocking) by checking:

1. **Type Annotations Match**: `__annotations__` identical (param types, return type)
2. **Defaults Match**: `__defaults__` tuple identical
3. **Keyword Defaults Match**: `__kwdefaults__` dict identical

**Usage Example**:
```python
from nuplan.common.utils.test_utils.function_validation import assert_functions_swappable

# Original function
def api_call(x: int, y: str, timeout: float = 5.0) -> bool:
    ...

# Mock function
def mock_api_call(x: int, y: str, timeout: float = 5.0) -> bool:
    return True

# Validate mock is compatible
assert_functions_swappable(api_call, mock_api_call)  # Pass

# ❌ This would fail (wrong type):
def bad_mock(x: str, y: str, timeout: float = 5.0) -> bool:  # x is str, not int
    return True

assert_functions_swappable(api_call, bad_mock)  # TypeError!
```

**Internal Functions**:
- `_assert_function_signature_types_match()` - Compare `__annotations__`
- `_assert_function_defaults_match()` - Compare `__defaults__`
- `_assert_function_kwdefaults_match()` - Compare `__kwdefaults__`

**Error Messages**: Show mismatched annotations/defaults with both functions' signatures

### S3 Mocking (`mock_s3_utils.py`)

**`mock_async_s3` Context Manager** - Mock AWS S3 for async code
- Wraps `moto.mock_s3` for compatibility with `aioboto3` (async boto3)
- Patches `aiobotocore` internals to work with moto's synchronous mocks

**How It Works**:
1. **Patch aiobotocore** to wrap moto responses (`MockAWSResponse`, `MockHttpClientResponse`)
2. **Start moto S3 mock** (in-memory S3 implementation)
3. **Yield control** to test code
4. **Stop mock and unpatch** on exit

**Usage Example**:
```python
from nuplan.common.utils.test_utils.mock_s3_utils import (
    mock_async_s3, create_mock_bucket, setup_mock_s3_directory
)
from nuplan.common.utils.io_utils import save_pickle, read_pickle
from pathlib import Path

async def test_s3_save_load():
    with mock_async_s3():
        # Create mock bucket
        await create_mock_bucket("test-bucket")

        # Use S3 as normal
        s3_path = Path("s3://test-bucket/model.pkl")
        await save_pickle_async(s3_path, {"weights": [1, 2, 3]})

        loaded = await read_pickle_async(s3_path)
        assert loaded == {"weights": [1, 2, 3]}
```

**Helper Functions**:
- `create_mock_bucket(bucket_name)` - Create bucket in mock S3
- `setup_mock_s3_directory(expected_files, directory, bucket)` - Populate mock S3 with files
  - `expected_files: Dict[str, str]` - `{relative_path: contents}`
  - Creates directory structure and uploads files
- `set_mock_object_from_aws(s3_key, s3_bucket)` - Copy real S3 object to mock S3

**`MockAWSResponse` Class** - Wraps moto's `AWSResponse` for aiobotocore
- `_content_prop()` - Async property returning response bytes
- `_text_prop()` - Async property returning response text
- `status_code` - HTTP status

**`MockHttpClientResponse` Class** - Wraps moto's response for aiohttp
- `content.read(n)` - Async read handler
- `raw_headers` - Headers encoded for aioboto

**Patches Applied**:
- `aiobotocore.endpoint.convert_to_response_dict` → `convert_to_response_dict_patch`
- `aiobotocore.handlers._looks_like_special_case_error` → `_looks_like_special_case_error_patch`

### Enhanced Patching (`patch.py`)

**`patch_with_validation(method_to_patch, patch_function, override_function=None, **kwargs)`**

Context manager wrapping `unittest.mock.patch` with signature validation:

1. **Import original function** from dot-string path
2. **Validate signatures** using `assert_functions_swappable()`
3. **Patch with validated function**
4. **Yield patched object** to test
5. **Unpatch on exit**

**Usage Example**:
```python
from nuplan.common.utils.test_utils.patch import patch_with_validation

def mock_expensive_api_call(param: str, timeout: int = 30) -> dict:
    return {"status": "mocked"}

def test_planner_with_mocked_api():
    with patch_with_validation(
        "my.module.expensive_api_call",
        mock_expensive_api_call
    ) as mock:
        # Test code here
        result = my_planner.run()  # Uses mock instead of real API
        assert result.is_valid()
```

**Internal Function**:
- `_get_method_from_import(import_str)` - Import function from dot-string
  - Parses `"foo.bar.baz.qux"` → imports `qux` from `foo.bar.baz`
  - **Limitation**: Only supports `import x.y.z` form (not `from x import y`)

**Parameters**:
- `method_to_patch: str` - Dot-string path (e.g., `"nuplan.common.utils.s3_utils.upload_file_to_s3"`)
- `patch_function: Callable` - Replacement function (validated)
- `override_function: Optional[Callable]` - Custom function for validation (if auto-import fails)
- `**kwargs` - Passed to `unittest.mock.patch`

### Pytest Plugin (`plugin.py`)

**`pytest_configure(config)`** - Register `nuplan_test` marker
- Adds marker definition to pytest config

**`pytest_collection_finish(session)`** - Register tests in global registry
- Iterates all collected test items
- Extracts `@nuplan_test` marker metadata
- Registers in `instances.REGISTRY`

**`pytest_runtest_makereport(item, call)`** - Hook for test result reporting
- Stores test result on item as `rep_setup`, `rep_call`, `rep_teardown`

**`@pytest.fixture scene(nuplan_test, request)`** - Main fixture for nuplan tests
- Queries registry for test type
- Loads JSON data if file-based
- Yields to test function
- Skips if test type is "newable" (placeholder)

**Workflow**:
```
1. pytest discovery → @nuplan_test tests collected
2. pytest_collection_finish → tests registered in REGISTRY
3. Test execution → scene fixture queries REGISTRY
4. Fixture loads JSON → yields to test
5. Test runs with scene data
```

---

## Architecture & Design Patterns

### 1. **Registry Pattern**
Global singleton (`instances.REGISTRY`) stores test metadata during collection phase:
```python
# Collection phase (pytest hooks)
REGISTRY.add(test_id, params, absdirpath, relpath)

# Execution phase (fixtures)
test_type = REGISTRY.get_type(test_id)
data = REGISTRY.get_data(test_id)
```

**Benefits**: Decouples test discovery from execution, enables dynamic test generation from filesystem

### 2. **Decorator-Based Test Generation**
`@nuplan_test` converts directories into parameterized tests:
```python
@nuplan_test("fixtures/")  # 10 JSON files
def test_foo(scene):       # Becomes 10 separate tests
    ...
```

**Implementation**: Uses `@pytest.mark.parametrize` under the hood with custom IDs

### 3. **Fixture-Driven Data Loading**
Test data loaded via pytest fixture, not directly in test:
```python
# ❌ WRONG - Explicit loading in test
def test_planner():
    with open("fixtures/scenario.json") as f:
        scene = json.load(f)
    ...

# ✅ CORRECT - Fixture handles loading
@nuplan_test("fixtures/scenario.json")
def test_planner(scene):  # scene is pre-loaded
    ...
```

**Benefits**: Cleaner tests, reusable loading logic, supports visual test creation

### 4. **Mocking with Validation**
Patches validated to ensure signature compatibility:
```python
# Standard mock.patch - no validation
with mock.patch("foo.bar", my_mock):
    ...  # Runtime error if signature wrong!

# patch_with_validation - fails fast
with patch_with_validation("foo.bar", my_mock):
    ...  # TypeError at patch time if signature wrong
```

**Benefits**: Catch mocking errors early (test setup) instead of late (test execution)

### 5. **Layered S3 Mocking**
Three-layer mock architecture for async S3:
```
Test Code
    ↓
aioboto3 (async boto3)
    ↓
MockAWSResponse (compatibility layer)
    ↓
moto (in-memory S3)
```

**Why Needed**: moto is synchronous, aioboto3 is async → compatibility layer bridges the gap

---

## Dependencies (What We Import)

### External Libraries
- **pytest**: Test framework and fixtures
- **moto**: AWS service mocking (S3, etc.)
- **aiohttp**: HTTP client library (for mock response types)
- **aiobotocore**: Async boto3 internals (for patching)
- **botocore**: Boto3 request/response models
- **aiofiles**: Async file I/O (for mock S3 setup)
- **unittest.mock**: Standard library mocking (`patch`, `MagicMock`)
- **contextlib**: Context manager utilities
- **importlib**: Dynamic module importing
- **inspect**: Function introspection
- **json, os, tempfile, uuid**: Standard library utilities

### Internal nuPlan Dependencies
- `nuplan.common.utils.s3_utils` - S3 utilities being mocked
  - `get_async_s3_session`, `upload_file_to_s3`, `download_file_from_s3`

**Dependency Direction**: `test_utils` is a **leaf module** - only imported by test files, doesn't import from application code (except for utilities being tested).

---

## Dependents (Who Imports Us)

**All test files throughout the nuPlan codebase**. Usage patterns:

### Unit Tests (`nuplan/*/test/`)
- **Interface Tests**: Validate planners, metrics, scenario builders implement interfaces correctly
- **Function Tests**: Test individual utility functions with mocked S3
- **Integration Tests**: Test workflows with file-based scenarios

### Specific Test Examples
```python
# nuplan/planning/simulation/planner/test/test_planners.py
from nuplan.common.utils.test_utils.interface_validation import (
    assert_class_properly_implements_interface
)

def test_simple_planner_interface():
    assert_class_properly_implements_interface(
        AbstractPlanner,
        SimplePlanner
    )

# nuplan/common/utils/test/test_s3_utils.py
from nuplan.common.utils.test_utils.mock_s3_utils import mock_async_s3

def test_s3_upload_download():
    with mock_async_s3():
        # Test S3 operations without real AWS
        ...

# nuplan/planning/scenario_builder/test/test_scenario_builder.py
from nuplan.common.utils.test_utils.nuplan_test import nuplan_test

@nuplan_test("fixtures/scenarios/")
def test_build_scenario(scene):
    scenario = build_scenario_from_json(scene)
    assert scenario.is_valid()
```

### Visual Regression Testing
File-based test approach enables:
1. **Capture Workflow**: Run planner, serialize results to JSON, save to `fixtures/`
2. **Regression Testing**: `@nuplan_test("fixtures/")` runs on all captured scenarios
3. **Debugging**: Inspect JSON fixtures to understand failures

---

## Critical Files (Prioritized)

### Tier 1: Core Testing Infrastructure

**1. `mock_s3_utils.py` (219 lines)** ⭐⭐⭐
- **Why Critical**: Enables testing all S3-dependent code (caching, checkpoints, metrics) without real AWS
- **Key Classes**: `mock_async_s3`, `MockAWSResponse`, `MockHttpClientResponse`
- **Gotchas**:
  - Complex patching of aiobotocore internals (fragile to aiobotocore updates)
  - Must create buckets explicitly (`create_mock_bucket`) before use
  - Moto has different behaviors than real S3 (e.g., eventual consistency not modeled)

**2. `nuplan_test.py` (97 lines)** ⭐⭐⭐
- **Why Critical**: Enables file-based parameterization for visual regression testing
- **Key Decorator**: `@nuplan_test(path)`
- **Gotchas**:
  - JSON files must be valid (no parsing error handling)
  - Test IDs derived from filenames (must be unique)
  - `<newname>` placeholder skipped (can confuse developers)

**3. `interface_validation.py` (109 lines)** ⭐⭐⭐
- **Why Critical**: Ensures interface contracts honored across 50+ planner/metric implementations
- **Key Function**: `assert_class_properly_implements_interface()`
- **Gotchas**:
  - Only checks public methods (private methods ignored)
  - Doesn't validate runtime behavior (only signatures)
  - Can't detect covariant/contravariant type changes (Python typing limitation)

### Tier 2: Important Utilities

**4. `function_validation.py` (104 lines)** ⭐⭐
- **Why Important**: Prevents mocking errors by validating signatures
- **Key Function**: `assert_functions_swappable()`
- **Gotchas**:
  - Doesn't check `__code__` or `__closure__` (TODO at line 103)
  - Type annotations compared by equality (generics may cause false negatives)

**5. `plugin.py` (47 lines)** ⭐⭐
- **Why Important**: Pytest integration for nuplan test workflow
- **Key Hooks**: `pytest_collection_finish`, `scene` fixture
- **Gotchas**:
  - Requires `NUPLAN_TEST_PLUGIN` in `pytest_plugins` (conftest.py)
  - Marker registration can conflict with other plugins

**6. `patch.py` (46 lines)** ⭐
- **Why Important**: Safer mocking with signature validation
- **Key Function**: `patch_with_validation()`
- **Gotchas**:
  - Only supports dot-string imports (`x.y.z`, not `from x import y`)
  - Auto-import can fail for dynamically generated functions

### Tier 3: Supporting Classes

**7. `instances.py` (100 lines)** ⭐
- **Why Important**: Global registry for test metadata
- **Key Classes**: `Registry`, `TestInfo`
- **Gotchas**: Global state shared across all tests (can cause test pollution if not careful)

---

## Common Usage Patterns

### Pattern 1: File-Based Test Parameterization
```python
from nuplan.common.utils.test_utils.nuplan_test import nuplan_test

# Single JSON file
@nuplan_test("fixtures/test_scenario_1.json")
def test_planner_on_single_scenario(scene):
    # scene = {"ego_state": {...}, "agents": [...], ...}
    result = my_planner.plan(scene)
    assert result.is_valid()

# Directory of JSON files (runs once per file)
@nuplan_test("fixtures/scenarios/")
def test_planner_on_all_scenarios(scene):
    # Runs 10 times if 10 JSON files in fixtures/scenarios/
    result = my_planner.plan(scene)
    assert result.is_valid()

# Hardcoded test (no JSON)
@nuplan_test()
def test_planner_with_hardcoded_data(scene):
    # scene = {} (empty dict)
    scene = create_test_scenario()  # Create in code
    result = my_planner.plan(scene)
    assert result.is_valid()
```

### Pattern 2: Interface Implementation Validation
```python
from nuplan.common.utils.test_utils.interface_validation import (
    assert_class_properly_implements_interface
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from my_planner import MyCustomPlanner

def test_my_planner_implements_interface():
    """Validates MyCustomPlanner correctly implements AbstractPlanner"""
    assert_class_properly_implements_interface(
        AbstractPlanner,  # Interface (abstract class)
        MyCustomPlanner   # Implementation (concrete class)
    )
    # Raises TypeError if:
    # - MyCustomPlanner not subclass of AbstractPlanner
    # - Missing methods: initialize, name, compute_planner_trajectory
    # - Method signatures don't match
```

### Pattern 3: Mocking S3 Operations
```python
import asyncio
from pathlib import Path
from nuplan.common.utils.test_utils.mock_s3_utils import (
    mock_async_s3, create_mock_bucket, setup_mock_s3_directory
)
from nuplan.common.utils.io_utils import save_pickle_async, read_pickle_async

async def test_checkpoint_save_load():
    with mock_async_s3():
        # Setup mock S3
        await create_mock_bucket("checkpoints")

        # Use S3 utilities as normal
        checkpoint = {"epoch": 10, "loss": 0.5}
        s3_path = Path("s3://checkpoints/model.pkl")

        await save_pickle_async(s3_path, checkpoint)
        loaded = await read_pickle_async(s3_path)

        assert loaded == checkpoint

# Synchronous wrapper
def test_checkpoint_sync():
    with mock_async_s3():
        asyncio.run(create_mock_bucket("checkpoints"))
        # ... rest of test
```

### Pattern 4: Populating Mock S3 Directory
```python
from nuplan.common.utils.test_utils.mock_s3_utils import (
    mock_async_s3, setup_mock_s3_directory
)

async def test_directory_download():
    with mock_async_s3():
        # Setup mock S3 directory with files
        expected_files = {
            "metadata/train.csv": "scenario,log\nA,log1\nB,log2",
            "metadata/val.csv": "scenario,log\nC,log3",
            "features/feat1.pkl": "binary_data_here",
        }

        await setup_mock_s3_directory(
            expected_files,
            directory=Path("experiments/cache_123"),
            bucket="ml-caches"
        )

        # Test code that downloads from S3
        cache = CacheLoader("s3://ml-caches/experiments/cache_123")
        metadata = cache.load_metadata()

        assert len(metadata) == 3  # A, B, C scenarios
```

### Pattern 5: Validated Function Patching
```python
from nuplan.common.utils.test_utils.patch import patch_with_validation

def mock_download_file_from_s3(local_path, s3_key, s3_bucket):
    """Mock that writes fake data to local_path"""
    local_path.write_text("fake checkpoint data")

def test_planner_loads_checkpoint():
    with patch_with_validation(
        "nuplan.common.utils.s3_utils.download_file_from_s3",
        mock_download_file_from_s3
    ):
        # Planner tries to download checkpoint from S3
        # Mock intercepts and writes fake data instead
        planner = MyPlanner(checkpoint_path="s3://bucket/model.pkl")

        assert planner.is_initialized()

# If mock signature wrong, patch_with_validation raises TypeError:
def bad_mock(wrong_param_name, s3_key, s3_bucket):  # ❌ Wrong!
    ...

with patch_with_validation(
    "nuplan.common.utils.s3_utils.download_file_from_s3",
    bad_mock  # TypeError: signature mismatch!
):
    ...
```

### Pattern 6: Function Signature Validation
```python
from nuplan.common.utils.test_utils.function_validation import (
    assert_functions_swappable
)

# Original API
def process_scenario(scenario: AbstractScenario, timeout: float = 30.0) -> SimulationResult:
    ...

# Mock implementation
def mock_process_scenario(scenario: AbstractScenario, timeout: float = 30.0) -> SimulationResult:
    return SimulationResult(success=True, metrics={})

# Validate mock can replace original
def test_mock_signature_compatibility():
    assert_functions_swappable(process_scenario, mock_process_scenario)  # Pass

# This would fail:
def bad_mock(scenario: AbstractScenario, timeout: int = 30) -> SimulationResult:
    # ❌ timeout is int, not float
    ...

assert_functions_swappable(process_scenario, bad_mock)  # TypeError!
```

---

## Gotchas & Pitfalls

### S3 Mocking

**1. Must Create Buckets Before Use**
```python
with mock_async_s3():
    # ❌ WRONG - Bucket doesn't exist yet
    await upload_file_to_s3_async(local_path, s3_key, "my-bucket")
    # Error: NoSuchBucket

    # ✅ CORRECT - Create bucket first
    await create_mock_bucket("my-bucket")
    await upload_file_to_s3_async(local_path, s3_key, "my-bucket")
```

**2. Moto Doesn't Model Eventual Consistency**
```python
with mock_async_s3():
    await create_mock_bucket("bucket")
    await upload_file_to_s3_async(local_path, "key", "bucket")

    # Real S3: might return False (eventual consistency)
    # Moto: always returns True (immediately consistent)
    exists = await check_s3_object_exists_async("key", "bucket")
    assert exists  # Passes with moto, might fail with real S3
```

**3. S3 Session Caching Persists Across Tests**
```python
# Test 1: Uses real S3
def test_real_s3():
    session = get_async_s3_session()  # Cached globally
    ...

# Test 2: Uses mock S3
def test_mock_s3():
    with mock_async_s3():
        session = get_async_s3_session()  # Still using cached REAL session!
        # Fix: Force new session
        session = get_async_s3_session(force_new=True)
```

**4. aiobotocore Version Sensitivity**
```python
# MockAWSResponse patches aiobotocore internals
# Patches at lines 134-138 can break with aiobotocore updates
# Symptom: AttributeError on aiobotocore.endpoint methods

# Solution: Pin aiobotocore version in requirements
aiobotocore==2.5.0  # Or compatible version
```

### File-Based Testing

**5. JSON Parsing Errors Not Handled**
```python
@nuplan_test("fixtures/broken.json")
def test_scenario(scene):
    ...

# If broken.json is invalid JSON → JSONDecodeError at fixture load
# No graceful error message about which file is broken

# Workaround: Validate JSONs in CI before tests
find fixtures/ -name "*.json" -exec python -m json.tool {} \; > /dev/null
```

**6. Test IDs Must Be Unique**
```python
# fixtures/scenarios/scenario_1.json
# fixtures/more_scenarios/scenario_1.json

@nuplan_test("fixtures/scenarios/")
@nuplan_test("fixtures/more_scenarios/")
def test_all(scene):
    ...

# Both generate test ID "scenario_1" → pytest collision!
# Solution: Use subdirectory in ID or rename files
```

**7. `<newname>` Placeholder Confusing**
```python
@nuplan_test("fixtures/scenarios/")
def test_scenarios(scene):
    ...

# Generates test IDs:
# - scenario_1
# - scenario_2
# - <newname>  # ← What is this?

# Answer: Placeholder for creating new fixtures interactively
# Always skipped during normal test runs
# To avoid: Don't have empty fixtures/ directory
```

### Interface Validation

**8. Only Validates Signatures, Not Behavior**
```python
class MyPlanner(AbstractPlanner):
    def compute_planner_trajectory(self, current_input):
        # ✅ Signature matches
        # ❌ But implementation raises NotImplementedError!
        raise NotImplementedError()

# Passes interface validation (signature matches)
assert_class_properly_implements_interface(AbstractPlanner, MyPlanner)

# Fails at runtime (not caught by validation)
planner = MyPlanner()
planner.compute_planner_trajectory(...)  # NotImplementedError!
```

**9. Doesn't Detect Covariant/Contravariant Type Changes**
```python
# Interface
class AbstractPlanner:
    def plan(self, input: PlannerInput) -> Trajectory:
        ...

# Implementation with subtype (should be valid)
class MyPlanner(AbstractPlanner):
    def plan(self, input: PlannerInput) -> InterpolatedTrajectory:
        # InterpolatedTrajectory is subclass of Trajectory (covariant)
        ...

# Validation fails (type annotations compared by equality, not subtyping)
assert_class_properly_implements_interface(AbstractPlanner, MyPlanner)
# TypeError: return type mismatch!
```

### Function Validation

**10. Doesn't Check Function Body**
```python
def original(x: int) -> int:
    return x * 2

def mock(x: int) -> int:
    # Signature matches, but behavior totally different
    return 999

# Validation passes (signatures match)
assert_functions_swappable(original, mock)

# But mock doesn't actually replicate behavior!
```

**11. Type Annotation Equality Can Be Too Strict**
```python
from typing import List

def func1(items: List[int]) -> None:
    ...

def func2(items: list[int]) -> None:  # Python 3.9+ lowercase syntax
    ...

# These are semantically equivalent but...
assert_functions_swappable(func1, func2)  # TypeError: List != list
```

### Patching

**12. Auto-Import Fails for Dynamic Functions**
```python
# This won't work:
with patch_with_validation(
    "dynamically.generated.function",  # Created at runtime
    mock_func
):
    ...
# ImportError: No module named 'dynamically.generated.function'

# Solution: Use override_function parameter
with patch_with_validation(
    "some.patchable.path",
    mock_func,
    override_function=actual_runtime_function  # Validate against this
):
    ...
```

---

## Related Documentation

### Internal nuPlan Modules
- **`nuplan/common/utils/CLAUDE.md`** - Parent utilities module (S3, I/O, distributed)
- **`nuplan/planning/simulation/planner/CLAUDE.md`** - Planner interfaces (validated by these utilities)
- **`nuplan/planning/simulation/CLAUDE.md`** - Simulation infrastructure (tested with these utilities)

### External References
- **pytest Documentation**: https://docs.pytest.org/
- **moto Documentation**: https://docs.getmoto.org/ (AWS mocking)
- **aiobotocore**: https://aiobotocore.readthedocs.io/ (async boto3)
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html

### Testing Best Practices
- **Visual Regression Testing**: https://www.browserstack.com/guide/visual-regression-testing
- **Interface Testing**: https://en.wikipedia.org/wiki/Interface_testing
- **Test Parameterization**: https://docs.pytest.org/en/stable/how-to/parametrize.html

---

## AIDEV-NOTES

**AIDEV-NOTE**: `mock_async_s3` patches aiobotocore internals (lines 134-138) - fragile to aiobotocore version updates. Pin aiobotocore version or monitor for breaking changes.

**AIDEV-NOTE**: `assert_class_properly_implements_interface()` doesn't validate covariant/contravariant types (Python typing limitation). Consider using `typing.get_args()` / `typing.get_origin()` for deeper type checking.

**AIDEV-NOTE**: `@nuplan_test` generates `<newname>` placeholder test (always skipped) - consider removing to reduce confusion (see `nuplan_test.py:42-44`).

**AIDEV-TODO**: Add validation of function `__code__` and `__closure__` in `assert_functions_swappable()` (see `function_validation.py:103`).

**AIDEV-TODO**: Handle JSON parsing errors gracefully in `scene` fixture with informative error messages about which file is malformed.
