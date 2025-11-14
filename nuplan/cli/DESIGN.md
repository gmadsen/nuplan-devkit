# nuPlan CLI Dataset Management - Design Document

## Overview
Extend the nuplan CLI with comprehensive dataset management capabilities including selective downloads, inventory tracking, and scenario-to-blob mapping.

## Architecture

### Module Structure
```
nuplan/cli/
├── nuplan_cli.py           # Main CLI entry point (existing)
├── db_cli.py               # Database commands (existing)
├── download_cli.py         # NEW: Download commands
├── inventory_cli.py        # NEW: Inventory/status commands
├── dataset_utils.py        # NEW: Shared utilities
└── DESIGN.md               # This file
```

### Dependencies
- **typer**: CLI framework (already used)
- **httpx**: Modern HTTP client for S3 scraping
- **rich**: Beautiful terminal output (progress bars, tables)
- **pathlib**: Modern path handling

## CLI Interface Design

### 1. Download Commands

#### `nuplan_cli download mini`
Download the mini dataset with selective sensor blob filtering.

**Usage:**
```bash
# Download everything (database + maps + all sensor blobs)
nuplan_cli download mini

# Download only specific camera sets
nuplan_cli download mini --camera=0,1

# Download only specific lidar sets
nuplan_cli download mini --lidar=0

# Download both specific cameras and lidar
nuplan_cli download mini --camera=0 --lidar=0,1

# Preview without downloading
nuplan_cli download mini --dry-run

# Control parallelism
nuplan_cli download mini --parallel=8
```

**Options:**
- `--camera`: Comma-separated camera set indices (0-7)
- `--lidar`: Comma-separated lidar set indices (0-8)
- `--skip-db`: Skip database files (default: False)
- `--skip-maps`: Skip map files (default: False)
- `--dry-run`: Show what would be downloaded (default: False)
- `--parallel`: Number of parallel downloads (default: 4)
- `--resume`: Resume from state file (default: True)

**State Management:**
- State file: `$NUPLAN_DATA_ROOT/.nuplan_download_state.json`
- Tracks: completed files, total bytes, last update
- Enables resume capability

#### `nuplan_cli download full`
Download the full dataset (~10TB).

**Usage:**
```bash
# Download everything
nuplan_cli download full

# Maps only
nuplan_cli download full --maps-only
```

**Options:**
- Same as `mini` but no camera/lidar filtering (full dataset is all-or-nothing for sensors)

### 2. Inventory Commands

#### `nuplan_cli inventory`
Show comprehensive dataset inventory with local vs remote comparison.

**Usage:**
```bash
# Show all datasets
nuplan_cli inventory

# Focus on sensor blobs
nuplan_cli inventory --sensors

# Show only missing files
nuplan_cli inventory --missing

# Export to JSON
nuplan_cli inventory --format=json > inventory.json
```

**Output Format (table):**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Dataset                     ┃ Local   ┃ Remote  ┃ Size    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ nuplan-v1.1_mini.zip        │ ✓       │ ✓       │ 8.0 GB  │
│ nuplan-maps-v1.1.zip        │ ✓       │ ✓       │ 927 MB  │
│ nuplan-v1.1_mini_camera_0   │ ✗       │ ✓       │ 350 GB  │
│ nuplan-v1.1_mini_camera_1   │ ✗       │ ✓       │ 340 GB  │
│ ...                         │         │         │         │
│ nuplan-v1.1_mini_lidar_0    │ ✗       │ ✓       │ 60 GB   │
│ nuplan-v1.1_mini_lidar_8    │ ✓       │ ✓       │ 61 GB   │
└─────────────────────────────┴─────────┴─────────┴─────────┘

Total Local: 69.9 GB / 3.8 TB (1.8%)
Missing: 64 sensor blob sets (3.7 TB)
```

**Options:**
- `--sensors`: Show only sensor blobs
- `--missing`: Show only missing files
- `--format`: Output format (table|json|csv)

#### `nuplan_cli inventory logs`
Show which logs have sensor blobs locally available.

**Usage:**
```bash
# List all logs and their sensor blob status
nuplan_cli inventory logs

# Show only logs missing sensor data
nuplan_cli inventory logs --missing

# Filter by date range
nuplan_cli inventory logs --from=2021-05 --to=2021-08
```

**Output Format:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Log                                         ┃ Camera  ┃ LiDAR   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ 2021.05.12.22.00.38_veh-35_01008_01518     │ ✗       │ ✗       │
│ 2021.07.16.20.45.29_veh-35_01095_01486     │ ✗       │ ✗       │
│ 2021.08.09.17.55.59_veh-28_00021_00307     │ ✗       │ ✓       │
│ 2021.10.11.02.57.41_veh-50_01522_02088     │ ✗       │ ✓       │
└─────────────────────────────────────────────┴─────────┴─────────┘

Total: 64 logs (9 with LiDAR, 0 with camera)
```

### 3. Map Commands

#### `nuplan_cli map log <log_name>`
Show which sensor blob zip sets contain a specific log.

**Usage:**
```bash
# Find which zips contain a specific log
nuplan_cli map log 2021.05.12.22.00.38_veh-35_01008_01518

# Output:
# Log: 2021.05.12.22.00.38_veh-35_01008_01518
# Camera: nuplan-v1.1_mini_camera_0.zip (~350 GB)
# LiDAR:  nuplan-v1.1_mini_lidar_0.zip (~60 GB)
#
# Download with:
#   nuplan_cli download mini --camera=0 --lidar=0
```

#### `nuplan_cli map db <db_file>`
Show which sensor blob zips are needed for a DB file's log.

**Usage:**
```bash
# Check sensor requirements for a DB file
nuplan_cli map db /path/to/2021.05.12.22.00.38_veh-35_01008_01518.db

# Check multiple DBs (e.g., all mini DBs)
nuplan_cli map db /path/to/splits/mini/*.db --summary
```

**Output Format (summary):**
```
Required sensor blob zips for 64 mini DB files:

Camera sets needed: 0, 1, 2, 3, 4, 5, 6, 7 (all)
LiDAR sets needed:  0, 1, 2, 3, 4, 5, 6, 7, 8 (all)

Download all with:
  nuplan_cli download mini
```

## Implementation Details

### dataset_utils.py - Shared Utilities

#### S3 Scraping
```python
def scrape_s3_bucket(base_url: str, pattern: str) -> list[str]:
    """
    Scrape S3 bucket HTML index page for files matching pattern.

    Args:
        base_url: S3 bucket URL
        pattern: Regex pattern to match files

    Returns:
        List of file URLs
    """
    # Implementation: httpx GET, BeautifulSoup parsing, regex filtering
```

#### Blob Mapping
```python
def get_blob_set_for_log(log_name: str, blob_type: str) -> int:
    """
    Determine which sensor blob zip set contains a log.

    Args:
        log_name: Log file name (e.g., "2021.05.12.22.00.38_veh-35_01008_01518")
        blob_type: "camera" or "lidar"

    Returns:
        Set index (0-7 for camera, 0-8 for lidar)

    Logic:
        - Parse log date from name
        - Map to blob set based on date ranges
        - Use lookup table derived from S3 bucket structure
    """
```

#### State Management
```python
@dataclass
class DownloadState:
    """Download state tracking."""
    version: str = "1.0"
    dataset_mode: str = "none"  # "mini" | "full"
    completed_files: list[str] = field(default_factory=list)
    total_downloaded_bytes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def save(self, path: Path) -> None:
        """Save state to JSON."""

    @classmethod
    def load(cls, path: Path) -> DownloadState:
        """Load state from JSON."""
```

### download_cli.py - Download Commands

#### Architecture
- Reuse bash script's proven download logic
- Use `httpx` for S3 requests (async capable)
- Use `rich.progress` for progress bars
- Validate downloads with zip integrity checks
- Resume capability via state file

#### Download Function
```python
async def download_file(
    url: str,
    dest: Path,
    progress: Progress,
    task_id: TaskID,
    max_retries: int = 5
) -> bool:
    """
    Download a file with progress tracking and retry logic.

    Uses httpx streaming to download large files efficiently.
    Validates zip files after download.
    """
```

### inventory_cli.py - Inventory Commands

#### Local Scanning
```python
def scan_local_dataset(data_root: Path) -> dict[str, FileInfo]:
    """
    Scan NUPLAN_DATA_ROOT for existing files.

    Returns:
        Dict mapping file names to FileInfo (size, path, modified)
    """
```

#### Remote Querying
```python
async def query_remote_dataset(dataset: str) -> dict[str, FileInfo]:
    """
    Query S3 bucket for available files.

    Args:
        dataset: "mini" or "full"

    Returns:
        Dict mapping file names to FileInfo (size, url)
    """
```

#### Log-to-Blob Matching
```python
def match_logs_to_blobs(
    db_files: list[Path],
    local_blobs: set[str]
) -> dict[str, BlobStatus]:
    """
    Match DB log files to sensor blobs and check availability.

    Returns:
        Dict mapping log names to BlobStatus (has_camera, has_lidar, sets_needed)
    """
```

## Integration Points

### 1. Main CLI (`nuplan_cli.py`)
```python
from nuplan.cli import download_cli, inventory_cli

cli.add_typer(download_cli.cli, name="download")
cli.add_typer(inventory_cli.cli, name="inventory")
cli.add_typer(inventory_cli.map_cli, name="map")  # Separate namespace
```

### 2. Environment Variables
- `NUPLAN_DATA_ROOT`: Dataset root (required)
- `NUPLAN_DOWNLOAD_PARALLEL`: Default parallelism (optional, default=4)
- `NUPLAN_DOWNLOAD_STATE`: State file path (optional, default=`$NUPLAN_DATA_ROOT/.nuplan_download_state.json`)

### 3. Justfile Recipes
```make
# Download mini dataset with camera_0 and lidar_0 for tutorials
download-tutorial:
    uv run nuplan_cli download mini --camera=0 --lidar=0

# Check dataset status
inventory:
    uv run nuplan_cli inventory

# Show missing sensor blobs
inventory-missing:
    uv run nuplan_cli inventory --missing --sensors

# Map a DB to required sensor zips
map-db file:
    uv run nuplan_cli map db {{file}}
```

## Testing Strategy

### Unit Tests
- `test_dataset_utils.py`: S3 scraping, blob mapping, state management
- `test_download_cli.py`: Download logic, retry handling, validation
- `test_inventory_cli.py`: Local scanning, remote querying, matching

### Integration Tests
- Test download with `--dry-run` flag
- Test resume capability (interrupt and resume)
- Test inventory accuracy (compare with actual S3 bucket)

### Manual Testing
- Download `camera_0` and `lidar_0` for tutorial
- Verify tutorial works with downloaded blobs
- Check inventory output matches reality

## Migration from Bash Script

### What to Keep
- State file format (JSON with completed_files tracking)
- aria2c for parallel downloads (or implement in Python with httpx)
- Retry logic with exponential backoff
- Zip integrity verification

### What to Improve
- Better error messages (typer's rich integration)
- Selective filtering (camera/lidar sets)
- Progress visualization (rich progress bars)
- Dry-run mode
- JSON/CSV export for inventory

### Deprecation Plan
1. Implement Python CLI with feature parity
2. Add migration note to bash script
3. Keep bash script for 1-2 releases
4. Eventually remove (after verifying Python version is stable)

## Security Considerations

### S3 Bucket Access
- Public bucket, no authentication required
- Use HTTPS for all requests
- Verify downloaded files (zip integrity)

### Path Traversal
- Validate all file paths are within `NUPLAN_DATA_ROOT`
- Sanitize user-provided DB file paths
- Use `pathlib` for safe path operations

### Download Safety
- Set max file size limits (prevent runaway downloads)
- Verify zip file integrity before extraction
- Handle corrupted downloads gracefully (retry)

## Future Enhancements (Phase 2)

### 1. Torrent Support
- Faster downloads via BitTorrent (if Motional provides torrents)
- Peer-to-peer distribution reduces server load

### 2. Cloud Storage Integration
- Upload/sync to personal cloud storage (S3, GCS, Azure)
- Share datasets across team members

### 3. Dataset Versioning
- Track dataset version changes
- Migration tools for version upgrades
- Changelog for dataset updates

### 4. Advanced Filtering
- Download by scenario type (e.g., only lane changes)
- Download by location (e.g., only Las Vegas logs)
- Download by time range (e.g., 2021-05 to 2021-08)

### 5. Storage Optimization
- Compress sensor blobs with better codecs
- Deduplicate identical frames across logs
- Stream directly to analysis (no local storage)

## Success Criteria

### MVP (Phase 1)
- ✅ Download mini dataset with selective sensor filtering
- ✅ Show inventory of local vs remote files
- ✅ Map DB files to required sensor blob zips
- ✅ Resume interrupted downloads
- ✅ Tutorial sensor data downloaded successfully

### Complete (Phase 2)
- ✅ Full dataset download support
- ✅ Advanced filtering (scenario type, location, date)
- ✅ Performance optimization (async downloads)
- ✅ Comprehensive testing (unit + integration)
- ✅ Documentation and examples in CLAUDE.md
