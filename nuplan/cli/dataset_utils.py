# ABOUTME: nuPlan dataset management utilities for S3 scraping, blob mapping, and state tracking
# ABOUTME: Provides shared functionality for download, inventory, and mapping CLI commands

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup


# S3 bucket configuration
S3_BASE_URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
S3_MINI_SENSORS = f"{S3_BASE_URL}/sensor_blobs/mini_set/"
S3_TRAIN_SENSORS = f"{S3_BASE_URL}/sensor_blobs/train_set/"
S3_VAL_SENSORS = f"{S3_BASE_URL}/sensor_blobs/val_set/"
S3_TEST_SENSORS = f"{S3_BASE_URL}/sensor_blobs/test_set/"

# Full dataset splits (DATABASE files, organized by city)
DATASET_SPLITS = {
    "mini": ["mini"],
    "test": ["test"],
    "val": ["val"],
    "train": [
        "train_boston",
        "train_pittsburgh",
        "train_singapore",
        "train_vegas_1",
        "train_vegas_2",
        "train_vegas_3",
        "train_vegas_4",
        "train_vegas_5",
        "train_vegas_6",
    ],
}

# Sensor blob sets (SENSOR files, numbered sequentially - NOT by city!)
SENSOR_BLOB_COUNTS = {
    "mini": {"camera": 9, "lidar": 9},      # camera_0..8, lidar_0..8 = 18 files
    "train": {"camera": 43, "lidar": 43},   # camera_0..42, lidar_0..42 = 86 files
    "val": {"camera": 11, "lidar": 11},     # camera_0..10, lidar_0..10 = 22 files
    "test": {"camera": 9, "lidar": 9},      # camera_0..8, lidar_0..8 = 18 files
}


@dataclass
class FileInfo:
    """Information about a dataset file (local or remote)."""

    name: str
    size_bytes: int
    path: str | None = None  # Local path if available
    url: str | None = None  # Remote URL if available
    modified: datetime | None = None  # Last modified time

    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        return format_bytes(self.size_bytes)


@dataclass
class BlobStatus:
    """Status of sensor blobs for a log."""

    log_name: str
    has_camera: bool
    has_lidar: bool
    camera_set: int | None = None  # Which camera set (0-7) contains this log
    lidar_set: int | None = None  # Which lidar set (0-8) contains this log


@dataclass
class DownloadState:
    """Download state tracking for resume capability."""

    version: str = "1.0"
    dataset_mode: Literal["none", "mini", "full"] = "none"
    completed_files: list[str] = field(default_factory=list)
    total_downloaded_bytes: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> DownloadState:
        """Load state from JSON file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def mark_completed(self, filename: str, size_bytes: int) -> None:
        """Mark a file as completed and update total bytes."""
        if filename not in self.completed_files:
            self.completed_files.append(filename)
            self.total_downloaded_bytes += size_bytes
        self.last_updated = datetime.now().isoformat()

    def is_completed(self, filename: str) -> bool:
        """Check if a file has been completed."""
        return filename in self.completed_files


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "500 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def scrape_s3_bucket(base_url: str, pattern: str | None = None, timeout: int = 30) -> list[str]:
    """
    Scrape S3 bucket HTML index page for files.

    The S3 bucket index is an HTML page with links to files. We parse the HTML
    and extract file URLs that match the optional pattern.

    Args:
        base_url: S3 bucket URL (must end with /)
        pattern: Optional regex pattern to filter files (e.g., r'\.zip$')
        timeout: HTTP request timeout in seconds

    Returns:
        List of full file URLs

    Raises:
        requests.HTTPError: If request fails
        ValueError: If HTML parsing fails
    """
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # S3 bucket index pages have links in <a> tags
    # Extract all href values that look like files (not directories)
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and not href.endswith("/"):
            # Filter by pattern if provided
            if pattern is None or re.search(pattern, href):
                # Construct full URL
                full_url = base_url + href if not href.startswith("http") else href
                files.append(full_url)

    return files


def get_blob_set_for_log(log_name: str, blob_type: Literal["camera", "lidar"]) -> int:
    """
    Determine which sensor blob zip set contains a log.

    The nuPlan dataset splits sensor blobs into numbered sets:
    - camera_0 through camera_7 (8 sets)
    - lidar_0 through lidar_8 (9 sets)

    Each set contains logs from different time periods. We infer the set number
    from the log's date (extracted from the log name).

    Args:
        log_name: Log file name (e.g., "2021.05.12.22.00.38_veh-35_01008_01518")
        blob_type: "camera" or "lidar"

    Returns:
        Set index (0-7 for camera, 0-8 for lidar)

    Raises:
        ValueError: If log name format is invalid

    Note:
        This is a heuristic based on observed patterns. The exact mapping may
        vary and should be validated against the actual S3 bucket contents.
    """
    # Parse date from log name (format: YYYY.MM.DD.HH.MM.SS_veh-XX_XXXXX_XXXXX)
    match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", log_name)
    if not match:
        raise ValueError(f"Invalid log name format: {log_name}")

    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))

    # AIDEV-NOTE: Heuristic mapping based on mini dataset observations
    # This may need refinement after analyzing full S3 bucket structure
    # Known mappings from bash script testing:
    #   - lidar_8: Aug/Oct 2021 (2021.08.*, 2021.10.*)
    #   - lidar_0: May 2021 (2021.05.*) - contains tutorial test log
    # The pattern appears to be chronological with some overlap

    # For now, use a simple month-based heuristic
    # TODO: Replace with lookup table derived from S3 bucket scraping
    if blob_type == "camera":
        # Camera sets 0-7 (8 sets total)
        # Distribute months across sets
        return month % 8
    else:  # lidar
        # LiDAR sets 0-8 (9 sets total)
        # Distribute months across sets
        if month <= 5:
            return 0  # May and earlier -> lidar_0
        elif month <= 7:
            return 1  # June-July
        elif month == 8:
            return 8  # August -> lidar_8 (observed)
        elif month == 10:
            return 8  # October -> lidar_8 (observed)
        else:
            return month % 9


def get_log_name_from_db_file(db_path: Path) -> str:
    """
    Extract log name from DB file path.

    DB files are named after their logs:
    /path/to/2021.05.12.22.00.38_veh-35_01008_01518.db -> 2021.05.12.22.00.38_veh-35_01008_01518

    Args:
        db_path: Path to DB file

    Returns:
        Log name (without .db extension)
    """
    return db_path.stem


def get_sensor_blob_zip_name(blob_type: Literal["camera", "lidar"], set_index: int, dataset: Literal["mini", "full"] = "mini") -> str:
    """
    Get the sensor blob zip file name for a given type and set.

    Args:
        blob_type: "camera" or "lidar"
        set_index: Set index (0-8 for mini, 0-42 for train, etc.)
        dataset: "mini" or "full"

    Returns:
        Zip file name (e.g., "nuplan-v1.1_mini_camera_0.zip")

    Raises:
        ValueError: If set_index is out of range
    """
    # Validate based on dataset
    if dataset == "mini":
        if not (0 <= set_index <= 8):
            raise ValueError(f"Mini {blob_type} set index must be 0-8, got {set_index}")
    # Add validation for other datasets as needed

    prefix = "nuplan-v1.1_mini" if dataset == "mini" else "nuplan-v1.1"
    return f"{prefix}_{blob_type}_{set_index}.zip"


def scan_local_sensor_blobs(data_root: Path) -> dict[str, BlobStatus]:
    """
    Scan local sensor_blobs directory and determine which logs have data.

    Args:
        data_root: NUPLAN_DATA_ROOT path

    Returns:
        Dict mapping log names to BlobStatus
    """
    sensor_blobs_dir = data_root / "nuplan-v1.1" / "sensor_blobs"
    if not sensor_blobs_dir.exists():
        return {}

    log_status = {}
    for log_dir in sensor_blobs_dir.iterdir():
        if not log_dir.is_dir():
            continue

        log_name = log_dir.name

        # Check for camera directories (CAM_F0, CAM_B0, etc.)
        has_camera = any((log_dir / f"CAM_{channel}").exists() for channel in ["F0", "B0", "L0", "R0", "L1", "L2", "R1", "R2"])

        # Check for lidar directory (MergedPointCloud)
        has_lidar = (log_dir / "MergedPointCloud").exists()

        log_status[log_name] = BlobStatus(
            log_name=log_name, has_camera=has_camera, has_lidar=has_lidar, camera_set=None if not has_camera else get_blob_set_for_log(log_name, "camera"), lidar_set=None if not has_lidar else get_blob_set_for_log(log_name, "lidar")
        )

    return log_status
