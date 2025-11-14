# ABOUTME: Dataset inventory management showing local vs remote files
# ABOUTME: Provides visibility into what's downloaded, what's available, and what's missing

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nuplan.cli.dataset_utils import (
    S3_MINI_SENSORS,
    format_bytes,
    scan_local_sensor_blobs,
    scrape_s3_bucket,
)


cli = typer.Typer()
console = Console()

NUPLAN_DATA_ROOT = Path(os.getenv("NUPLAN_DATA_ROOT", "/data/sets/nuplan/"))


@cli.command()
def main(
    sensors: bool = typer.Option(False, "--sensors", help="Show only sensor blob files"),
    missing: bool = typer.Option(False, "--missing", help="Show only missing files"),
    data_root: Path = typer.Option(NUPLAN_DATA_ROOT, help="Dataset root directory"),
) -> None:
    """
    Show dataset inventory with local vs remote comparison.

    Example:
        nuplan_cli inventory
        nuplan_cli inventory --sensors
        nuplan_cli inventory --missing
    """
    console.print("\n[bold]Scanning local dataset...[/bold]")

    # Scan local files
    local_files = scan_local_files(data_root)

    console.print(f"[green]✓[/green] Found {len(local_files)} local files\n")

    # Build comparison table
    table = Table(title="nuPlan Dataset Inventory")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Local", justify="center", style="green")
    table.add_column("Size", justify="right", style="yellow")

    # Core dataset files
    if not sensors:
        core_files = [
            ("nuplan-v1.1_mini.zip", "Database (mini)"),
            ("nuplan-maps-v1.1.zip", "Maps"),
        ]

        for filename, description in core_files:
            local = filename in local_files
            if missing and local:
                continue

            size = format_bytes(local_files[filename].st_size) if local else "-"
            status = "✓" if local else "✗"
            table.add_row(f"{description} ({filename})", status, size)

    # Sensor blob files
    sensor_files = [f for f in local_files if "camera_" in f or "lidar_" in f]

    if sensor_files or not missing:
        table.add_section()

        # Camera sets
        for i in range(8):
            filename = f"nuplan-v1.1_mini_camera_{i}.zip"
            local = filename in local_files

            if missing and local:
                continue

            size = format_bytes(local_files[filename].st_size) if local else "~350 GB"
            status = "✓" if local else "✗"
            table.add_row(f"Camera set {i}", status, size)

        # LiDAR sets
        for i in range(9):
            filename = f"nuplan-v1.1_mini_lidar_{i}.zip"
            local = filename in local_files

            if missing and local:
                continue

            size = format_bytes(local_files[filename].st_size) if local else "~60 GB"
            status = "✓" if local else "✗"
            table.add_row(f"LiDAR set {i}", status, size)

    console.print(table)

    # Summary statistics
    total_size = sum(f.st_size for f in local_files.values())
    local_count = len(local_files)

    console.print(f"\n[bold]Total Local:[/bold] {format_bytes(total_size)} ({local_count} files)")

    # Count missing sensor blobs
    missing_camera = sum(1 for i in range(8) if f"nuplan-v1.1_mini_camera_{i}.zip" not in local_files)
    missing_lidar = sum(1 for i in range(9) if f"nuplan-v1.1_mini_lidar_{i}.zip" not in local_files)

    if missing_camera > 0 or missing_lidar > 0:
        console.print(f"[yellow]Missing:[/yellow] {missing_camera} camera sets, {missing_lidar} lidar sets\n")
    else:
        console.print("[green]✓ All sensor blobs downloaded![/green]\n")


@cli.command()
def logs(
    missing: bool = typer.Option(False, "--missing", help="Show only logs missing sensor data"),
    from_date: str = typer.Option(None, "--from", help="Filter logs from date (YYYY-MM)"),
    to_date: str = typer.Option(None, "--to", help="Filter logs to date (YYYY-MM)"),
    data_root: Path = typer.Option(NUPLAN_DATA_ROOT, help="Dataset root directory"),
) -> None:
    """
    Show which logs have sensor blobs locally available.

    Example:
        nuplan_cli inventory logs
        nuplan_cli inventory logs --missing
        nuplan_cli inventory logs --from=2021-05 --to=2021-08
    """
    # Scan local sensor blobs
    log_status = scan_local_sensor_blobs(data_root)

    if not log_status:
        console.print("[yellow]No sensor blobs found locally[/yellow]")
        console.print(f"Expected location: {data_root / 'nuplan-v1.1' / 'sensor_blobs'}\n")
        return

    # Apply date filtering
    filtered_logs = log_status

    if from_date or to_date:
        filtered_logs = {}
        for log_name, status in log_status.items():
            # Extract date from log name (YYYY.MM.DD.HH.MM.SS_veh-XX_XXXXX_XXXXX)
            log_date = log_name[:7]  # YYYY.MM

            if from_date and log_date < from_date.replace("-", "."):
                continue
            if to_date and log_date > to_date.replace("-", "."):
                continue

            filtered_logs[log_name] = status

    # Apply missing filter
    if missing:
        filtered_logs = {name: status for name, status in filtered_logs.items() if not (status.has_camera and status.has_lidar)}

    if not filtered_logs:
        console.print("[green]No logs match the filters[/green]\n")
        return

    # Build table
    table = Table(title=f"Sensor Blob Status for {len(filtered_logs)} Log(s)")
    table.add_column("Log", style="cyan")
    table.add_column("Camera", justify="center", style="magenta")
    table.add_column("LiDAR", justify="center", style="green")

    for log_name in sorted(filtered_logs.keys()):
        status = filtered_logs[log_name]
        camera_status = "✓" if status.has_camera else "✗"
        lidar_status = "✓" if status.has_lidar else "✗"
        table.add_row(log_name, camera_status, lidar_status)

    console.print(table)

    # Summary
    with_camera = sum(1 for s in filtered_logs.values() if s.has_camera)
    with_lidar = sum(1 for s in filtered_logs.values() if s.has_lidar)

    console.print(f"\n[bold]Total:[/bold] {len(filtered_logs)} logs")
    console.print(f"[bold]With Camera:[/bold] {with_camera}")
    console.print(f"[bold]With LiDAR:[/bold] {with_lidar}\n")


def scan_local_files(data_root: Path) -> dict[str, os.stat_result]:
    """
    Scan data_root for existing dataset files.

    Returns:
        Dict mapping filenames to stat results
    """
    local_files = {}

    # Check for core files in data_root
    for file in data_root.glob("*.zip"):
        if file.is_file():
            local_files[file.name] = file.stat()

    return local_files


if __name__ == "__main__":
    cli()
