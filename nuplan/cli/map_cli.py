# ABOUTME: Map scenarios and DB files to required sensor blob zip sets
# ABOUTME: Helps users determine which downloads are needed for specific logs

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nuplan.cli.dataset_utils import (
    get_blob_set_for_log,
    get_log_name_from_db_file,
    get_sensor_blob_zip_name,
)


cli = typer.Typer()
console = Console()

NUPLAN_DATA_ROOT = Path(os.getenv("NUPLAN_DATA_ROOT", "/data/sets/nuplan/"))


@cli.command()
def log(
    log_name: str = typer.Argument(..., help="Log name (e.g., 2021.05.12.22.00.38_veh-35_01008_01518)"),
    dataset: str = typer.Option("mini", help="Dataset type (mini or full)"),
) -> None:
    """
    Show which sensor blob zip sets contain a specific log.

    Example:
        nuplan_cli map log 2021.05.12.22.00.38_veh-35_01008_01518
    """
    try:
        # Determine which sets contain this log
        camera_set = get_blob_set_for_log(log_name, "camera")
        lidar_set = get_blob_set_for_log(log_name, "lidar")

        # Get zip file names
        camera_zip = get_sensor_blob_zip_name("camera", camera_set, dataset)  # type: ignore
        lidar_zip = get_sensor_blob_zip_name("lidar", lidar_set, dataset)  # type: ignore

        # Display results
        console.print(f"\n[bold]Log:[/bold] {log_name}")
        console.print(f"[bold]Dataset:[/bold] {dataset}\n")

        table = Table(title="Required Sensor Blob Zips")
        table.add_column("Type", style="cyan")
        table.add_column("Set Index", style="magenta")
        table.add_column("Zip File", style="green")
        table.add_column("Approx Size", style="yellow")

        # Estimated sizes (from bash script observations)
        camera_size = "~350 GB" if dataset == "mini" else "~1.5 TB"
        lidar_size = "~60 GB" if dataset == "mini" else "~250 GB"

        table.add_row("Camera", str(camera_set), camera_zip, camera_size)
        table.add_row("LiDAR", str(lidar_set), lidar_zip, lidar_size)

        console.print(table)

        # Show download command
        console.print("\n[bold]Download with:[/bold]")
        console.print(f"  nuplan_cli download {dataset} --camera={camera_set} --lidar={lidar_set}\n")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold red")
        raise typer.Exit(1)


@cli.command()
def db(
    db_files: list[Path] = typer.Argument(..., help="Path to DB file(s)"),
    summary: bool = typer.Option(False, help="Show summary of all required sets"),
    data_root: Path = typer.Option(NUPLAN_DATA_ROOT, help="Dataset root directory"),
) -> None:
    """
    Show which sensor blob zips are needed for DB file(s).

    Example:
        nuplan_cli map db /path/to/2021.05.12.22.00.38_veh-35_01008_01518.db
        nuplan_cli map db /path/to/splits/mini/*.db --summary
    """
    # Collect all logs from DB files
    logs = []
    for db_path in db_files:
        if not db_path.exists():
            console.print(f"[yellow]Warning:[/yellow] DB file not found: {db_path}")
            continue

        # Extract log name from DB file name (not from inside the DB)
        log_name = get_log_name_from_db_file(db_path)
        logs.append(log_name)

    if not logs:
        console.print("[red]Error:[/red] No valid DB files found")
        raise typer.Exit(1)

    if summary:
        # Show summary of all required sets
        camera_sets = set()
        lidar_sets = set()

        for log_name in logs:
            try:
                camera_sets.add(get_blob_set_for_log(log_name, "camera"))
                lidar_sets.add(get_blob_set_for_log(log_name, "lidar"))
            except ValueError:
                console.print(f"[yellow]Warning:[/yellow] Could not parse log name: {log_name}")
                continue

        console.print(f"\n[bold]Summary for {len(logs)} DB files:[/bold]\n")

        table = Table(title="Required Sensor Blob Sets")
        table.add_column("Type", style="cyan")
        table.add_column("Sets Needed", style="green")
        table.add_column("Download Command", style="yellow")

        camera_list = ",".join(map(str, sorted(camera_sets)))
        lidar_list = ",".join(map(str, sorted(lidar_sets)))

        table.add_row("Camera", camera_list, f"--camera={camera_list}")
        table.add_row("LiDAR", lidar_list, f"--lidar={lidar_list}")

        console.print(table)

        console.print("\n[bold]Download all with:[/bold]")
        console.print(f"  nuplan_cli download mini --camera={camera_list} --lidar={lidar_list}\n")

    else:
        # Show individual mappings
        table = Table(title=f"Sensor Blob Mapping for {len(logs)} Log(s)")
        table.add_column("Log Name", style="cyan")
        table.add_column("Camera Set", style="magenta")
        table.add_column("LiDAR Set", style="green")

        for log_name in logs:
            try:
                camera_set = get_blob_set_for_log(log_name, "camera")
                lidar_set = get_blob_set_for_log(log_name, "lidar")
                table.add_row(log_name, str(camera_set), str(lidar_set))
            except ValueError:
                table.add_row(log_name, "[red]Error[/red]", "[red]Error[/red]")

        console.print(table)
        console.print("\n[dim]Tip: Use --summary to see aggregate download commands[/dim]\n")


if __name__ == "__main__":
    cli()
