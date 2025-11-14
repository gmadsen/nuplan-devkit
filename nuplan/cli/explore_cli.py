# ABOUTME: Explore nuPlan S3 bucket structure to discover available datasets
# ABOUTME: Helps understand the full breadth of training/val/test splits

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from nuplan.cli.dataset_utils import (
    DATASET_SPLITS,
    SENSOR_BLOB_COUNTS,
    S3_BASE_URL,
    S3_MINI_SENSORS,
    S3_TRAIN_SENSORS,
    S3_VAL_SENSORS,
    S3_TEST_SENSORS,
    scrape_s3_bucket,
)


cli = typer.Typer()
console = Console()


@cli.command()
def datasets() -> None:
    """
    Show all available dataset splits in nuPlan v1.1.

    Example:
        nuplan_cli explore datasets
    """
    console.print("\n[bold cyan]nuPlan v1.1 Dataset Structure[/bold cyan]\n")

    # Full dataset table
    table = Table(title="Available Dataset Splits")
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Files", style="green")
    table.add_column("Purpose", style="yellow")
    table.add_column("Approx Size", style="magenta")

    # Mini dataset
    table.add_row(
        "Mini",
        "nuplan-v1.1_mini.zip",
        "Tutorials & quick testing",
        "~8 GB (DB only)\n+ sensor blobs (camera_0..7, lidar_0..8)"
    )

    table.add_section()

    # Test split
    table.add_row(
        "Test",
        "nuplan-v1.1_test.zip",
        "Final evaluation",
        "~??? GB"
    )

    # Validation split
    table.add_row(
        "Validation",
        "nuplan-v1.1_val.zip",
        "Hyperparameter tuning",
        "~??? GB"
    )

    table.add_section()

    # Training splits
    train_files = "\n".join([
        "nuplan-v1.1_train_boston.zip",
        "nuplan-v1.1_train_pittsburgh.zip",
        "nuplan-v1.1_train_singapore.zip",
        "nuplan-v1.1_train_vegas_1.zip",
        "nuplan-v1.1_train_vegas_2.zip",
        "nuplan-v1.1_train_vegas_3.zip",
        "nuplan-v1.1_train_vegas_4.zip",
        "nuplan-v1.1_train_vegas_5.zip",
        "nuplan-v1.1_train_vegas_6.zip",
    ])

    table.add_row(
        "Training",
        train_files,
        "Model training\n(9 city/region splits)",
        "~??? GB total\n(~10TB with sensors)"
    )

    console.print(table)

    # Sensor blob organization
    console.print("\n[bold]Sensor Blob Organization:[/bold]")
    console.print("  [dim]Folders containing numbered sensor blob zip collections:[/dim]")
    console.print("  [dim](NOT organized by city - numbered sequentially instead!)[/dim]")
    console.print()

    for set_name, counts in SENSOR_BLOB_COUNTS.items():
        camera_range = f"0..{counts['camera']-1}"
        lidar_range = f"0..{counts['lidar']-1}"
        total_files = counts['camera'] + counts['lidar']

        console.print(f"  📁 [cyan]sensor_blobs/{set_name}_set/[/cyan]")
        console.print(f"     ├─ {counts['camera']} camera zips: nuplan-v1.1_{set_name}_camera_{{{camera_range}}}.zip")
        console.print(f"     ├─ {counts['lidar']} lidar zips: nuplan-v1.1_{set_name}_lidar_{{{lidar_range}}}.zip")
        console.print(f"     └─ Total: {total_files} sensor zips + 1 txt file (public_set_{set_name}_sensor.txt)")
        console.print()

    # URLs
    console.print(f"\n[bold]Base URL:[/bold] {S3_BASE_URL}")

    console.print("\n[bold]Cities Covered:[/bold]")
    console.print("  🌆 Boston (US-MA)")
    console.print("  🌆 Pittsburgh (US-PA)")
    console.print("  🏙️  Las Vegas (US-NV) - 6 splits")
    console.print("  🌏 Singapore")

    console.print("\n[dim]Note: Use 'nuplan_cli explore sensors' to scrape S3 and see actual zip files[/dim]\n")


@cli.command()
def sensors(
    sensor_set: str = typer.Option("mini", help="Sensor set to explore (mini, train, val, test)")
) -> None:
    """
    Scrape S3 bucket to show available sensor blob zips for a specific set.

    Example:
        nuplan_cli explore sensors --sensor-set=mini
        nuplan_cli explore sensors --sensor-set=train
    """
    # Map sensor set to URL
    sensor_urls = {
        "mini": S3_MINI_SENSORS,
        "train": S3_TRAIN_SENSORS,
        "val": S3_VAL_SENSORS,
        "test": S3_TEST_SENSORS,
    }

    if sensor_set not in sensor_urls:
        console.print(f"[red]Error:[/red] Unknown sensor set '{sensor_set}'")
        console.print("Valid options: mini, train, val, test")
        raise typer.Exit(1)

    url = sensor_urls[sensor_set]

    console.print(f"\n[bold cyan]Scraping S3 Bucket:[/bold cyan] {url}\n")

    # Scrape with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching file list from S3...", total=None)

        try:
            files = scrape_s3_bucket(url, pattern=r'\.zip$')
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error scraping S3:[/red] {e}")
            console.print("\n[yellow]Note:[/yellow] S3 bucket may not allow directory listing")
            console.print("Try checking the nuPlan website or documentation for file lists\n")
            raise typer.Exit(1)

    if not files:
        console.print("[yellow]No zip files found or directory listing not allowed[/yellow]\n")
        return

    # Parse filenames to organize by type
    camera_files = [f for f in files if "camera" in f]
    lidar_files = [f for f in files if "lidar" in f]
    other_files = [f for f in files if "camera" not in f and "lidar" not in f]

    # Display results
    console.print(f"[bold]Found {len(files)} zip files in {sensor_set}_set:[/bold]\n")

    if camera_files:
        console.print("[bold cyan]Camera Zips:[/bold cyan]")
        for f in sorted(camera_files):
            filename = f.split("/")[-1]
            console.print(f"  • {filename}")
        console.print()

    if lidar_files:
        console.print("[bold green]LiDAR Zips:[/bold green]")
        for f in sorted(lidar_files):
            filename = f.split("/")[-1]
            console.print(f"  • {filename}")
        console.print()

    if other_files:
        console.print("[bold yellow]Other Zips:[/bold yellow]")
        for f in sorted(other_files):
            filename = f.split("/")[-1]
            console.print(f"  • {filename}")
        console.print()

    console.print(f"[bold]Total:[/bold] {len(files)} files")
    console.print(f"[bold]Download from:[/bold] {url}\n")


@cli.command()
def splits() -> None:
    """
    Show the dataset split configuration.

    Example:
        nuplan_cli explore splits
    """
    console.print("\n[bold cyan]Dataset Split Configuration[/bold cyan]\n")

    for split_name, files in DATASET_SPLITS.items():
        console.print(f"[bold]{split_name.upper()}:[/bold]")
        for file in files:
            console.print(f"  • {file}")
        console.print()


if __name__ == "__main__":
    cli()
