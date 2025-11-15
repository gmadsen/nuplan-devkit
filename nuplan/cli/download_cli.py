# ABOUTME: Flexible download helper CLI for nuPlan datasets with configurable extraction
# ABOUTME: Supports interactive selection, batch flags, and multiple decompression tools

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from nuplan.cli.dataset_utils import (
    DATASET_SPLITS,
    S3_BASE_URL,
    SENSOR_BLOB_COUNTS,
    get_sensor_blob_zip_name,
)


cli = typer.Typer()
console = Console()

NUPLAN_DATA_ROOT = Path(os.getenv("NUPLAN_DATA_ROOT", "/data/sets/nuplan/"))


def get_all_database_files() -> list[str]:
    """Get list of all database zip files (test, val, train_*)."""
    files = []

    # Add test
    files.append("nuplan-v1.1_test.zip")

    # Add val
    files.append("nuplan-v1.1_val.zip")

    # Add all train splits
    for split in DATASET_SPLITS["train"]:
        files.append(f"nuplan-v1.1_{split}.zip")

    return files


def check_file_exists(data_root: Path, filename: str) -> bool:
    """Check if a file already exists (downloaded or extracted)."""
    zip_path = data_root / filename

    # Check if zip exists
    if zip_path.exists():
        return True

    # Check if already extracted (for DB files)
    if filename.endswith(".zip"):
        split_name = filename.replace("nuplan-v1.1_", "").replace(".zip", "")
        extracted_dir = data_root / "nuplan-v1.1" / "splits" / split_name
        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            return True

    return False


def estimate_size(filename: str) -> int:
    """Estimate file size in GB (rough estimates)."""
    if "mini" in filename and "camera" in filename:
        return 350
    elif "mini" in filename and "lidar" in filename:
        return 60
    elif "mini.zip" in filename:
        return 8
    elif "maps" in filename:
        return 1
    elif "test" in filename:
        return 10  # Rough estimate
    elif "val" in filename:
        return 15  # Rough estimate
    elif "train" in filename:
        return 30  # Rough estimate per train split
    else:
        return 5  # Default estimate


@cli.command()
def datasets(
    # Database selection flags
    mini: bool = typer.Option(False, help="Include mini dataset"),
    test: bool = typer.Option(False, help="Include test dataset"),
    val: bool = typer.Option(False, help="Include validation dataset"),
    all_train: bool = typer.Option(False, help="Include all training datasets"),
    all_db: bool = typer.Option(False, help="Include all database files (test, val, all train)"),

    # Individual dataset selection
    datasets_list: str = typer.Option(None, help="Comma-separated list of datasets (e.g., 'test,val,train_boston')"),

    # Sensor blob selection (for mini dataset)
    camera: str = typer.Option(None, help="Camera sets to download (e.g., '0,1,2' or '0')"),
    lidar: str = typer.Option(None, help="LiDAR sets to download (e.g., '0,1' or '0')"),

    # Maps
    skip_maps: bool = typer.Option(False, help="Skip map files"),

    # Extraction configuration
    extract: bool = typer.Option(False, help="Extract files after download (default: no)"),
    extract_tool: Literal["unzip", "7z", "pigz"] = typer.Option("7z", help="Extraction tool to use"),
    extract_parallel: int = typer.Option(4, help="Number of parallel extraction jobs"),

    # General options
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode with selection prompts"),
    data_root: Path = typer.Option(NUPLAN_DATA_ROOT, help="Dataset root directory"),
) -> None:
    """
    Download nuPlan datasets with flexible selection and extraction options.

    Examples:
        # Interactive mode
        nuplan_cli download datasets --interactive

        # All database files (no sensors)
        nuplan_cli download datasets --all-db

        # Test + validation only
        nuplan_cli download datasets --test --val

        # Specific datasets
        nuplan_cli download datasets --datasets=test,val,train_boston

        # Mini with sensors (tutorial setup)
        nuplan_cli download datasets --mini --camera=0 --lidar=0

        # With extraction enabled
        nuplan_cli download datasets --all-db --extract --extract-tool=7z
    """
    console.print("\n[bold cyan]nuPlan Dataset Download Manager[/bold cyan]\n")

    # Build initial selection
    selected_files = []

    # Handle flags
    if all_db:
        test = True
        val = True
        all_train = True

    if mini:
        selected_files.append("nuplan-v1.1_mini.zip")

    if test:
        selected_files.append("nuplan-v1.1_test.zip")

    if val:
        selected_files.append("nuplan-v1.1_val.zip")

    if all_train:
        for split in DATASET_SPLITS["train"]:
            selected_files.append(f"nuplan-v1.1_{split}.zip")

    # Handle datasets_list
    if datasets_list:
        for dataset in datasets_list.split(","):
            dataset = dataset.strip()
            # Handle both "test" and "nuplan-v1.1_test.zip" formats
            if not dataset.startswith("nuplan-v1.1_"):
                dataset = f"nuplan-v1.1_{dataset}"
            if not dataset.endswith(".zip"):
                dataset = f"{dataset}.zip"
            if dataset not in selected_files:
                selected_files.append(dataset)

    # Add maps
    if not skip_maps:
        selected_files.append("nuplan-maps-v1.1.zip")

    # Handle sensor blobs for mini
    if camera or lidar:
        camera_sets = [int(x.strip()) for x in camera.split(",")] if camera else []
        lidar_sets = [int(x.strip()) for x in lidar.split(",")] if lidar else []

        # Validate indices
        for i in camera_sets:
            if not (0 <= i <= 8):
                console.print(f"[red]Error:[/red] Camera set must be 0-8, got {i}")
                raise typer.Exit(1)

        for i in lidar_sets:
            if not (0 <= i <= 8):
                console.print(f"[red]Error:[/red] LiDAR set must be 0-8, got {i}")
                raise typer.Exit(1)

        # Add sensor blob files
        for i in camera_sets:
            zip_name = get_sensor_blob_zip_name("camera", i, "mini")
            selected_files.append(zip_name)

        for i in lidar_sets:
            zip_name = get_sensor_blob_zip_name("lidar", i, "mini")
            selected_files.append(zip_name)

    # Interactive mode
    if interactive and not selected_files:
        console.print("[bold]Available datasets:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Dataset", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Est. Size", justify="right")

        all_datasets = (
            ["nuplan-v1.1_mini.zip"]
            + ["nuplan-v1.1_test.zip", "nuplan-v1.1_val.zip"]
            + [f"nuplan-v1.1_{s}.zip" for s in DATASET_SPLITS["train"]]
            + ["nuplan-maps-v1.1.zip"]
        )

        for dataset in all_datasets:
            exists = check_file_exists(data_root, dataset)
            status = "✅ Downloaded" if exists else "❌ Missing"
            size = f"~{estimate_size(dataset)} GB"
            table.add_row(dataset, status, size)

        console.print(table)
        console.print("\n[dim]Note: For sensor blobs, use --camera and --lidar flags[/dim]\n")

        # Prompt for selection
        selection = Prompt.ask(
            "[bold]Enter datasets to download[/bold] (comma-separated, or 'all-db' for all databases)",
            default="all-db"
        )

        if selection == "all-db":
            selected_files = get_all_database_files()
            selected_files.append("nuplan-maps-v1.1.zip")
        else:
            for dataset in selection.split(","):
                dataset = dataset.strip()
                if not dataset.startswith("nuplan-v1.1_"):
                    dataset = f"nuplan-v1.1_{dataset}"
                if not dataset.endswith(".zip"):
                    dataset = f"{dataset}.zip"
                selected_files.append(dataset)

    # If nothing selected, show help
    if not selected_files:
        console.print("[yellow]No datasets selected.[/yellow]")
        console.print("Use flags like --all-db, --test, --val, or --interactive\n")
        console.print("Examples:")
        console.print("  nuplan_cli download datasets --all-db")
        console.print("  nuplan_cli download datasets --test --val")
        console.print("  nuplan_cli download datasets --interactive\n")
        return

    # Remove duplicates and filter already downloaded
    selected_files = list(dict.fromkeys(selected_files))  # Remove duplicates

    # Show what's already downloaded
    already_downloaded = [f for f in selected_files if check_file_exists(data_root, f)]
    to_download = [f for f in selected_files if not check_file_exists(data_root, f)]

    if already_downloaded:
        console.print(f"[green]Already downloaded ({len(already_downloaded)}):[/green]")
        for f in already_downloaded:
            console.print(f"  ✅ {f}")
        console.print()

    if not to_download:
        console.print("[green]All selected files are already downloaded![/green]\n")
        return

    # Calculate total size
    total_size_gb = sum(estimate_size(f) for f in to_download)

    # Display summary
    console.print(f"[bold]Files to download:[/bold] {len(to_download)}")
    for f in to_download:
        console.print(f"  📥 {f}")
    console.print(f"\n[bold]Estimated total size:[/bold] ~{total_size_gb} GB")
    console.print(f"[bold]Destination:[/bold] {data_root}\n")

    # Build download URLs
    urls = []
    for filename in to_download:
        if "sensor_blobs" in filename or filename in [f"nuplan-v1.1_mini_{t}_{i}.zip"
                                                       for t in ["camera", "lidar"]
                                                       for i in range(9)]:
            # Sensor blob URL
            if "camera" in filename:
                urls.append(f"{S3_BASE_URL}/sensor_blobs/mini_set/{filename}")
            elif "lidar" in filename:
                urls.append(f"{S3_BASE_URL}/sensor_blobs/mini_set/{filename}")
        else:
            # Database or map URL
            urls.append(f"{S3_BASE_URL}/{filename}")

    # Generate wget script
    wget_script = f"""#!/bin/bash
# Generated by nuplan_cli download datasets
# Destination: {data_root}
# Date: $(date)

set -euo pipefail

cd "{data_root}"

echo "Downloading {len(urls)} files (~{total_size_gb} GB)..."
echo ""

# Download files with wget
"""

    for url in urls:
        filename = url.split("/")[-1]
        wget_script += f'echo "📥 Downloading {filename}..."\n'
        wget_script += f'wget -c "{url}" -O "{filename}"\n\n'

    wget_script += 'echo ""\n'
    wget_script += 'echo "✅ All downloads complete!"\n'

    # Add extraction if enabled
    if extract:
        wget_script += f'\necho ""\necho "📦 Extracting archives with {extract_tool}..."\n\n'

        if extract_tool == "7z":
            wget_script += f"""# Extract with 7z (fast, parallel)
for zip in *.zip; do
    echo "Extracting $zip..."
    7z x -y "$zip" || echo "Warning: Failed to extract $zip"
done
"""
        elif extract_tool == "pigz":
            wget_script += f"""# Extract with pigz (parallel gzip)
for zip in *.zip; do
    echo "Extracting $zip..."
    pigz -dc "$zip" | tar -xf - || unzip -n "$zip" || echo "Warning: Failed to extract $zip"
done
"""
        else:  # unzip
            wget_script += f"""# Extract with unzip
for zip in *.zip; do
    echo "Extracting $zip..."
    unzip -n "$zip" || echo "Warning: Failed to extract $zip"
done
"""

        wget_script += '\necho "✅ Extraction complete!"\n'

    wget_script += '\necho ""\necho "🎉 Setup complete!"\n'

    # Generate aria2c script (faster, recommended)
    aria2c_script = f"""#!/bin/bash
# Generated by nuplan_cli download datasets (aria2c - faster!)
# Destination: {data_root}
# Date: $(date)

set -euo pipefail

cd "{data_root}"

echo "Downloading {len(urls)} files (~{total_size_gb} GB) with aria2c..."
echo ""

# Create URL list file
cat > nuplan_download_urls.txt <<'EOF'
"""

    for url in urls:
        aria2c_script += f"{url}\n"

    aria2c_script += """EOF

# Download with aria2c (parallel, resume-capable)
aria2c \\
  --input-file=nuplan_download_urls.txt \\
  --continue=true \\
  --max-connection-per-server=16 \\
  --split=16 \\
  --min-split-size=4M \\
  --max-concurrent-downloads=2 \\
  --file-allocation=falloc \\
  --summary-interval=10

echo ""
echo "✅ All downloads complete!"
"""

    # Add extraction if enabled
    if extract:
        aria2c_script += f'\necho ""\necho "📦 Extracting archives with {extract_tool}..."\n\n'

        if extract_tool == "7z":
            aria2c_script += f"""# Extract with 7z (fast, parallel)
# Extract {extract_parallel} files in parallel
ls *.zip | xargs -P {extract_parallel} -I {{}} sh -c '
    echo "Extracting {{}}..."
    7z x -y "{{}}" || echo "Warning: Failed to extract {{}}"
'
"""
        elif extract_tool == "pigz":
            aria2c_script += f"""# Extract with pigz (parallel gzip)
# Extract {extract_parallel} files in parallel
ls *.zip | xargs -P {extract_parallel} -I {{}} sh -c '
    echo "Extracting {{}}..."
    pigz -dc "{{}}" | tar -xf - || unzip -n "{{}}" || echo "Warning: Failed to extract {{}}"
'
"""
        else:  # unzip
            aria2c_script += f"""# Extract with unzip
# Extract {extract_parallel} files in parallel
ls *.zip | xargs -P {extract_parallel} -I {{}} sh -c '
    echo "Extracting {{}}..."
    unzip -n "{{}}" || echo "Warning: Failed to extract {{}}"
'
"""

        aria2c_script += '\necho "✅ Extraction complete!"\n'

    aria2c_script += '\necho ""\necho "🎉 Setup complete!"\n'

    # Clean up URL list
    if not extract:
        aria2c_script += 'rm -f nuplan_download_urls.txt\n'

    # Display scripts
    console.print(Panel(
        Syntax(wget_script, "bash", theme="monokai"),
        title="[bold green]Option 1: wget (simple, reliable)[/bold green]",
        border_style="green"
    ))

    console.print()

    console.print(Panel(
        Syntax(aria2c_script, "bash", theme="monokai"),
        title="[bold cyan]Option 2: aria2c (faster, recommended)[/bold cyan]",
        border_style="cyan"
    ))

    # Save scripts to files
    wget_file = data_root / "download_nuplan_wget.sh"
    aria2c_file = data_root / "download_nuplan_aria2c.sh"

    with open(wget_file, "w") as f:
        f.write(wget_script)
    wget_file.chmod(0o755)

    with open(aria2c_file, "w") as f:
        f.write(aria2c_script)
    aria2c_file.chmod(0o755)

    console.print(f"\n[bold]Scripts saved:[/bold]")
    console.print(f"  wget:   {wget_file}")
    console.print(f"  aria2c: {aria2c_file}")

    console.print("\n[bold]To download:[/bold]")
    console.print(f"  {aria2c_file}")

    if extract:
        console.print(f"\n[bold yellow]⚠️  Auto-extraction enabled with {extract_tool}[/bold yellow]")
        console.print(f"   Parallel jobs: {extract_parallel}")
        console.print("   This may take several hours depending on file sizes.")
    else:
        console.print("\n[dim]💡 Tip: Use --extract to auto-extract after download[/dim]")
        console.print("[dim]   Extraction tools: --extract-tool={unzip,7z,pigz}[/dim]")

    console.print("\n[dim]💡 Tip: aria2c is much faster for large files[/dim]")
    console.print("[dim]   Install: sudo apt install aria2 p7zip-full[/dim]\n")


if __name__ == "__main__":
    cli()
