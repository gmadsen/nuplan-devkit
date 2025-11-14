#!/usr/bin/env python

import typer

from nuplan.cli import db_cli, download_cli, explore_cli, inventory_cli, map_cli

# Construct main cli interface
cli = typer.Typer()

# Add database CLI
cli.add_typer(db_cli.cli, name="db")

# Add dataset management CLIs
cli.add_typer(download_cli.cli, name="download")
cli.add_typer(explore_cli.cli, name="explore")
cli.add_typer(inventory_cli.cli, name="inventory")
cli.add_typer(map_cli.cli, name="map")


def main() -> None:
    """
    Main entry point for the CLI
    """
    cli()


if __name__ == '__main__':
    main()
