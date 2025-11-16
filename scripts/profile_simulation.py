#!/usr/bin/env python3
"""
ABOUTME: Profile nuPlan simulation with cProfile to identify performance bottlenecks.
ABOUTME: Usage: .venv/bin/python scripts/profile_simulation.py [args for run_simulation.py]
"""
import cProfile
import pstats
import sys
import os
from pathlib import Path

# Add nuplan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Profile the simulation
profiler = cProfile.Profile()
profiler.enable()

# Import and run simulation
from nuplan.planning.script.run_simulation import main
main()

profiler.disable()

# Save stats
output_dir = Path("profiling_output")
output_dir.mkdir(exist_ok=True)
stats_file = output_dir / "simulation_profile.stats"
profiler.dump_stats(str(stats_file))

# Print summary
print(f"\n{'='*80}")
print(f"Profile saved to: {stats_file}")
print(f"{'='*80}\n")

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
print("\nTop 50 functions by cumulative time:")
stats.print_stats(50)

print("\n" + "="*80)
print("Top 50 functions by total time (excluding subcalls):")
print("="*80)
stats.sort_stats('tottime')
stats.print_stats(50)
