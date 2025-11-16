#!/bin/bash
# ABOUTME: Profile a single scenario with simple planner to establish baseline performance.
# ABOUTME: Used for comparison against ML planner to isolate ML-specific overhead.

.venv/bin/python scripts/profile_simulation.py \
  experiment_name=profile_simple \
  planner=simple_planner \
  +simulation=open_loop_boxes \
  scenario_builder=nuplan_mini \
  'scenario_filter.scenario_types=[near_multiple_vehicles]' \
  scenario_filter.num_scenarios_per_type=1 \
  worker=sequential
