#!/bin/bash
# ABOUTME: Profile a single scenario with ML planner to identify performance bottlenecks.
# ABOUTME: Uses best checkpoint and runs sequentially for clean profiling data.

CHECKPOINT="/home/garrett/nuplan/exp/tutorial_nuplan_framework/training_raster_experiment/train_default_raster/2025.11.14.06.09.43/best_model/best_model.ckpt"

.venv/bin/python scripts/profile_simulation.py \
  experiment_name=profile_test \
  planner=ml_planner \
  model=raster_model \
  planner.ml_planner.model_config=\${model} \
  planner.ml_planner.checkpoint_path="$CHECKPOINT" \
  +simulation=open_loop_boxes \
  scenario_builder=nuplan_mini \
  'scenario_filter.scenario_types=[near_multiple_vehicles]' \
  scenario_filter.num_scenarios_per_type=1 \
  worker=sequential
