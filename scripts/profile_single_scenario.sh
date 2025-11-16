#!/bin/bash
# Profile a single scenario with ML planner
#
# NOTE: Edit CHECKPOINT path below to point to your trained model checkpoint
# Or find latest with: find $NUPLAN_EXP_ROOT -name "best_model.ckpt" -o -name "epoch*.ckpt" | sort | tail -1
# Or set ML_CHECKPOINT environment variable
#
CHECKPOINT="${ML_CHECKPOINT:-AUTO_DETECT}"  # Set ML_CHECKPOINT env var or edit this default

.venv/bin/python scripts/profile_simulation.py \
  experiment_name=profile_test \
  planner=ml_planner \
  planner.ml_planner.checkpoint_path="$CHECKPOINT" \
  +simulation=open_loop_boxes \
  scenario_builder=nuplan_mini \
  'scenario_filter.scenario_types=[near_multiple_vehicles]' \
  scenario_filter.num_scenarios_per_type=1 \
  worker=sequential
