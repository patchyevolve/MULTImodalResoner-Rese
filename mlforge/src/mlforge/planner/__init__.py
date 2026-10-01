"""Capability-based execution planner — build step 9 (pending).

Normative: 12_training_system.md §13.

Feasibility solver: find micro_batch × grad_accum × world_size ==
frozen global_batch subject to VRAM/precision/model-min constraints.
Hard namespace separation: execution → semantic mutation is forbidden;
adapting micro/accum/workers is free, LR/optimizer/loss/dataset are not.
"""
