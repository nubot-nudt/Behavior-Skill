"""Evaluation result analysis tools (metrics).

Three hierarchical aggregation CLIs (pure statistics tools, no simulator
dependency)::

    benchmark/metrics/episode_summary.py      single episode, multiple runs → episode_summary.json
    benchmark/metrics/task_summary.py         single task, multiple episodes → task_summary.json
    benchmark/metrics/cross_task_summary.py   cross-task skill matrix → cross_task_skill_summary.json

Usage (either way)::

    # Option A: direct execution (zero extra dependencies, recommended for the stats stage)
    python benchmark/metrics/episode_summary.py --log_dir ...

    # Option B: module mode (requires an environment that can import the benchmark
    # package, i.e. with omnigibson installed)
    python -m benchmark.metrics.episode_summary --log_dir ...

Evaluation log directory layout (matching benchmark/scripts/run_auto_eval.sh)::

    {log_root}/{task}/episode_{ep}/run_XX/metrics/skill_YY_ep00.json   ← raw records
    {log_root}/{task}/episode_{ep}/episode_summary.json                ← from episode_summary
    {log_root}/{task}/task_summary.json                                ← from task_summary
    {log_root}/cross_task_skill_summary.json                           ← from cross_task_summary
"""
