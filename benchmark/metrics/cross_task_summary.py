#!/usr/bin/env python3
"""
cross_task_summary.py — cross-task skill success rates and per-task TSCR

Given a set of tasks, aggregate the success rates of each skill type (e.g.
move to, pick up from, place in) across all tasks, reporting each skill's
results per task plus the overall cross-task success rate, and summarize each
task's TSCR.

Outputs:
  - cross-task success rate matrix (rows = skill, columns = tasks + Overall)
  - overall success rate ranking (sorted by SR, descending)
  - per-task TSCR summary
  - {log_root_prefix}/cross_task_skill_summary.json

Usage (from the repo root):
  python -m benchmark.metrics.cross_task_summary \
      --tasks task_a,task_b \
      --log_root_prefix output/eval \
      [--configs_root_prefix data/skill_eval_configs] \
      [--group_by skill_description,type] \\     # aggregate by skill_description,type
      [--regen]                    # rebuild each episode_summary.json from raw records first
      [--num_runs N]               # only effective with --regen; default: auto-scan run_XX dirs

Notes:
  - Data source: {log_root_prefix}/{task}/episode_*/episode_summary.json;
    if the episode_summary.json files are stale after a re-run, pass --regen to
    rebuild them from the raw records (run_XX/metrics/*.json) before aggregating.
  - Per-episode STSR is preferably rebuilt from per_run.skill_results + YAML
    metadata (most reliable); when the YAML is unavailable it falls back to the
    prebuilt stsr_by_{field} inside episode_summary.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

try:
    from .episode_summary import aggregate_episode, load_skill_meta
    from .task_summary import merge_stsr
except ImportError:  # direct script invocation (python benchmark/metrics/cross_task_summary.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from episode_summary import aggregate_episode, load_skill_meta
    from task_summary import merge_stsr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_ep_stsr(summary: dict, field: str, task: str, ep_id: str,
                configs_root: str) -> dict:
    """Get the single episode's STSR grouped by the given field.

    Prefer rebuilding from per_run.skill_results + YAML metadata; when the
    YAML is unavailable, fall back to the prebuilt stsr_by_{field} inside
    episode_summary.
    """
    yaml_path = (
        os.path.join(configs_root, task, f"episode_{ep_id}.yaml")
        if configs_root else ""
    )
    skill_meta = load_skill_meta(yaml_path)
    if skill_meta:
        cnt, succ = {}, {}
        for run_data in summary.get("per_run", []):
            for r in run_data.get("skill_results", []):
                val = skill_meta.get(r["skill_id"], {}).get(
                    field, f"<unknown:{field}>"
                )
                cnt[val] = cnt.get(val, 0) + 1
                if r["success"]:
                    succ[val] = succ.get(val, 0) + 1
        return {v: {"n": cnt[v], "n_success": succ.get(v, 0)} for v in cnt}

    prebuilt = summary.get(f"stsr_by_{field}")
    return prebuilt if prebuilt else {}


def pct(ns, n):
    return "-" if not n else f"{100.0 * ns / n:.1f}%"


def cell(ns, n):
    return f"{ns}/{n} ({pct(ns, n)})" if n else "-"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-task skill success rate summary"
    )
    parser.add_argument("--tasks", required=True,
                        help="comma-separated list of task names")
    parser.add_argument("--log_root_prefix", default="output/eval",
                        help="parent dir of per-task log dirs (each subdirectory name = task name)")
    parser.add_argument("--configs_root_prefix",
                        default=str(PROJECT_ROOT / "data" / "skill_eval_configs"),
                        help="parent dir of episode YAMLs ({task}/episode_*.yaml below)")
    parser.add_argument("--group_by", default="skill_description,type",
                        help="aggregation dimensions, comma-separated (default skill_description,type)")
    parser.add_argument("--regen", action="store_true",
                        help="rebuild each episode's episode_summary.json from raw records first")
    parser.add_argument("--num_runs", type=int, default=0,
                        help="only effective with --regen; default 0 = auto-scan the number of run_XX dirs")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    fields = [f.strip() for f in args.group_by.split(",") if f.strip()] \
        or ["skill_description"]

    print("=" * 60)
    print("Cross-task skill success rate summary")
    print(f"  log_root_prefix:   {args.log_root_prefix}")
    print(f"  configs_root:      {args.configs_root_prefix}")
    print(f"  group_by:          {args.group_by}")
    print(f"  tasks:             {len(tasks)}  ({','.join(tasks)})")
    if args.regen:
        print(f"  regen:             true (num_runs={args.num_runs or 'auto'})")
    print("=" * 60)

    # ── Stage 1 (optional): rebuild episode_summary.json from raw records ───
    if args.regen:
        print("\n── Stage 1: rebuild episode_summary ──────────────────────────")
        ok = fail = 0
        for task in tasks:
            task_root = os.path.join(args.log_root_prefix, task)
            for ep_dir in sorted(glob.glob(os.path.join(task_root, "episode_*"))):
                if not os.path.isdir(ep_dir):
                    continue
                if not glob.glob(os.path.join(ep_dir, "run_*")):
                    continue
                ep_id = os.path.basename(ep_dir)[len("episode_"):]
                eval_config = os.path.join(
                    args.configs_root_prefix, task, f"episode_{ep_id}.yaml"
                )
                summary = aggregate_episode(
                    log_dir=ep_dir,
                    episode=ep_id,
                    num_runs=args.num_runs,
                    eval_config=eval_config if os.path.isfile(eval_config) else "",
                    group_fields=fields,
                )
                if summary is not None:
                    ok += 1
                else:
                    print(f"[FAIL] {task}/episode_{ep_id}")
                    fail += 1
        print(f"Stage 1 done: {ok} succeeded / {fail} failed")

    # ── Stage 2: cross-task aggregation of per-skill success rates ──────────
    print("\n── Stage 2: cross-task skill success rate aggregation ──────────────────────────")

    task_stsr = {}    # task -> {field -> {label -> {n, n_success, SR}}}
    task_tcr = {}     # task -> mean TSCR (float | None)
    task_eps = {}     # task -> n_episodes
    all_labels = {f: set() for f in fields}

    for task in tasks:
        root = os.path.join(args.log_root_prefix, task)
        ep_paths = sorted(glob.glob(
            os.path.join(root, "episode_*", "episode_summary.json")
        ))
        if not ep_paths:
            print(f"[WARN] no episode_summary.json found for task '{task}' "
                  f"(searched: {root})")
            task_eps[task] = 0
            task_stsr[task] = {f: {} for f in fields}
            task_tcr[task] = None
            continue

        summaries = []
        for p in ep_paths:
            try:
                with open(p, encoding="utf-8") as f:
                    summaries.append(json.load(f))
            except Exception as e:
                print(f"[WARN] failed to read {p}: {e}")
        task_eps[task] = len(summaries)

        # aggregate per field
        task_stsr[task] = {}
        for f in fields:
            stsr_list = []
            for s in summaries:
                stsr = get_ep_stsr(s, f, task, s.get("episode", ""),
                                   args.configs_root_prefix)
                if stsr:
                    stsr_list.append(stsr)
            merged = merge_stsr(stsr_list)
            task_stsr[task][f] = merged
            all_labels[f].update(merged.keys())

        # per-task TSCR
        tcrs = []
        for s in summaries:
            t = s.get("TSCR", {})
            tm = t.get("mean") if isinstance(t, dict) else t
            if isinstance(tm, (int, float)):
                tcrs.append(float(tm))
        task_tcr[task] = (sum(tcrs) / len(tcrs)) if tcrs else None

    # overall aggregation
    overall_stsr = {}
    for f in fields:
        overall_stsr[f] = merge_stsr(
            [task_stsr[t][f] for t in tasks if task_stsr.get(t, {}).get(f)]
        )

    # ── Print matrix ────────────────────────────────────────────────────────
    for f in fields:
        labels = sorted(all_labels[f])
        print()
        print("=" * 80)
        print(f"  Cross-task skill success rate matrix  (group_by: {f})")
        print("=" * 80)
        if not labels:
            print("  [no data]")
            if f == "skill_description":
                print("  Hint: make sure --configs_root_prefix points to a valid "
                      "skill_eval_configs directory,")
                print("        or that episode_summary was generated with "
                      "--group_by skill_description")
            continue

        header = ["skill"] + tasks + ["Overall"]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for label in labels:
            row = [label]
            tot_n = tot_s = 0
            for t in tasks:
                v = task_stsr.get(t, {}).get(f, {}).get(label)
                if v:
                    row.append(cell(v["n_success"], v["n"]))
                    tot_n += v["n"]
                    tot_s += v["n_success"]
                else:
                    row.append("-")
            row.append(cell(tot_s, tot_n) if tot_n else "-")
            print("| " + " | ".join(row) + " |")

    # ── Print overall success rate ranking ────────────────────────────────────
    for f in fields:
        ov = overall_stsr.get(f, {})
        if not ov:
            continue
        print()
        print(f"── Overall success rate ranking (group_by: {f}) ──────────────────────────")
        print(f"  {'skill':<24} {'succ':>6} / {'total':<6}  {'SR':>7}")
        print("  " + "-" * 52)
        for label in sorted(ov, key=lambda k: ov[k]["SR"], reverse=True):
            v = ov[label]
            print(f"  {label:<24} {v['n_success']:>6} / {v['n']:<6}  "
                  f"{v['SR']*100:>6.1f}%")

    # ── Print per-task TSCR ────────────────────────────────────────────────────────
    print()
    print("── Per-task TSCR summary ────────────────────────────────────────")
    print(f"  {'task':<32} {'episodes':>8}  {'mean_TSCR':>8}")
    print("  " + "-" * 54)
    for t in tasks:
        tcr = task_tcr.get(t)
        tcr_str = f"{tcr:.4f}" if tcr is not None else "-"
        print(f"  {t:<32} {task_eps.get(t, 0):>8}  {tcr_str:>8}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "tasks": tasks,
        "group_by": fields,
        "per_task": {
            t: {
                "n_episodes": task_eps.get(t, 0),
                "TSCR": task_tcr.get(t),
                "stsr": task_stsr.get(t, {}),
            }
            for t in tasks
        },
        "overall": overall_stsr,
    }
    out_path = os.path.join(args.log_root_prefix, "cross_task_skill_summary.json")
    try:
        os.makedirs(args.log_root_prefix, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  → {out_path}")
    except Exception as e:
        print(f"[WARN] failed to save JSON: {e}")


if __name__ == "__main__":
    main()
