"""
skill_eval.py — standalone sub-task (skill) evaluation CLI entry point

Drives the benchmark package modules for standalone evaluation:
    core/            evaluation core domain (main() here only orchestrates)
        evaluator.py     SkillEvaluator + run_batch
        suite.py         eval_config YAML / state_manifest.json loading + prompt override table
        skill_bddl.py    BDDL construction, parsing and wildcard expansion
        observation.py   policy observation preprocessing + eval env config generation
    policies/        websocket remote / hydra local policy integration
    states/          skill initial state loading and post-restore fixes
    utils/           video frame extraction and saving (video), diagnostic logging (diagnostics)

Usage:
  python -m benchmark.skill_eval \\
      --task setting_the_fire \\
      --eval_config skill_eval_configs/setting_the_fire.yaml \\
      --manifest skill_init_states/task-xxxx/state_manifest.json \\
      --websocket_port 8001 \\
      --n_episodes 1

  # evaluate only the specified skills
  python -m benchmark.skill_eval --task setting_the_fire --skill_id 2 9 14 ...
"""

import argparse
import json
import logging
import os
import sys
import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from pathlib import Path
from .core.evaluator import run_batch
from .core.observation import build_env_cfg
from .policies.hydra_policy import load_hydra_policy
from .policies.websocket import WebsocketPolicyWrapper
from .core.suite import build_prompt_overrides, load_eval_config, load_manifest

logger = logging.getLogger("skill_eval")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval_config", default=None,
                        help="skill_eval_configs/{task}.yaml (auto-inferred by default)")
    parser.add_argument("--manifest", default=None,
                        help="skill_init_states/task-xxxx/state_manifest.json (auto-inferred by default)")
    parser.add_argument("--skill_id", type=int, nargs="*",
                        help="skill ids to evaluate (defaults to all evaluable skills)")
    # policy loading: choose one — local hydra config or remote websocket service
    parser.add_argument("--policy_cfg", default=None,
                        help="local policy config (hydra config dir)")
    parser.add_argument("--websocket_host", default=None,
                        help="websocket policy server address (e.g. 127.0.0.1), used with --websocket_port")
    parser.add_argument("--websocket_port", type=int, default=8001,
                        help="websocket policy server port (default 8001)")
    parser.add_argument("--n_episodes", type=int, default=1,help="number of evaluation episodes")
    parser.add_argument("--log_dir", default="~/skill_eval_logs")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--no_video", action="store_true",
                        help="disable video recording (by default recorded to log_dir/videos/)")
    # prompt overrides (no YAML edits needed): set skill prompts manually
    parser.add_argument("--prompt", default=None,
                        help="override the instruction of all evaluated skills (handy for single-skill tests)")
    parser.add_argument("--prompt_file", default=None,
                        help="JSON file, format {\"skill_id\": \"prompt\", ...}; exact per-skill_id override")
    args = parser.parse_args()

    if args.policy_cfg is None and args.websocket_host is None:
        parser.error("either --policy_cfg or --websocket_host must be specified")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    eval_config_path = args.eval_config or str(
        PROJECT_ROOT / "data" / "skill_eval_configs" / f"{args.task}.yaml"
    )

    manifest_path = args.manifest or str(PROJECT_ROOT / "data" / "skill_init_states/task-xxxx" / "state_manifest.json")

    for path, name in [(eval_config_path, "eval_config"), (manifest_path, "manifest")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    task_cfg = load_eval_config(eval_config_path)
    manifest = load_manifest(manifest_path, base_dir=PROJECT_ROOT / "data")

    # ── Diagnostic log: manifest loading result ──────────────────────────
    logger.info(f"\n[main] manifest path: {manifest_path}")
    logger.info(f"[main] loaded {len(manifest)} skill entries: {sorted(manifest.keys())}")
    missing_state = [sid for sid, e in manifest.items() if not os.path.exists(e.get("state_file", ""))]
    missing_scene = [sid for sid, e in manifest.items() if e.get("scene_file") and not os.path.exists(e.get("scene_file", ""))]
    if missing_state:
        logger.error(f"[main] skills with a missing state_file: {missing_state}")
    else:
        logger.info(f"[main] state_file exists for all {len(manifest)} skills")
    if missing_scene:
        logger.warning(f"[main] skills with a missing scene_file: {missing_scene}")
    # ────────────────────────────────────────────────────────

    log_dir = str(Path(args.log_dir).expanduser())

    register_omegaconf_resolvers()
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True
    gm.HEADLESS = args.headless.lower() == "true"
    with macros.unlocked():
        macros.robots.manipulation_robot.GRASP_WINDOW = 0.75
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False

    if args.websocket_host:
        # connect to the remote policy service over websocket (served by serve_b1k.py)
        policy = WebsocketPolicyWrapper(host=args.websocket_host, port=args.websocket_port)
        logger.info(f"using websocket policy: ws://{args.websocket_host}:{args.websocket_port}")
    else:
        policy = load_hydra_policy(args.policy_cfg)

    env_cfg = build_env_cfg(args.task)

    # build the prompt override table
    prompt_overrides = build_prompt_overrides(
        prompt_file=args.prompt_file,
        prompt=args.prompt,
        skill_ids=args.skill_id,
        task_cfg=task_cfg,
    )

    try:
        all_results = run_batch(
            task_cfg=task_cfg,
            manifest=manifest,
            env_cfg=env_cfg,
            policy=policy,
            n_episodes=args.n_episodes,
            log_dir=log_dir,
            skill_ids=args.skill_id,
            prompt_overrides=prompt_overrides if prompt_overrides else None,
            save_video=not args.no_video,
        )
    except Exception:
        logger.error("[main] run_batch() crashed with an exception:", exc_info=True)
        all_results = {}

    # all skills evaluated: shut down Isaac Sim once
    try:
        og.shutdown()
        logger.info("[main] og.shutdown() done.")
    except Exception:
        pass

    os.makedirs(log_dir, exist_ok=True)
    summary_file = os.path.join(log_dir, f"{args.task}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task {args.task} — sub-task evaluation summary")
    print(f"{'='*60}")
    for sid, res in sorted(all_results.items()):
        rate = res["success_rate"]
        bar = "█" * round(rate * 10) + "░" * (10 - round(rate * 10))
        print(f"  Skill {sid:2d}: [{bar}] {rate:.0%}  {res['instruction'][:55]}")
    print(f"{'='*60}\nDetailed results: {summary_file}")
    if not args.no_video:
        print(f"Video directory: {os.path.join(log_dir, 'videos')}")


if __name__ == "__main__":
    main()
