"""Evaluation config loading: skill_eval_configs/*.yaml, state_manifest.json and prompt overrides."""

import json
import os
import sys
import yaml
import logging
from pathlib import Path

logger = logging.getLogger("skill_eval")


def load_eval_config(config_path: str) -> dict:
    """Load a skill_eval_configs/*.yaml file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: str, base_dir=None) -> dict:
    """Load state_manifest.json, returning a skill_idx → entry mapping.
    base_dir: base directory for resolving relative paths; when omitted, the
    parent directory of the manifest is used."""
    manifest_dir = Path(manifest_path).parent
    resolve_base = Path(base_dir) if base_dir else manifest_dir.parent
    with open(manifest_path) as f:
        manifest = json.load(f)
    result = {}
    for entry in manifest["skills"]:
        entry = dict(entry)  # shallow copy so the original data is not mutated
        for key in ("state_file", "scene_file"):
            val = entry.get(key)
            if val and not Path(val).is_absolute():
                entry[key] = str(resolve_base / val)
        result[entry["skill_idx"]] = entry
    return result


def build_prompt_overrides(prompt_file=None, prompt=None, skill_ids=None, task_cfg=None) -> dict:
    """Build the prompt override table (backend for the --prompt_file / --prompt CLI args).

    - prompt_file: path to a JSON file of the form {"skill_id": "prompt", ...};
      exact per-skill_id overrides
    - prompt: fallback value applied to every evaluated skill (lower priority
      than sids already present in prompt_file)
    - skill_ids: list of skill ids to evaluate (--skill_id); when None, all
      skills in task_cfg that have a bddl_goal
    """
    prompt_overrides: dict = {}
    if prompt_file:
        if not os.path.exists(prompt_file):
            logger.error(f"--prompt_file not found: {prompt_file}")
            sys.exit(1)
        with open(prompt_file) as _f:
            raw = json.load(_f)
        prompt_overrides = {int(k): v for k, v in raw.items()}
        logger.info(f"loaded {len(prompt_overrides)} prompt overrides from {prompt_file}")
    if prompt:
        # --prompt covers every evaluated skill (lower priority than sids already in --prompt_file)
        target_ids = skill_ids or [s["id"] for s in task_cfg["skills"] if s.get("bddl_goal")]
        for sid in target_ids:
            if sid not in prompt_overrides:
                prompt_overrides[sid] = prompt
        logger.info(f"--prompt: instruction overrides active for {len(prompt_overrides)} skills")
    return prompt_overrides
