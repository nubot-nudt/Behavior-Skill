"""Evaluation core: SkillEvaluator (per-skill evaluator) + run_batch (batch evaluation)."""

import json
import logging
import os
import sys
import omnigibson as og
from signal import SIGINT, signal
from typing import Optional

from .observation import preprocess_obs
from .skill_bddl import (
    _expand_wildcards_from_scene,
    _select_objects_bddl,
    build_predefined_problem,
)
from ..states.loader import load_skill_initial_state
from ..utils.diagnostics import (
    _log_gripper_init_state,
    _log_object_scope_wildcards,
    _log_proprio_gripper,
)
from ..utils.video import _extract_video_frame, _save_video


logger = logging.getLogger("skill_eval")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SkillEvaluator:
    """
    Standalone evaluator for a single skill.

    Mapping to the full-task evaluator:
      Full-task eval.py                   This class
      ─────────────────────────────────── ─────────────────────────────────────
      predefined_problem=None             predefined_problem=<skill_bddl_str>
      load_task_instance(id)              og.sim.restore(skill_XX_scene.json)
      policy receives full-task text      policy receives skill instruction text
      info["done"]["success"]             info["done"]["success"] (same BDDL engine)

    All skill types are judged uniformly via bddl_goal:
      move to       → (near agent target)                 → NearPredicate
      pick up from  → (grasped/inside/ontop/on_fire/...)  → the corresponding ObjectState
      push to       → (and (ontop obj surface) (overhanging_edge obj surface))
    """

    def __init__(
        self,
        skill_id: int,
        skill_cfg: dict,
        task_cfg: dict,
        state_file: str,
        env_cfg: dict,
        scene_file: str = None,
        frame_duration: int = None,
        existing_env=None,
        **kwargs,
    ):
        """
        existing_env: if provided, reuse that og.Environment (only the BDDL task
                      is updated) instead of creating a new one, and close()
                      becomes a no-op. This keeps a single entry in
                      og.sim.scenes for the whole batch, so the
                      load_from_scratch=False path of og.sim.restore() keeps working.
        """
        self.skill_id = skill_id
        # instruction: external override first, then the default from skill_eval_configs
        instruction_override = kwargs.get("instruction_override")
        self.instruction = instruction_override or skill_cfg["instruction"]
        self.bddl_goal = skill_cfg["bddl_goal"]
        self.state_file = state_file
        self.scene_file = scene_file
        self.default_max_steps = int(frame_duration * 2) if frame_duration else 500

        if not self.bddl_goal:
            raise ValueError(f"Skill {skill_id} has no bddl_goal; cannot build the evaluation environment")

        _task_name = task_cfg["task_name"]
        _act_def_id = task_cfg.get("activity_definition_id", 0)

        if existing_env is not None:
            # ── Reuse the existing environment, only switch the BDDL task (no new scene) ──
            # env.close() is a no-op and og.Environment.__init__ requires sim stopped;
            # import_scene() accumulates entries in og.sim.scenes and breaks restore().
            # Solution: keep a single og.Environment and update the goal via update_task().
            self.env = existing_env
            self._owns_env = False  # not closed in close(); managed by run_batch

            # ── Wildcard expansion ──────────────────────────────────────────────
            # The scene is ready: expand wildcards such as tree.n.01_* directly from
            # env.scene so that exists/forall quantifiers cover every instance of
            # the object class in the scene.
            _exp_objects, _exp_init = _expand_wildcards_from_scene(_task_name, _act_def_id, existing_env)
            _final_objects, _final_init = _select_objects_bddl(
                yaml_objects=task_cfg["objects_bddl"],
                yaml_init=task_cfg["init_bddl"],
                exp_objects=_exp_objects,
                exp_init=_exp_init,
            )
            predefined_problem = build_predefined_problem(
                task_name=_task_name,
                objects_bddl=_final_objects,
                init_bddl=_final_init,
                bddl_goal=self.bddl_goal,
            )
            logger.debug(f"predefined_problem (reuse env, merged):\n{predefined_problem}")

            new_task_cfg = dict(self.env.config.get("task", {}))
            new_task_cfg["predefined_problem"] = predefined_problem
            if task_cfg.get("extra_object_name_map"):
                new_task_cfg["extra_inst_to_name"] = task_cfg["extra_object_name_map"]
            logger.info(f"  [SkillEvaluator] reusing env, switching task: skill {skill_id}")
            # update_task() requires the sim to be playing (it internally does stop → load → play → post_play_load)
            if not og.sim.is_playing():
                og.sim.play()
            self.env.update_task(task_config=new_task_cfg)
            logger.info(f"  [SkillEvaluator] BDDL task switched, obs/action space rebuilt")
            _log_object_scope_wildcards(self.env, label=f"skill {skill_id} reuse")
        else:
            # ── First skill: create a new og.Environment ─────────────────────────
            # ⚠ Key: do NOT pass predefined_problem here. Let BehaviorTask.update_activity()
            #   call get_processed_bddl() automatically, which expands tree.n.01_* into
            #   all tree instances in the scene (tree.n.01_1 ~ tree.n.01_26) and writes
            #   the full inst_to_name into scene metadata via update_bddl_scope_metadata().
            #
            #   If predefined_problem (with tree.n.01_*) were passed at this point:
            #     object_scope would contain only [tree.n.01_1, tree.n.01_*]
            #     update_bddl_scope_metadata() would overwrite inst_to_name with 2 entries
            #     the later update_task(expanded) would then raise an AssertionError in
            #     assign_object_scope_with_cache() for tree.n.01_2..26
            #
            #   og.sim.restore([scene_file]) happens later, inside run_episode();
            #   even though the snapshot's inst_to_name contains all 26 trees, it
            #   cannot be relied upon at this point.
            self._owns_env = True
            cfg = dict(env_cfg)
            cfg["task"] = dict(cfg.get("task", {}))
            # No predefined_problem → use activity_name + get_processed_bddl()
            cfg["task"].pop("predefined_problem", None)
            if task_cfg.get("extra_object_name_map"):
                cfg["task"]["extra_inst_to_name"] = task_cfg["extra_object_name_map"]
            self.env = og.Environment(configs=cfg)

            # Scene loaded; inst_to_name fully written by get_processed_bddl() (all trees).
            # Now expand wildcards and rebuild the task with the skill-specific bddl_goal.
            _exp_objects, _exp_init = _expand_wildcards_from_scene(_task_name, _act_def_id, self.env)
            _final_objects, _final_init = _select_objects_bddl(
                yaml_objects=task_cfg["objects_bddl"],
                yaml_init=task_cfg["init_bddl"],
                exp_objects=_exp_objects,
                exp_init=_exp_init,
            )
            predefined_problem = build_predefined_problem(
                task_name=_task_name,
                objects_bddl=_final_objects,
                init_bddl=_final_init,
                bddl_goal=self.bddl_goal,
            )
            logger.debug(f"predefined_problem (first env, merged):\n{predefined_problem}")
            new_task_cfg = dict(self.env.config.get("task", {}))
            new_task_cfg["predefined_problem"] = predefined_problem
            if task_cfg.get("extra_object_name_map"):
                new_task_cfg["extra_inst_to_name"] = task_cfg["extra_object_name_map"]
            if not og.sim.is_playing():
                og.sim.play()
            self.env.update_task(task_config=new_task_cfg)
            logger.info(f"  [SkillEvaluator] first env created, wildcards expanded, skill goal injected")
            _log_object_scope_wildcards(self.env, label=f"skill {skill_id} first")

        self.robot = self.env.robots[0]

        logger.info(
            f"Skill {skill_id} initialized\n"
            f"  instruction: {self.instruction}\n"
            f"  type: {skill_cfg.get('type')}\n"
            f"  BDDL goal: {self.bddl_goal}"
        )

    def _reset_to_skill_start(self) -> dict:
        """Reset the environment and load the skill start state (prefers the full scene snapshot)."""
        import omnigibson as og
        load_skill_initial_state(
            env=self.env,
            state_file=self.state_file,
            scene_file=self.scene_file,
        )

        # ── Reset the episode step counter and termination conditions ──────────
        # load_skill_initial_state() only calls env.scene.reset(), not env.reset(),
        # so env._current_step keeps accumulating across episodes. The Timeout
        # condition checks episode_steps >= max_steps (5000); without the reset,
        # later episodes would time out early — or immediately.
        # Fix: manually zero _current_step and reset the _done flag of every
        # termination condition.
        self.env._current_step = 0
        for _tc in self.env.task._termination_conditions.values():
            _tc.reset(self.env.task, self.env)
        logger.info(f"[EPISODE RESET] _current_step=0, termination conditions reset")
        # ──────────────────────────────────────────────────────────────────────

        # Log the actual gripper positions and grasp state, to verify initialization
        _log_gripper_init_state(self.env, self.skill_id)
        # og rendering is asynchronous: >=3 render calls are needed to fill the camera
        # buffers. Use render() rather than env.step(zero_action) to avoid actuating
        # the gripper controller.
        for _ in range(3):
            og.sim.render()
        obs, _ = self.env.get_obs()
        return self._preprocess_obs(obs)

    def _preprocess_obs(self, obs: dict) -> dict:
        """Convert raw env obs to the policy format (thin wrapper over
        observation.preprocess_obs; binds this evaluator's robot/env, so
        call sites only need to pass obs)."""
        return preprocess_obs(obs, self.robot, self.env)

    def run_episode(self, policy, max_steps: int = None, video_path: str = None) -> tuple:
        """
        Run one episode, returning (success, n_steps).

        All skills are judged via bddl_goal (BDDL PredicateGoal engine):
          - navigation: (near agent target) — robot XY distance <= 1.5m
          - manipulation: (grasped/inside/ontop/...) — physical state
          - push: (and (ontop obj surface) (overhanging_edge obj surface))

        max_steps priority: caller argument > self.default_max_steps (2x frame_duration) > 500
        video_path: when not None, each step's RGB frame is written to the .mp4 file
        """
        if max_steps is None:
            max_steps = self.default_max_steps

        frames = [] if video_path else None
        obs = self._reset_to_skill_start()
        last_frame = None  # fallback cache: if _extract_video_frame returns None, reuse the previous frame so that 1 step = 1 frame
        if frames is not None:
            f = _extract_video_frame(obs)
            if f is not None:
                frames.append(f)
                last_frame = f
        policy.reset()
        # Log the gripper DOFs as observed by the policy; compare with
        # _log_gripper_init_state to verify consistency
        _log_proprio_gripper(obs, self.skill_id)

        # ── Initial goal check: is the goal already satisfied before the episode starts? ──
        # A True at step 0 means the initial state itself already satisfies the goal:
        #   - navigation skill: start pose already within threshold (NearPredicate) → snapshot start frame chosen too late
        #   - grasp skill: _recover_grasps_after_restore() wrongly established a grasp constraint on the target object
        try:
            from bddl.activity import evaluate_goal_conditions as _eval_goal
            _goal_sat, _sat_preds = _eval_goal(self.env.task.ground_goal_state_options[0])
            if _goal_sat:
                logger.warning(
                    f"  ⚠ [TRIVIAL SUCCESS] Skill {self.skill_id}: "
                    f"BDDL goal '{self.bddl_goal}' is already satisfied before the episode starts (step=0)!\n"
                    f"    Possible causes:\n"
                    f"    1. NearPredicate: start pose already within the threshold ({self.bddl_goal[:60]}...)\n"
                    f"    2. GraspedPredicate: _recover_grasps_after_restore() wrongly "
                    f"established a grasp constraint on the target object\n"
                    f"    3. Other predicates: the initial state already satisfies the BDDL condition\n"
                    f"    Suggestion: check _ag_obj_in_hand or adjust the snapshot start frame"
                )
        except Exception as _e:
            logger.debug(f"  [initial goal check] skipped ({_e})")
        # ─────────────────────────────────────────────────────────────────────

        done = False
        success = False
        step = 0

        while not done and step < max_steps:
            action = policy.forward(obs=obs, instruction=self.instruction)
            raw_obs, _, terminated, truncated, info = self.env.step(action, n_render_iterations=1)
            obs = self._preprocess_obs(raw_obs)
            if frames is not None:
                f = _extract_video_frame(obs)
                if f is not None:
                    frames.append(f)
                    last_frame = f
                elif last_frame is not None:
                    # Frame-drop fallback: reuse the previous frame so video frames = step + 1
                    frames.append(last_frame)
            step += 1

            if terminated or truncated:
                done = True
                success = info.get("done", {}).get("success", False)

            if step % 100 == 0:
                logger.info(f"Current step: {step} / {max_steps}")

        if not done:
            success = False

        # Freeze the last frame for 1 second (= fps frames) to make the final outcome
        # easier to see. Also force extra renders: step the env a few more times
        # (repeating the last action) to capture the visual state after success,
        # compensating for the PhysX→render lag that can miss the final moment.
        if frames is not None and len(frames) > 0:
            try:
                # Render a few extra frames to capture the post-success visual state
                for _ in range(5):
                    raw_obs_extra, _, _, _, _ = self.env.step(action, n_render_iterations=1)
                    f_extra = _extract_video_frame(self._preprocess_obs(raw_obs_extra))
                    if f_extra is not None:
                        frames.append(f_extra)
            except Exception as _e:
                logger.debug(f"  extra video rendering failed (ignored): {_e}")
            # Hold the last frame for 1 second
            last = frames[-1]
            video_fps = 15
            frames.extend([last] * video_fps)

        if video_path and frames:
            _save_video(frames, video_path, fps=15)
            logger.info(f"  video saved: {video_path} ({len(frames)} frames)")

        mark = "\u2713" if success else "\u2717"
        logger.info(
            f"Evaluation finished at step {step}."
            f" [{mark}] skill {self.skill_id}: {'success' if success else 'fail'}"
        )
        return success, step

    def evaluate(self, policy, n_episodes: int, video_dir: str = None, metrics_dir: str = None) -> dict:
        results = []
        n_success = 0
        for ep in range(n_episodes):
            logger.info(f"\n" + "=" * 50)
            logger.info(f"[Skill {self.skill_id}] Episode {ep+1}/{n_episodes}")
            logger.info(f"  instruction: {self.instruction}")
            logger.info("=" * 50)
            video_path = None
            if video_dir:
                video_path = os.path.join(
                    video_dir, f"skill_{self.skill_id:02d}_ep{ep:02d}.mp4"
                )
            success, n_steps = self.run_episode(policy, video_path=video_path)
            if success:
                n_success += 1
            ep_result = {"episode": ep, "success": success, "n_steps": n_steps}
            if video_path:
                ep_result["video"] = video_path
            results.append(ep_result)

            # Save per-episode metrics (aligned with eval.py's metrics/{task}_{idx}_{epi}.json)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
                ep_metrics = {
                    "task_name": self.env.task.activity_name,
                    "skill_id": self.skill_id,
                    "episode": ep,
                    "success": bool(success),
                    "n_steps": n_steps,
                    "instruction": self.instruction,
                    "bddl_goal": self.bddl_goal,
                    "n_success_so_far": n_success,
                    "success_rate_so_far": n_success / (ep + 1),
                }
                mfile = os.path.join(
                    metrics_dir, f"skill_{self.skill_id:02d}_ep{ep:02d}.json"
                )
                with open(mfile, "w") as f:
                    json.dump(ep_metrics, f, indent=2)
                logger.info(f"Metrics saved to {mfile}")

            logger.info(
                f"Total trials: {ep + 1}  Total success: {n_success}  "
                f"Success rate: {n_success / (ep + 1):.0%}"
            )

        rate = n_success / n_episodes
        return {
            "skill_id": self.skill_id,
            "instruction": self.instruction,
            "bddl_goal": self.bddl_goal,
            "n_trials": n_episodes,
            "n_success": n_success,
            "success_rate": rate,
            "episodes": results,
        }

    def close(self):
        """Close the Environment (Isaac Sim keeps running so the next skill can reuse it).
        - _owns_env=True (this evaluator created the env): call env.close().
        - _owns_env=False (reusing an external env): no-op; run_batch closes it.
        og.shutdown() is called once by main() after all skills finish."""
        if getattr(self, "_owns_env", True) and hasattr(self, "env"):
            try:
                self.env.close()
                logger.info(f"  [SkillEvaluator] Skill {self.skill_id} env closed.")
            except Exception as e:
                logger.warning(f"  [SkillEvaluator] env.close() failed: {e}")
        else:
            logger.info(f"  [SkillEvaluator] Skill {self.skill_id}: reusing env, skipping close.")


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    task_cfg: dict,
    manifest: dict,
    env_cfg: dict,
    policy,
    n_episodes: int,
    log_dir: str,
    skill_ids: Optional[list] = None,
    prompt_overrides: Optional[dict] = None,
    save_video: bool = True,
) -> dict:
    """
    Evaluate all skills that have a bddl_goal (navigation/manipulation/push are
    unified through the BDDL engine; see the SkillEvaluator docstring);
    skills without a bddl_goal are skipped.
    """
    skills = {s["id"]: s for s in task_cfg["skills"]}

    # By default, evaluate all skills that have a bddl_goal
    if skill_ids:
        target_ids = sorted(skill_ids)
    else:
        target_ids = sorted(
            s["id"] for s in task_cfg["skills"]
            if s.get("bddl_goal") is not None
        )

    all_results = {}
    # ── Shared env (single entry in og.sim.scenes) ─────────────────────────────
    # Root cause: every og.Environment(configs=cfg) appends a new scene to
    # og.sim.scenes, which causes two problems:
    #   1. og.sim requires is_stopped()=True to import a scene, but env.close()
    #      is a no-op, so after the previous skill the sim is still playing → AssertionError
    #   2. og.sim.restore([scene_file]) requires len(scenes)==1 (the non-empty-scene
    #      restore path); multiple scenes break it → silent error
    # Fix: share a single og.Environment across the whole batch:
    #   - first skill: created normally (existing_env=None)
    #   - later skills: pass existing_env, only the BDDL task goal is updated
    # ──────────────────────────────────────────────────────────────────────────
    shared_env = None  # saved after the first SkillEvaluator creates it; reused afterwards

    logger.info(
        f"\n" + "=" * 60 + "\n"
        f"[run_batch] starting batch evaluation\n"
        f"  task: {task_cfg['task_name']}  skills to evaluate: {target_ids}\n"
        f"  n_episodes={n_episodes}  log_dir={log_dir}\n"
        + "=" * 60
    )
    for sid in target_ids:
        skill_cfg = skills.get(sid)
        if skill_cfg is None:
            logger.error(f"[SKIP] Skill {sid}: not in eval_config")
            continue
        if not skill_cfg.get("bddl_goal"):
            logger.warning(f"[SKIP] Skill {sid}: no bddl_goal in eval_config")
            continue
        if sid not in manifest:
            logger.error(f"[SKIP] Skill {sid}: not in manifest ({len(manifest)} skills total)")
            continue
        state_file = manifest[sid]["state_file"]
        scene_file = manifest[sid].get("scene_file")
        if not os.path.exists(state_file):
            logger.error(f"[SKIP] Skill {sid}: state_file not found: {state_file}")
            continue
        if scene_file and not os.path.exists(scene_file):
            logger.warning(f"[WARN] Skill {sid}: scene_file not found: {scene_file}")

        # Read frame_duration from the manifest, used to compute max_steps
        manifest_entry = manifest.get(sid, {})
        start_frame = manifest_entry.get("start_frame", 0)
        end_frame = manifest_entry.get("end_frame", 0)
        frame_duration = max(end_frame - start_frame, 1)

        # Apply prompt overrides (from --prompt or --prompt_file)
        instruction_override = (prompt_overrides or {}).get(sid)
        video_dir = os.path.join(log_dir, "videos") if save_video else None
        metrics_dir = os.path.join(log_dir, "metrics")
        evaluator = None
        orig_handler = signal(SIGINT, lambda s, f: (evaluator.close() if evaluator else None, sys.exit(0)))
        try:
            logger.info(f"\n[run_batch] initializing SkillEvaluator: skill {sid} ({skill_cfg.get('type', '?')})")
            evaluator = SkillEvaluator(
                skill_id=sid,
                skill_cfg=skill_cfg,
                task_cfg=task_cfg,
                state_file=state_file,
                scene_file=scene_file,
                env_cfg=env_cfg,
                frame_duration=frame_duration,
                instruction_override=instruction_override,
                existing_env=shared_env,  # None → create a new env; otherwise reuse + switch task
            )
            # After the first skill creates the env, save it for later skills to reuse
            if shared_env is None:
                shared_env = evaluator.env
                logger.info(f"  [run_batch] shared_env created; later skills will reuse it")
            logger.info(f"[run_batch] SkillEvaluator initialized, starting evaluation...")
            summary = evaluator.evaluate(policy, n_episodes, video_dir=video_dir, metrics_dir=metrics_dir)
            all_results[sid] = summary
            os.makedirs(log_dir, exist_ok=True)
            out = os.path.join(log_dir, f"{task_cfg['task_name']}_skill_{sid:02d}.json")
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                f"\n" + "=" * 50 + "\n"
                f"Skill {sid} evaluation finished\n"
                f"  success rate: {summary['success_rate']:.0%}  "
                f"({summary['n_success']}/{summary['n_trials']})\n"
                f"  results saved to: {out}\n"
                + "=" * 50
            )
        except Exception as e:
            logger.error(
                f"[ERROR] Skill {sid} evaluation failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
        finally:
            if evaluator is not None:
                evaluator.close()
            signal(SIGINT, orig_handler)

    # Close the shared env after all skills are done (env.close() is a no-op, kept for symmetry)
    if shared_env is not None:
        try:
            shared_env.close()
            logger.info("[run_batch] shared_env closed.")
        except Exception as e:
            logger.warning(f"[run_batch] shared_env.close() failed: {e}")

    n_done = len(all_results)
    logger.info(
        f"\n[run_batch] finished: {n_done}/{len(target_ids)} skills completed"
        + (" (all skipped or failed! check the [SKIP]/[ERROR] logs above)" if n_done == 0 else "")
    )
    return all_results
