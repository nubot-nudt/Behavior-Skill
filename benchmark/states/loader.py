"""Skill initial state loading: og.sim.restore + post-restore fixes + physics stabilization."""

import logging
from pathlib import Path
import omnigibson as og

import numpy as np

from ..utils.diagnostics import _log_scene_state_summary
from .restore_fixes import (
    _fix_all_joint_drive_targets_after_restore,
    _fix_gripper_targets_after_restore,
    _recover_grasps_after_restore,
)

logger = logging.getLogger("skill_eval")


def load_skill_initial_state(env, state_file: str, scene_file: str = None, n_stabilize: int = 25):
    """
    Load the skill start state.

    Official Disk API flow (the correct one):
      og.sim.restore([scene_file])  ← no og.clear() needed; with scenes already present it calls scene.restore()
      → syncs the object set (add/remove to match objects_info)
      → loads all object positions/states (including sleeping objects):
          - object position/orientation: fully saved/restored
          - Temperature/OnFire: saved via TensorizedValueState._dump/load_state()
          - ag_obj_constraint_params (grasp constraints): saved via ManipulationRobot._dump/load_state()
      → grasp constraint recovery (new-format files: done by _load_state(); legacy files: done by _recover_grasps)
      → step_physics × n to stabilize physics (env.step() is NOT called, so a zero action cannot release grasps)

    ⚠ Critical: never call env.reset() after restore()
      env.reset() calls scene.restore(self._initial_file) and overwrites the just-loaded state!

    ── State maintenance guarantees during evaluation ──
      ✅ Temperature/OnFire: every og.sim.step_physics() calls Temperature.global_update()
         → the temperature is maintained (OnFire objects stay at their high T) → OnFire stays True automatically
      ✅ Grasp: og.sim.restore() restores ag_obj_constraint_params → _load_state()
         → _establish_grasp() creates a PhysX FixedJoint → the physical constraint holds the grasped object
         ⚠ During env.step(action): _handle_assisted_grasping() checks the gripper controller's
           control signal to decide whether to maintain the grasp.
           → if the action's gripper signal < 0 (close): applying_grasp=True → grasp maintained
           → if the action's gripper signal >= 0 (open/neutral): applying_grasp=False → release window starts
           Therefore: the evaluated policy must send a "close gripper" signal while holding an object.

    Reference: https://behavior.stanford.edu/tutorials/save_load.html#disk
    """
    if scene_file and Path(scene_file).exists():
        # ── Before restore: release grasp constraints left over from the previous run ──
        # og.sim.restore()'s internal _load_state() calls release_grasp_immediately,
        # but ag_constraint prim deletion has a timing issue: it conflicts with the
        # new joint created at the same path by the recovery pass.
        # Note: _release_grasp() does not clear _ag_obj_in_hand, so it must be set
        # to None manually; otherwise _recover_grasps_after_restore()'s "is not
        # None" check would skip that arm.
        for _robot in env.robots:
            if _robot.grasping_mode == "physical":
                continue
            for _arm in _robot.arm_names:
                if (_robot._ag_obj_in_hand.get(_arm) is not None
                        or _robot._ag_obj_constraints.get(_arm) is not None):
                    try:
                        _robot._release_grasp(arm=_arm)
                    except Exception:
                        pass
                    _robot._ag_obj_in_hand[_arm] = None
                    _robot._ag_release_counter[_arm] = None
                    logger.info(f"  [PRE-RESTORE] released leftover grasp on {_arm}")
        logger.info(f"  [og.sim.restore] {Path(scene_file).name}")
        og.sim.restore([str(scene_file)])

        # grasp constraint recovery (must run before step_physics, before objects fall)
        _recover_grasps_after_restore(env)
        _fix_gripper_targets_after_restore(env)  # set gripper PD targets = actual grasped positions
        _fix_all_joint_drive_targets_after_restore(env)  # fix arm joint PD targets to prevent stabilization jitter

        # physics stabilization: only step_physics(), which does not trigger _handle_assisted_grasping()
        # PhysX position drives hold the arm; the PhysX fixed joint holds the grasped object
        for _obj in env.scene.objects:
            _obj.keep_still()
        for _ in range(n_stabilize):
            og.sim.step_physics()

        # ── Align with eval.py load_task_instance()'s full init flow ──────────
        # eval.py: step_physics×25 → update_initial_file() → env.scene.reset()
        #   env.scene.reset() calls scene.restore(initial_file)
        #     → load_state() → robot._load_state()
        #       → set_joint_positions(positions, drive=True)  ← all joint targets synced
        #       → _establish_grasp()                          ← rebuilds the grasp FixedJoint
        # This makes the PD controller targets exactly match the joint positions,
        # removing the root cause of post-restore oscillation.
        try:
            env.scene.update_initial_file()   # save the current stabilized state as "initial_file"
            env.scene.reset()                 # full reset from initial_file (joint targets + grasps)
            # ── Safety check: env.scene.reset() rebuilds grasp constraints internally via
            # robot._load_state() → _establish_grasp(). If the JSON did not fully serialize
            # ag_obj_constraint_params (legacy-format fallback), recover explicitly again
            # to guarantee the grasp FixedJoint is always valid.
            _recover_grasps_after_restore(env)
            _fix_gripper_targets_after_restore(env)
            logger.info("  [env.scene.reset()] full controller reset + grasp safety check done")
        except AssertionError as _ae:
            if "initialized before dumping" in str(_ae):
                # Some future object is uninitialized (e.g. diced__steak), so the whole
                # scene cannot be serialized. Safe to skip: the joint target sync and
                # grasp rebuild were already done above via _recover_grasps /
                # _fix_gripper_targets / _fix_all_joint_drive_targets; update_initial_file
                # + scene.reset is a redundant second confirmation in this case.
                logger.warning(
                    f"  [WARN] skipping update_initial_file + scene.reset"
                    f" (uninitialized future object present): {_ae}"
                )
            else:
                raise
        # ────────────────────────────────────────────────────────────────────

        # print a key state summary after loading
        _log_scene_state_summary(env)

        logger.info(f"  scene restore complete (physics stabilized {n_stabilize} steps)")
    else:
        # degraded fallback: partial state (robot only; scene objects at BDDL initial positions)
        logger.warning(
            f"  [WARN] scene_file not found, falling back to partial state: {Path(state_file).name}\n"
            f"         objects moved in the scene will sit at their BDDL initial positions;"
            f" not suitable for evaluating intermediate steps"
        )
        env.reset()
        data = np.load(state_file, allow_pickle=True)
        if "state" not in data:
            logger.warning("  [WARN] no 'state' field in npz, skipping state loading")
            return
        state_vec = data["state"]
        logger.info(f"  loaded partial state: {Path(state_file).name} (size={len(state_vec)})")
        import torch as _th
        if isinstance(state_vec, np.ndarray):
            state_vec = _th.from_numpy(state_vec.astype(np.float32))
        og.sim.load_state(state_vec, serialized=True)
        for _ in range(n_stabilize):
            og.sim.step_physics()
        logger.info(f"  physics stabilized ({n_stabilize} steps)")
