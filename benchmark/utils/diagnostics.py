"""Diagnostic logging: object_scope wildcards, gripper state, proprio, scene state summary."""
import datetime
import torch as _th
import numpy as _np
import logging

logger = logging.getLogger("skill_eval")


def _log_object_scope_wildcards(env, label: str = "") -> None:
    """
    Print the resolution of every wildcard category (synset with multiple
    instances) in env.task.object_scope.

    Purpose: verify that the exists/forall quantifiers cover all instances of
    the category present in the scene.
    Sample output:
      [SCOPE] tree.n.01: 26 instances → [tree.n.01_1, tree.n.01_2, ..., tree.n.01_26]
                all bound to entities: tree_wtyipq_6, tree_wtyipq_1, ...
    """
    lines = []
    prefix = f"[SCOPE{' ' + label if label else ''}]"
    lines.append(f"\n{'='*60}")
    lines.append(f"{prefix} @ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*60}")

    try:
        scope = env.task.object_scope  # dict: inst_str → BDDLEntity
        if scope is None:
            msg = f"  {prefix} object_scope is None, skipping diagnostics"
            logger.warning(msg)
            lines.append(msg)
        else:
            from collections import defaultdict
            by_synset = defaultdict(list)
            for inst, entity in scope.items():
                synset = "_".join(inst.split("_")[:-1])
                by_synset[synset].append((inst, entity))

            for synset, entries in sorted(by_synset.items()):
                entries_sorted = sorted(entries, key=lambda x: x[0])
                n = len(entries_sorted)
                none_insts = [i for i, e in entries_sorted if e is None or not e.exists]
                bound_names = []
                for inst, entity in entries_sorted:
                    try:
                        bound_names.append(entity.name if (entity is not None and entity.exists) else "None")
                    except Exception:
                        bound_names.append("?")

                if n > 1:
                    inst_list = [i for i, _ in entries_sorted]
                    msg = (
                        f"  {prefix} {synset}: {n} instances\n"
                        f"    instances : {inst_list}\n"
                        f"    bound     : {bound_names}"
                        + (f"\n    ⚠ unbound : {none_insts}" if none_insts else "")
                    )
                    logger.info(msg)
                    lines.append(msg)
                elif none_insts:
                    msg = f"  {prefix} ⚠ {synset}: instance {none_insts[0]} is not bound to any entity!"
                    logger.warning(msg)
                    lines.append(msg)
                else:
                    # single instance and already bound: collected in lines only, not logged
                    lines.append(f"  {prefix} {synset}: {bound_names[0]}")

    except Exception as _e:
        msg = f"  {prefix} diagnostic print failed: {_e}"
        logger.warning(msg)
        lines.append(msg)

    # results are consumed via the `lines` buffer (log output only)


def _log_gripper_init_state(env, skill_id: int) -> None:
    """
    After initialization, read the gripper joint positions and held-object state
    straight from the simulator and print them. Used to verify that the initial
    observation the policy will receive matches the simulator's actual physical
    state.

    Key checks:
      1. gripper_pos   — physical joint positions of the gripper
      2. open ratio    — 0.0 = fully closed, 1.0 = fully open
      3. held object   — _ag_obj_in_hand (valid only once the FixedJoint constraint exists)

    If "held" is empty while the gripper is closed (ratio ~0), the grasp
    constraint was not established: during env.step() the gripper will be driven
    to the fully closed position → the gripper DOFs in later proprio read low.
    """
    for robot in env.robots:
        for arm in robot.arm_names:
            obj = robot._ag_obj_in_hand.get(arm)
            gripper_idx = robot.gripper_control_idx[arm]
            try:
                all_pos = robot.get_joint_positions()
                gripper_pos = all_pos[gripper_idx]
                lower = robot.joint_lower_limits[gripper_idx]
                upper = robot.joint_upper_limits[gripper_idx]
                denom = (upper - lower).clamp(min=1e-6)
                ratio = ((gripper_pos - lower) / denom).clamp(0.0, 1.0)
                logger.info(
                    f"  [Skill {skill_id} init] {arm}: "
                    f"gripper_pos={[round(float(v), 4) for v in gripper_pos.tolist()]}  "
                    f"open_ratio={[round(float(r), 2) for r in ratio.tolist()]}  "
                    f"held={obj.name if obj else 'empty'}"
                )
            except Exception as e:
                logger.warning(f"  [Skill {skill_id} init] {arm}: failed to read gripper state: {e}")


def _log_proprio_gripper(obs: dict, skill_id: int) -> None:
    """
    Extract and print the gripper-related DOFs from obs['robot_r1::proprio'],
    exactly as the policy receives it.

    Policy action space (23 DOF): index[-9] = left gripper, index[-1] = right gripper.
    Comparing this output with _log_gripper_init_state() verifies the full
    simulator-state → policy-observation chain.
    A large mismatch means the proprio_obs config is wrong or the
    flatten_obs_dict dimensions do not match.
    """
    key = "robot_r1::proprio"
    if key not in obs:
        logger.warning(f"  [Skill {skill_id} proprio] '{key}' not in obs, skipping check")
        return
    proprio = obs[key]
    if isinstance(proprio, _th.Tensor):
        proprio = proprio.cpu().numpy()
    proprio = _np.asarray(proprio).flatten()
    n = len(proprio)
    logger.info(
        f"  [Skill {skill_id} proprio] shape={n}  "
        f"idx[-9]={round(float(proprio[-9]), 4) if n >= 9 else 'N/A'} (left gripper)  "
        f"idx[-1]={round(float(proprio[-1]), 4) if n >= 1 else 'N/A'} (right gripper)"
    )


def _log_scene_state_summary(env):
    """Print a key state summary (grasps, burning) after loading, for debugging."""
    try:
        from omnigibson.object_states.on_fire import OnFire
        from omnigibson.object_states.temperature import Temperature
    except ImportError:
        return

    for robot in env.robots:
        for arm in robot.arm_names:
            obj = robot._ag_obj_in_hand[arm]
            if obj is not None:
                logger.info(f"  [GRASP] {arm} holding: {obj.name}")
            else:
                logger.info(f"  [GRASP] {arm}: empty")

    burning = []
    for obj in env.scene.objects:
        if not getattr(obj, '_initialized', False):
            continue
        if hasattr(obj, "states") and Temperature in obj.states:
            temp = obj.states[Temperature].get_value()
            if OnFire in obj.states and obj.states[OnFire].get_value():
                burning.append(f"{obj.name}(T={temp:.0f}°C)")
    if burning:
        logger.info(f"  [BURNING] {', '.join(burning)}")
    else:
        logger.info(f"  [BURNING] no burning objects")
