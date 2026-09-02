"""Post-og.sim.restore() state fixes:
grasp constraint recovery (_recover_grasps_after_restore),
gripper joint drive target fix (_fix_gripper_targets_after_restore),
all-joint drive target fix (_fix_all_joint_drive_targets_after_restore).
"""
import torch as th
import os as _os
import omnigibson as og
import logging

logger = logging.getLogger("skill_eval")


def _recover_grasps_after_restore(env) -> None:
    """
    Recover grasp constraints after og.sim.restore() (same logic as visualize_states.py).

    ① New-format scene files (with ag_obj_constraint_params): _load_state() already
       established the constraints; this is a no-op.
    ② Legacy files (ag_obj_constraint_params empty): proximity detection, calling
       _establish_grasp() directly.

    Never calls env.step() (a zero action → gripper opens → constraints released).

    ★ Do not teleport the object: the FixedJoint automatically locks the EEF-object
      relative pose at establishment time. The extraction stage already placed the
      object at the correct grasping pose, so establishing the constraint directly
      is enough; teleporting it to the EEF center would break the correct
      object-finger relationship.

    ★ Controller-goal guard (against false positives):
      When ag_obj_constraint_params is empty (e.g. extraction scripts that failed
      to serialize it), every grasp goes through the proximity fallback. When the
      controller goal is OPEN (target ≈ upper), a grasp constraint should NOT be
      established even if the joint positions happen to be semi-closed —
      otherwise GraspedPredicate would produce a false positive (success at step=0).
      The gripper controller goal distinguishes "holding / was holding" from
      "opening / fully open" states.
    """
    # ── Master switch: disabled by default (scene_file carries correct AG) ──
    # Since extraction switched to a fully sequential rollout, the scene_file's
    # ag_obj_constraint_params are correctly written by og.sim.save → _dump_state
    # (dict mode), and og.sim.restore → _load_state (dict mode) automatically
    # rebuilds the PhysX FixedJoint. The proximity fallback is therefore no longer
    # needed.
    #   - default: BEHAVIOR_DISABLE_GRASP_RECOVERY="1" → no-op
    #   - emergency: explicitly set to "0" to enable the legacy contact + proximity
    #     fallback (debug only)
    if _os.environ.get("BEHAVIOR_DISABLE_GRASP_RECOVERY", "1") == "1":
        logger.info("[GRASP RECOVERY] disabled (trusting the scene_file's built-in AG; set BEHAVIOR_DISABLE_GRASP_RECOVERY=0 to enable)")
        # still print the current state for diagnostics
        for robot in env.robots:
            if robot.grasping_mode == "physical":
                continue
            for arm in robot.arm_names:
                held = robot._ag_obj_in_hand.get(arm)
                tag = held.name if held is not None else "empty"
                logger.info(f"  [GRASP DIAG] {arm}: held={tag}")
        return

    # ── step_physics once first to refresh the PhysX contact cache ──────────
    # After og.sim.restore() the RigidContactAPI cache is still empty; one step
    # is needed before contact data exists.
    # og.sim.step() is NOT called (it would trigger action/render); a pure
    # physics step only.
    try:
        og.sim.step_physics()
    except Exception as _e:
        logger.warning(f"  [GRASP RECOVERY] step_physics failed: {_e}")

    # ── Fallback mode switch, debug only: for snapshots whose grasp relation was
    # not established correctly ──────────────────────────────────────────────
    # BEHAVIOR_GRASP_RECOVERY_MODE:
    #   "contact_only"     - real contact only
    #   "proximity"        - real contact + proximity fallback:
    #                        when contact fails, find the nearest movable object
    #                        within PROXIMITY_THRESHOLD of the EEF and grasp it.
    #                        No finger-state check; the object position in the
    #                        snapshot is the sole intent signal.
    # Threshold: BEHAVIOR_GRASP_PROXIMITY=0.20 (default, meters)
    #   - an object within 0.20 m of the EEF counts as "demo intended to hold"
    #   - outside the range → no grasp (the policy grasps on its own)
    _mode = _os.environ.get("BEHAVIOR_GRASP_RECOVERY_MODE", "proximity")
    _prox_thresh = float(_os.environ.get("BEHAVIOR_GRASP_PROXIMITY", "0.20"))

    for robot in env.robots:
        if robot.grasping_mode == "physical":
            continue
        for arm in robot.arm_names:
            if robot._ag_obj_in_hand[arm] is not None:
                logger.info(f"  [GRASP STATE] {arm}: already holding {robot._ag_obj_in_hand[arm].name} (rebuilt by _load_state)")
                continue

            # ── ① Real contact detection (highest priority) ──────────────────
            ag_data = None
            try:
                ag_data = robot._calculate_in_hand_object(arm=arm)
            except Exception as _e:
                logger.warning(f"  [GRASP RECOVERY] {arm}: _calculate_in_hand_object failed: {_e}")

            if ag_data is not None:
                obj, link = ag_data
                try:
                    eef_pos, _ = robot.eef_links[arm].get_position_orientation()
                    robot._establish_grasp(arm=arm, ag_data=(obj, link), contact_pos=eef_pos)
                    logger.info(f"  [GRASP RECOVERY] ✓ {arm} → {obj.name} (based on real physical contact)")
                except Exception as e:
                    logger.warning(f"  [GRASP RECOVERY] {arm}: _establish_grasp failed: {e}")
                continue

            if _mode == "contact_only":
                logger.info(f"  [GRASP STATE] {arm}: no contact + contact_only mode → skip")
                continue

            # ── Controller-goal guard (against proximity false positives) ──
            # The most reliable "demo intent" signal is controller.goal.target:
            #   - target ≈ upper → demo commanded OPEN (empty-hand intent) → no grasp
            #   - target ≈ lower → demo commanded CLOSE (holding intent) → proximity allowed
            # This guard works after og.sim.restore() (restore also restores controller goals).
            controller_key = f"gripper_{arm}"
            if controller_key in robot.controllers:
                try:
                    ctrl_goal = robot.controllers[controller_key].goal
                    if isinstance(ctrl_goal, dict) and "target" in ctrl_goal:
                        tgt = ctrl_goal["target"]
                        tgt_val = float(tgt[0]) if hasattr(tgt, '__iter__') else float(tgt)
                        upper_mean = float(robot.joint_upper_limits[
                            robot.gripper_control_idx[arm]].mean())
                        if tgt_val >= upper_mean * 0.5:
                            logger.info(
                                f"  [GRASP STATE] {arm}: controller target={tgt_val:.3f} "
                                f"(≥ {upper_mean*0.5:.3f}=OPEN) → skipping proximity"
                            )
                            continue
                except Exception as _eg:
                    logger.debug(f"  [GRASP RECOVERY] {arm}: controller goal check failed ({_eg})")
            # ─────────────────────────────────────────────────────────────────

            # ── ② Proximity fallback (no finger-state check) ────────────────
            # Rationale: finger joints in extraction-stage snapshots are unreliable
            # (replay load_state corrupts the _handle_assisted_grasping accumulator,
            # so snapshots often show one finger open and one half-closed), hence
            # the gripper joints are not a reliable "demo intends to hold" signal.
            # The reliable signal is the object's position relative to the EEF in
            # the snapshot:
            #   - really holding during the demo → load_state places the object
            #     near the EEF (within the threshold)
            #   - empty-handed during the demo → the object sits on a table/on the
            #     ground (far beyond the threshold)
            # Find the nearest movable object within the EEF proximity range;
            # found → establish the grasp; not found → skip.
            eef_pos, _ = robot.eef_links[arm].get_position_orientation()
            best_obj, best_link, best_dist = None, None, _prox_thresh

            for obj in robot.scene.objects:
                if obj is robot:
                    continue
                try:
                    if obj.fixed_base:
                        continue
                except AttributeError:
                    continue
                for link in obj.links.values():
                    try:
                        pos = link.get_position_orientation()[0]
                    except Exception:
                        continue
                    dist = float(th.norm(pos - eef_pos).item())
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = obj
                        best_link = link

            if best_obj is None:
                logger.info(
                    f"  [GRASP STATE] {arm}: no graspable object within {_prox_thresh} m of the EEF → no grasp"
                )
                continue

            try:
                robot._establish_grasp(arm=arm, ag_data=(best_obj, best_link), contact_pos=eef_pos)
                logger.info(
                    f"  [GRASP RECOVERY] ✓ {arm} → {best_obj.name}  dist={best_dist:.3f}m"
                    f" (proximity fallback)"
                )
            except Exception as e:
                logger.warning(f"  [GRASP RECOVERY] {arm}: _establish_grasp failed: {e}")

    # ── Diagnostics: print the final grasp state, default_arm, EEF positions ──
    for robot in env.robots:
        try:
            logger.info(f"  [GRASP DIAG] default_arm = {robot.default_arm}")
            for arm in robot.arm_names:
                held = robot._ag_obj_in_hand[arm]
                held_name = held.name if held is not None else "None"
                eef_pos, _ = robot.eef_links[arm].get_position_orientation()
                fz = robot._ag_freeze_gripper[arm]
                logger.info(
                    f"  [GRASP DIAG] {arm}: held={held_name}  "
                    f"EEF=({eef_pos[0]:.3f},{eef_pos[1]:.3f},{eef_pos[2]:.3f})  "
                    f"freeze={fz}"
                )
        except Exception as _e:
            logger.warning(f"  [GRASP DIAG] print failed: {_e}")


def _fix_gripper_targets_after_restore(env) -> None:
    """
    After og.sim.restore() + _recover_grasps_after_restore(), set the gripper
    joints' PhysX PD drive targets to the saved grasped position (gripper_pos).

    Without this fix: restore() leaves the joint drive targets at the lower
    limits (fully closed), so the step_physics() PD controller drives the gripper
    fully closed, which looks unnatural.

    gripper_pos is the actual joint position while grasping; after setting it,
    the gripper keeps its natural "wrapped around the object" pose.
    """
    for robot in env.robots:
        if robot.grasping_mode == "physical":
            continue
        for arm in robot.arm_names:
            if robot._ag_obj_in_hand[arm] is None:
                continue
            params = robot._ag_obj_constraint_params.get(arm, {})
            if not params:
                continue
            gripper_pos = params.get("gripper_pos")
            if gripper_pos is None:
                continue
            gripper_idx = robot.gripper_control_idx[arm]
            try:
                robot.set_joint_positions(gripper_pos, indices=gripper_idx, drive=True)
                logger.info(f"  [GRIPPER FIX] {arm}: drive target → {[round(v,4) for v in gripper_pos.tolist()]}")
            except Exception as e:
                logger.warning(f"  [GRIPPER FIX] {arm}: set failed: {e}")


def _fix_all_joint_drive_targets_after_restore(env) -> None:
    """
    og.sim.restore() only restores joint positions; it does not update the PhysX
    PD controller drive targets. If a target disagrees with the restored position,
    the PD controller drives the arm back to the wrong target with a large torque,
    causing violent arm jitter during the first few dozen steps of an episode
    (visible as gripper/arm trembling in the videos).

    This function sets every joint's (arm + gripper) drive target to the actual
    restored position, eliminating the mismatch between the PD controllers and
    the simulation state — the same effect env.scene.reset() has in eval.py.
    """
    for robot in env.robots:
        try:
            current_pos = robot.get_joint_positions()
            robot.set_joint_positions(current_pos, drive=True)
            logger.info(f"  [JOINT DRIVE FIX] {robot.name}: all {len(current_pos)} joint drive targets synced to the restored positions")
        except Exception as e:
            logger.warning(f"  [JOINT DRIVE FIX] {robot.name}: set failed: {e}")
