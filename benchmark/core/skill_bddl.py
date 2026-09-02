"""BDDL utilities: predefined_problem building, block parsing and scene wildcard expansion.

(Named skill_bddl rather than bddl to avoid shadowing the third-party bddl package.)
"""
import re
import logging
from omnigibson.utils.bddl_utils import get_processed_bddl
logger = logging.getLogger("skill_eval")


def build_predefined_problem(task_name: str, objects_bddl: str, init_bddl: str, bddl_goal: str) -> str:
    """
    Build a predefined_problem BDDL string.

    The :init content does not matter (it gets overwritten by og.sim.load_state());
    it only needs to be logically consistent with :objects so that BehaviorTask's
    object_scope resolution succeeds.
    """
    return (
        f"(define (problem {task_name}-0)\n"
        f"    (:domain omnigibson)\n\n"
        f"    {objects_bddl.strip()}\n\n"
        f"    {init_bddl.strip()}\n\n"
        f"    (:goal\n"
        f"        {bddl_goal.strip()}\n"
        f"    )\n"
        f")"
    )


def _extract_bddl_block(bddl_str: str, keyword: str) -> str:
    """
    Extract a top-level block such as (:objects ...) or (:init ...) from a BDDL
    string via parenthesis-depth tracking. More robust than regex; supports
    arbitrarily nested () content.
    """
    start = bddl_str.find(f"(:{keyword}")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(bddl_str)):
        if bddl_str[i] == "(":
            depth += 1
        elif bddl_str[i] == ")":
            depth -= 1
            if depth == 0:
                return bddl_str[start : i + 1]
    return None


def _parse_bddl_object_instances(objects_bddl: str) -> dict:
    """
    Parse the (:objects ...) block, returning an {instance_name: category} mapping.
    Wildcard instances (ending with _*) are included in the result.
    """
    content = objects_bddl.strip()
    inner = re.sub(r'^\(:objects\s*', '', content, flags=re.IGNORECASE)
    inner = re.sub(r'\)\s*$', '', inner)
    result = {}
    for line in inner.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if " - " in line:
            left, category = line.rsplit(" - ", 1)
            for inst in left.strip().split():
                result[inst.strip()] = category.strip()
    return result


def _select_objects_bddl(
    yaml_objects: str, yaml_init: str, exp_objects, exp_init
) -> tuple:
    """
    Select or merge objects_bddl / init_bddl depending on wildcard presence in the YAML.

    Case                          Result
    YAML has no _*                use the YAML as-is (extra objects such as countertop.n.01_1 are kept)
    YAML has _*, expansion failed  fall back to the YAML (wildcards kept, quantifier coverage may be partial)
    YAML has _*, expansion worked  use the expansion as the base and inject YAML-only instances

    Args:
        yaml_objects: objects_bddl from the YAML (raw, may contain _*)
        yaml_init:    init_bddl from the YAML
        exp_objects:  get_processed_bddl expansion result (may be None)
        exp_init:     get_processed_bddl expansion result (may be None)

    Returns:
        (final_objects_bddl, final_init_bddl)
    """
    yaml_has_wildcard = "_*" in yaml_objects

    if not yaml_has_wildcard or exp_objects is None:
        # No wildcards, or expansion failed: use the YAML as-is, keeping all extra objects
        return yaml_objects, yaml_init

    # ── Wildcards present and expansion succeeded: merge YAML-only instances ──
    yaml_inst_map = _parse_bddl_object_instances(yaml_objects)
    exp_inst_map = _parse_bddl_object_instances(exp_objects)

    # Non-wildcard instances present in the YAML but missing from the expansion (e.g. countertop.n.01_1)
    yaml_extra = {
        inst: cat for inst, cat in yaml_inst_map.items()
        if not inst.endswith("_*") and inst not in exp_inst_map
    }

    if not yaml_extra:
        return exp_objects, exp_init

    # Append yaml_extra just before the closing ) of exp_objects
    extra_obj_lines = "\n".join(
        f"      {inst} - {cat}" for inst, cat in yaml_extra.items()
    )
    merged_objects = exp_objects.rstrip()
    if merged_objects.endswith(")"):
        merged_objects = merged_objects[:-1].rstrip() + "\n" + extra_obj_lines + "\n  )"
    else:
        merged_objects = merged_objects + "\n" + extra_obj_lines

    # init: append the YAML init entries that involve the yaml_extra instances to exp_init
    extra_insts = set(yaml_extra.keys())
    extra_init_lines = []
    for raw_line in yaml_init.splitlines():
        s = raw_line.strip()
        if s and not s.startswith(";"):
            for inst in extra_insts:
                if inst in s:
                    extra_init_lines.append(f"      {s}")
                    break

    if extra_init_lines:
        merged_init = exp_init.rstrip()
        if merged_init.endswith(")"):
            merged_init = merged_init[:-1].rstrip() + "\n" + "\n".join(extra_init_lines) + "\n  )"
        else:
            merged_init = merged_init + "\n" + "\n".join(extra_init_lines)
    else:
        merged_init = exp_init

    logger.info(
        f"  [select_objects] injected YAML-only instances into the expansion: {list(yaml_extra.keys())}"
    )
    return merged_objects, merged_init


def _expand_wildcards_from_scene(task_name: str, activity_definition_id: int, env) -> tuple:
    """
    Expand BDDL wildcards (e.g. tree.n.01_*) using the loaded scene, returning the
    expanded (objects_bddl_str, init_bddl_str).

    Background:
      When a predefined_problem is passed directly to BehaviorTask, its
      update_activity() skips get_processed_bddl(), so tree.n.01_* stays as a
      literal instance name instead of being expanded into all tree instances
      actually present in the scene (tree.n.01_2, tree.n.01_3, ...).

      The BDDL engine's existential/universal quantifiers only iterate over
      object_map[category]; without expansion only [tree.n.01_1, tree.n.01_*]
      are iterable, which means:
        - (exists (?tree.n.01 ...) (near agent ?tree)) checks only 2 specific trees
        - (nextto egg ?tree.n.01) is likewise evaluated against only those 2 trees
      When the robot walks to (or places an object next to) any other tree, the
      BDDL engine still returns False even though it physically succeeded.

    Fix: call get_processed_bddl() with the loaded env.scene to expand the
    wildcards in objects_bddl into concrete instances matching the actual scene
    objects, then build the predefined_problem from the expansion.

    Returns:
        (objects_bddl, init_bddl): the expanded BDDL block strings;
        (None, None) if the expansion fails.
    """
    try:
        expanded = get_processed_bddl(task_name, activity_definition_id, env.scene)
        objects_bddl = _extract_bddl_block(expanded, "objects")
        init_bddl = _extract_bddl_block(expanded, "init")
        if objects_bddl and init_bddl:
            logger.info(
                f"  [expand_wildcards] wildcard expansion succeeded, "
                f"objects_bddl length: {len(objects_bddl)} chars"
            )
            return objects_bddl, init_bddl
        logger.warning(f"  [expand_wildcards] failed to parse the expanded BDDL blocks; falling back to the raw objects_bddl")
    except Exception as _e:
        logger.warning(f"  [expand_wildcards] get_processed_bddl failed; falling back to the raw objects_bddl: {_e}")
    return None, None
