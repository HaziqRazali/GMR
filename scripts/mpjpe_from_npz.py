"""
Compute MPJPE between _generated and _true robot motion files using FK.

FK is run on the already-saved (root_pos, root_rot, dof_pos) from smplx_to_robot.py output,
so no IK is needed — this is fast.

Usage:
    # Evaluate all pairs in the workspace:
    python scripts/mpjpe_from_npz.py --robot unitree_g1

    # Or specify explicit pairs:
    python scripts/mpjpe_from_npz.py \
        --generated unitree_g1_warmup_9_generated.npz \
        --true      unitree_g1_warmup_9_true.npz \
        --robot     unitree_g1

Categories reported:
    hands  - bodies whose name contains 'hand' or 'wrist'
    feet   - bodies whose name contains 'toe' or 'ankle'
    all    - all non-world bodies
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import mujoco as mj

from general_motion_retargeting import ROBOT_XML_DICT


# ── FK helper ─────────────────────────────────────────────────────────────────

def run_fk(npz_path: str, model: mj.MjModel) -> tuple[np.ndarray, list[str]]:
    """Return (joint_pos, body_names) where joint_pos is (T, nbody, 3)."""
    with open(npz_path, "rb") as f:
        data = pickle.load(f)

    root_pos = np.array(data["root_pos"])   # (T, 3)
    root_rot = np.array(data["root_rot"])   # (T, 4)  xyzw
    dof_pos  = np.array(data["dof_pos"])    # (T, N_dof)
    T        = root_pos.shape[0]

    mj_data = mj.MjData(model)
    body_names = [model.body(i).name for i in range(model.nbody)]
    joint_pos  = np.zeros((T, model.nbody, 3), dtype=np.float64)

    for i in range(T):
        quat_xyzw = root_rot[i]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]   # mujoco wants wxyz
        mj_data.qpos[:3]  = root_pos[i]
        mj_data.qpos[3:7] = quat_wxyz
        mj_data.qpos[7:]  = dof_pos[i]
        mj.mj_kinematics(model, mj_data)
        joint_pos[i] = mj_data.xpos.copy()

    return joint_pos, body_names


# ── MPJPE helper ──────────────────────────────────────────────────────────────

def mpjpe(pos_gen: np.ndarray, pos_true: np.ndarray) -> float:
    """Mean per-joint position error (metres) over all frames and selected joints."""
    diff = pos_gen - pos_true               # (T, J, 3)
    dist = np.linalg.norm(diff, axis=-1)    # (T, J)
    return float(dist.mean())


def body_indices(body_names: list[str], pattern: str) -> list[int]:
    """Return indices of bodies whose name matches a regex pattern."""
    return [i for i, n in enumerate(body_names) if re.search(pattern, n)]


# ── per-pair evaluation ───────────────────────────────────────────────────────

def evaluate_pair(gen_path: str, true_path: str, model: mj.MjModel,
                  body_names: list[str]) -> dict:
    T_gen  = pickle.load(open(gen_path,  "rb"))["root_pos"].shape[0]
    T_true = pickle.load(open(true_path, "rb"))["root_pos"].shape[0]
    T = min(T_gen, T_true)

    pos_gen,  _ = run_fk(gen_path,  model)
    pos_true, _ = run_fk(true_path, model)

    pos_gen  = pos_gen [:T]
    pos_true = pos_true[:T]

    # skip body index 0 ("world" – always at origin)
    world_idx = body_names.index("world") if "world" in body_names else -1

    idx_all   = [i for i in range(len(body_names)) if i != world_idx]
    idx_hands = body_indices(body_names, r"hand|wrist")
    idx_feet  = body_indices(body_names, r"toe|ankle")

    results = {}
    for label, idx in [("all", idx_all), ("hands", idx_hands), ("feet", idx_feet)]:
        if idx:
            results[label] = mpjpe(pos_gen[:, idx], pos_true[:, idx]) * 1000  # → mm
        else:
            results[label] = float("nan")
    return results


# ── auto-discover pairs ───────────────────────────────────────────────────────

def find_pairs(robot: str) -> list[tuple[str, str, str]]:
    """Return list of (label, gen_path, true_path)."""
    pairs = []
    gen_files = sorted(glob.glob(f"{robot}_*_generated.npz"))
    for gen in gen_files:
        true = gen.replace("_generated.npz", "_true.npz")
        if os.path.exists(true):
            label = os.path.basename(gen).replace(f"{robot}_", "").replace("_generated.npz", "")
            pairs.append((label, gen, true))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True,
                        help="Robot name (e.g. unitree_g1, tienkung)")
    parser.add_argument("--generated", default=None,
                        help="Explicit generated .npz path")
    parser.add_argument("--true",      default=None,
                        help="Explicit true .npz path")
    parser.add_argument("--label",     default=None,
                        help="Label for explicit pair")
    args = parser.parse_args()

    xml_path = str(ROBOT_XML_DICT[args.robot])
    model = mj.MjModel.from_xml_path(xml_path)
    body_names = [model.body(i).name for i in range(model.nbody)]

    print(f"\nRobot : {args.robot}  ({len(body_names)} bodies)")
    print(f"Hands bodies : {[body_names[i] for i in body_indices(body_names, r'hand|wrist')]}")
    print(f"Feet  bodies : {[body_names[i] for i in body_indices(body_names, r'toe|ankle')]}")
    print()

    # Build pairs list
    if args.generated and args.true:
        label = args.label or os.path.basename(args.generated)
        pairs = [(label, args.generated, args.true)]
    else:
        pairs = find_pairs(args.robot)
        if not pairs:
            print(f"No *_generated.npz / *_true.npz pairs found for robot '{args.robot}'")
            sys.exit(1)

    # Header
    col = 28
    print(f"{'Motion':<{col}}  {'MPJPE-all':>12}  {'MPJPE-hands':>12}  {'MPJPE-feet':>12}")
    print("-" * (col + 42))

    all_metrics: dict[str, list] = {"all": [], "hands": [], "feet": []}

    for label, gen_path, true_path in pairs:
        print(f"  {label:<{col-2}}", end="  ", flush=True)
        res = evaluate_pair(gen_path, true_path, model, body_names)
        print(f"{res['all']:>11.2f}mm  {res['hands']:>11.2f}mm  {res['feet']:>11.2f}mm")
        for k in all_metrics:
            if not np.isnan(res[k]):
                all_metrics[k].append(res[k])

    if len(pairs) > 1:
        print("-" * (col + 42))
        print(f"{'MEAN':<{col}}  "
              f"{np.mean(all_metrics['all']):>11.2f}mm  "
              f"{np.mean(all_metrics['hands']):>11.2f}mm  "
              f"{np.mean(all_metrics['feet']):>11.2f}mm")

    print()


if __name__ == "__main__":
    main()
