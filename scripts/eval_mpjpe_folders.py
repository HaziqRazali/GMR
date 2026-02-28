"""
Compute per-motion MPJPE (hands / feet / all) by comparing two folders of
robot motion .npz files produced by batch_smplx_to_robot_fit3d_val.sh.

Default usage (fit3d val, unitree_g1, true vs generated):
    python scripts/eval_mpjpe_folders.py

Custom usage:
    python scripts/eval_mpjpe_folders.py \
        --true_dir   results/fit3d_val_g1 \
        --gen_dir    results/fit3d_val_g1_generated \
        --robot      unitree_g1
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import mujoco as mj

from general_motion_retargeting import ROBOT_XML_DICT


# ── FK ────────────────────────────────────────────────────────────────────────

def run_fk(path: str, model: mj.MjModel, mj_data: mj.MjData) -> np.ndarray:
    """Return (T, nbody, 3) world-frame positions."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    root_pos = np.array(d["root_pos"])
    root_rot = np.array(d["root_rot"])   # xyzw
    dof_pos  = np.array(d["dof_pos"])
    T        = root_pos.shape[0]

    out = np.zeros((T, model.nbody, 3), dtype=np.float64)
    for i in range(T):
        quat_wxyz = root_rot[i][[3, 0, 1, 2]]
        mj_data.qpos[:3]  = root_pos[i]
        mj_data.qpos[3:7] = quat_wxyz
        mj_data.qpos[7:]  = dof_pos[i]
        mj.mj_kinematics(model, mj_data)
        out[i] = mj_data.xpos.copy()
    return out


def body_indices(body_names: list[str], pattern: str) -> list[int]:
    return [i for i, n in enumerate(body_names) if re.search(pattern, n)]


def mpjpe_mm(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(pos_a - pos_b, axis=-1).mean()) * 1000


# ── pair discovery ────────────────────────────────────────────────────────────

def find_pairs(true_dir: str, gen_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (label, true_path, gen_path) for files present in both."""
    pairs = []
    for root, _, files in os.walk(true_dir):
        for fname in sorted(files):
            if not fname.endswith(".npz"):
                continue
            rel   = os.path.relpath(os.path.join(root, fname), true_dir)
            gen_p = os.path.join(gen_dir, rel)
            if os.path.exists(gen_p):
                label = rel.replace(os.sep, "/").replace(".npz", "")
                pairs.append((label, os.path.join(root, fname), gen_p))
    return sorted(pairs)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_dir", default="results/fit3d_val_g1")
    parser.add_argument("--gen_dir",  default="results/fit3d_val_g1_generated")
    parser.add_argument("--robot",    default="unitree_g1")
    parser.add_argument("--csv",      default=None,
                        help="Optional path to save results as CSV")
    args = parser.parse_args()

    # Load model once
    model  = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    mj_data = mj.MjData(model)
    body_names = [model.body(i).name for i in range(model.nbody)]

    world_idx  = body_names.index("world") if "world" in body_names else -1
    idx_all    = [i for i in range(len(body_names)) if i != world_idx]
    # Auto-detect best available arm end effectors: hand > wrist > elbow
    idx_hands = body_indices(body_names, r"hand")
    if not idx_hands:
        idx_hands = body_indices(body_names, r"wrist")
    if not idx_hands:
        idx_hands = body_indices(body_names, r"elbow")
    arm_label = ("hands" if body_indices(body_names, r"hand") else
                 "wrists" if body_indices(body_names, r"wrist") else
                 "elbows" if idx_hands else "arms")

    # Auto-detect best available leg end effectors: toe > ankle > knee
    idx_feet = body_indices(body_names, r"toe")
    if not idx_feet:
        idx_feet = body_indices(body_names, r"ankle")
    if not idx_feet:
        idx_feet = body_indices(body_names, r"knee")
    leg_label = ("toes" if body_indices(body_names, r"toe") else
                 "ankles" if body_indices(body_names, r"ankle") else
                 "knees" if idx_feet else "legs")

    print(f"\nRobot : {args.robot}")
    print(f"True  : {args.true_dir}")
    print(f"Gen   : {args.gen_dir}")
    print(f"Arm end-effectors ({arm_label}): {[body_names[i] for i in idx_hands]}")
    print(f"Leg end-effectors ({leg_label}): {[body_names[i] for i in idx_feet]}")

    pairs = find_pairs(args.true_dir, args.gen_dir)
    if not pairs:
        print("\nNo matching pairs found. Check that both directories exist and have .npz files.")
        sys.exit(1)
    print(f"\nPairs found: {len(pairs)}\n")

    col = 32
    print(f"{'Motion':<{col}}  {'all':>10}  {arm_label:>10}  {leg_label:>10}")
    print("-" * (col + 36))

    rows = []
    agg = {"all": [], "hands": [], "feet": []}

    for label, true_p, gen_p in pairs:
        pos_true = run_fk(true_p, model, mj_data)
        pos_gen  = run_fk(gen_p,  model, mj_data)
        T = min(len(pos_true), len(pos_gen))
        pos_true, pos_gen = pos_true[:T], pos_gen[:T]

        r = {
            "all":   mpjpe_mm(pos_gen[:, idx_all],   pos_true[:, idx_all]),
            "hands": mpjpe_mm(pos_gen[:, idx_hands],  pos_true[:, idx_hands]) if idx_hands else float("nan"),
            "feet":  mpjpe_mm(pos_gen[:, idx_feet],   pos_true[:, idx_feet])  if idx_feet  else float("nan"),
        }
        print(f"  {label:<{col-2}}  {r['all']:>9.2f}mm  {r['hands']:>9.2f}mm  {r['feet']:>9.2f}mm")
        rows.append((label, r["all"], r["hands"], r["feet"]))
        for k in agg:
            if not np.isnan(r[k]):
                agg[k].append(r[k])

    print("-" * (col + 36))
    means = {k: np.mean(v) for k, v in agg.items()}
    print(f"  {'MEAN (n=' + str(len(pairs)) + ')':<{col-2}}  "
          f"{means['all']:>9.2f}mm  {means['hands']:>9.2f}mm  {means['feet']:>9.2f}mm")
    print(f"  (arm column = {arm_label}, leg column = {leg_label})\n")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["motion", "mpjpe_all_mm", "mpjpe_hands_mm", "mpjpe_feet_mm"])
            for row in rows:
                w.writerow(row)
            w.writerow(["MEAN", means["all"], means["hands"], means["feet"]])
        print(f"Saved CSV → {args.csv}")


if __name__ == "__main__":
    main()
