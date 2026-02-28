"""
Evaluate MPJPE (Mean Per Joint Position Error) between two IK configs on the
Fit3D validation set.

For each JSON file in /home/haziq/datasets/mocap/data/fit3d/val/*/smplx/*.json:
  1. Retarget with the "generated" IK config  → joint XYZ via MuJoCo FK
  2. Retarget with the "true" (reference) IK config → joint XYZ via MuJoCo FK
  3. Compute MPJPE(generated, true) for:
        - "all"   : all robot bodies (excluding the fixed world body)
        - "hands" : left_rubber_hand, right_rubber_hand
        - "feet"  : left_toe_link, right_toe_link

Usage:
    conda run -n gmr python scripts/eval_mpjpe_fit3d_val.py

Optional flags:
    --robot      <robot_name>   (default: unitree_g1)
    --gen_config <path>         (default: smplx_to_g1_generated.json)
    --true_config <path>        (default: smplx_to_g1.json)
    --val_dir    <path>         (default: /home/haziq/datasets/mocap/data/fit3d/val)
    --max_files  <int>          Limit number of files (for quick testing)
    --output     <csv_path>     Save per-file results to CSV
"""

import argparse
import json
import pathlib
import sys
import traceback

import mujoco as mj
import numpy as np

HERE = pathlib.Path(__file__).parent.parent  # project root

sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR, ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl_json import load_smplx_json_file
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

SMPLX_FOLDER = HERE / "assets" / "body_models"

# ── joint category definitions per robot ────────────────────────────────────
HAND_JOINT_KEYWORDS = ["rubber_hand", "gripper", "hand_link", "wrist_yaw"]
FOOT_JOINT_KEYWORDS = ["toe_link", "ankle_roll", "foot_link"]


def categorise_joints(body_names):
    """Return index sets for hands, feet, and all (non-world)."""
    idx_all   = [i for i, n in enumerate(body_names) if n != "world"]
    idx_hands = [i for i, n in enumerate(body_names)
                 if any(kw in n for kw in HAND_JOINT_KEYWORDS)]
    idx_feet  = [i for i, n in enumerate(body_names)
                 if any(kw in n for kw in FOOT_JOINT_KEYWORDS)]
    return idx_all, idx_hands, idx_feet


def retarget_file(smplx_path: pathlib.Path, robot: str, ik_config: str):
    """
    Retarget one SMPL-X JSON file → returns list of qpos arrays (wxyz root quat).
    """
    smplx_data_raw, body_model, smplx_output, actual_human_height = load_smplx_json_file(
        str(smplx_path), SMPLX_FOLDER, fps=50  # fit3d is 50 fps
    )

    tgt_fps = 30
    smplx_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data_raw, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        ik_config_path=ik_config,
    )

    qpos_list = []
    for frame_data in smplx_frames[1:]:   # skip frame 0 (init frame)
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos)

    return qpos_list, aligned_fps


def qpos_to_joint_pos(qpos_list, model, mj_data):
    """
    Run MuJoCo FK for every qpos → return (T, nbody, 3) world-frame positions.
    qpos root quaternion convention: wxyz (MuJoCo native from GMR.retarget).
    """
    T = len(qpos_list)
    joint_pos = np.zeros((T, model.nbody, 3), dtype=np.float32)
    for i, qpos in enumerate(qpos_list):
        mj_data.qpos[:3]  = qpos[:3]
        mj_data.qpos[3:7] = qpos[3:7]   # already wxyz
        mj_data.qpos[7:]  = qpos[7:]
        mj.mj_kinematics(model, mj_data)
        joint_pos[i] = mj_data.xpos.copy()
    return joint_pos


def mpjpe(pos_gen, pos_true, idx_joints):
    """
    MPJPE in metres.
    pos_gen, pos_true: (T, nbody, 3)
    idx_joints: list of body indices to include
    """
    diff = pos_gen[:, idx_joints, :] - pos_true[:, idx_joints, :]  # (T, J, 3)
    per_joint_err = np.linalg.norm(diff, axis=-1)                   # (T, J)
    return float(per_joint_err.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot",       default="unitree_g1")
    parser.add_argument("--gen_config",  default=None,
                        help="Path to generated IK config. Defaults to smplx_to_g1_generated.json")
    parser.add_argument("--true_config", default=None,
                        help="Path to true/reference IK config. Defaults to smplx_to_g1.json")
    parser.add_argument("--val_dir",     default="/home/haziq/datasets/mocap/data/fit3d/val")
    parser.add_argument("--max_files",   type=int, default=None,
                        help="Limit number of files processed (for quick tests)")
    parser.add_argument("--output",      default=None,
                        help="Save per-file CSV results to this path")
    args = parser.parse_args()

    # Resolve config paths based on robot if not specified
    robot_config_suffix = args.robot.replace("unitree_", "").replace("_", "")
    # Map robot name to config suffix
    robot_to_config = {
        "unitree_g1": "g1",
        "tienkung": "tienkung",
        "booster_t1": "t1",
        "booster_k1": "k1",
        "stanford_toddy": "toddy",
        "fourier_n1": "n1",
        "engineai_pm01": "pm01",
        "kuavo_s45": "kuavo",
        "hightorque_hi": "hi",
        "galaxea_r1pro": "r1pro",
    }
    suffix = robot_to_config.get(args.robot, args.robot)
    ik_cfg_dir = HERE / "general_motion_retargeting" / "ik_configs"

    gen_config  = args.gen_config  or str(ik_cfg_dir / f"smplx_to_{suffix}_generated.json")
    true_config = args.true_config or str(ik_cfg_dir / f"smplx_to_{suffix}.json")

    print(f"Robot        : {args.robot}")
    print(f"Gen config   : {gen_config}")
    print(f"True config  : {true_config}")
    print(f"Val dir      : {args.val_dir}")

    # ── collect all smplx json files ────────────────────────────────────────
    val_dir = pathlib.Path(args.val_dir)
    all_files = sorted(val_dir.glob("*/smplx/*.json"))
    if args.max_files:
        all_files = all_files[:args.max_files]
    print(f"Files found  : {len(all_files)}\n")

    # ── set up MuJoCo model (shared across files) ────────────────────────────
    xml_path = str(ROBOT_XML_DICT[args.robot])
    model  = mj.MjModel.from_xml_path(xml_path)
    mj_data = mj.MjData(model)
    body_names = [model.body(i).name for i in range(model.nbody)]
    idx_all, idx_hands, idx_feet = categorise_joints(body_names)

    print(f"Bodies total : {len(body_names)}")
    print(f"  all (non-world) : {len(idx_all)} joints")
    print(f"  hands           : {[body_names[i] for i in idx_hands]}")
    print(f"  feet            : {[body_names[i] for i in idx_feet]}")
    print()

    # ── per-file results ──────────────────────────────────────────────────────
    results = []   # list of dicts

    for file_idx, smplx_path in enumerate(all_files):
        rel = smplx_path.relative_to(val_dir)
        print(f"[{file_idx+1}/{len(all_files)}] {rel} ...", end=" ", flush=True)

        try:
            # retarget with both configs
            qpos_gen,  _ = retarget_file(smplx_path, args.robot, gen_config)
            qpos_true, _ = retarget_file(smplx_path, args.robot, true_config)

            # align lengths (very rare floating-point FPS rounding difference)
            T = min(len(qpos_gen), len(qpos_true))
            qpos_gen  = qpos_gen[:T]
            qpos_true = qpos_true[:T]

            # FK → world-space joint positions
            pos_gen  = qpos_to_joint_pos(qpos_gen,  model, mj_data)
            pos_true = qpos_to_joint_pos(qpos_true, model, mj_data)

            mpjpe_all   = mpjpe(pos_gen, pos_true, idx_all)   * 1000  # → mm
            mpjpe_hands = mpjpe(pos_gen, pos_true, idx_hands) * 1000
            mpjpe_feet  = mpjpe(pos_gen, pos_true, idx_feet)  * 1000

            print(f"all={mpjpe_all:.1f}mm  hands={mpjpe_hands:.1f}mm  feet={mpjpe_feet:.1f}mm  (T={T})")

            results.append({
                "file":        str(rel),
                "subject":     smplx_path.parts[-3],
                "action":      smplx_path.stem,
                "frames":      T,
                "mpjpe_all_mm":   mpjpe_all,
                "mpjpe_hands_mm": mpjpe_hands,
                "mpjpe_feet_mm":  mpjpe_feet,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            continue

    # ── aggregate ────────────────────────────────────────────────────────────
    if results:
        mpjpe_all_vals   = [r["mpjpe_all_mm"]   for r in results]
        mpjpe_hands_vals = [r["mpjpe_hands_mm"] for r in results]
        mpjpe_feet_vals  = [r["mpjpe_feet_mm"]  for r in results]

        print("\n" + "="*60)
        print(f"RESULTS  ({len(results)}/{len(all_files)} files succeeded)")
        print("="*60)
        print(f"  MPJPE All   : {np.mean(mpjpe_all_vals):.2f} ± {np.std(mpjpe_all_vals):.2f} mm")
        print(f"  MPJPE Hands : {np.mean(mpjpe_hands_vals):.2f} ± {np.std(mpjpe_hands_vals):.2f} mm")
        print(f"  MPJPE Feet  : {np.mean(mpjpe_feet_vals):.2f} ± {np.std(mpjpe_feet_vals):.2f} mm")
        print("="*60)

        # per-subject breakdown
        subjects = sorted(set(r["subject"] for r in results))
        if len(subjects) > 1:
            print("\nPer-subject breakdown:")
            for subj in subjects:
                sr = [r for r in results if r["subject"] == subj]
                print(f"  {subj}  ({len(sr)} files)")
                print(f"    all={np.mean([r['mpjpe_all_mm']   for r in sr]):.2f}mm  "
                      f"hands={np.mean([r['mpjpe_hands_mm'] for r in sr]):.2f}mm  "
                      f"feet={np.mean([r['mpjpe_feet_mm']   for r in sr]):.2f}mm")

        # ── save CSV ─────────────────────────────────────────────────────────
        if args.output:
            import csv
            out_path = pathlib.Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\nPer-file results saved to {out_path}")


if __name__ == "__main__":
    main()
