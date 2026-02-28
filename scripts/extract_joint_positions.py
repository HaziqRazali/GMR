"""
Extract world-space XYZ positions for every robot body/joint from a saved .npz motion file.

Usage:
    python scripts/extract_joint_positions.py \
        --motion_file unitree_g1_warmup_9_true.npz \
        --robot unitree_g1 \
        --output joint_positions.npz

Output .npz contains:
    joint_names  : list of body names (N_bodies,)
    joint_pos    : float32 array of shape (N_frames, N_bodies, 3)  [x, y, z] in world frame
    fps          : scalar
"""

import argparse
import pickle
import numpy as np
import mujoco as mj

from general_motion_retargeting import ROBOT_XML_DICT


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def extract(motion_path: str, robot: str, output_path: str | None = None, verbose: bool = True):
    # ── load saved motion ────────────────────────────────────────────────────
    data = load_motion(motion_path)
    root_pos = np.array(data["root_pos"])          # (T, 3)
    root_rot = np.array(data["root_rot"])          # (T, 4)  stored as xyzw
    dof_pos  = np.array(data["dof_pos"])           # (T, N_dof)
    fps      = float(data["fps"])
    T        = root_pos.shape[0]

    # ── load MuJoCo model ────────────────────────────────────────────────────
    xml_path = str(ROBOT_XML_DICT[robot])
    model = mj.MjModel.from_xml_path(xml_path)
    mj_data = mj.MjData(model)

    # body names (index 0 is the "world" body in MuJoCo, skip it)
    body_names = [model.body(i).name for i in range(model.nbody)]

    if verbose:
        print(f"Robot : {robot}")
        print(f"Frames: {T}  |  FPS: {fps}")
        print(f"Bodies: {len(body_names)}")
        print("Body list:", body_names)

    # ── run FK for every frame ───────────────────────────────────────────────
    joint_pos = np.zeros((T, model.nbody, 3), dtype=np.float32)

    for i in range(T):
        # MuJoCo stores root quaternion as wxyz; the file saves xyzw → convert back
        quat_xyzw = root_rot[i]                         # xyzw
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]             # → wxyz

        mj_data.qpos[:3]  = root_pos[i]
        mj_data.qpos[3:7] = quat_wxyz
        mj_data.qpos[7:]  = dof_pos[i]

        mj.mj_kinematics(model, mj_data)                # forward kinematics only (fast)

        joint_pos[i] = mj_data.xpos.copy()              # (nbody, 3) world-frame positions

    if verbose:
        print(f"\njoint_pos shape: {joint_pos.shape}  (frames, bodies, xyz)")
        print(f"Sample frame 0:\n")
        for name, pos in zip(body_names, joint_pos[0]):
            print(f"  {name:30s}  {pos}")

    # ── save ─────────────────────────────────────────────────────────────────
    if output_path is not None:
        np.savez(
            output_path,
            joint_names=np.array(body_names),
            joint_pos=joint_pos,
            fps=fps,
        )
        print(f"\nSaved to {output_path}")

    return body_names, joint_pos, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-joint XYZ from a robot motion .npz")
    parser.add_argument("--motion_file", required=True, help="Input .npz motion file")
    parser.add_argument(
        "--robot",
        required=True,
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
            "tienkung", "fourier_gr3",
        ],
    )
    parser.add_argument("--output", default=None, help="Where to save the output .npz (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-joint printout")
    args = parser.parse_args()

    extract(args.motion_file, args.robot, args.output, verbose=not args.quiet)
