"""
Retarget MHR .npz motion to a humanoid robot.
Mirrors smplx_to_robot.py exactly — no normalisation applied.
Raw MHR global orientations are fed directly into the IK.

This is intentional: the script is used to demonstrate what happens when
the rest pose of the source body model differs from SMPL-X T-pose.

Usage (mhr_new env):
    python scripts/mhr_to_robot.py \
        --mhr_file /path/to/motion.npz \
        --robot booster_t1 \
        --rate_limit \
        --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

from rich import print

# ---------------------------------------------------------------------------
# MHR joint index → name used in mhr_to_*.json IK configs
# ---------------------------------------------------------------------------
_MHR_IK_JOINTS = {
    1:  "root",
    2:  "l_upleg",
    18: "r_upleg",
    3:  "l_lowleg",
    19: "r_lowleg",
    37: "c_spine3",
    8:  "l_ball",
    24: "r_ball",
    75: "l_uparm",
    39: "r_uparm",
    76: "l_lowarm",
    40: "r_lowarm",
}
_MHR_IK_INDICES = sorted(_MHR_IK_JOINTS.keys())

_HEAD_IDX  = 113
_LBALL_IDX = 8
_RBALL_IDX = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slerp(r1, r2, t):
    q1 = r1.as_quat(); q2 = r2.as_quat()
    q1 /= np.linalg.norm(q1); q2 /= np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0: q2 = -q2; dot = -dot
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    theta0 = np.arccos(np.clip(dot, -1, 1))
    theta  = theta0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return R.from_quat(s0 * q1 + s1 * q2)


# ---------------------------------------------------------------------------
# MHR loading — raw orientations, no normalisation
# ---------------------------------------------------------------------------

def load_mhr_npz(mhr_file, mhr_root="~/MHR", device="cpu", batch_size=256, fps=None):
    mhr_root = os.path.expanduser(mhr_root)
    if mhr_root not in sys.path:
        sys.path.insert(0, mhr_root)
    from mhr.mhr import MHR  # type: ignore

    data    = np.load(mhr_file, allow_pickle=True)
    T       = data["param_lbs_model_params"].shape[0]
    model_p = torch.tensor(data["param_lbs_model_params"], dtype=torch.float32)
    shape_p = torch.tensor(data["param_identity_coeffs"],  dtype=torch.float32)
    expr_p  = torch.tensor(data["param_face_expr_coeffs"], dtype=torch.float32)

    dev = torch.device(device)
    print(f"[MHR] Loading model (device={dev}) ...")
    mhr_model = MHR.from_files(device=dev, lod=1)

    all_skel = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        with torch.no_grad():
            _, skel = mhr_model(shape_p[start:end].to(dev),
                                model_p[start:end].to(dev),
                                expr_p [start:end].to(dev))
        all_skel.append(skel.cpu())
        if start == 0 or start % (batch_size * 4) == 0:
            print(f"  frames {start}-{end} / {T}")
    skel_state = torch.cat(all_skel, dim=0).numpy()  # (T, 127, 8)

    # T-pose forward pass — only used for height estimation
    with torch.no_grad():
        _, tpose_skel = mhr_model(
            shape_p[0:1].to(dev),
            torch.zeros(1, 204, device=dev),
            torch.zeros(1,  72, device=dev),
        )
    tpose_np = tpose_skel[0].cpu().numpy()  # (127, 8)

    ik_idxs = np.array(_MHR_IK_INDICES)

    # Raw quaternions: xyzw from MHR -> wxyz for GMR
    q_xyzw = skel_state[:, ik_idxs, 3:7]              # (T, N_ik, 4)
    quats_wxyz = q_xyzw[:, :, [3, 0, 1, 2]]           # (T, N_ik, 4) wxyz

    # Positions: cm -> m
    positions = skel_state[:, ik_idxs, :3] / 100.0    # (T, N_ik, 3)

    head_y = tpose_np[_HEAD_IDX,  1] / 100.0
    foot_y = min(tpose_np[_LBALL_IDX, 1], tpose_np[_RBALL_IDX, 1]) / 100.0
    human_height = (head_y - foot_y) + 0.15
    print(f"  Estimated human height: {human_height:.3f} m")

    joint_names = [_MHR_IK_JOINTS[idx] for idx in _MHR_IK_INDICES]
    src_fps = fps if fps is not None else 50
    print(f"[MHR] {T} frames at {src_fps} FPS")

    return positions, quats_wxyz, joint_names, src_fps, human_height


def get_mhr_frames(positions, quats_wxyz, joint_names, src_fps, tgt_fps=30):
    T = positions.shape[0]

    if tgt_fps < src_fps:
        frame_skip  = int(src_fps / tgt_fps)
        new_T       = T // frame_skip
        orig_time   = np.arange(T)
        target_time = np.linspace(0, T - 1, new_T)
        N_j         = positions.shape[1]

        pos_out = np.empty((new_T, N_j, 3), dtype=np.float32)
        for j in range(N_j):
            for d in range(3):
                pos_out[:, j, d] = interp1d(orig_time, positions[:, j, d])(target_time)

        quat_out = np.empty((new_T, N_j, 4), dtype=np.float32)
        for j in range(N_j):
            for k, t_val in enumerate(target_time):
                i1 = int(np.floor(t_val)); i2 = min(i1 + 1, T - 1)
                alpha = t_val - i1
                q1 = quats_wxyz[i1, j]; q2 = quats_wxyz[i2, j]
                r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
                r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
                quat_out[k, j] = _slerp(r1, r2, alpha).as_quat(scalar_first=True)

        positions   = pos_out
        quats_wxyz  = quat_out
        aligned_fps = float(new_T) / T * src_fps
    else:
        aligned_fps = float(tgt_fps)

    frames = [{name: (positions[t, j], quats_wxyz[t, j])
               for j, name in enumerate(joint_names)}
              for t in range(len(positions))]

    return frames, aligned_fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--mhr_file", required=True)
    parser.add_argument("--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
                 "tienkung", "fourier_gr3"],
        default="booster_t1",
    )
    parser.add_argument("--mhr_root",         default="~/MHR")
    parser.add_argument("--device",           default="cpu")
    parser.add_argument("--fps",              type=int,   default=None)
    parser.add_argument("--save_path",        default=None)
    parser.add_argument("--loop",             action="store_true")
    parser.add_argument("--record_video",     action="store_true")
    parser.add_argument("--video_path",       type=str,   default=None)
    parser.add_argument("--rate_limit",       action="store_true")
    parser.add_argument("--camera_distance",  type=float, default=None)
    parser.add_argument("--camera_elevation", type=float, default=-10)
    parser.add_argument("--camera_height",    type=float, default=0.0)
    parser.add_argument("--rotate_roll",      type=float, default=0.0)
    parser.add_argument("--rotate_yaw",       type=float, default=0.0)
    parser.add_argument("--rotate_pitch",     type=float, default=0.0)
    parser.add_argument("--hide_floor",       action="store_true")
    parser.add_argument("--ik_config",        type=str,   default=None)
    parser.add_argument("--no_viewer",        action="store_true")
    parser.add_argument("--freeze_at_end",    action="store_true")
    args = parser.parse_args()

    # 1. Load
    positions, quats_wxyz, joint_names, src_fps, human_height = load_mhr_npz(
        mhr_file=args.mhr_file,
        mhr_root=args.mhr_root,
        device=args.device,
        fps=args.fps,
    )

    # 2. Optional global rotation (same as smplx_to_robot.py)
    if args.rotate_roll != 0.0 or args.rotate_yaw != 0.0 or args.rotate_pitch != 0.0:
        combined_rot = (
            R.from_euler("x", args.rotate_roll,  degrees=True) *
            R.from_euler("z", args.rotate_yaw,   degrees=True) *
            R.from_euler("y", args.rotate_pitch, degrees=True)
        )
        T, N_j, _ = positions.shape
        positions = combined_rot.apply(positions.reshape(-1, 3)).reshape(T, N_j, 3).astype(np.float32)
        for t in range(T):
            for j in range(N_j):
                q = quats_wxyz[t, j]
                r_orig = R.from_quat([q[1], q[2], q[3], q[0]])
                quats_wxyz[t, j] = (combined_rot * r_orig).as_quat(scalar_first=True)

    # 3. FPS alignment
    tgt_fps = 30
    mhr_frames, aligned_fps = get_mhr_frames(
        positions, quats_wxyz, joint_names, src_fps, tgt_fps=tgt_fps
    )
    print(f"[MHR] {len(mhr_frames)} frames at {aligned_fps:.1f} FPS")

    # 4. GMR
    retarget = GMR(
        actual_human_height=human_height,
        src_human="mhr",
        tgt_robot=args.robot,
        ik_config_path=args.ik_config,
    )

    # 5. Viewer
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=(args.video_path if args.video_path is not None
                        else f"videos/{args.robot}_{os.path.basename(args.mhr_file).split('.')[0]}.mp4"),
            camera_distance=args.camera_distance,
            camera_elevation=args.camera_elevation,
            camera_height=args.camera_height,
            hide_floor=args.hide_floor,
        )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    fps_counter    = 0
    fps_start_time = time.time()

    # 6. Main loop
    i = -1
    while True:
        if args.loop:
            i = (i + 1) % len(mhr_frames)
        else:
            i += 1
            if i >= len(mhr_frames):
                if args.freeze_at_end and not args.no_viewer:
                    print("[freeze_at_end] Holding last frame. Close the window to exit.")
                    while robot_motion_viewer.viewer.is_running():
                        robot_motion_viewer.step(
                            root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                            human_motion_data=retarget.scaled_human_data,
                            human_pos_offset=np.array([0.0, 0.0, 0.0]),
                            show_human_body_name=False,
                            rate_limit=args.rate_limit,
                            follow_camera=False,
                        )
                break

        fps_counter += 1
        now = time.time()
        if now - fps_start_time >= 2.0:
            print(f"Actual rendering FPS: {fps_counter / (now - fps_start_time):.2f}")
            fps_counter = 0
            fps_start_time = now

        qpos = retarget.retarget(mhr_frames[i])

        if not args.no_viewer:
            robot_motion_viewer.step(
                root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )

        if args.save_path is not None:
            qpos_list.append(qpos)

    # 7. Save
    if args.save_path is not None:
        import pickle
        motion_data = {
            "fps":            aligned_fps,
            "root_pos":       np.array([q[:3]             for q in qpos_list]),
            "root_rot":       np.array([q[3:7][[1,2,3,0]] for q in qpos_list]),
            "dof_pos":        np.array([q[7:]              for q in qpos_list]),
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if not args.no_viewer:
        robot_motion_viewer.close()
