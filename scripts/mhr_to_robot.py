"""
mhr_to_robot.py — retarget MHR skeleton motion to a robot using GMR.

Loads an MHR NPZ file (output of sam-3d-body demo.py, i.e. img.npz),
runs the MHR forward pass to obtain per-frame skeleton joint positions and
orientations, builds the human-motion frame dict using the MHR→T1 joint
mapping, then drives GMR's IK solver.

MHR → robot joint mapping (from compare_mhr_smplx_joints.py):
    root      ← MHR idx  1  (pelvis / root)
    l_upleg   ← MHR idx  2  (left hip)
    r_upleg   ← MHR idx 18  (right hip)
    l_lowleg  ← MHR idx  3  (left knee)
    r_lowleg  ← MHR idx 19  (right knee)
    c_spine3  ← MHR idx 37  (spine3, OVERRIDE)
    l_ball    ← MHR idx  8  (left foot)
    r_ball    ← MHR idx 24  (right foot)
    l_uparm   ← MHR idx 75  (left shoulder)
    r_uparm   ← MHR idx 39  (right shoulder)
    l_lowarm  ← MHR idx 76  (left elbow)
    r_lowarm  ← MHR idx 40  (right elbow)

Usage:
    conda activate mhr_new
    cd /home/haziq/GMR
    python scripts/mhr_to_robot.py \\
        --mhr_file /home/haziq/sam-3d-body/example_data/results/img.npz \\
        --robot booster_t1 \\
        --rotate_roll 90 \\
        --rate_limit --freeze_at_end

    # Save retargeted motion:
    python scripts/mhr_to_robot.py \\
        --mhr_file /home/haziq/sam-3d-body/example_data/results/img.npz \\
        --robot booster_t1 \\
        --rotate_roll 90 \\
        --save_path output_mhr_t1.pkl
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch

# ── locate the MHR package ────────────────────────────────────────────────────
_MHR_ROOT = pathlib.Path("/home/haziq/MHR")
if str(_MHR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MHR_ROOT))

from mhr.mhr import MHR

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

from rich import print


# ── MHR joint index → human name mapping (matches mhr_to_t1.json) ────────────
_MHR_IK_JOINT_IDX: dict[str, int] = {
    "root":     1,   # pelvis
    "l_upleg":  2,   # left hip
    "r_upleg":  18,  # right hip
    "l_lowleg": 3,   # left knee
    "r_lowleg": 19,  # right knee
    "c_spine3": 37,  # spine3
    "l_ball":   8,   # left foot
    "r_ball":   24,  # right foot
    "l_uparm":  75,  # left shoulder
    "r_uparm":  39,  # right shoulder
    "l_lowarm": 76,  # left elbow
    "r_lowarm": 40,  # right elbow
}

# ── default input file ────────────────────────────────────────────────────────
_DEFAULT_MHR_FILE = "/home/haziq/sam-3d-body/example_data/results/img.npz"


def load_mhr_file(mhr_file: str, device: torch.device):
    """
    Load an MHR NPZ file, run the MHR forward pass for every frame, and
    return:
        skel_states  : (N, 127, 8) numpy float32 — global joint transforms
                       [:, :, :3]  = position in cm
                       [:, :, 3:7] = quaternion (w, x, y, z) scalar-first
        fps          : float — motion fps (30 if not stored in NPZ)
    """
    data = np.load(mhr_file, allow_pickle=True)

    if "body_pose_params" not in data.files:
        raise KeyError(
            f"'{mhr_file}' is missing 'body_pose_params'. "
            "Re-run demo.py to regenerate the NPZ."
        )

    body_pose_params = data["body_pose_params"]  # (133,) or (N, 133)
    if body_pose_params.ndim == 1:
        body_pose_params = body_pose_params[np.newaxis, :]  # → (1, 133)

    N = body_pose_params.shape[0]
    print(f"  MHR file: {mhr_file}  ({N} frame(s))")

    # ── load MHR model ──────────────────────────────────────────────────────
    print(f"  Loading MHR model (device={device}, lod=1) ...")
    mhr_model = MHR.from_files(device=device, lod=1)

    # ── build batched model params ──────────────────────────────────────────
    # model_params (N, 204):
    #   [0:6]    zeros — global trans (3) + global rot (3)
    #   [6:136]  body_pose_params[:, :130]
    #   [136:204] zeros — scale params (68)
    model_params = torch.zeros(N, 204, dtype=torch.float32, device=device)
    model_params[:, 6:136] = torch.tensor(
        body_pose_params[:, :130], dtype=torch.float32, device=device
    )

    shape_params = torch.zeros(N, 45, dtype=torch.float32, device=device)
    expr_params  = torch.zeros(N, 72, dtype=torch.float32, device=device)

    # ── MHR forward pass (batched) ──────────────────────────────────────────
    print(f"  Running MHR forward pass (batch={N}) ...")
    with torch.no_grad():
        _, skel_state = mhr_model(shape_params, model_params, expr_params)
    # skel_state: (N, 127, 8)  — positions in cm, quat wxyz

    skel_states = skel_state.cpu().numpy().astype(np.float32)  # (N, 127, 8)

    # Try to read fps from NPZ; default to 30
    if "mocap_frame_rate" in data.files:
        fps = float(np.array(data["mocap_frame_rate"]).item())
    else:
        fps = 30.0

    return skel_states, fps


def compute_human_height(skel_states: np.ndarray) -> float:
    """
    Estimate the standing height of the MHR character (metres) using the
    first frame's skeleton: 3-D distance from l_ball (joint 8) to c_head
    (joint 113), converted from cm to metres.
    """
    skel0 = skel_states[0]  # (127, 8)
    head_pos  = skel0[113, :3]  # c_head
    foot_pos  = skel0[8,   :3]  # l_ball
    height_cm = float(np.linalg.norm(head_pos - foot_pos))
    return height_cm / 100.0


def build_frame_data(skel_frame: np.ndarray) -> dict:
    """
    Build the per-frame human-motion dict expected by GMR.retarget():
        { joint_name: (pos_m, quat_wxyz), ... }

    skel_frame : (127, 8) — one row per MHR joint
                   [:3]  = position in cm
                   [3:7] = quaternion (w, x, y, z) scalar-first
    """
    frame_data = {}
    for name, idx in _MHR_IK_JOINT_IDX.items():
        pos_m     = skel_frame[idx, :3] / 100.0   # cm → metres
        quat_wxyz = skel_frame[idx, 3:7]           # (w, x, y, z)
        frame_data[name] = (pos_m, quat_wxyz)
    return frame_data


def apply_rotation(skel_states: np.ndarray,
                   roll_deg: float,
                   yaw_deg: float,
                   pitch_deg: float) -> np.ndarray:
    """
    Apply roll → yaw → pitch rotation to all joint positions and orientations
    in skel_states.  Mirrors the rotate logic in smplx_to_robot.py.

    skel_states : (N, 127, 8)  positions in cm, quat wxyz
    Returns     : (N, 127, 8) rotated
    """
    from scipy.spatial.transform import Rotation as R_scipy

    combined_rot = (
        R_scipy.from_euler('x', roll_deg,  degrees=True) *
        R_scipy.from_euler('z', yaw_deg,   degrees=True) *
        R_scipy.from_euler('y', pitch_deg, degrees=True)
    )

    N, J, _ = skel_states.shape
    out = skel_states.copy()

    # rotate positions (cm — rotation doesn't care about scale)
    pos = out[:, :, :3].reshape(-1, 3)
    out[:, :, :3] = combined_rot.apply(pos).reshape(N, J, 3)

    # rotate orientations (wxyz → xyzw for scipy → rotate → back to wxyz)
    quat_wxyz = out[:, :, 3:7].reshape(-1, 4)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rotated_xyzw = (combined_rot * R_scipy.from_quat(quat_xyzw)).as_quat()
    out[:, :, 3:7] = rotated_xyzw[:, [3, 0, 1, 2]].reshape(N, J, 4)

    return out


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Retarget MHR skeleton motion (img.npz) to a robot via GMR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mhr_file",
        type=str,
        default=_DEFAULT_MHR_FILE,
        help="Path to MHR NPZ file (output of sam-3d-body demo.py).",
    )

    parser.add_argument(
        "--robot",
        choices=["booster_t1"],   # extend here as more mhr_to_<robot>.json files are added
        default="booster_t1",
        help="Target robot name (default: booster_t1).",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for MHR forward pass: cpu or cuda (default: cpu).",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the retargeted robot motion as a .pkl file.",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record viewer output to video.",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the recorded video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit playback rate to match input motion FPS.",
    )

    parser.add_argument(
        "--camera_distance",
        type=float,
        default=None,
        help="Camera distance (zoom level).",
    )

    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=-10,
        help="Camera elevation angle in degrees (default: -10).",
    )

    parser.add_argument(
        "--camera_height",
        type=float,
        default=0.0,
        help="Vertical camera lookat offset in metres (default: 0).",
    )

    parser.add_argument(
        "--rotate_roll",
        type=float,
        default=0.0,
        help="Rotate around global X axis (roll) in degrees, applied first.",
    )

    parser.add_argument(
        "--rotate_yaw",
        type=float,
        default=0.0,
        help="Rotate around global Z axis (yaw) in degrees, applied second.",
    )

    parser.add_argument(
        "--rotate_pitch",
        type=float,
        default=0.0,
        help="Rotate around global Y axis (pitch) in degrees, applied third.",
    )

    parser.add_argument(
        "--hide_floor",
        default=False,
        action="store_true",
        help="Hide the floor/ground plane in the viewer.",
    )

    parser.add_argument(
        "--ik_config",
        type=str,
        default=None,
        help="Path to a custom IK config JSON. Overrides the default mhr_to_<robot>.json.",
    )

    parser.add_argument(
        "--no_viewer",
        default=False,
        action="store_true",
        help="Run headless (no MuJoCo viewer). Only IK + save.",
    )

    parser.add_argument(
        "--freeze_at_end",
        default=False,
        action="store_true",
        help="Keep the viewer open on the last frame instead of closing.",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 1. Load MHR file and run forward pass ─────────────────────────────────
    skel_states, motion_fps = load_mhr_file(args.mhr_file, device)

    # ── 2. Optional rotation of all joint positions/orientations ──────────────
    if args.rotate_roll != 0.0 or args.rotate_yaw != 0.0 or args.rotate_pitch != 0.0:
        print(f"  Applying rotation: roll={args.rotate_roll}°  yaw={args.rotate_yaw}°  "
              f"pitch={args.rotate_pitch}°")
        skel_states = apply_rotation(
            skel_states, args.rotate_roll, args.rotate_yaw, args.rotate_pitch
        )

    # ── 3. Estimate human height from T-pose ──────────────────────────────────
    actual_human_height = compute_human_height(skel_states)
    print(f"  Estimated human height: {actual_human_height:.3f} m")

    # ── 4. Pre-build all frame dicts ───────────────────────────────────────────
    N = skel_states.shape[0]
    frame_dicts = [build_frame_data(skel_states[i]) for i in range(N)]

    # ── 5. Initialise GMR retargeting system ───────────────────────────────────
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="mhr",
        tgt_robot=args.robot,
        ik_config_path=args.ik_config,
    )

    # ── 6. Optionally open the MuJoCo viewer ───────────────────────────────────
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=(
                args.video_path
                if args.video_path is not None
                else f"videos/{args.robot}_{pathlib.Path(args.mhr_file).stem}.mp4"
            ),
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

    # ── 7. Main retargeting loop ────────────────────────────────────────────────
    i = -1
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while True:
        if args.loop:
            i = (i + 1) % N
        else:
            i += 1
            if i >= N:
                if args.freeze_at_end and not args.no_viewer:
                    print("[freeze_at_end] Holding last frame. Close the window to exit.")
                    while robot_motion_viewer.viewer.is_running():
                        robot_motion_viewer.step(
                            root_pos=qpos[:3],
                            root_rot=qpos[3:7],
                            dof_pos=qpos[7:],
                            human_motion_data=retarget.scaled_human_data,
                            human_pos_offset=np.array([0.0, 0.0, 0.0]),
                            show_human_body_name=False,
                            rate_limit=args.rate_limit,
                            follow_camera=False,
                        )
                break

        # FPS display
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"  Rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Retarget current frame
        qpos = retarget.retarget(frame_dicts[i])

        # Visualise
        if not args.no_viewer:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )

        if args.save_path is not None:
            qpos_list.append(qpos)

    # ── 8. Optionally save ─────────────────────────────────────────────────────
    if args.save_path is not None:
        import pickle

        root_pos = np.array([q[:3]          for q in qpos_list])
        root_rot = np.array([q[3:7][[1,2,3,0]] for q in qpos_list])  # wxyz → xyzw
        dof_pos  = np.array([q[7:]           for q in qpos_list])

        motion_data = {
            "fps":            motion_fps,
            "root_pos":       root_pos,
            "root_rot":       root_rot,
            "dof_pos":        dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"  Saved to {args.save_path}")

    if not args.no_viewer:
        robot_motion_viewer.close()
