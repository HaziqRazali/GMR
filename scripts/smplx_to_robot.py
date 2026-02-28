import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.utils.smpl_json import load_smplx_json_file
from general_motion_retargeting.utils.smpl_pkl import load_smplx_4dhumans_pkl

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
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
        help="Record the video.",
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
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--camera_distance",
        type=float,
        default=None,
        help="Camera distance (zoom level). Larger = more zoomed out. Defaults to robot-specific preset.",
    )

    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=-10,
        help="Camera elevation angle in degrees (negative = looking down). Default: -10.",
    )

    parser.add_argument(
        "--camera_height",
        type=float,
        default=0.0,
        help="Vertical offset (metres) added to the camera lookat point. "
             "Positive = camera looks at a higher point, effectively raising the viewpoint. Default: 0.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Input motion FPS (used for JSON files). Fit3D defaults to 50, MotionX defaults to 30.",
    )

    parser.add_argument(
        "--rotate_roll",
        type=float,
        default=0.0,
        help="Rotate around the global X axis (roll) in degrees, applied first.",
    )

    parser.add_argument(
        "--rotate_yaw",
        type=float,
        default=0.0,
        help="Rotate around the global Z axis (yaw) in degrees, applied second.",
    )

    parser.add_argument(
        "--rotate_pitch",
        type=float,
        default=0.0,
        help="Rotate around the global Y axis (pitch) in degrees, applied third.",
    )

    parser.add_argument(
        "--hide_floor",
        default=False,
        action="store_true",
        help="Hide the floor/ground plane in the viewer and recorded video.",
    )

    parser.add_argument(
        "--ik_config",
        type=str,
        default=None,
        help="Path to a custom IK config JSON file. Overrides the default config for the chosen robot.",
    )

    parser.add_argument(
        "--zero_global_orient",
        default=False,
        action="store_true",
        help="Zero out the initial global orientation of the human SMPL-X after loading, "
             "so all frames are expressed relative to the first frame's orientation.",
    )

    parser.add_argument(
        "--no_viewer",
        default=False,
        action="store_true",
        help="Run headless (no MuJoCo viewer). Only IK + save. Use for batch processing.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory (support .npz, .json, and .pkl)
    if args.smplx_file.endswith(".pkl"):
        pkl_fps = args.fps if args.fps is not None else 30
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_4dhumans_pkl(
            args.smplx_file, SMPLX_FOLDER, fps=pkl_fps
        )
    elif args.smplx_file.endswith(".json"):
        # Auto-detect fps: fit3d is 50 fps, MotionX/annotations format is 30 fps
        import json as _json
        with open(args.smplx_file) as _f:
            _probe = _json.load(_f)
        is_fit3d = 'annotations' not in _probe and 'transl' in _probe
        json_fps = args.fps if args.fps is not None else (50 if is_fit3d else 30)
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_json_file(
            args.smplx_file, SMPLX_FOLDER, fps=json_fps
        )
    else:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            args.smplx_file, SMPLX_FOLDER
        )
    
    # align fps
    tgt_fps = 30

    # Apply optional rotation (roll → yaw → pitch) to global orientation and joint positions
    if args.rotate_roll != 0.0 or args.rotate_yaw != 0.0 or args.rotate_pitch != 0.0:
        from scipy.spatial.transform import Rotation as R_scipy
        combined_rot = (
            R_scipy.from_euler('x', args.rotate_roll,  degrees=True) *
            R_scipy.from_euler('z', args.rotate_yaw,   degrees=True) *
            R_scipy.from_euler('y', args.rotate_pitch, degrees=True)
        )

        # Rotate global_orient in-place
        go_np = smplx_output.global_orient.detach().numpy().reshape(-1, 3)
        rotated_go = (combined_rot * R_scipy.from_rotvec(go_np)).as_rotvec().astype(np.float32)
        smplx_output.global_orient.data.copy_(
            torch.tensor(rotated_go).reshape(smplx_output.global_orient.shape))

        # Rotate all joint positions in-place
        joints_np = smplx_output.joints.detach().numpy()
        N_f, J, _ = joints_np.shape
        rotated_joints = combined_rot.apply(joints_np.reshape(-1, 3)).reshape(N_f, J, 3).astype(np.float32)
        smplx_output.joints.data.copy_(torch.tensor(rotated_joints))

    # Zero out the initial global orientation so frame 0 becomes identity
    if args.zero_global_orient:
        from scipy.spatial.transform import Rotation as R_scipy
        go_np = smplx_output.global_orient.detach().numpy().reshape(-1, 3)
        init_rot_inv = R_scipy.from_rotvec(go_np[0]).inv()
        zeroed_go = (init_rot_inv * R_scipy.from_rotvec(go_np)).as_rotvec().astype(np.float32)
        smplx_output.global_orient.data.copy_(
            torch.tensor(zeroed_go).reshape(smplx_output.global_orient.shape))
        joints_np = smplx_output.joints.detach().numpy()
        N_f, J, _ = joints_np.shape
        zeroed_joints = init_rot_inv.apply(joints_np.reshape(-1, 3)).reshape(N_f, J, 3).astype(np.float32)
        smplx_output.joints.data.copy_(torch.tensor(zeroed_joints))

    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        ik_config_path=args.ik_config,
    )
    
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=aligned_fps,
                                                transparent_robot=0,
                                                record_video=args.record_video,
                                                video_path=args.video_path if args.video_path is not None else f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",
                                                camera_distance=args.camera_distance,
                                                camera_elevation=args.camera_elevation,
                                                camera_height=args.camera_height,
                                                hide_floor=args.hide_floor,)

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        if not args.no_viewer:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if not args.no_viewer:
        robot_motion_viewer.close()
