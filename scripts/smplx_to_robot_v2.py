import argparse
import pathlib
import os
import time
import json
import sys
from glob import glob

import numpy as np
import torch
import smplx
import pandas as pd

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

from rich import print

def load_mib_data(idx, result_foldername, obj_idx, rep_idx):
    """
    Load MIB data from JSON result files.
    
    Args:
        idx: Index of the data sample (row index in CSV)
        result_foldername: Path to folder containing result JSON files
        obj_idx: Object index to extract
        rep_idx: Repetition index to extract
        
    Returns:
        mib_body_pose: [t, 21, 3] array of body poses
        mib_global_translation: [t, 3] array of translations
        mib_global_orientation: [t, 3] array of orientations
        metadata: dict with scene info
    """
    # Load CSV to get scene info for this idx
    csv_path = "/home/haziq/datasets/gimo/data/dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if idx >= len(df):
        raise ValueError(f"idx={idx} is out of range for CSV with {len(df)} rows")
    
    csv_row = df.iloc[idx]
    scene_name = csv_row["scene"]
    sequence_name = csv_row["sequence_path"]
    inp_frame = csv_row["start_frame"]
    
    print(f"[cyan]Looking for: scene={scene_name}, sequence={sequence_name}, inp_frame={inp_frame}[/cyan]")
    
    # Search for JSON files in result folder
    result_foldernames = glob(os.path.join(result_foldername, "*"))
    
    for result_foldername_sub in result_foldernames:
        result_filenames = glob(os.path.join(result_foldername_sub, "*.json"))
        
        for result_filename in result_filenames:
            # Load JSON
            with open(result_filename, "r") as f:
                data = json.load(f)
            
            # Check if this matches our scene/sequence/frame
            if (scene_name == data.get("scene_name") and 
                sequence_name == data.get("sequence_name") and 
                inp_frame == data.get("inp_frame")):
                print(f"[green]Found matching data in: {result_filename}[/green]")
                
                # Load centering data (like visualize_mib.py does)
                center = np.array(data["center"])
                is_all_objects = data.get("pred_all_objects", 0) == 1
                
                if is_all_objects:
                    all_object1_cloud_xz_center = np.array(data["all_object1_cloud_xz_center"])  # [max_num_objects, 1, 3]
                    all_object1_cloud_xz_center = np.squeeze(all_object1_cloud_xz_center)  # [max_num_objects, 3]
                
                # Extract MIB data
                mib_body_pose_all = np.array(data["mib_body_pose"][obj_idx])  # [3, t, 1, 21, 3]
                mib_global_translation_all = np.array(data["mib_global_translation"][obj_idx])  # [3, t, 3]
                mib_global_orientation_all = np.array(data["mib_global_orientation"][obj_idx])  # [3, t, 3]
                
                # Extract specific repetition
                mib_body_pose = mib_body_pose_all[rep_idx]  # [t, 1, 21, 3]
                mib_global_translation = mib_global_translation_all[rep_idx]  # [t, 3]
                mib_global_orientation = mib_global_orientation_all[rep_idx]  # [t, 3]
                
                # Apply offsets (exactly like visualize_mib.py)
                if is_all_objects:
                    mib_global_translation += all_object1_cloud_xz_center[obj_idx]
                mib_global_translation += center
                
                # Remove singleton dimension from body pose
                mib_body_pose = mib_body_pose.squeeze(1)  # [t, 21, 3]
                
                metadata = {
                    "scene_name": data.get("scene_name", "unknown"),
                    "sequence_name": data.get("sequence_name", "unknown"),
                    "object_name": data.get("object1_name", "unknown"),
                    "num_frames": mib_body_pose.shape[0],
                }
                
                print(f"[cyan]Scene: {metadata['scene_name']}[/cyan]")
                print(f"[cyan]Sequence: {metadata['sequence_name']}[/cyan]")
                print(f"[cyan]Object: {metadata['object_name']}[/cyan]")
                print(f"[cyan]Frames: {metadata['num_frames']}[/cyan]")
                
                return mib_body_pose, mib_global_translation, mib_global_orientation, metadata
    
    raise ValueError(
        f"Could not find matching data for idx={idx} "
        f"(scene={scene_name}, sequence={sequence_name}, inp_frame={inp_frame}) "
        f"in {result_foldername}"
    )


def create_smplx_data_from_mib(mib_body_pose, mib_global_translation, mib_global_orientation, fps=30):
    """
    Convert MIB data to SMPLX format.
    
    Args:
        mib_body_pose: [t, 21, 3] array of body poses in axis-angle
        mib_global_translation: [t, 3] array of translations
        mib_global_orientation: [t, 3] array of orientations in axis-angle
        fps: frames per second (default 30)
        
    Returns:
        smplx_data: dict with keys matching SMPLX .npz format
    """
    num_frames = mib_body_pose.shape[0]
    
    # Reshape body pose from [t, 21, 3] to [t, 63]
    pose_body = mib_body_pose.reshape(num_frames, 63)
    
    # Create SMPLX data dict
    smplx_data = {
        "pose_body": pose_body,  # [t, 63]
        "root_orient": mib_global_orientation,  # [t, 3]
        "trans": mib_global_translation,  # [t, 3]
        "betas": np.zeros(10),  # Default body shape
        "gender": "neutral",
        "mocap_frame_rate": np.array(fps),
    }
    
    return smplx_data


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    
    # MIB data arguments (similar to visualize_mib.py)
    parser.add_argument(
        "args_list",
        nargs='*',
        help="Arguments in format key=value (idx, result_foldername, obj_idx, rep_idx)",
    )
    
    # Robot arguments (from original smplx_to_robot.py)
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", 
                "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", 
                "tienkung", "fourier_gr3"],
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
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS of the MIB motion data (default: 30)",
    )
    
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.66,
        help="Actual human height in meters (default: 1.66)",
    )

    args = parser.parse_args()
    
    # Parse key=value arguments
    idx = None
    result_foldername = None
    obj_idx = None
    rep_idx = None
    
    for arg in args.args_list:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key == "idx":
                idx = int(value)
            elif key == "result_foldername":
                result_foldername = value
            elif key == "obj_idx":
                obj_idx = int(value)
            elif key == "rep_idx":
                rep_idx = int(value)
    
    # Validate required arguments
    if idx is None or result_foldername is None or obj_idx is None or rep_idx is None:
        print("[red]Error: Required arguments: idx=X result_foldername=X obj_idx=X rep_idx=X[/red]")
        sys.exit(1)
    
    print(f"[yellow]Loading MIB data...[/yellow]")
    print(f"  idx={idx}, obj_idx={obj_idx}, rep_idx={rep_idx}")
    
    # Load MIB data from JSON
    mib_body_pose, mib_global_translation, mib_global_orientation, metadata = load_mib_data(
        idx, result_foldername, obj_idx, rep_idx
    )
    
    # Convert to SMPLX format
    print(f"[yellow]Converting to SMPLX format...[/yellow]")
    smplx_data = create_smplx_data_from_mib(
        mib_body_pose, mib_global_translation, mib_global_orientation, fps=args.fps
    )
    
    # Create body model
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    body_model = smplx.create(
        SMPLX_FOLDER,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
        num_betas=10,
    )
    
    # Forward pass through body model
    print(f"[yellow]Running SMPLX forward pass...[/yellow]")
    num_frames = smplx_data["pose_body"].shape[0]
    
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    
    # Align FPS
    print(f"[yellow]Aligning FPS to {args.fps}...[/yellow]")
    tgt_fps = args.fps
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    # Initialize the retargeting system
    print(f"[yellow]Initializing retargeting for {args.robot}...[/yellow]")
    retarget = GMR(
        actual_human_height=args.actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    # Create viewer
    video_name = f"{args.robot}_{metadata['scene_name']}_{metadata['sequence_name']}_obj{obj_idx}_rep{rep_idx}"
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{video_name}.mp4",
    )
    
    curr_frame = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0
    print(f"[green]Starting retargeting and visualization...[/green]")
    
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
        
        # Get SMPLX data for current frame
        smplx_data_frame = smplx_data_frames[i]

        # Retarget to robot
        qpos = retarget.retarget(smplx_data_frame)

        # Visualize
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
    
    # Save motion data
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # Save from wxyz to xyzw
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
        print(f"[green]Saved to {args.save_path}[/green]")
    
    robot_motion_viewer.close()
    print(f"[green]Done![/green]")
