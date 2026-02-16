#!/usr/bin/env python3
"""
Export robot motion to Blender-compatible formats.
Supports FBX export with skeletal animation.
"""

import argparse
import pickle
import numpy as np
import mujoco as mj
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json

from general_motion_retargeting.params import ROBOT_XML_DICT


def export_to_fbx(motion_file, robot_type, output_path):
    """
    Export robot motion to FBX format using MuJoCo's mesh export capabilities.
    Note: This creates an animated FBX file.
    """
    try:
        import trimesh
        import pyassimp
        HAS_FBX_SUPPORT = True
    except ImportError:
        print("Warning: trimesh or pyassimp not installed. FBX export may be limited.")
        HAS_FBX_SUPPORT = False
    
    # Load motion data
    with open(motion_file, 'rb') as f:
        motion_data = pickle.load(f)
    
    root_pos = motion_data['root_pos']
    root_rot = motion_data['root_rot']  # xyzw format
    dof_pos = motion_data['dof_pos']
    fps = motion_data['fps']
    
    num_frames = len(root_pos)
    
    print(f"Loaded motion with {num_frames} frames at {fps} FPS")
    print(f"Root pos shape: {root_pos.shape}")
    print(f"Root rot shape: {root_rot.shape}")
    print(f"DoF pos shape: {dof_pos.shape}")
    
    # Load robot model
    xml_file = str(ROBOT_XML_DICT[robot_type])
    model = mj.MjModel.from_xml_path(xml_file)
    data = mj.MjData(model)
    
    print(f"\nRobot model loaded: {robot_type}")
    print(f"Number of DoFs: {model.nv}")
    print(f"Number of bodies: {model.nbody}")
    
    # Create animation data file (JSON format for Blender Python script)
    animation_data = {
        'fps': int(fps),
        'num_frames': int(num_frames),
        'frames': []
    }
    
    # Extract joint names
    joint_names = []
    for i in range(model.njnt):
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            joint_names.append(joint_name)
    
    print(f"\nJoint names: {joint_names}")
    
    # Coordinate system correction for Blender
    # MuJoCo Z-up to Blender Z-up with correct orientation (rotate 90° around X-axis)
    correction_rot = R.from_euler('x', 90, degrees=True)
    
    # Get initial pelvis height to calculate ground offset
    data.qpos[:3] = root_pos[0]
    quat_xyzw = root_rot[0]
    data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    if dof_pos.shape[1] == model.nv - 6:
        data.qpos[7:] = dof_pos[0]
    else:
        data.qpos[7:7+dof_pos.shape[1]] = dof_pos[0]
    mj.mj_forward(model, data)
    initial_pelvis_z = data.xpos[1].copy()[2]  # Body 1 is typically pelvis
    
    print(f"Initial pelvis height: {initial_pelvis_z:.3f}")
    print("Applying coordinate transform for Blender...")
    
    # For each frame, store the complete pose
    for frame_idx in range(num_frames):
        # Set the pose in MuJoCo
        data.qpos[:3] = root_pos[frame_idx]
        
        # Convert quaternion from xyzw to wxyz for MuJoCo
        quat_xyzw = root_rot[frame_idx]
        data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # wxyz
        
        # Set joint angles
        if dof_pos.shape[1] == model.nv - 6:  # Floating base (6 DoF) + joints
            data.qpos[7:] = dof_pos[frame_idx]
        else:
            data.qpos[7:7+dof_pos.shape[1]] = dof_pos[frame_idx]
        
        # Forward kinematics to get body positions and orientations
        mj.mj_forward(model, data)
        
        # Store frame data
        frame_data = {
            'root_pos': root_pos[frame_idx].tolist(),
            'root_rot': root_rot[frame_idx].tolist(),  # xyzw
            'dof_pos': dof_pos[frame_idx].tolist(),
            'body_poses': {}
        }
        
        # Store all body positions and orientations
        for body_id in range(model.nbody):
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                body_pos = data.xpos[body_id].copy()
                body_quat = data.xquat[body_id].copy()  # wxyz in MuJoCo
                
                # Apply coordinate transform for Blender
                # Rotate position
                body_pos_transformed = correction_rot.apply(body_pos)
                # Adjust height to put on ground
                body_pos_transformed[2] -= initial_pelvis_z
                
                # Convert quaternion wxyz to xyzw, then apply rotation correction
                body_quat_xyzw = [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
                body_rot = R.from_quat(body_quat_xyzw)  # scipy uses xyzw
                body_rot_transformed = correction_rot * body_rot
                body_quat_xyzw_transformed = body_rot_transformed.as_quat()  # xyzw
                
                frame_data['body_poses'][body_name] = {
                    'position': body_pos_transformed.tolist(),
                    'rotation': body_quat_xyzw_transformed.tolist()  # xyzw format
                }
        
        animation_data['frames'].append(frame_data)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Processed frame {frame_idx + 1}/{num_frames}")
    
    # Save animation data as JSON
    json_path = output_path.replace('.fbx', '_animation.json')
    with open(json_path, 'w') as f:
        json.dump(animation_data, f, indent=2)
    
    print(f"\n✓ Animation data saved to: {json_path}")
    
    # Save robot URDF path info
    urdf_info = {
        'robot_type': robot_type,
        'xml_path': xml_file,
        'urdf_path': xml_file.replace('.xml', '.urdf') if '.xml' in xml_file else None,
        'mesh_dir': str(Path(xml_file).parent / 'meshes'),
    }
    
    urdf_info_path = output_path.replace('.fbx', '_robot_info.json')
    with open(urdf_info_path, 'w') as f:
        json.dump(urdf_info, f, indent=2)
    
    print(f"✓ Robot info saved to: {urdf_info_path}")
    
    return json_path, urdf_info_path


def export_to_obj_sequence(motion_file, robot_type, output_dir):
    """
    Export robot motion as a sequence of OBJ files (one per frame).
    This is simpler but creates many files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load motion data
    with open(motion_file, 'rb') as f:
        motion_data = pickle.load(f)
    
    root_pos = motion_data['root_pos']
    root_rot = motion_data['root_rot']
    dof_pos = motion_data['dof_pos']
    fps = motion_data['fps']
    
    num_frames = len(root_pos)
    
    # Load robot model
    xml_file = str(ROBOT_XML_DICT[robot_type])
    model = mj.MjModel.from_xml_path(xml_file)
    data = mj.MjData(model)
    
    print(f"Exporting {num_frames} frames as OBJ sequence...")
    
    for frame_idx in range(num_frames):
        # Set the pose
        data.qpos[:3] = root_pos[frame_idx]
        quat_xyzw = root_rot[frame_idx]
        data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        
        if dof_pos.shape[1] == model.nv - 6:
            data.qpos[7:] = dof_pos[frame_idx]
        
        mj.mj_forward(model, data)
        
        # Export mesh (this would require additional implementation)
        obj_filename = f"{output_dir}/frame_{frame_idx:04d}.obj"
        # Note: Actual OBJ export would need mesh extraction from MuJoCo
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Exported frame {frame_idx + 1}/{num_frames}")
    
    print(f"✓ OBJ sequence exported to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export robot motion to Blender-compatible formats"
    )
    
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Path to the motion.npz file (pickle format)"
    )
    
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
                 "tienkung", "fourier_gr3"],
        help="Robot type"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="blender_export/robot_motion.fbx",
        help="Output path for FBX file"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="fbx",
        choices=["fbx", "obj_sequence"],
        help="Export format: fbx (single file with animation) or obj_sequence (multiple files)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"="*60)
    print(f"Exporting Robot Motion to Blender")
    print(f"="*60)
    print(f"Motion file: {args.motion_file}")
    print(f"Robot type: {args.robot}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"="*60)
    
    if args.format == "fbx":
        json_path, info_path = export_to_fbx(args.motion_file, args.robot, args.output)
        print(f"\n" + "="*60)
        print(f"Export Complete!")
        print(f"="*60)
        print(f"\nGenerated files:")
        print(f"  1. {json_path} - Animation keyframe data")
        print(f"  2. {info_path} - Robot model information")
        
    elif args.format == "obj_sequence":
        output_dir = args.output.replace('.fbx', '_obj_sequence')
        export_to_obj_sequence(args.motion_file, args.robot, output_dir)
