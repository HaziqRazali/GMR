"""
Utilities for loading SMPLX data from JSON format (e.g., MotionX dataset)
"""
import json
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d


def load_smplx_json_file(json_file, smplx_body_model_path, gender="neutral", fps=30):
    """
    Load SMPLX data from JSON file format.
    
    Args:
        json_file: Path to JSON file containing per-frame annotations with smplx_params
        smplx_body_model_path: Path to SMPLX body models
        gender: Gender for body model ('neutral', 'male', 'female')
        fps: Frames per second for the motion
    
    Returns:
        smplx_data: Dictionary with pose_body, betas, root_orient, trans, etc.
        body_model: SMPLX body model
        smplx_output: SMPLX forward pass output
        human_height: Estimated human height
    """
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    num_frames = len(annotations)
    
    print(f"Loaded {num_frames} frames from JSON file")
    
    # Detect format by checking first annotation's keys
    first_params = annotations[0]['smplx_params']
    is_global_motion = 'pose_hand' in first_params  # global_motion format
    is_local_motion = 'lhand_pose' in first_params or 'root_pose' in first_params  # local_motion format
    
    format_name = "global_motion" if is_global_motion else "local_motion"
    print(f"Detected format: {format_name}")
    
    # Initialize arrays
    root_pose_list = []
    body_pose_list = []
    trans_list = []
    lhand_pose_list = []
    rhand_pose_list = []
    jaw_pose_list = []
    shape_list = []
    expr_list = []
    
    # Extract parameters from each frame
    for ann in annotations:
        params = ann['smplx_params']
        
        if is_global_motion:
            # Global motion format
            root_pose_list.append(params['root_orient'])
            body_pose_list.append(params['pose_body'])
            trans_list.append(params['trans'])
            
            # Split pose_hand (90,) into left (45,) and right (45,)
            pose_hand = np.array(params['pose_hand'])
            lhand_pose_list.append(pose_hand[:45])
            rhand_pose_list.append(pose_hand[45:])
            
            jaw_pose_list.append(params['pose_jaw'])
            
            # Use betas (10,) for shape
            shape_list.append(params['betas'])
            
            # Use face_expr, but only take first 10 components
            face_expr = np.array(params['face_expr'])
            expr_list.append(face_expr[:10])
        else:
            # Local motion format
            root_pose_list.append(params['root_pose'])
            body_pose_list.append(params['body_pose'])
            trans_list.append(params['trans'])
            lhand_pose_list.append(params['lhand_pose'])
            rhand_pose_list.append(params['rhand_pose'])
            jaw_pose_list.append(params['jaw_pose'])
            shape_list.append(params['shape'])
            expr_list.append(params['expr'])
    
    # Convert to numpy arrays
    root_orient = np.array(root_pose_list)  # (N, 3)
    pose_body = np.array(body_pose_list)    # (N, 63)
    trans = np.array(trans_list)            # (N, 3)
    lhand_pose = np.array(lhand_pose_list)  # (N, 45)
    rhand_pose = np.array(rhand_pose_list)  # (N, 45)
    jaw_pose = np.array(jaw_pose_list)      # (N, 3)
    shape = np.array(shape_list)            # (N, 10)
    expr = np.array(expr_list)              # (N, 10)
    
    # Use the first frame's shape (betas) and average across frames for stability
    betas = np.mean(shape, axis=0)  # (10,)
    
    # Pad betas from 10 to 16 dimensions (SMPLX expects 10, but can handle 16)
    betas_padded = np.pad(betas, (0, 6), mode='constant', constant_values=0)  # (16,)
    
    # Create smplx_data dictionary in the format expected by the pipeline
    smplx_data = {
        'pose_body': pose_body,
        'betas': betas_padded,
        'root_orient': root_orient,
        'trans': trans,
        'gender': np.array(gender),
        'mocap_frame_rate': np.array(fps),
    }
    
    # Create body model
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
    )
    
    # Run forward pass
    smplx_output = body_model(
        betas=torch.tensor(betas_padded).float().view(1, -1),  # (1, 16)
        global_orient=torch.tensor(root_orient).float(),        # (N, 3)
        body_pose=torch.tensor(pose_body).float(),              # (N, 63)
        transl=torch.tensor(trans).float(),                     # (N, 3)
        left_hand_pose=torch.tensor(lhand_pose).float(),        # (N, 45)
        right_hand_pose=torch.tensor(rhand_pose).float(),       # (N, 45)
        jaw_pose=torch.tensor(jaw_pose).float(),                # (N, 3)
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.tensor(expr).float(),                  # (N, 10)
        return_full_pose=True,
    )
    
    # Estimate human height from betas
    human_height = 1.66 + 0.1 * betas_padded[0]
    
    print(f"Estimated human height: {human_height:.2f}m")
    print(f"FPS: {fps}")
    
    return smplx_data, body_model, smplx_output, human_height
