"""
FIXED: Blender script to import robot motion with proper rotation.
This fixes the issue where empties don't rotate properly.
"""

import bpy
import json
from mathutils import Vector, Quaternion

def clear_scene():
    """Clear all objects in the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_animation_with_rotation(animation_json_path):
    """Import animation with BOTH location and rotation."""
    
    # Load animation data
    with open(animation_json_path, 'r') as f:
        animation_data = json.load(f)
    
    fps = animation_data['fps']
    num_frames = animation_data['num_frames']
    
    print(f"Importing {num_frames} frames at {fps} FPS")
    
    # Set scene settings
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    # Create empties for each body
    empties = {}
    first_frame = animation_data['frames'][0]
    
    print("Creating empties...")
    for body_name in first_frame['body_poses'].keys():
        bpy.ops.object.empty_add(type='SPHERE', radius=0.05, location=(0, 0, 0))
        empty = bpy.context.active_object
        empty.name = body_name
        # CRITICAL: Set rotation mode to QUATERNION before any keyframing
        empty.rotation_mode = 'QUATERNION'
        empty.show_name = True
        empties[body_name] = empty
    
    print(f"Created {len(empties)} empties with QUATERNION rotation mode")
    
    # Verify rotation mode is set
    for body_name, empty in empties.items():
        if empty.rotation_mode != 'QUATERNION':
            print(f"WARNING: {body_name} is not in QUATERNION mode!")
            empty.rotation_mode = 'QUATERNION'
    
    # Animate empties
    print("Adding keyframes (this will take a moment)...")
    for frame_idx, frame_data in enumerate(animation_data['frames']):
        frame_num = frame_idx + 1
        bpy.context.scene.frame_set(frame_num)
        
        for body_name, pose_data in frame_data['body_poses'].items():
            if body_name not in empties:
                continue
                
            empty = empties[body_name]
            
            # Set location
            pos = Vector(pose_data['position'])
            empty.location = pos
            empty.keyframe_insert(data_path="location", frame=frame_num)
            
            # Set rotation (convert xyzw to wxyz for Blender)
            quat_xyzw = pose_data['rotation']
            quat = Quaternion((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
            empty.rotation_quaternion = quat
            empty.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Keyframed {frame_idx + 1}/{num_frames}")
    
    print("✓ Animation import complete with rotation!")
    return empties

# =============================================================================
# RUN THIS
# =============================================================================

if __name__ == "__main__":
    ANIMATION_JSON = "/home/haziq/GMR/blender_export/robot_motion_animation.json"
    
    print("="*60)
    print("Importing Robot Animation (WITH ROTATION)")
    print("="*60)
    
    # Clear scene
    clear_scene()
    
    # Import animation
    empties = import_animation_with_rotation(ANIMATION_JSON)
    
    print("\n" + "="*60)
    print("✓ Import Complete!")
    print("="*60)
    print(f"Created {len(empties)} animated empties")
    print("\nNow:")
    print("1. Import robot meshes (File → Import → STL)")
    print("2. Run the parenting script")
    print("3. Press SPACE to see the animated robot!")
