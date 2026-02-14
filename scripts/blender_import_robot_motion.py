"""
Blender script to import and animate robot motion.
Run this script inside Blender's Scripting workspace.

Usage in Blender:
1. Open Blender
2. Go to Scripting workspace
3. Open this script or paste its content
4. Modify the paths at the bottom to point to your exported files
5. Run the script (Alt+P or click Run Script button)
"""

import bpy
import json
import math
from mathutils import Vector, Quaternion, Euler
from pathlib import Path


def clear_scene():
    """Clear all mesh objects in the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_robot_urdf(urdf_path):
    """
    Import robot URDF into Blender.
    Requires URDF importer addon to be installed.
    """
    try:
        # Try to import using urdf_importer addon if available
        bpy.ops.import_scene.urdf(filepath=urdf_path)
        print(f"✓ Imported URDF: {urdf_path}")
        return True
    except AttributeError:
        print("⚠ URDF importer addon not found.")
        print("Install it from: https://github.com/ros/robot_model")
        return False


def import_robot_obj(mesh_dir):
    """
    Import robot meshes from directory (OBJ/STL files).
    """
    mesh_path = Path(mesh_dir)
    
    if not mesh_path.exists():
        print(f"✗ Mesh directory not found: {mesh_dir}")
        return []
    
    imported_objects = []
    
    # Import all mesh files
    for mesh_file in mesh_path.glob("*.obj"):
        bpy.ops.import_scene.obj(filepath=str(mesh_file))
        imported_objects.append(bpy.context.selected_objects[0])
        print(f"✓ Imported: {mesh_file.name}")
    
    for mesh_file in mesh_path.glob("*.stl"):
        bpy.ops.import_mesh.stl(filepath=str(mesh_file))
        imported_objects.append(bpy.context.selected_objects[0])
        print(f"✓ Imported: {mesh_file.name}")
    
    return imported_objects


def create_armature_from_animation(animation_data, robot_info):
    """
    Create an armature (skeleton) from the animation data.
    Each robot body becomes a bone.
    """
    # Create armature
    bpy.ops.object.armature_add()
    armature_obj = bpy.context.active_object
    armature_obj.name = f"{robot_info['robot_type']}_Armature"
    
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    armature = armature_obj.data
    
    # Remove default bone
    armature.edit_bones.remove(armature.edit_bones[0])
    
    # Get first frame to establish bone structure
    first_frame = animation_data['frames'][0]
    body_poses = first_frame['body_poses']
    
    # Create bones for each body
    bone_map = {}
    for body_name, pose_data in body_poses.items():
        bone = armature.edit_bones.new(body_name)
        pos = Vector(pose_data['position'])
        
        # Set bone head and tail
        bone.head = pos
        bone.tail = pos + Vector([0, 0, 0.1])  # Small offset for bone length
        
        bone_map[body_name] = bone
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"✓ Created armature with {len(bone_map)} bones")
    
    return armature_obj, bone_map


def animate_armature(armature_obj, animation_data):
    """
    Animate the armature using keyframe animation.
    """
    fps = animation_data['fps']
    num_frames = animation_data['num_frames']
    
    # Set scene frame rate
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    print(f"Animating {num_frames} frames at {fps} FPS...")
    
    # Animate each bone
    for frame_idx, frame_data in enumerate(animation_data['frames']):
        frame_num = frame_idx + 1  # Blender frames start at 1
        
        # Set current frame
        bpy.context.scene.frame_set(frame_num)
        
        # Animate pose bones
        for body_name, pose_data in frame_data['body_poses'].items():
            if body_name in armature_obj.pose.bones:
                pose_bone = armature_obj.pose.bones[body_name]
                
                # Set location
                pos = Vector(pose_data['position'])
                pose_bone.location = pos
                pose_bone.keyframe_insert(data_path="location", frame=frame_num)
                
                # Set rotation (quaternion xyzw -> wxyz for Blender)
                quat_xyzw = pose_data['rotation']
                quat = Quaternion((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
                pose_bone.rotation_quaternion = quat
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        
        # Progress indicator
        if (frame_idx + 1) % 10 == 0:
            print(f"  Keyframed {frame_idx + 1}/{num_frames} frames")
    
    print("✓ Animation complete!")


def create_simple_robot_visualization(animation_data):
    """
    Create a simple stick-figure visualization of the robot motion.
    Useful when URDF import is not available.
    """
    fps = animation_data['fps']
    num_frames = animation_data['num_frames']
    
    # Set scene settings
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    # Create empties for each body
    empties = {}
    first_frame = animation_data['frames'][0]
    
    for body_name in first_frame['body_poses'].keys():
        bpy.ops.object.empty_add(type='SPHERE', radius=0.05)
        empty = bpy.context.active_object
        empty.name = body_name
        empty.rotation_mode = 'QUATERNION'  # Important: set to quaternion mode
        empties[body_name] = empty
    
    print(f"Created {len(empties)} markers")
    
    # Animate empties
    for frame_idx, frame_data in enumerate(animation_data['frames']):
        frame_num = frame_idx + 1
        bpy.context.scene.frame_set(frame_num)
        
        for body_name, pose_data in frame_data['body_poses'].items():
            if body_name in empties:
                empty = empties[body_name]
                
                # Set location
                pos = Vector(pose_data['position'])
                empty.location = pos
                empty.keyframe_insert(data_path="location", frame=frame_num)
                
                # Set rotation (quaternion xyzw -> wxyz for Blender)
                quat_xyzw = pose_data['rotation']
                quat = Quaternion((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
                empty.rotation_quaternion = quat
                empty.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Animated {frame_idx + 1}/{num_frames} frames")
    
    print("✓ Simple visualization complete!")
    return empties


def main(animation_json_path, robot_info_json_path, import_robot_model=True):
    """
    Main function to import and animate robot in Blender.
    
    Args:
        animation_json_path: Path to the animation JSON file
        robot_info_json_path: Path to the robot info JSON file
        import_robot_model: Whether to try importing the actual robot model (URDF/meshes)
    """
    print("="*60)
    print("Importing Robot Motion to Blender")
    print("="*60)
    
    # Load animation data
    with open(animation_json_path, 'r') as f:
        animation_data = json.load(f)
    
    print(f"✓ Loaded animation: {animation_data['num_frames']} frames at {animation_data['fps']} FPS")
    
    # Load robot info
    with open(robot_info_json_path, 'r') as f:
        robot_info = json.load(f)
    
    print(f"✓ Robot type: {robot_info['robot_type']}")
    
    # Clear scene
    clear_scene()
    
    if import_robot_model:
        # Try to import robot model
        urdf_path = robot_info.get('urdf_path')
        if urdf_path and Path(urdf_path).exists():
            success = import_robot_urdf(urdf_path)
            if not success:
                print("Falling back to simple visualization...")
                create_simple_robot_visualization(animation_data)
        else:
            # Try importing meshes directly
            mesh_dir = robot_info.get('mesh_dir')
            if mesh_dir and Path(mesh_dir).exists():
                import_robot_obj(mesh_dir)
                create_simple_robot_visualization(animation_data)
            else:
                print("No robot model found, creating simple visualization...")
                create_simple_robot_visualization(animation_data)
    else:
        # Just create simple visualization
        create_simple_robot_visualization(animation_data)
    
    print("="*60)
    print("✓ Import Complete!")
    print("="*60)
    print("\nPress SPACE to play the animation")


# =============================================================================
# MODIFY THESE PATHS TO MATCH YOUR EXPORTED FILES
# =============================================================================

if __name__ == "__main__":
    # Update these paths to your exported files
    ANIMATION_JSON = "/home/haziq/GMR/blender_export/robot_motion_animation.json"
    ROBOT_INFO_JSON = "/home/haziq/GMR/blender_export/robot_motion_robot_info.json"
    
    # Run the import
    main(ANIMATION_JSON, ROBOT_INFO_JSON, import_robot_model=True)
