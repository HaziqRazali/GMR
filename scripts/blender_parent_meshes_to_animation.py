"""
Blender script to automatically parent imported meshes to animated empties.
Run this AFTER importing both the animation (empties) and the robot meshes.

This script matches mesh names to body names and parents them automatically.
"""

import bpy
import re

# Mapping from mesh names to body names (adjust based on your mesh naming)
# This is for Unitree G1 robot
MESH_TO_BODY_MAPPING = {
    # Pelvis/Base
    'pelvis': 'pelvis',
    'base_link': 'pelvis',
    'torso': 'torso_link',
    
    # Left leg
    'left_hip_pitch_link': 'left_hip_pitch_link',
    'left_hip_roll_link': 'left_hip_roll_link',
    'left_hip_yaw_link': 'left_hip_yaw_link',
    'left_knee_link': 'left_knee_link',
    'left_ankle_pitch': 'left_ankle_pitch',
    'left_ankle_roll': 'left_ankle_roll',
    'left_foot': 'left_ankle_roll',
    
    # Right leg
    'right_hip_pitch_link': 'right_hip_pitch_link',
    'right_hip_roll_link': 'right_hip_roll_link',
    'right_hip_yaw_link': 'right_hip_yaw_link',
    'right_knee_link': 'right_knee_link',
    'right_ankle_pitch': 'right_ankle_pitch',
    'right_ankle_roll': 'right_ankle_roll',
    'right_foot': 'right_ankle_roll',
    
    # Waist
    'waist_yaw_link': 'waist_yaw_link',
    'waist_roll_link': 'waist_roll_link',
    'waist_pitch_link': 'logo_link',
    'logo_link': 'logo_link',
    
    # Left arm
    'left_shoulder_pitch_link': 'left_shoulder_pitch_link',
    'left_shoulder_roll_link': 'left_shoulder_roll_link',
    'left_shoulder_yaw_link': 'left_shoulder_yaw_link',
    'left_elbow_link': 'left_elbow_link',
    'left_wrist_roll': 'left_wrist_roll',
    'left_wrist_pitch': 'left_wrist_pitch',
    'left_wrist_yaw': 'left_wrist_yaw',
    'left_hand': 'left_wrist_yaw',
    
    # Right arm
    'right_shoulder_pitch_link': 'right_shoulder_pitch_link',
    'right_shoulder_roll_link': 'right_shoulder_roll_link',
    'right_shoulder_yaw_link': 'right_shoulder_yaw_link',
    'right_elbow_link': 'right_elbow_link',
    'right_wrist_roll': 'right_wrist_roll',
    'right_wrist_pitch': 'right_wrist_pitch',
    'right_wrist_yaw': 'right_wrist_yaw',
    'right_hand': 'right_wrist_yaw',
    
    # Head
    'head_link': 'imu_link',
    'imu_link': 'imu_link',
    
    # Logo - the waist logo link
    'logo_link': 'logo_link',
}

# Mesh local position offsets (from MuJoCo URDF/XML)
# Format: 'mesh_name': ('parent_body', (x, y, z))
MESH_OFFSETS = {
    'logo_link': ('torso_link', (0.0039635, 0, -0.044)),
    # Add more as needed
}


def find_matching_body_name(mesh_name, available_empties):
    """
    Try to find the matching empty/body for a mesh.
    Uses fuzzy matching to handle naming variations.
    """
    # Remove file extensions and Blender suffixes
    base_mesh_name = mesh_name.replace('.001', '').replace('.002', '').replace('.003', '').replace('.stl', '').replace('.obj', '')
    mesh_lower = base_mesh_name.lower().replace('_', '')
    
    # First try EXACT name match (case-sensitive, with base name)
    if base_mesh_name in available_empties:
        return base_mesh_name
    
    # Then try exact mapping from the table (case-sensitive first)
    for mesh_key, body_name in MESH_TO_BODY_MAPPING.items():
        if mesh_key == base_mesh_name:  # Exact match
            if body_name in available_empties:
                return body_name
    
    # Then try case-insensitive from table
    for mesh_key, body_name in MESH_TO_BODY_MAPPING.items():
        mesh_key_clean = mesh_key.lower().replace('_', '')
        if mesh_key_clean == mesh_lower or mesh_key.lower() == base_mesh_name.lower():
            if body_name in available_empties:
                return body_name
    
    # Try partial matching with mapping table
    for mesh_key, body_name in MESH_TO_BODY_MAPPING.items():
        if mesh_key.lower().replace('_', '') in mesh_lower:
            if body_name in available_empties:
                return body_name
    
    # Try fuzzy matching with empties
    best_match = None
    best_score = 0
    
    for empty_name in available_empties:
        empty_lower = empty_name.lower().replace('_', '')
        
        # Calculate similarity score
        common_parts = 0
        for part in empty_lower.split():
            if part in mesh_lower:
                common_parts += len(part)
        
        if common_parts > best_score:
            best_score = common_parts
            best_match = empty_name
    
    if best_score > 3:  # Minimum similarity threshold
        return best_match
    
    return None


def parent_meshes_to_empties():
    """
    Automatically parent all mesh objects to their corresponding empties.
    """
    print("="*60)
    print("Parenting Meshes to Animated Empties")
    print("="*60)
    
    # Get all empties (animated markers)
    empties = {obj.name: obj for obj in bpy.data.objects if obj.type == 'EMPTY'}
    print(f"\nFound {len(empties)} empties (animated markers)")
    
    # Get all mesh objects
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    print(f"Found {len(meshes)} mesh objects")
    
    if not empties:
        print("\n❌ ERROR: No empties found!")
        print("Make sure you've run the animation import script first.")
        return
    
    if not meshes:
        print("\n❌ ERROR: No meshes found!")
        print("Make sure you've imported the robot mesh files (STL/OBJ).")
        return
    
    print("\n" + "-"*60)
    print("Matching and parenting meshes...")
    print("-"*60)
    
    parented_count = 0
    unmatched_meshes = []
    
    for mesh_obj in meshes:
        mesh_name = mesh_obj.name
        base_mesh_name = mesh_name.replace('.001', '').replace('.002', '').replace('.003', '').replace('.stl', '').replace('.obj', '')
        
        # Check if this mesh has a special offset defined
        parent_body = None
        local_offset = None
        if base_mesh_name in MESH_OFFSETS:
            parent_body, local_offset = MESH_OFFSETS[base_mesh_name]
            print(f"  Found special offset for {base_mesh_name}: parent={parent_body}, offset={local_offset}")
        
        # Try to find matching empty
        if parent_body and parent_body in empties:
            # Use the specified parent from MESH_OFFSETS
            body_name = parent_body
        else:
            # Use normal matching
            body_name = find_matching_body_name(mesh_name, empties.keys())
        
        if body_name:
            # Make sure the empty has correct rotation mode
            empties[body_name].rotation_mode = 'QUATERNION'
            
            # Parent the mesh to the empty
            mesh_obj.parent = empties[body_name]
            
            # Clear the parent inverse matrix so mesh uses empty's local coordinates
            mesh_obj.matrix_parent_inverse.identity()
            
            # Set mesh to local origin of the empty (or custom offset)
            if local_offset:
                mesh_obj.location = local_offset
            else:
                mesh_obj.location = (0, 0, 0)
            
            mesh_obj.rotation_euler = (0, 0, 0)
            mesh_obj.scale = (1, 1, 1)
            
            print(f"✓ {mesh_name:40s} → {body_name}")
            parented_count += 1
        else:
            print(f"✗ {mesh_name:40s} → NO MATCH FOUND")
            unmatched_meshes.append(mesh_name)
    
    print("\n" + "="*60)
    print(f"Parenting Complete!")
    print("="*60)
    print(f"Successfully parented: {parented_count}/{len(meshes)} meshes")
    
    if unmatched_meshes:
        print(f"\n⚠ Warning: {len(unmatched_meshes)} meshes couldn't be matched:")
        for mesh_name in unmatched_meshes[:10]:  # Show first 10
            print(f"  - {mesh_name}")
        if len(unmatched_meshes) > 10:
            print(f"  ... and {len(unmatched_meshes) - 10} more")
        print("\nYou may need to parent these manually or add them to the mapping.")
    
    print("\n✓ Ready! Press SPACEBAR to play the animation.")
    print("  The robot meshes should now follow the motion.\n")


def list_all_names():
    """
    Helper function to list all empties and meshes for debugging.
    """
    print("\n" + "="*60)
    print("Available Empties (Body Names):")
    print("="*60)
    for obj in sorted(bpy.data.objects, key=lambda x: x.name):
        if obj.type == 'EMPTY':
            print(f"  - {obj.name}")
    
    print("\n" + "="*60)
    print("Available Meshes:")
    print("="*60)
    for obj in sorted(bpy.data.objects, key=lambda x: x.name):
        if obj.type == 'MESH':
            print(f"  - {obj.name}")
    print("\n")


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Uncomment the next line to see all available names (for debugging)
    # list_all_names()
    
    # Parent meshes to empties
    parent_meshes_to_empties()
