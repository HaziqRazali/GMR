"""
Debug script to check which empties have rotation animation.
Run this in Blender to diagnose the issue.
"""

import bpy

def check_animation_data():
    """Check which empties have location and rotation keyframes."""
    
    print("="*60)
    print("Checking Animation Data on Empties")
    print("="*60)
    
    empties = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
    
    # Try simple approach - check at different frames
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(1)
    
    for empty in sorted(empties, key=lambda x: x.name):
        rotation_mode = empty.rotation_mode
        
        # Check if object has animation by testing if values change across frames
        bpy.context.scene.frame_set(1)
        loc1 = empty.location.copy()
        if rotation_mode == 'QUATERNION':
            rot1 = empty.rotation_quaternion.copy()
        else:
            rot1 = empty.rotation_euler.copy()
        
        bpy.context.scene.frame_set(30)
        loc2 = empty.location.copy()
        if rotation_mode == 'QUATERNION':
            rot2 = empty.rotation_quaternion.copy()
        else:
            rot2 = empty.rotation_euler.copy()
        
        has_location = (loc1 - loc2).length > 0.001
        
        # Check rotation differently based on mode
        if rotation_mode == 'QUATERNION':
            has_rotation = (rot1 - rot2).length > 0.001
        else:
            # For Euler, check each axis
            has_rotation = abs(rot1[0] - rot2[0]) > 0.001 or abs(rot1[1] - rot2[1]) > 0.001 or abs(rot1[2] - rot2[2]) > 0.001
        
        status = ""
        if has_location and has_rotation:
            status = f"✓ LOC + ROT [{rotation_mode}]"
        elif has_location:
            status = f"⚠ LOC only [{rotation_mode}]"
        elif has_rotation:
            status = f"⚠ ROT only [{rotation_mode}]"
        else:
            status = f"✗ No animation [{rotation_mode}]"
        
        print(f"{empty.name:35s} {status}")
    
    # Restore frame
    bpy.context.scene.frame_set(current_frame)
    
    print("\n" + "="*60)

def check_mesh_parents():
    """Check which empty each mesh is parented to."""
    
    print("\nMesh Parenting:")
    print("="*60)
    
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    for mesh in sorted(meshes, key=lambda x: x.name):
        if mesh.parent:
            print(f"{mesh.name:35s} → {mesh.parent.name}")
        else:
            print(f"{mesh.name:35s} → NOT PARENTED")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_animation_data()
    check_mesh_parents()
