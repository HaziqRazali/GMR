"""
Apply a delta transform (translation) to the entire robot animation.
Run this AFTER parenting meshes to empties.

This moves the entire robot by a constant offset in any direction.
"""

import bpy

def apply_delta_transform(offset_x=0, offset_y=0, offset_z=0):
    """
    Apply a constant offset to all empties across all frames.
    
    Args:
        offset_x: X-axis offset
        offset_y: Y-axis offset  
        offset_z: Z-axis offset
    """
    print("="*60)
    print("Applying Delta Transform to Robot")
    print("="*60)
    print(f"Offset: ({offset_x}, {offset_y}, {offset_z})")
    
    # Get all empties
    empties = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
    
    if not empties:
        print("❌ No empties found!")
        return
    
    print(f"Found {len(empties)} empties to transform")
    
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    
    # Apply offset to all empties across all frames
    for frame_num in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame_num)
        
        for empty in empties:
            # Add offset to current location
            empty.location.x += offset_x
            empty.location.y += offset_y
            empty.location.z += offset_z
            
            # Update keyframe
            empty.keyframe_insert(data_path="location", frame=frame_num)
        
        if (frame_num - frame_start + 1) % 50 == 0:
            print(f"  Processed frame {frame_num}/{frame_end}")
    
    print("\n✓ Delta transform applied!")
    print(f"✓ Entire robot moved by ({offset_x}, {offset_y}, {offset_z})")
    print("="*60)
    
    # Reset to first frame
    bpy.context.scene.frame_set(frame_start)


# =============================================================================
# MODIFY THE OFFSET VALUES HERE
# =============================================================================

if __name__ == "__main__":
    # Change these values to move the robot:
    OFFSET_X = 2.0   # Move right (+) or left (-)
    OFFSET_Y = 0.0   # Move forward (+) or backward (-)
    OFFSET_Z = 0.0   # Move up (+) or down (-)
    
    apply_delta_transform(OFFSET_X, OFFSET_Y, OFFSET_Z)
