"""
Create static robot duplicates at multiple frames to visualize entire motion.
This creates "ghost frames" showing the robot at different timesteps simultaneously.

Run this AFTER importing animation and parenting meshes.
"""

import bpy

def create_motion_trail(frame_step=10, start_frame=None, end_frame=None, opacity=0.5):
    """
    Create static robot copies at different timesteps.
    
    Args:
        frame_step: Create a copy every N frames (e.g., 10 = copy at frame 1, 11, 21, 31...)
        start_frame: First frame to copy (None = scene start)
        end_frame: Last frame to copy (None = scene end)
        opacity: Transparency for ghost robots (0=invisible, 1=opaque)
    """
    print("="*60)
    print("Creating Motion Trail (Ghost Frames)")
    print("="*60)
    
    # Get frame range
    if start_frame is None:
        start_frame = bpy.context.scene.frame_start
    if end_frame is None:
        end_frame = bpy.context.scene.frame_end
    
    # Get all meshes (robot parts)
    original_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not original_meshes:
        print("❌ No meshes found! Import the robot first.")
        return
    
    print(f"Found {len(original_meshes)} mesh objects")
    print(f"Creating copies from frame {start_frame} to {end_frame}, step={frame_step}")
    
    # Create a collection for each frame
    trail_collection = bpy.data.collections.new("Motion_Trail")
    bpy.context.scene.collection.children.link(trail_collection)
    
    frame_list = range(start_frame, end_frame + 1, frame_step)
    total_frames = len(frame_list)
    
    print(f"Will create {total_frames} ghost robots")
    
    for idx, frame_num in enumerate(frame_list):
        # Set to this frame
        bpy.context.scene.frame_set(frame_num)
        
        # Create collection for this frame
        frame_collection = bpy.data.collections.new(f"Frame_{frame_num:04d}")
        trail_collection.children.link(frame_collection)
        
        # Duplicate all meshes at this frame
        for mesh_obj in original_meshes:
            # Duplicate the mesh
            new_obj = mesh_obj.copy()
            new_obj.data = mesh_obj.data.copy()
            new_obj.name = f"{mesh_obj.name}_frame_{frame_num:04d}"
            
            # Add to frame collection first (before removing parent)
            frame_collection.objects.link(new_obj)
            
            # Store world transform BEFORE clearing parent
            world_matrix = mesh_obj.matrix_world.copy()
            
            # Clear parent while keeping transform
            new_obj.parent = None
            new_obj.matrix_parent_inverse.identity()
            
            # Apply the stored world transform
            new_obj.matrix_world = world_matrix
            
            # Clear animation
            new_obj.animation_data_clear()
            
            # Adjust material for transparency
            if new_obj.data.materials:
                for mat_slot in new_obj.data.materials:
                    if mat_slot:
                        # Create a copy of the material
                        mat = mat_slot.copy()
                        mat.name = f"{mat_slot.name}_ghost_{frame_num}"
                        
                        # Set transparency
                        if mat.use_nodes:
                            bsdf = mat.node_tree.nodes.get('Principled BSDF')
                            if bsdf:
                                bsdf.inputs['Alpha'].default_value = opacity
                        
                        mat.blend_method = 'BLEND'
                        
                        # Replace material
                        new_obj.data.materials[0] = mat
        
        if (idx + 1) % 5 == 0:
            print(f"  Created ghost robot {idx + 1}/{total_frames} (frame {frame_num})")
    
    print("\n✓ Motion trail created!")
    print(f"✓ Created {total_frames} ghost robots")
    print(f"\nTo hide/show specific frames:")
    print("  - Find 'Motion_Trail' collection in Outliner")
    print("  - Click eye icon next to Frame_XXXX to hide/show that timestep")
    print("="*60)
    
    # Reset to start frame
    bpy.context.scene.frame_set(start_frame)


def delete_motion_trail():
    """Delete all motion trail ghost robots."""
    if "Motion_Trail" in bpy.data.collections:
        collection = bpy.data.collections["Motion_Trail"]
        
        # Delete all objects in the collection
        for obj in collection.all_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Delete the collection
        bpy.data.collections.remove(collection)
        
        print("✓ Motion trail deleted")
    else:
        print("⚠ No motion trail found")


# =============================================================================
# MODIFY SETTINGS HERE
# =============================================================================

if __name__ == "__main__":
    # Settings
    FRAME_STEP = 10      # Create a ghost every 10 frames
    START_FRAME = None   # None = use scene start
    END_FRAME = None     # None = use scene end
    OPACITY = 0.3        # 0.0 = invisible, 1.0 = fully opaque
    
    # To create motion trail:
    create_motion_trail(
        frame_step=FRAME_STEP,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        opacity=OPACITY
    )
    
    # To delete motion trail (uncomment):
    # delete_motion_trail()
