"""
Apply materials and colors to robot meshes based on MuJoCo XML definitions.
Run this AFTER parenting meshes to empties.
"""

import bpy

# Color definitions from MuJoCo XML (RGBA format: R, G, B, Alpha)
MESH_COLORS = {
    # Dark grey/black parts (0.2, 0.2, 0.2, 1)
    'pelvis': (0.2, 0.2, 0.2, 1.0),
    'left_hip_pitch_link': (0.2, 0.2, 0.2, 1.0),
    'right_hip_pitch_link': (0.2, 0.2, 0.2, 1.0),
    'left_ankle_roll_link': (0.2, 0.2, 0.2, 1.0),
    'right_ankle_roll_link': (0.2, 0.2, 0.2, 1.0),
    'logo_link': (0.2, 0.2, 0.2, 1.0),
    
    # Light grey parts (0.7, 0.7, 0.7, 1)
    'pelvis_contour_link': (0.7, 0.7, 0.7, 1.0),
    'left_hip_roll_link': (0.7, 0.7, 0.7, 1.0),
    'left_hip_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'left_knee_link': (0.7, 0.7, 0.7, 1.0),
    'left_ankle_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'right_hip_roll_link': (0.7, 0.7, 0.7, 1.0),
    'right_hip_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'right_knee_link': (0.7, 0.7, 0.7, 1.0),
    'right_ankle_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'waist_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'waist_roll_link': (0.7, 0.7, 0.7, 1.0),
    'torso_link': (0.7, 0.7, 0.7, 1.0),
    'head_link': (0.7, 0.7, 0.7, 1.0),
    'left_shoulder_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'left_shoulder_roll_link': (0.7, 0.7, 0.7, 1.0),
    'left_shoulder_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'left_elbow_link': (0.7, 0.7, 0.7, 1.0),
    'left_wrist_roll_link': (0.7, 0.7, 0.7, 1.0),
    'left_wrist_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'left_wrist_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'left_rubber_hand': (0.7, 0.7, 0.7, 1.0),
    'right_shoulder_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'right_shoulder_roll_link': (0.7, 0.7, 0.7, 1.0),
    'right_shoulder_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'right_elbow_link': (0.7, 0.7, 0.7, 1.0),
    'right_wrist_roll_link': (0.7, 0.7, 0.7, 1.0),
    'right_wrist_pitch_link': (0.7, 0.7, 0.7, 1.0),
    'right_wrist_yaw_link': (0.7, 0.7, 0.7, 1.0),
    'right_rubber_hand': (0.7, 0.7, 0.7, 1.0),
}


def create_material(name, color_rgba):
    """Create a material with the given color."""
    # Check if material already exists
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name=name)
    
    # Use nodes for better rendering
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Create Principled BSDF shader
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Set color (RGB)
    bsdf.inputs['Base Color'].default_value = color_rgba
    
    # Set metallic and roughness for plastic-like appearance
    bsdf.inputs['Metallic'].default_value = 0.1
    bsdf.inputs['Roughness'].default_value = 0.5
    
    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    
    # Link nodes
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def apply_colors_to_meshes():
    """Apply colors to all robot meshes based on their names."""
    
    print("="*60)
    print("Applying Colors to Robot Meshes")
    print("="*60)
    
    # Get all mesh objects
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    colored_count = 0
    
    for mesh_obj in meshes:
        mesh_name = mesh_obj.name
        
        # Remove Blender suffixes like .001, .002, etc.
        base_name = mesh_name.replace('.001', '').replace('.002', '').replace('.003', '').replace('.004', '')
        
        # Check if we have a color for this mesh
        if base_name in MESH_COLORS:
            color = MESH_COLORS[base_name]
            
            # Create or get material
            mat_name = f"Material_{base_name}"
            mat = create_material(mat_name, color)
            
            # Apply material to mesh
            if len(mesh_obj.data.materials) == 0:
                mesh_obj.data.materials.append(mat)
            else:
                mesh_obj.data.materials[0] = mat
            
            print(f"✓ {mesh_name:40s} → RGB{color[:3]}")
            colored_count += 1
        else:
            # Default grey color for unknown meshes
            default_color = (0.5, 0.5, 0.5, 1.0)
            mat = create_material(f"Material_Default_{base_name}", default_color)
            
            if len(mesh_obj.data.materials) == 0:
                mesh_obj.data.materials.append(mat)
            else:
                mesh_obj.data.materials[0] = mat
            
            print(f"⚠ {mesh_name:40s} → Default Grey (no color defined)")
    
    print("\n" + "="*60)
    print(f"✓ Applied colors to {colored_count}/{len(meshes)} meshes")
    print("="*60)
    
    # Optional: Switch to Material Preview or Rendered view to see colors
    # Commented out to avoid lag - manually press 'Z' → 'Material Preview' when ready
    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         for space in area.spaces:
    #             if space.type == 'VIEW_3D':
    #                 space.shading.type = 'MATERIAL'
    #                 break
    
    print("\n✓ Colors applied!")
    print("  To see colors: Press 'Z' → 'Material Preview' (may cause lag)")
    print("  Or press 'Z' → 'Solid' with 'Matcap' enabled for faster preview\n")


# =============================================================================
# RUN THIS
# =============================================================================

if __name__ == "__main__":
    apply_colors_to_meshes()
