"""
Quick script to hide/show all empties (markers) in Blender.
"""

import bpy

# Hide all empties
for obj in bpy.data.objects:
    if obj.type == 'EMPTY':
        obj.hide_set(True)  # Hide in viewport
        obj.hide_render = True  # Hide in render

print("✓ All empties (markers) hidden")

# To SHOW them again, use:
# for obj in bpy.data.objects:
#     if obj.type == 'EMPTY':
#         obj.hide_set(False)
#         obj.hide_render = False
