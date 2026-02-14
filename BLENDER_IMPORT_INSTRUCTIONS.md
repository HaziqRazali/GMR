# Import Robot Motion into Blender - Complete Instructions

## ✅ Files Generated

Your robot motion has been exported to:
```
/home/haziq/GMR/blender_export/
├── robot_motion_animation.json      # Animation keyframe data (682 frames at 30 FPS)
├── robot_motion_robot_info.json     # Robot model metadata
└── (robot_motion.fbx)                # Placeholder
```

## 📋 Step-by-Step Instructions

### Method 1: Simple Visualization (Recommended - Easiest)

This creates animated markers showing the robot's body parts moving through space.

1. **Open Blender** (tested with Blender 3.x+)

2. **Open Scripting Workspace**
   - Click "Scripting" tab at the top of Blender

3. **Load the import script**
   - Click "Open" in the text editor
   - Navigate to: `/home/haziq/GMR/scripts/blender_import_robot_motion.py`
   - Or copy-paste the script content

4. **Update file paths** (lines 271-272 in the script):
   ```python
   ANIMATION_JSON = "/home/haziq/GMR/blender_export/robot_motion_animation.json"
   ROBOT_INFO_JSON = "/home/haziq/GMR/blender_export/robot_motion_robot_info.json"
   ```

5. **Run the script**
   - Press `Alt + P` or click the "Run Script" button
   - You should see: "✓ Import Complete!"

6. **Play the animation**
   - Press `SPACEBAR` to play
   - Use mouse to rotate view (Middle Mouse Button drag)
   - Scroll to zoom

**Result:** You'll see 39 animated spheres (empties) representing the Unitree G1 robot's body parts performing the jogging motion.

---

### Method 2: With Robot Mesh (Advanced)

To see the actual robot 3D model instead of just markers:

#### Option A: Import URDF (Requires Plugin)

1. **Install URDF Importer addon for Blender**
   
   **Option 1 - Phobos (Recommended):**
   ```bash
   # Download from Blender Extensions or:
   # https://github.com/dfki-ric/phobos
   # Then in Blender: Edit → Preferences → Add-ons → Install → Select the .zip file
   ```
   
   **Option 2 - bpy_building_blocks:**
   ```bash
   git clone https://github.com/AIS-Bonn/bpy_building_blocks.git
   cd bpy_building_blocks
   # Copy to Blender addons folder: ~/.config/blender/[version]/scripts/addons/
   ```
   
   **Option 3 - Manual URDF Parser (simplest):**
   - Use the built-in script to import meshes directly (see Option B below)

2. **Import robot URDF manually** (if addon installed)
   - In Blender: File → Import → URDF (.urdf)
   - Select: `/home/haziq/GMR/assets/unitree_g1/g1_custom_collision_29dof.urdf`

3. **Run the animation script** (from Method 1)
   - The script will animate the imported robot

#### Option B: Import robot meshes (Recommended - Works Now!)

1. **First, run the animation import** (from Method 1 above)
   - This creates the animated empties/markers

2. **Import robot meshes into Blender**
   - File → Import → STL (.stl)
   - Navigate to: `/home/haziq/GMR/assets/unitree_g1/meshes/`
   - Select all STL files (Ctrl+A) and click "Import STL"
   - All meshes will be imported at origin

3. **Automatically parent meshes to animated markers**
   - In Scripting workspace, open a new text file
   - Load script: `/home/haziq/GMR/scripts/blender_parent_meshes_to_animation.py`
   - Run the script (Alt+P)
   - The script will automatically match and parent each mesh to its corresponding animated marker
   - You should see output like:
     ```
     ✓ left_hip_pitch_link → left_hip_pitch_link
     ✓ right_knee_link → right_knee_link
     ...
     Successfully parented: XX/XX meshes
     ```

4. **Play the animation**
   - Press SPACEBAR
   - The robot meshes should now follow the jogging motion!

**Troubleshooting:**
- If some meshes don't match, run `list_all_names()` in the script to see all available names
- Manually parent unmatched meshes: Select mesh → Shift+Select empty → Ctrl+P → "Object (Keep Transform)"

---

## 🎬 Animation Details

- **Total Frames:** 682
- **Frame Rate:** ~30 FPS
- **Duration:** ~22.7 seconds
- **Motion:** Jogging motion from HumanEva dataset (S1/Jog_1)
- **Robot Bodies:** 39 body parts tracked
- **Joints:** 29 degrees of freedom

---

## 🔧 Troubleshooting

### "No module named bpy"
- This is normal if you run the Blender script outside of Blender
- The script must be run **inside Blender's scripting environment**

### Animation looks strange
- Make sure coordinate system is correct (Blender uses Z-up)
- The script handles coordinate conversions automatically

### Want to export from Blender?
After importing, you can export the animation to other formats:
- File → Export → FBX (.fbx) - for Unity/Unreal
- File → Export → Alembic (.abc) - for VFX pipelines
- File → Render → Render Animation - to create video

---

## 📊 What's in the JSON files?

### robot_motion_animation.json
```json
{
  "fps": 30,
  "num_frames": 682,
  "frames": [
    {
      "root_pos": [x, y, z],
      "root_rot": [x, y, z, w],  // quaternion
      "dof_pos": [29 joint angles],
      "body_poses": {
        "pelvis": {"position": [...], "rotation": [...]},
        "left_hip": {...},
        // ... 39 body parts total
      }
    },
    // ... 682 frames
  ]
}
```

### robot_motion_robot_info.json
```json
{
  "robot_type": "unitree_g1",
  "xml_path": "/home/haziq/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
  "urdf_path": "/home/haziq/GMR/assets/unitree_g1/g1_mocap_29dof.urdf",
  "mesh_dir": "/home/haziq/GMR/assets/unitree_g1/meshes"
}
```

---

## 🎯 Quick Start (TL;DR)

```bash
# 1. You already ran this:
python scripts/smplx_to_robot.py --smplx_file /path/to/Jog_1.npz --robot unitree_g1 --save_path motion.npz

# 2. You already exported:
conda activate gmr
python scripts/export_robot_motion_to_blender.py --motion_file motion.npz --robot unitree_g1 --output blender_export/robot_motion.fbx

# 3. Now in Blender:
# - Open Blender
# - Scripting workspace
# - Open/paste: scripts/blender_import_robot_motion.py
# - Run script (Alt+P)
# - Press SPACE to play animation
```

---

## 💡 Tips

1. **Camera setup:** In Blender, press `Numpad 0` to view through camera, then position it to see the robot
2. **Lighting:** Add lights (Shift+A → Light) for better visualization
3. **Ground plane:** Add a plane (Shift+A → Mesh → Plane) and scale it up for ground reference
4. **Slow motion:** In Timeline, change FPS to 15 to slow down playback
5. **Export video:** Set output format (PNG/MP4) and click Render → Render Animation

---

## 📁 File Locations Summary

```
/home/haziq/GMR/
├── motion.npz                                    # Original exported robot motion
├── blender_export/
│   ├── robot_motion_animation.json              # ← Import this in Blender
│   └── robot_motion_robot_info.json             # ← Import this in Blender
├── scripts/
│   ├── export_robot_motion_to_blender.py        # Export script (already ran)
│   └── blender_import_robot_motion.py           # ← Run this INSIDE Blender
└── assets/unitree_g1/
    ├── g1_custom_collision_29dof.urdf           # Robot URDF (optional)
    └── meshes/                                   # Robot 3D meshes (optional)
```

---

## ✨ You're all set!

The exported files are ready to be imported into Blender. Follow Method 1 (Simple Visualization) for the quickest results.
