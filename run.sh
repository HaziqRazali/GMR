python scripts/smplx_to_robot.py --smplx_file /data/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot unitree_g1 --save_path motion.npz --rate_limit
blender --background --python scripts/blender_import_motion.py -- motion.npz --mesh_dir /data/assets/unitree_g1/meshes

# local motion
python scripts/smplx_to_robot_json.py \
  --smplx_file /data/VITRA/motionx/data/local_motion/idea400/local_motion/Act_cute_and_sitting_at_the_same_time_clip1.json \
  --robot unitree_g1 \
  --save_path local_motion.npz \
  --rate_limit

python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path local_motion.npz

# global motion - single file
python scripts/smplx_to_robot_json.py \
  --smplx_file /data/VITRA/motionx/data/global_motion/idea400/global_motion/Act_cute_and_sitting_at_the_same_time_clip1.json \
  --robot unitree_g1 \
  --save_path global_motion.npz \
  --rate_limit

python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path global_motion.npz

# global motion - batch process all files in directory
INPUT_DIR="/data/VITRA/motionx/data/global_motion/idea400/global_motion"
OUTPUT_DIR="/data/VITRA/motionx/data/global_robot_motion/idea400/global_robot_motion"
ROBOT="unitree_g1"

mkdir -p "$OUTPUT_DIR"

for json_file in "$INPUT_DIR"/*.json; do
  if [ -f "$json_file" ]; then
    filename=$(basename "$json_file" .json)
    echo "Processing: $filename"
    
    python scripts/smplx_to_robot_json.py \
      --smplx_file "$json_file" \
      --robot "$ROBOT" \
      --save_path "$OUTPUT_DIR/${filename}.npz" \
      --no_viewer
    
    echo "Saved: $OUTPUT_DIR/${filename}.npz"
    echo "---"
  fi
done

echo "All files processed!"