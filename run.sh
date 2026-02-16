
#################### default smplx_to_robot
python scripts/smplx_to_robot.py --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot unitree_g1 --save_path motion.npz --rate_limit

####################
python scripts/smplx_to_robot_v2.py idx=47 result_foldername="/home/haziq/Collab_AI/results/AAAI2026/2024_11_18/gimo/obj2pose/cvae=all_v2_transl=mse_orient=mse_pred_all_objects_dtype=train_mib/" obj_idx=1 rep_idx=0 --robot unitree_g1 --save_path motion.npz --rate_limit
python scripts/export_robot_motion_to_blender.py --motion_file motion.npz --robot unitree_g1 --output blender_export/robot_motion.fbx

#################### global motion - single file
python scripts/smplx_to_robot_json.py \
  --smplx_file /data/VITRA/motionx/data/global_motion/idea400/global_motion/Act_cute_and_sitting_at_the_same_time_clip1.json \
  --robot unitree_g1 \
  --save_path global_motion.npz \
  --rate_limit

python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path global_motion.npz

#################### global motion - batch process all files in directory
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


#################### global motion - process all subfolders and robots
INPUT_BASE_DIR="/media/haziq/Haziq/motionx/Motion-X++/motion/mesh_recovery/global_motion"
OUTPUT_ROOT_BASE="/media/haziq/Haziq/motionx/Motion-X++/motion/robot_motion/global_motion"
ROBOTS=("unitree_g1" "unitree_h1" "unitree_h1_2" "booster_t1" "booster_t1_29dof" "berkeley_humanoid_lite" "booster_k1")

# Iterate each subfolder under INPUT_BASE_DIR (e.g. idea400, ikea400, ...)
for FOLDER_PATH in "$INPUT_BASE_DIR"/*/; do
  [ -d "$FOLDER_PATH" ] || continue
  FOLDER=$(basename "$FOLDER_PATH")
  INPUT_DIR="$FOLDER_PATH"
  OUTPUT_BASE="$OUTPUT_ROOT_BASE/$FOLDER"

  echo "=== Processing folder: $FOLDER ==="

  # Launch one background job per robot for this folder, then wait for them
  for ROBOT in "${ROBOTS[@]}"; do
    (
      OUTPUT_DIR="$OUTPUT_BASE/$ROBOT"
      mkdir -p "$OUTPUT_DIR"

      echo "Processing robot: $ROBOT (folder: $FOLDER)"

      for json_file in "$INPUT_DIR"/*.json; do
        [ -f "$json_file" ] || continue
        filename=$(basename "$json_file" .json)
        echo "[$FOLDER][$ROBOT] Processing: $filename"

        python scripts/smplx_to_robot_json.py \
          --smplx_file "$json_file" \
          --robot "$ROBOT" \
          --save_path "$OUTPUT_DIR/${filename}.npz" \
          --no_viewer

        echo "[$FOLDER][$ROBOT] Saved: $OUTPUT_DIR/${filename}.npz"
      done

      echo "[$FOLDER][$ROBOT] Finished processing!"
    ) &
  done

  # Wait for all robots in this folder to finish before moving to next folder
  wait
  echo "Finished folder: $FOLDER"
  echo "======"
done

echo "All folders processed!"