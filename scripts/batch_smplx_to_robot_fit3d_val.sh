#!/usr/bin/env bash
# Batch retarget all Fit3D val SMPL-X JSON files → unitree_g1 robot motions.
# Output: results/fit3d_val_g1/{subject}/{motion}.npz
# No viewer, no video, no rate-limiting (pure IK + save).

# Args:  $1 = ik_config   $2 = output_root   $3 = robot
IK_CONFIG="${1:-general_motion_retargeting/ik_configs/smplx_to_g1.json}"
OUTPUT_ROOT="${2:-results/fit3d_val_g1}"
ROBOT="${3:-unitree_g1}"
INPUT_GLOB="/home/haziq/datasets/mocap/data/fit3d/val/*/smplx/*.json"

total=0
done_count=0
failed=0

# Count total files first
for f in $INPUT_GLOB; do
    [ -f "$f" ] && total=$((total + 1))
done
echo "Found $total files to process"
echo "Output root: $OUTPUT_ROOT"
echo "IK config  : $IK_CONFIG"
echo "---"

for json_file in $INPUT_GLOB; do
    [ -f "$json_file" ] || continue

    # Extract subject (e.g. s10) and motion name (e.g. warmup_9)
    subject=$(echo "$json_file" | grep -oP '(?<=/val/)[^/]+')
    motion=$(basename "$json_file" .json)

    out_dir="$OUTPUT_ROOT/$subject"
    out_file="$out_dir/${motion}.npz"
    mkdir -p "$out_dir"

    # Skip if already computed
    if [ -f "$out_file" ]; then
        echo "[$((done_count+1))/$total] SKIP (exists): $subject/$motion"
        done_count=$((done_count + 1))
        continue
    fi

    echo "[$((done_count+1))/$total] Processing: $subject/$motion"

    python scripts/smplx_to_robot.py \
        --smplx_file "$json_file" \
        --robot "$ROBOT" \
        --ik_config "$IK_CONFIG" \
        --save_path "$out_file" \
        --no_viewer
    if [ $? -eq 0 ]; then
        done_count=$((done_count + 1))
    else
        echo "  FAILED: $subject/$motion"
        failed=$((failed + 1))
    fi

done

echo "---"
echo "Done: $done_count / $total"
[ "$failed" -gt 0 ] && echo "Failed: $failed"
