#!/usr/bin/env bash
# Launch batch retargeting for tienkung, unitree_h1, and openloong
# each with their true (hand-tuned) and generated IK configs.
# Jobs run sequentially to avoid overloading the CPU.

cd "$(dirname "$0")/.."

run() {
    local robot="$1" ik_config="$2" output_root="$3" logfile="$4"
    echo "=== START: $robot | $(basename $ik_config) ==="
    conda run -n gmr bash scripts/batch_smplx_to_robot_fit3d_val.sh \
        "$ik_config" "$output_root" "$robot" \
        2>&1 | tee "$logfile"
    echo "=== DONE:  $robot | $(basename $ik_config) ==="
}

# tienkung
run tienkung \
    general_motion_retargeting/ik_configs/smplx_to_tienkung.json \
    results/fit3d_val_tienkung \
    results_batch_tienkung_true.log

run tienkung \
    general_motion_retargeting/ik_configs/smplx_to_tienkung_generated2.json \
    results/fit3d_val_tienkung_generated \
    results_batch_tienkung_generated.log

# unitree_h1
run unitree_h1 \
    general_motion_retargeting/ik_configs/smplx_to_h1.json \
    results/fit3d_val_h1 \
    results_batch_h1_true.log

run unitree_h1 \
    general_motion_retargeting/ik_configs/smplx_to_h1_generated.json \
    results/fit3d_val_h1_generated \
    results_batch_h1_generated.log

# openloong
run openloong \
    general_motion_retargeting/ik_configs/smplx_to_openloong.json \
    results/fit3d_val_openloong \
    results_batch_openloong_true.log

run openloong \
    general_motion_retargeting/ik_configs/smplx_to_openloong_generated.json \
    results/fit3d_val_openloong_generated \
    results_batch_openloong_generated.log

echo "All done!"
