"""
Generate a 100-frame MHR rest pose (T-pose) sequence in MHR NPZ format.

Output can be used directly with mhr_to_robot.py:

    python scripts/mhr_to_robot.py \
        --mhr_file /tmp/mhr_rest_pose.npz \
        --robot booster_t1 \
        --rate_limit \
        --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90

Run in the mhr_new conda environment:
    conda activate mhr_new
"""

import argparse
import numpy as np


def main(args):
    N = args.num_frames

    # MHR NPZ format expected by load_mhr_npz():
    #   param_lbs_model_params  (T, 204)  — pose parameters; zero = T-pose / rest
    #   param_identity_coeffs   (T,  45)  — shape/identity; zero = mean body
    #   param_face_expr_coeffs  (T,  72)  — facial expression; zero = neutral

    data = {
        "param_lbs_model_params": np.zeros((N, 204), dtype=np.float32),
        "param_identity_coeffs":  np.zeros((N,  45), dtype=np.float32),
        "param_face_expr_coeffs": np.zeros((N,  72), dtype=np.float32),
    }

    np.savez(args.output, **data)

    # np.savez appends .npz if not present — normalise for display
    out_path = args.output if args.output.endswith(".npz") else args.output + ".npz"
    print(f"Saved {N}-frame MHR T-pose sequence to: {out_path}")
    print()
    print("Verify with (conda activate mhr_new first):")
    print(f"  python scripts/mhr_to_robot.py \\")
    print(f"      --mhr_file {out_path} \\")
    print(f"      --robot booster_t1 \\")
    print(f"      --rate_limit \\")
    print(f"      --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/mhr_rest_pose.npz",
                        help="Output NPZ path (default: /tmp/mhr_rest_pose.npz)")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Number of frames to generate (default: 100)")
    main(parser.parse_args())
