"""
Generate a 100-frame SMPL-X T-pose (rest pose) sequence in Fit3D JSON format.

Output can be used directly with smplx_to_robot.py:

    python scripts/smplx_to_robot.py \
        --smplx_file /tmp/smplx_rest_pose.json \
        --robot booster_t1 \
        --rate_limit \
        --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90
"""

import json
import argparse
import numpy as np

def main(args):
    N = args.num_frames

    # Identity rotation matrix (3x3)
    I = np.eye(3)

    data = {
        # (N, 3) — root translation, set to zero so pelvis is at origin
        "transl": np.zeros((N, 3)).tolist(),

        # (N, 1, 3, 3) — root orientation: identity = no rotation from Y-up rest
        "global_orient": np.tile(I, (N, 1, 1, 1)).tolist(),

        # (N, 21, 3, 3) — body joints: all identity = T-pose
        "body_pose": np.tile(I, (N, 21, 1, 1)).tolist(),

        # (N, 10) — shape coefficients: zero = mean body shape
        "betas": np.zeros((N, 10)).tolist(),

        # (N, 15, 3, 3) — hand poses: identity
        "left_hand_pose":  np.tile(I, (N, 15, 1, 1)).tolist(),
        "right_hand_pose": np.tile(I, (N, 15, 1, 1)).tolist(),

        # (N, 1, 3, 3)
        "jaw_pose":  np.tile(I, (N, 1, 1, 1)).tolist(),
        "leye_pose": np.tile(I, (N, 1, 1, 1)).tolist(),
        "reye_pose": np.tile(I, (N, 1, 1, 1)).tolist(),

        # (N, 10) — facial expression: neutral
        "expression": np.zeros((N, 10)).tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(data, f)

    print(f"Saved {N}-frame SMPL-X T-pose sequence to: {args.output}")
    print()
    print("Verify with:")
    print(f"  python scripts/smplx_to_robot.py \\")
    print(f"      --smplx_file {args.output} \\")
    print(f"      --robot booster_t1 \\")
    print(f"      --rate_limit \\")
    print(f"      --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/smplx_rest_pose.json",
                        help="Output JSON path (default: /tmp/smplx_rest_pose.json)")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Number of frames to generate (default: 100)")
    main(parser.parse_args())
