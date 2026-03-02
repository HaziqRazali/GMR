

python scripts/smplx_to_robot.py \
--smplx_file /home/haziq/sam-3d-body/example_data/results/img_smplx.npz \
--robot booster_t1 \
--rate_limit --freeze_at_end  --rotate_roll 90

python scripts/smplx_to_robot.py --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot unitree_g1 --save_path motion.npz --rate_limit

python scripts/smplx_to_robot.py \
--smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
--robot booster_t1 \
--save_path motion.npz \
--rate_limit --record_video --video_path booster.mp4 \
--camera_distance 5.0  --camera_elevation -15 --rotate_yaw -90