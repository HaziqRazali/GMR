

python /home/haziq/GMR/scripts/smplx_to_robot.py \
--smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
--robot booster_t1 \
--rate_limit \
--camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 --hide_floor

python scripts/smplx_to_robot.py \
    --smplx_file /tmp/smplx_rest_pose.json \
    --robot booster_t1 \
    --rate_limit \
    --camera_distance 5.5 --camera_elevation -15 --rotate_roll 90 --hide_floor

# Fit3D MHR file
conda activate mhr_new
cd /home/haziq/GMR
python /home/haziq/GMR/scripts/mhr_to_robot.py \
--mhr_file /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz \
--robot booster_t1 \
--rate_limit \
--camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90

python scripts/mhr_to_robot.py \
    --mhr_file /tmp/mhr_rest_pose.npz \
    --robot booster_t1 \
    --rate_limit \
    --camera_distance 5.5 --camera_elevation -15 --rotate_roll 90 --hide_floor