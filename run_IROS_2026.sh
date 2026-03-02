
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                         #
# IROS : Vision–Language Pose Correction and Automatic IK Configuration for Human-to-Robot Pose Transfer  #
#                                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# example scripts
python scripts/smplx_to_robot.py --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot unitree_g1 --save_path motion.npz --rate_limit
python scripts/smplx_to_robot.py --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot tienkung --save_path motion.npz --rate_limit
python scripts/smplx_to_robot.py --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Jog_1.npz --robot openloong --save_path motion.npz --rate_limit

python3 scripts/generate_smplx_retarget.py assets/unitree_g1/g1_custom_collision_29dof.urdf --human-scale 0.9 --arm-scale 0.8 --ground-height 0.0 --no-head -o output4.json
python3 scripts/generate_smplx_retarget.py assets/tienkung/urdf/tienkung2_lite.urdf --human-scale 0.9 --arm-scale 0.8 --ground-height 0.0 --no-head -o /home/haziq/GMR/general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# # # # # # #
# h1        #
# # # # # # #

python3 scripts/generate_smplx_retarget.py \
  assets/unitree_h1/h1.urdf \
  --human-scale 0.9 \
  --arm-scale 0.8 \
  --ground-height 0.0 \
  --no-head \
  -o general_motion_retargeting/ik_configs/smplx_to_h1_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot unitree_h1 --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_h1_warmup_9_generated.npz \
  --video_path videos/unitree_h1_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_h1.json
  
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot unitree_h1 --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_h1_warmup_9_generated.npz \
  --video_path videos/unitree_h1_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_h1_generated.json

# # # # # # #
# openloong #
# # # # # # #

python3 scripts/generate_smplx_retarget.py \
  assets/openloong/AzureLoong.urdf \
  --human-scale 0.9 \
  --arm-scale 0.8 \
  --ground-height 0.0 \
  --no-head \
  -o general_motion_retargeting/ik_configs/smplx_to_openloong_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot openloong --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path openloong_warmup_9_generated.npz \
  --video_path videos/openloong_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_openloong_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot openloong --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path openloong_warmup_9_generated.npz \
  --video_path videos/openloong_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_openloong.json

# # # # # # #
# tienkung  #
# # # # # # #

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 --record_video  \
  --save_path tienkung_warmup_9_generated.npz \
  --video_path videos/tienkung_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_warmup_9_true.npz \
  --video_path videos/tienkung_warmup_9_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# mule_kick
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/mule_kick.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_mule_kick_generated.npz \
  --video_path videos/tienkung_mule_kick_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung_generated.json
  
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/mule_kick.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_mule_kick_true.npz \
  --video_path videos/tienkung_mule_kick_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# Box_1
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Box_1.npz \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_Box_1_generated.npz \
  --video_path videos/tienkung_Box_1_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Box_1.npz \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_Box_1_true.npz \
  --video_path videos/tienkung_Box_1_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# bicep curl
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/dumbbell_biceps_curls.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_bicep_curl_generated.npz \
  --video_path videos/tienkung_bicep_curl_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json
  
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/dumbbell_biceps_curls.json \
  --robot tienkung --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path tienkung_bicep_curl_true.npz \
  --video_path videos/tienkung_bicep_curl_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# open
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/motion-x++/data/video_4dh/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1_new.pkl \
  --robot tienkung --rate_limit --record_video --camera_distance 2.0 --camera_elevation -15 --camera_height 2.0 --hide_floor --rotate_roll -110 --rotate_pitch 45 \
  --save_path tienkung_open_generated.npz \
  --video_path videos/tienkung_open_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/motion-x++/data/video_4dh/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1_new.pkl \
  --robot tienkung --rate_limit --record_video --camera_distance 2.0 --camera_elevation -15 --camera_height 2.0 --hide_floor --rotate_roll -110 --rotate_pitch 45 \
  --save_path tienkung_open_true.npz \
  --video_path videos/tienkung_open_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_tienkung.json

# # # # # #
# unitree #
# # # # # #

# warmup_9
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_warmup_9_generated.npz \
  --video_path videos/unitree_g1_warmup_9_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/warmup_9.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_warmup_9_true.npz \
  --video_path videos/unitree_g1_warmup_9_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1.json

# mule_kick
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/mule_kick.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_mule_kick_generated.npz \
  --video_path videos/unitree_g1_mule_kick_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_generated.json
  
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/mule_kick.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_mule_kick_true.npz \
  --video_path videos/unitree_g1_mule_kick_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1.json

# Box_1
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Box_1.npz \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_Box_1_generated.npz \
  --video_path videos/unitree_g1_Box_1_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/humaneva/train/S1/smplx/Box_1.npz \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.0 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_Box_1_true.npz \
  --video_path videos/unitree_g1_Box_1_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1.json

# open
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/motion-x++/data/video_4dh/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1_new.pkl \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 2.0 --camera_elevation -15 --camera_height 2.0 --hide_floor --rotate_roll -140 --rotate_pitch 45 \
  --save_path unitree_g1_open_generated.npz \
  --video_path videos/unitree_g1_open_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_generated.json

python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/motion-x++/data/video_4dh/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1_new.pkl \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 2.0 --camera_elevation -15 --camera_height 2.0 --hide_floor --rotate_roll -140 --rotate_pitch 45 \
  --save_path unitree_g1_open_true.npz \
  --video_path videos/unitree_g1_open_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1.json

# bicep curl
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/dumbbell_biceps_curls.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_bicep_curl_generated.npz \
  --video_path videos/unitree_g1_bicep_curl_generated.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_generated.json
  
python scripts/smplx_to_robot.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s04/smplx/dumbbell_biceps_curls.json \
  --robot unitree_g1 --rate_limit --record_video --camera_distance 5.5 --camera_elevation -15 --rotate_yaw -90 \
  --save_path unitree_g1_bicep_curl_true.npz \
  --video_path videos/unitree_g1_bicep_curl_true.mp4 \
  --ik_config general_motion_retargeting/ik_configs/smplx_to_g1.json

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                         #
# IROS : Learning Intent‑Relevant Gaze and Time‑Aware Dynamics for Robust 3D Human Motion Forecasting     #
#                                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

python scripts/smplx_to_robot_v2.py idx=47 result_foldername="/home/haziq/Collab_AI/results/AAAI2026/2024_11_18/gimo/obj2pose/cvae=all_v2_transl=mse_orient=mse_pred_all_objects_dtype=train_mib/" obj_idx=1 rep_idx=0 --robot unitree_g1 --save_path objidx1_repidx0.npz --rate_limit
python scripts/smplx_to_robot_v2.py idx=47 result_foldername="/home/haziq/Collab_AI/results/AAAI2026/2024_11_18/gimo/obj2pose/cvae=all_v2_transl=mse_orient=mse_pred_all_objects_dtype=train_mib/" obj_idx=7 rep_idx=0 --robot unitree_g1 --save_path objidx7_repidx0.npz --rate_limit
python scripts/smplx_to_robot_v2.py idx=47 result_foldername="/home/haziq/Collab_AI/results/AAAI2026/2024_11_18/gimo/obj2pose/cvae=all_v2_transl=mse_orient=mse_pred_all_objects_dtype=train_mib/" obj_idx=4 rep_idx=0 --robot unitree_g1 --save_path objidx4_repidx0.npz --rate_limit
python scripts/export_robot_motion_to_blender.py --motion_file objidx1_repidx0.npz --robot unitree_g1 --output blender_export/objidx1_repidx0/robot_motion.fbx
python scripts/export_robot_motion_to_blender.py --motion_file objidx7_repidx0.npz --robot unitree_g1 --output blender_export/objidx7_repidx0/robot_motion.fbx
python scripts/export_robot_motion_to_blender.py --motion_file objidx4_repidx0.npz --robot unitree_g1 --output blender_export/objidx4_repidx0/robot_motion.fbx

python scripts/smplx_to_robot_v2.py idx=212 result_foldername="/home/haziq/Collab_AI/results/AAAI2026/2024_11_18/gimo/obj2pose/cvae=all_v2_transl=mse_orient=mse_pred_all_objects_dtype=val_seminar_room1_0221_2022-02-21-014246_255_mib/" obj_idx=6 rep_idx=0 --robot unitree_g1 --save_path objidx6_repidx0.npz --rate_limit
python scripts/export_robot_motion_to_blender.py --motion_file objidx6_repidx0.npz --robot unitree_g1 --output blender_export/objidx6_repidx0/robot_motion.fbx