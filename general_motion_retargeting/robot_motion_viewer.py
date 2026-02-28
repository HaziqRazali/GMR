import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                hide_floor=False,
                # video recording
                record_video=False,
                video_path=None,
                video_width=1920,
                video_height=1080,
                keyboard_callback=None,
                # camera override
                camera_distance=None,
                camera_elevation=-10,
                camera_height=0.0,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = camera_distance if camera_distance is not None else VIEWER_CAM_DISTANCE_DICT[robot_type]
        self.viewer_cam_elevation = camera_elevation
        self.viewer_cam_height = camera_height
        mj.mj_step(self.model, self.data)
        
        if hide_floor:
            floor_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
            if floor_id >= 0:
                self.model.geom_rgba[floor_id, 3] = 0.0  # set alpha to 0
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
            # Dedicated camera for the renderer – independent of the passive viewer window
            self.render_cam = mj.MjvCamera()
            self.render_cam.type = mj.mjtCamera.mjCAMERA_FREE
            self.render_cam.distance = self.viewer_cam_distance
            self.render_cam.elevation = self.viewer_cam_elevation
            self.render_cam.azimuth = self.viewer.cam.azimuth  # inherit initial azimuth
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)

        # Compute effective distance/elevation so the camera is physically
        # camera_height metres higher in world-Z while still aimed at the robot.
        # MuJoCo convention: cam_z_above_lookat = distance * sin(-elevation_rad)
        # We raise that by camera_height, then back-solve distance and elevation.
        el_rad = np.radians(self.viewer_cam_elevation)
        h_dist  = self.viewer_cam_distance * np.cos(el_rad)       # horizontal dist (unchanged)
        v_above = self.viewer_cam_distance * np.sin(-el_rad) + self.viewer_cam_height  # new Z offset
        eff_dist = float(np.sqrt(h_dist ** 2 + v_above ** 2))
        eff_elev = float(-np.degrees(np.arctan2(v_above, h_dist)))  # negative = above

        # Always apply camera distance/elevation; only follow lookat when requested
        self.viewer.cam.distance = eff_dist
        self.viewer.cam.elevation = eff_elev
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id].copy()
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人

        # Keep the dedicated render camera in sync (immune to viewer.sync() overrides)
        if self.record_video:
            self.render_cam.distance = eff_dist
            self.render_cam.elevation = eff_elev
            self.render_cam.azimuth = self.viewer.cam.azimuth
            if follow_camera:
                self.render_cam.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id].copy()
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use the dedicated render camera so viewer.sync() cannot override our settings
            self.renderer.update_scene(self.data, camera=self.render_cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
