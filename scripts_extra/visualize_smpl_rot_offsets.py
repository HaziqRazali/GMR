"""
Visualize the effect of rot_offset on SMPL-X IK joint orientations.

For each IK-relevant joint, draws coordinate axes (RGB = XYZ) in two colours:
  - BEFORE rot_offset: solid bright axes
  - AFTER  rot_offset: transparent/pastel axes

Two sets of joints are shown side-by-side (offset along X) so you can compare
without overlap.

usage (mhr_new env or gmr env with smplx installed):
    python /home/haziq/GMR/scripts_extra/visualize_smpl_rot_offsets.py \
        --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
        --frame 0

    python /home/haziq/GMR/scripts_extra/visualize_smpl_rot_offsets.py \
        --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
        --frame 100 --axis_len 0.12
"""

import argparse
import json
import sys
import os

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

# ── SMPL-X IK joints (from smplx_to_t1.json) ─────────────────────────────────
# joint_name → rot_offset wxyz
_IK_ROT_OFFSETS = {
    "pelvis":           [ 0.5, -0.5, -0.5, -0.5],
    "left_hip":         [ 0.5, -0.5, -0.5, -0.5],
    "right_hip":        [ 0.5, -0.5, -0.5, -0.5],
    "left_knee":        [ 0.5, -0.5, -0.5, -0.5],
    "right_knee":       [ 0.5, -0.5, -0.5, -0.5],
    "left_foot":        [-0.5,  0.5,  0.5,  0.5],
    "right_foot":       [-0.5,  0.5,  0.5,  0.5],
    "spine3":           [ 0.5, -0.5, -0.5, -0.5],
    "left_shoulder":    [ 0.5, -0.5, -0.5, -0.5],
    "right_shoulder":   [ 0.5, -0.5, -0.5, -0.5],
    "left_elbow":       [ 0.5, -0.5, -0.5, -0.5],
    "right_elbow":      [ 0.5, -0.5, -0.5, -0.5],
}

# SMPL-X joint index map (from smplx.joint_names.JOINT_NAMES order)
_SMPLX_JOINT_IDX = {
    "pelvis":        0,
    "left_hip":      1,
    "right_hip":     2,
    "spine3":        9,
    "left_foot":    10,
    "right_foot":   11,
    "left_knee":     4,
    "right_knee":    5,
    "left_shoulder":16,
    "right_shoulder":17,
    "left_elbow":   18,
    "right_elbow":  19,
}

# SMPL-X parent indices (22 body joints)
_SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Skeleton edges for the 12 IK joints (approximate, for context)
_SKEL_EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"), ("pelvis", "spine3"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_foot"), ("right_knee", "right_foot"),
    ("spine3", "left_shoulder"), ("spine3", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
]


# ── geometry helpers ──────────────────────────────────────────────────────────

def make_axis_arrows(pos, rot_mat, length=0.08, radius=0.004, offset=np.zeros(3)):
    """
    Draw 3 cylinders+cones (XYZ = RGB) for a coordinate frame.
    pos    : (3,) world position
    rot_mat: (3,3) rotation matrix, columns = X,Y,Z axes in world space
    offset : (3,) optional shift so before/after don't overlap
    """
    geoms = []
    colours = [[1, 0, 0], [0, 0.8, 0], [0, 0, 1]]   # R G B = X Y Z
    for i, col in enumerate(colours):
        axis_world = rot_mat[:, i]
        tip = pos + offset + axis_world * length
        base = pos + offset

        # cylinder body
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length * 0.8
        )
        cyl.paint_uniform_color(col)
        # default cylinder is along Z; rotate to axis_world
        z = np.array([0, 0, 1.0])
        v = np.cross(z, axis_world)
        s = np.linalg.norm(v)
        c = np.dot(z, axis_world)
        if s < 1e-6:
            rot_cyl = np.eye(3) if c > 0 else R.from_euler("x", np.pi).as_matrix()
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot_cyl = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        cyl.rotate(rot_cyl, center=[0, 0, 0])
        midpoint = base + axis_world * length * 0.4
        cyl.translate(midpoint)
        geoms.append(cyl)

        # cone tip
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=radius * 2.5, height=length * 0.2
        )
        cone.paint_uniform_color(col)
        cone.rotate(rot_cyl, center=[0, 0, 0])
        cone.translate(tip)
        geoms.append(cone)

    return geoms


def make_sphere(pos, radius=0.012, color=(1, 1, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(pos)
    s.paint_uniform_color(list(color))
    return s


def make_skeleton_lines(positions_dict, edges, offset=np.zeros(3), color=(0.6, 0.6, 0.6)):
    pts, lines = [], []
    idx_map = {}
    for i, (name, pos) in enumerate(positions_dict.items()):
        pts.append(pos + offset)
        idx_map[name] = i
    for a, b in edges:
        if a in idx_map and b in idx_map:
            lines.append([idx_map[a], idx_map[b]])
    if not lines:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


def make_label_sphere(pos, color, r=0.008):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s.translate(pos)
    s.paint_uniform_color(color)
    return s


# ── SMPL-X loading ────────────────────────────────────────────────────────────

def _rotmat_to_rotvec(arr):
    shape = arr.shape[:-2]
    return R.from_matrix(arr.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


def load_fit3d_frame(json_file, smplx_model_path, frame=0, fps=50):
    """
    Load one frame from a Fit3D SMPL-X JSON and run the body model forward pass.
    Returns:
        joints_world : (22, 3)  world positions
        world_rots   : dict joint_name → (3,3) world rotation matrix
    """
    import smplx as smplx_lib
    from smplx.joint_names import JOINT_NAMES

    with open(json_file) as f:
        data = json.load(f)

    # detect format
    is_fit3d = "annotations" not in data and "transl" in data
    if not is_fit3d:
        raise ValueError("Only Fit3D flat-dict JSON is supported currently.")

    N = np.array(data["transl"]).shape[0]
    f_idx = min(frame, N - 1)
    print(f"[load] {N} frames, using frame {f_idx}")

    transl        = np.array(data["transl"])[f_idx:f_idx+1]          # (1,3)
    global_orient = np.array(data["global_orient"])[f_idx:f_idx+1]   # (1,1,3,3)
    body_pose_mat = np.array(data["body_pose"])[f_idx:f_idx+1]       # (1,21,3,3)
    betas_arr     = np.array(data["betas"])[f_idx:f_idx+1]           # (1,10)
    lhand_mat     = np.array(data["left_hand_pose"])[f_idx:f_idx+1]
    rhand_mat     = np.array(data["right_hand_pose"])[f_idx:f_idx+1]
    jaw_mat       = np.array(data["jaw_pose"])[f_idx:f_idx+1]
    leye_mat      = np.array(data["leye_pose"])[f_idx:f_idx+1]
    reye_mat      = np.array(data["reye_pose"])[f_idx:f_idx+1]
    expr          = np.array(data["expression"])[f_idx:f_idx+1]

    root_orient = _rotmat_to_rotvec(global_orient[:, 0]).astype(np.float32)  # (1,3)
    pose_body   = _rotmat_to_rotvec(body_pose_mat).reshape(1, -1).astype(np.float32)
    lhand_pose  = _rotmat_to_rotvec(lhand_mat).reshape(1, -1).astype(np.float32)
    rhand_pose  = _rotmat_to_rotvec(rhand_mat).reshape(1, -1).astype(np.float32)
    jaw_pose    = _rotmat_to_rotvec(jaw_mat[:, 0]).astype(np.float32)
    leye_pose   = _rotmat_to_rotvec(leye_mat[:, 0]).astype(np.float32)
    reye_pose   = _rotmat_to_rotvec(reye_mat[:, 0]).astype(np.float32)

    betas = np.mean(betas_arr, axis=0)
    betas_padded = np.pad(betas, (0, 6), mode="constant").astype(np.float32)

    body_model = smplx_lib.create(
        smplx_model_path, "smplx", gender="neutral",
        use_pca=False, num_betas=len(betas_padded),
    )

    with torch.no_grad():
        out = body_model(
            betas=             torch.tensor(betas_padded).float().view(1, -1),
            global_orient=     torch.tensor(root_orient).float(),
            body_pose=         torch.tensor(pose_body).float(),
            transl=            torch.tensor(transl).float(),
            left_hand_pose=    torch.tensor(lhand_pose).float(),
            right_hand_pose=   torch.tensor(rhand_pose).float(),
            jaw_pose=          torch.tensor(jaw_pose).float(),
            leye_pose=         torch.tensor(leye_pose).float(),
            reye_pose=         torch.tensor(reye_pose).float(),
            expression=        torch.tensor(expr).float(),
            return_full_pose=  True,
        )

    joints = out.joints[0].detach().numpy()          # (J, 3)
    full_pose = out.full_pose[0].reshape(-1, 3).detach().numpy()  # (J, 3) axis-angle
    global_orient_aa = out.global_orient[0].detach().numpy().reshape(3)

    joint_names = JOINT_NAMES[:len(body_model.parents)]
    parents     = body_model.parents

    # Accumulate world-frame rotations (same as get_smplx_data_offline_fast)
    world_rots_list = []
    for i, jname in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient_aa)
        else:
            rot = world_rots_list[parents[i]] * R.from_rotvec(full_pose[i])
        world_rots_list.append(rot)

    world_rots = {}
    for jname, idx in _SMPLX_JOINT_IDX.items():
        world_rots[jname] = world_rots_list[idx].as_matrix()  # (3,3)

    joint_positions = {jname: joints[idx] for jname, idx in _SMPLX_JOINT_IDX.items()}

    return joint_positions, world_rots


def load_rest_pose(smplx_model_path):
    """
    Run SMPL-X at the rest pose (all-zero parameters → T-pose).
    Returns the same (joint_positions, world_rots) dict as load_fit3d_frame.
    """
    import smplx as smplx_lib
    from smplx.joint_names import JOINT_NAMES

    print("[load] No file supplied — using SMPL-X rest (T-pose)")

    body_model = smplx_lib.create(
        smplx_model_path, "smplx", gender="neutral",
        use_pca=False, num_betas=10,
    )

    with torch.no_grad():
        out = body_model(
            betas=           torch.zeros(1, 10),
            global_orient=   torch.zeros(1, 3),
            body_pose=       torch.zeros(1, 63),
            transl=          torch.zeros(1, 3),
            left_hand_pose=  torch.zeros(1, 45),
            right_hand_pose= torch.zeros(1, 45),
            jaw_pose=        torch.zeros(1, 3),
            leye_pose=       torch.zeros(1, 3),
            reye_pose=       torch.zeros(1, 3),
            expression=      torch.zeros(1, 10),
            return_full_pose=True,
        )

    joints    = out.joints[0].detach().numpy()                         # (J, 3)
    full_pose = out.full_pose[0].reshape(-1, 3).detach().numpy()      # (J, 3)

    joint_names = JOINT_NAMES[:len(body_model.parents)]
    parents     = body_model.parents

    world_rots_list = []
    for i, jname in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(full_pose[0])
        else:
            rot = world_rots_list[parents[i]] * R.from_rotvec(full_pose[i])
        world_rots_list.append(rot)

    world_rots      = {jname: world_rots_list[idx].as_matrix() for jname, idx in _SMPLX_JOINT_IDX.items()}
    joint_positions = {jname: joints[idx] for jname, idx in _SMPLX_JOINT_IDX.items()}

    return joint_positions, world_rots


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── find SMPL-X model path ────────────────────────────────────────────────
    smplx_candidates = [
        "/home/haziq/datasets/mocap/data/models_smplx_v1_1/models",
        "/home/haziq/datasets/motion-x++/data/models_smplx_v1_1/models",
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models"),
        "/home/haziq/GMR/assets/body_models",
    ]
    smplx_path = args.smplx_path
    if smplx_path is None:
        for p in smplx_candidates:
            if os.path.isdir(p):
                smplx_path = p
                break
    if smplx_path is None:
        print("[ERROR] SMPL-X model not found. Provide --smplx_path.")
        sys.exit(1)
    print(f"[smplx] path: {smplx_path}")

    # ── load data ─────────────────────────────────────────────────────────────
    if args.smplx_file is None:
        joint_positions, world_rots = load_rest_pose(smplx_path)
    else:
        joint_positions, world_rots = load_fit3d_frame(
            args.smplx_file, smplx_path, frame=args.frame
        )

    # Centre everything at pelvis so scene is near origin
    pelvis_pos = joint_positions["pelvis"].copy()
    joint_positions_c = {k: v - pelvis_pos for k, v in joint_positions.items()}

    ax = args.axis_len
    BEFORE_OFFSET = np.array([0.0,              0.0, 0.0])
    MID_OFFSET    = np.array([args.side_gap,    0.0, 0.0])
    AFTER_OFFSET  = np.array([args.side_gap*2,  0.0, 0.0])

    geoms = []

    # ── world frame axes for reference ───────────────────────────────────────
    # Place below the feet (≈ -1.1 m from pelvis) so they don't overlap joints.
    feet_y = min(v[1] for v in joint_positions_c.values()) - 0.15
    for off in [BEFORE_OFFSET, MID_OFFSET, AFTER_OFFSET]:
        origin = list(off + np.array([0.0, feet_y, 0.0]))
        geoms.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12, origin=origin)
        )

    # ── skeleton lines ────────────────────────────────────────────────────────
    for off, col in [
        (BEFORE_OFFSET, [0.5, 0.5, 0.5]),
        (MID_OFFSET,    [0.5, 0.5, 0.5]),
        (AFTER_OFFSET,  [0.5, 0.5, 0.5]),
    ]:
        ls = make_skeleton_lines(joint_positions_c, _SKEL_EDGES, offset=off, color=col)
        if ls:
            geoms.append(ls)

    print("\nJoint-by-joint orientation change:")
    print(f"  {'Joint':<18}  Δ angle")

    for jname, pos_c in joint_positions_c.items():
        rot_mat_before = world_rots[jname]   # (3,3)

        # Apply rot_offset: R_after = R_before * R_offset  (right-multiply, in joint local frame)
        q_off = _IK_ROT_OFFSETS[jname]      # wxyz
        R_off = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])  # xyzw for scipy
        rot_mat_after = (R.from_matrix(rot_mat_before) * R_off).as_matrix()

        # angular difference
        delta = R.from_matrix(rot_mat_before.T @ rot_mat_after)
        angle_deg = np.degrees(np.linalg.norm(delta.as_rotvec()))
        print(f"  {jname:<18}  Δ = {angle_deg:.1f}°")

        # ── LEFT: BEFORE only ────────────────────────────────────────────────
        geoms += make_axis_arrows(pos_c, rot_mat_before, length=ax, offset=BEFORE_OFFSET)
        geoms.append(make_label_sphere(pos_c + BEFORE_OFFSET, color=(1.0, 0.9, 0.0)))   # yellow

        # ── MIDDLE: BEFORE (shorter) + AFTER (longer) overlaid ───────────────
        # Before: shorter axes so both are visible when they diverge
        geoms += make_axis_arrows(pos_c, rot_mat_before, length=ax * 0.6, offset=MID_OFFSET)
        geoms.append(make_label_sphere(pos_c + MID_OFFSET, color=(0.8, 0.8, 0.8), r=0.006))
        # After: full-length axes
        geoms += make_axis_arrows(pos_c, rot_mat_after,  length=ax,       offset=MID_OFFSET)

        # ── RIGHT: AFTER only ────────────────────────────────────────────────
        geoms += make_axis_arrows(pos_c, rot_mat_after,  length=ax, offset=AFTER_OFFSET)
        geoms.append(make_label_sphere(pos_c + AFTER_OFFSET, color=(0.0, 1.0, 0.8)))    # cyan

    print()
    print("Legend:")
    print("  LEFT   (yellow spheres) = BEFORE rot_offset  (raw SMPL-X world orientations)")
    print("  MIDDLE                  = BEFORE (short axes) + AFTER (long axes) overlaid")
    print("  RIGHT  (cyan   spheres) = AFTER  rot_offset  (what GMR feeds into IK as targets)")
    print("  Axis colours: RED = X,  GREEN = Y,  BLUE = Z")
    print()
    print("Press Q in the Open3D window to exit.")

    source_label = "T-pose (rest)" if args.smplx_file is None else f"frame {args.frame}"
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"rot_offset effect — {source_label}  |  LEFT=before  MIDDLE=overlaid  RIGHT=after",
        mesh_show_back_face=True,
        width=1600, height=900,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize rot_offset effect on SMPL-X IK joint orientations."
    )
    parser.add_argument(
        "--smplx_file",
        default=None,
        help="Path to Fit3D SMPL-X JSON file. If omitted, uses the T-pose rest pose.",
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to visualize (default: 0).",
    )
    parser.add_argument(
        "--axis_len", type=float, default=0.08,
        help="Length of each coordinate axis arrow (metres, default: 0.08).",
    )
    parser.add_argument(
        "--side_gap", type=float, default=1.2,
        help="X offset between BEFORE and AFTER skeletons (default: 1.2m).",
    )
    parser.add_argument(
        "--smplx_path", default=None,
        help="Path to SMPL-X model directory (auto-detected if not given).",
    )
    main(parser.parse_args())
