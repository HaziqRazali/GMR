"""
Compute R_{A→T} per IK joint — the per-joint axis-convention correction that
makes normalized MHR rotation matrices speak the same language as SMPL-X.

Method  (T-pose quaternion — exact)
------------------------------------
After running MHR through a T-pose shape optimisation we have per-joint world
quaternions skel_tpose[j].  At T-pose the normalised rotation is:

  R_norm_tpose[j] = R_rest[j]⁻¹ · R_world_tpose[j]

We WANT R_AtoT · R_norm = I at T-pose, so:

  R_AtoT[j] = R_norm_tpose[j]⁻¹

This is exact by definition: no geometric assumptions, no twist ambiguity.
The bone positions from the same optimisation are used only for the skeleton
visualisation (COL 3/4 skeleton shapes).

Full correction chain used in IK:
  R_target = rot_offset · R_AtoT · R_rest⁻¹ · R_world

Output
------
Prints a copy-pasteable _R_ATO_T_WXYZ dict for use in mhr_to_robot.py /
visualize_mhr_rot_offsets.py.  Also opens an Open3D 4-column viewer:
  COL 1 — SMPL-X T-pose identity arrows  (target)
  COL 2 — SMPL-X canonical bone-frames   (diagnostic)
  COL 3 — MHR   canonical bone-frames    (diagnostic)
  COL 4 — MHR ~T-pose + R_AtoT·R_norm   (should equal COL 1)

Usage (mhr_new env):
python /home/haziq/GMR/scripts_extra/visualize_mhr_rot_offsets.py \
    --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
    --frame 100 --axis_len 0.12

"""

import os, sys, argparse
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/MHR"))

# ── Joint map:  SMPL-X body-joint idx → (readable name, MHR joint idx, IK name) ──
_JOINT_MAP = {
     0: ("root",            1, "root"),
     1: ("left_hip",        2, "l_upleg"),
     2: ("right_hip",      18, "r_upleg"),
     4: ("left_knee",       3, "l_lowleg"),
     5: ("right_knee",     19, "r_lowleg"),
     7: ("left_ankle",      8, "l_ball"),
     8: ("right_ankle",    24, "r_ball"),
     9: ("c_spine3",       37, "c_spine3"),
    16: ("left_shoulder",  75, "l_uparm"),
    17: ("right_shoulder", 39, "r_uparm"),
    18: ("left_elbow",     76, "l_lowarm"),
    19: ("right_elbow",    40, "r_lowarm"),
}

_IK_NAMES = [v[2] for v in _JOINT_MAP.values()]

# rot_offset (wxyz) from mhr_to_robot IK config — converts Z-up MHR axes to
# the Y-up world convention that GMR / mink IK expects.
_IK_ROT_OFFSETS = {
    "root":     [ 0.5, -0.5, -0.5, -0.5],
    "l_upleg":  [ 0.5, -0.5, -0.5, -0.5],
    "r_upleg":  [ 0.5, -0.5, -0.5, -0.5],
    "l_lowleg": [ 0.5, -0.5, -0.5, -0.5],
    "r_lowleg": [ 0.5, -0.5, -0.5, -0.5],
    "l_ball":   [-0.5,  0.5,  0.5,  0.5],
    "r_ball":   [-0.5,  0.5,  0.5,  0.5],
    "c_spine3": [ 0.5, -0.5, -0.5, -0.5],
    "l_uparm":  [ 0.5, -0.5, -0.5, -0.5],
    "r_uparm":  [ 0.5, -0.5, -0.5, -0.5],
    "l_lowarm": [ 0.5, -0.5, -0.5, -0.5],
    "r_lowarm": [ 0.5, -0.5, -0.5, -0.5],
}

_SMPLX_EDGES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),
]

_MHR_SKEL_EDGES_NAMES = [
    ("root","l_upleg"),("root","r_upleg"),("root","c_spine3"),
    ("l_upleg","l_lowleg"),("r_upleg","r_lowleg"),
    ("l_lowleg","l_ball"),("r_lowleg","r_ball"),
    ("c_spine3","l_uparm"),("c_spine3","r_uparm"),
    ("l_uparm","l_lowarm"),("r_uparm","r_lowarm"),
]

# Incoming-bone pairs for canonical-frame R_AtoT.
# Format: (smplx_parent_idx, smplx_current_idx, mhr_parent_name, mhr_current_name)
# Parent = nearest IK-listed ancestor (NOT anatomical parent).
#   e.g. r_uparm IK-parent = c_spine3, skipping the anatomical r_clavicle.
# None -> R_AtoT = identity (root and c_spine3 are equvalently upright in both skeletons).
_BONE_PAIRS = {
    "root":     None,
    "l_upleg":  ( 0,  1, "root",     "l_upleg"),   # pelvis    -> l_hip
    "r_upleg":  ( 0,  2, "root",     "r_upleg"),   # pelvis    -> r_hip
    "l_lowleg": ( 1,  4, "l_upleg",  "l_lowleg"),  # l_hip     -> l_knee
    "r_lowleg": ( 2,  5, "r_upleg",  "r_lowleg"),  # r_hip     -> r_knee
    "l_ball":   ( 4,  7, "l_lowleg", "l_ball"),    # l_knee    -> l_ankle  (IK parent)
    "r_ball":   ( 5,  8, "r_lowleg", "r_ball"),    # r_knee    -> r_ankle
    "c_spine3": None,
    "l_uparm":  ( 9, 16, "c_spine3", "l_uparm"),   # spine3    -> l_shoulder  (IK parent)
    "r_uparm":  ( 9, 17, "c_spine3", "r_uparm"),   # spine3    -> r_shoulder
    "l_lowarm": (16, 18, "l_uparm",  "l_lowarm"),  # l_shoulder -> l_elbow
    "r_lowarm": (17, 19, "r_uparm",  "r_lowarm"),  # r_shoulder -> r_elbow
}


# ── geometry helpers ──────────────────────────────────────────────────────────

def make_axis_arrows(pos, rot_mat, length=0.07, radius=0.004, offset=np.zeros(3)):
    geoms = []
    colours = [[1,0,0],[0,0.8,0],[0,0,1]]
    for i, col in enumerate(colours):
        axis_world = rot_mat[:, i]
        base = pos + offset
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length*0.8)
        cyl.paint_uniform_color(col)
        z = np.array([0,0,1.0])
        v = np.cross(z, axis_world); s = np.linalg.norm(v); c = np.dot(z, axis_world)
        if s < 1e-6:
            rc = np.eye(3) if c > 0 else R.from_euler("x", np.pi).as_matrix()
        else:
            vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            rc = np.eye(3) + vx + vx@vx*((1-c)/(s*s))
        cyl.rotate(rc, center=[0,0,0])
        cyl.translate(base + axis_world*length*0.4)
        geoms.append(cyl)
        cone = o3d.geometry.TriangleMesh.create_cone(radius=radius*2.5, height=length*0.2)
        cone.paint_uniform_color(col)
        cone.rotate(rc, center=[0,0,0])
        cone.translate(base + axis_world*length)
        geoms.append(cone)
    return geoms


def build_lineset(pts, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


def sphere_at(pos, radius=0.012, color=(1,1,0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(pos)
    s.paint_uniform_color(list(color))
    return s


# ── rotation helpers ──────────────────────────────────────────────────────────

def canonical_frame(bone_dir: np.ndarray) -> np.ndarray:
    """
    Build a deterministic right-handed orthonormal 3x3 frame from an incoming
    bone direction vector.  Applying the SAME function to both skeletons yields
    canonical frames whose difference is exactly R_AtoT = F_sx @ F_mhr.T.

    Column convention (matches Open3D arrow colours):
      col0 (red,  X) = normalize(bone_dir)          — along the bone
      col2 (blue, Z) = Z_world projected perp to bone  — toward viewer
      col1 (grn,  Y) = cross(col2, col0)             — right-hand completion

    Fallback: if the bone is nearly parallel to Z_world (|cos| > 0.9), switch
    the reference to Y_world so col2 stays well-conditioned.
    """
    d   = bone_dir / (np.linalg.norm(bone_dir) + 1e-9)
    ref = np.array([0., 0., 1.])                    # Z_world — toward viewer
    if abs(float(np.dot(d, ref))) > 0.9:
        ref = np.array([0., 1., 0.])                # fallback: Y_world
    # Gram-Schmidt: project ref onto plane perp to d
    col2 = ref - float(np.dot(ref, d)) * d
    col2 /= np.linalg.norm(col2) + 1e-9             # blue  (toward viewer)
    col1  = np.cross(col2, d)
    col1 /= np.linalg.norm(col1) + 1e-9             # green
    return np.column_stack([d, col1, col2])          # columns: [red, green, blue]


# ── SMPL-X helpers ────────────────────────────────────────────────────────────

def find_smplx_path():
    candidates = [
        "/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/smplx",
        "/home/haziq/datasets/motion-x++/data/models_smplx_v1_1/models/smplx",
        "/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx",
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def load_smplx_tpose(smplx_path, device):
    import smplx
    model = smplx.SMPLX(
        model_path=smplx_path, gender="neutral",
        use_pca=False, num_betas=10, num_expression_coeffs=10,
    ).to(device)
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1,10,device=device),
            global_orient=torch.zeros(1,3,device=device),
            body_pose=torch.zeros(1,63,device=device),
            left_hand_pose=torch.zeros(1,45,device=device),
            right_hand_pose=torch.zeros(1,45,device=device),
            jaw_pose=torch.zeros(1,3,device=device),
            leye_pose=torch.zeros(1,3,device=device),
            reye_pose=torch.zeros(1,3,device=device),
            expression=torch.zeros(1,10,device=device),
        )
    return out.joints[0].cpu().numpy()   # (J, 3) metres, world rot = I


# ── MHR helpers ───────────────────────────────────────────────────────────────

def load_mhr_model(device):
    from mhr.mhr import MHR
    print("[MHR] Loading model ...")
    return MHR.from_files(device=torch.device(device), lod=1)


def get_mhr_apose(mhr_model, device):
    """MHR A-pose (model_params = 0).  Returns positions in metres and R_rest.
    pos_apose contains both IK joints and extra joints needed for bone-direction pairs.
    """
    with torch.no_grad():
        shape_p = torch.zeros(1, 45, device=device)
        expr_p  = torch.zeros(1, 72, device=device)
        model_p = torch.zeros(1,204, device=device)
        _, skel = mhr_model(shape_p, model_p, expr_p)
    skel_np = skel[0].cpu().numpy()   # (127, 8)

    pos_apose = {}
    R_rest    = {}
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pos_apose[ik_name] = skel_np[mhr_idx, :3] / 100.0
        R_rest[ik_name]    = R.from_quat(skel_np[mhr_idx, 3:7])   # xyzw

    return pos_apose, R_rest, skel_np


def optimise_mhr_tpose(mhr_model, smplx_joints, device, iters=1500, lr=5e-3, reg=1e-3):
    """
    Position-only optimisation: move MHR into T-pose shape to match SMPL-X.
    No rotation loss — only used for the visualisation skeleton shape.
    Returns skel_tpose numpy (127, 8) and a pos_tpose dict (metres).
    """
    # Scale SMPL-X positions to MHR cm space
    with torch.no_grad():
        shape_p0 = torch.zeros(1, 45, device=device)
        expr_p0  = torch.zeros(1, 72, device=device)
        skel0    = mhr_model(shape_p0, torch.zeros(1,204,device=device), expr_p0)[1]
    skel0_np = skel0[0].cpu().numpy()
    mhr_root_cm = skel0_np[1, :3]
    mhr_head_cm = skel0_np[113, :3]
    mhr_h_cm    = np.linalg.norm(mhr_head_cm - mhr_root_cm)
    sx_root     = smplx_joints[0]
    sx_h        = np.linalg.norm(smplx_joints[15] - sx_root)
    scale       = mhr_h_cm / sx_h

    targets_cm = {}
    mhr_indices = []
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        sx_pt = (smplx_joints[sx_idx] - sx_root) * scale + mhr_root_cm
        targets_cm[ik_name] = sx_pt
        mhr_indices.append(mhr_idx)
    tgt_tensor = torch.tensor(
        np.stack([targets_cm[v[2]] for v in _JOINT_MAP.values()]),
        dtype=torch.float32, device=device,
    )

    model_params = torch.zeros(1, 204, device=device, requires_grad=True)
    optimiser    = torch.optim.Adam([model_params], lr=lr)
    shape_p = torch.zeros(1, 45, device=device)
    expr_p  = torch.zeros(1, 72, device=device)

    print(f"\n[T-pose opt] {iters} iters (position only, for visualisation) ...")
    import torch.nn.functional as F
    for it in range(iters):
        optimiser.zero_grad()
        _, skel = mhr_model(shape_p, model_params, expr_p)
        pred_pos = skel[0, mhr_indices, :3]
        loss = F.mse_loss(pred_pos, tgt_tensor) + reg * (model_params**2).mean()
        loss.backward()
        optimiser.step()
        if it % 300 == 0:
            print(f"  iter {it:4d}  pos={loss.item():.4f} cm²")

    with torch.no_grad():
        _, skel_final = mhr_model(shape_p, model_params.detach(), expr_p)
    skel_np = skel_final[0].cpu().numpy()

    pos_tpose = {}
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pos_tpose[ik_name] = skel_np[mhr_idx, :3] / 100.0
    print(f"  iter {iters:4d}  DONE")
    return skel_np, pos_tpose


# ── Core computation ──────────────────────────────────────────────────────────

def compute_f_frames(smplx_joints, pos_apose):
    """
    Compute canonical bone-frame pairs (F_sx, F_mhr) for diagnostic visualisation
    only (COL 2 and COL 3 in the viewer).  NOT used to compute R_AtoT.
    """
    F_frames = {}
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        bp = _BONE_PAIRS[ik_name]
        if bp is None:
            F_frames[ik_name] = (np.eye(3), np.eye(3))
            continue
        sx_par, sx_cur, mhr_par, mhr_cur = bp
        d_sx  = smplx_joints[sx_cur] - smplx_joints[sx_par]
        d_mhr = pos_apose[mhr_cur]   - pos_apose[mhr_par]
        F_frames[ik_name] = (canonical_frame(d_sx), canonical_frame(d_mhr))
    return F_frames


def compute_r_ato_t(skel_tpose, R_rest):
    """
    T-pose quaternion method — exact.

    At the T-pose optimised skeleton:
      R_norm_tpose[j] = R_rest[j]⁻¹ · R_world_tpose[j]

    Define:
      R_AtoT[j] = R_norm_tpose[j]⁻¹

    Then by construction: R_AtoT · R_norm_tpose = I  →  at T-pose every joint
    arrow equals the world axes (= COL 1 in the viewer).

    skel_tpose : (127, 8) numpy array from optimise_mhr_tpose()
    R_rest     : dict  ik_name → scipy Rotation  (A-pose rest frames)
    """
    R_AtoT = {}

    print("\n── R_{A->T}  (T-pose quaternion method) ────────────────────────────────")
    print(f"  {'joint':<18}  {'angle':>6}    wxyz quaternion")
    print("  " + "-"*60)

    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        q_xyzw       = skel_tpose[mhr_idx, 3:7]
        R_world_tp   = R.from_quat(q_xyzw)
        R_norm_tp    = R_rest[ik_name].inv() * R_world_tp
        R_AtoT[ik_name] = R_norm_tp.inv()

        angle_deg = np.degrees(np.linalg.norm(R_AtoT[ik_name].as_rotvec()))
        q = R_AtoT[ik_name].as_quat(scalar_first=True)
        print(f"  {ik_name:<18}  {angle_deg:5.1f} deg   "
              f"[{q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}]")

    print("\n# ─── Copy-paste into mhr_to_robot.py / visualize_mhr_rot_offsets.py ───")
    print("_R_ATO_T_WXYZ = {")
    for _, (_, _, ik_name) in _JOINT_MAP.items():
        q = R_AtoT[ik_name].as_quat(scalar_first=True)
        print(f'    "{ik_name}": [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}],')
    print("}")

    return R_AtoT


# ── Open3D visualisation ──────────────────────────────────────────────────────

def visualise(smplx_joints, pos_apose, R_rest, R_AtoT,
              axis_len=0.08, skel_tpose=None, pos_tpose=None):
    """
    Five-column layout (left → right):

    COL 1 (blue,   x=-2*GAP): SMPL-X T-pose — identity arrows (world X/Y/Z).
        THE TARGET.

    COL 2 (red,    x=-GAP)  : MHR A-pose — raw world rotations.
        Each joint carries MHR's rigging-defined A-pose prerotation (chaotic).

    COL 3 (yellow, x=0)     : MHR A-pose — after normalization R_rest⁻¹·R_world.
        At A-pose R_world = R_rest, so R_norm = I → all arrows = world axes.

    COL 4 (green,  x=+GAP)  : MHR ~T-pose — R_norm_tpose = R_rest⁻¹·R_world_tpose.
        Shows how the T-pose differs from A-pose after normalization.
        These are the rotations R_AtoT is designed to cancel.

    COL 5 (purple, x=+2*GAP): MHR ~T-pose — R_AtoT·R_norm_tpose.
        Full fix: by construction = I → arrows = world axes = COL 1.  ✓
    """
    GAP    = 1.5
    DISP_H = 1.0

    # ── SMPL-X point cloud (centred, normalised height) ───────────────────────
    sx_root = smplx_joints[0].copy()
    sx_pts  = (smplx_joints - sx_root).copy()
    sx_h    = np.linalg.norm(sx_pts[15] - sx_pts[0]) + 1e-9
    sx_pts  = sx_pts * (DISP_H / sx_h)

    sx_ik_indices = list(_JOINT_MAP.keys())
    sx_local      = {sx_idx: i for i, sx_idx in enumerate(sx_ik_indices)}
    _sx_ik_edges  = [
        (0, 1),(0, 2),(1, 4),(2, 5),(4, 7),(5, 8),
        (0, 9),(9,16),(9,17),(16,18),(17,19),
    ]
    sx_edge_list = [(sx_local[a], sx_local[b]) for a, b in _sx_ik_edges]

    # ── MHR A-pose point cloud (centred, normalised height) ───────────────────
    ik_names_ordered = [ik_name for _, (_, _, ik_name) in _JOINT_MAP.items()]
    mhr_pos_arr = np.array([pos_apose[n] for n in ik_names_ordered])
    mhr_root_p  = pos_apose["root"]
    mhr_pts_c   = mhr_pos_arr - mhr_root_p
    mhr_max_y   = mhr_pts_c[:, 1].max()
    mhr_h       = (mhr_max_y - 0.0) + 1e-9
    mhr_pts_c   = mhr_pts_c * (DISP_H / mhr_h)
    name_to_mhr = {n: mhr_pts_c[i] for i, n in enumerate(ik_names_ordered)}

    mhr_ik_pts   = list(name_to_mhr.values())
    mhr_name_idx = {n: i for i, n in enumerate(name_to_mhr.keys())}
    mhr_edge_list = [(mhr_name_idx[a], mhr_name_idx[b])
                     for a, b in _MHR_SKEL_EDGES_NAMES
                     if a in mhr_name_idx and b in mhr_name_idx]

    # ── MHR T-pose point cloud + per-joint R_norm_tpose ───────────────────────
    if skel_tpose is not None and pos_tpose is not None:
        tp_names     = ik_names_ordered
        tp_pos_arr   = np.array([pos_tpose[n] for n in tp_names])
        tp_root      = pos_tpose["root"]
        tp_pts_c     = tp_pos_arr - tp_root
        tp_max_y     = tp_pts_c[:, 1].max()
        tp_h         = (tp_max_y - 0.0) + 1e-9
        tp_pts_c     = tp_pts_c * (DISP_H / tp_h)
        tp_name_pt   = {n: tp_pts_c[i] for i, n in enumerate(tp_names)}
        tp_ik_pts    = [tp_name_pt[n] for n in tp_names]
        tp_name_idx  = {n: i for i, n in enumerate(tp_names)}
        tp_edge_list = [(tp_name_idx[a], tp_name_idx[b])
                        for a, b in _MHR_SKEL_EDGES_NAMES
                        if a in tp_name_idx and b in tp_name_idx]
        # R_norm_tpose[j] = R_rest[j]⁻¹ · R_world_tpose[j]
        r_norm_tp = {}
        for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
            q_xyzw = skel_tpose[mhr_idx, 3:7]
            R_world_tp   = R.from_quat(q_xyzw)
            r_norm_tp[ik_name] = (R_rest[ik_name].inv() * R_world_tp).as_matrix()
    else:
        tp_ik_pts    = mhr_ik_pts
        tp_edge_list = mhr_edge_list
        r_norm_tp    = {ik_name: np.eye(3) for ik_name in ik_names_ordered}

    geoms = []

    # ── COL 1 (x = -2*GAP):  SMPL-X T-pose — identity arrows ─────────────────
    x1 = -2 * GAP
    sx_shifted = [sx_pts[idx].copy() + np.array([x1, 0, 0]) for idx in sx_ik_indices]
    geoms.append(build_lineset(sx_shifted, sx_edge_list, color=[0.2, 0.5, 1.0]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt = sx_pts[sx_idx].copy() + np.array([x1, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.6, 1.0)))
        geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)

    # ── COL 2 (x = -GAP):  MHR A-pose — raw rotations (chaotic) ──────────────
    x2 = -GAP
    mhr_shifted2 = [p + np.array([x2, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(mhr_shifted2, mhr_edge_list, color=[0.9, 0.2, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x2, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.3, 0.15)))
        geoms += make_axis_arrows(pt, R_rest[ik_name].as_matrix(), length=axis_len)

    # ── COL 3 (x = 0):  MHR A-pose — normalized = identity ───────────────────
    # At A-pose R_world = R_rest, so R_norm = R_rest⁻¹·R_rest = I.
    x3 = 0.0
    mhr_shifted3 = [p + np.array([x3, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(mhr_shifted3, mhr_edge_list, color=[0.8, 0.75, 0.0]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x3, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.9, 0.1)))
        geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)  # R_norm = I at A-pose

    # ── COL 4 (x = +GAP):  MHR ~T-pose — R_norm_tpose (not identity) ─────────
    x4 = GAP
    tp_shifted4 = [p + np.array([x4, 0, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(tp_shifted4, tp_edge_list, color=[0.1, 0.7, 0.2]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([x4, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.85, 0.3)))
        geoms += make_axis_arrows(pt, r_norm_tp[ik_name], length=axis_len)

    # ── COL 5 (x = +2*GAP):  MHR ~T-pose — R_AtoT·R_norm_tpose = I ──────────
    x5 = 2 * GAP
    tp_shifted5 = [p + np.array([x5, 0, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(tp_shifted5, tp_edge_list, color=[0.6, 0.2, 0.9]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([x5, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.75, 0.3, 1.0)))
        col5_rot = (R_AtoT[ik_name] * R.from_matrix(r_norm_tp[ik_name])).as_matrix()
        geoms += make_axis_arrows(pt, col5_rot, length=axis_len)

    # ── COL 6 (x = +3*GAP):  rot_offset · R_AtoT · R_norm_tpose ─────────────
    # Full IK target: rot_offset converts the now-identity arrows into the
    # Z-up convention that GMR / mink IK expects.  All arrows should be
    # identical across joints (= the fixed rot_offset rotation).
    x6 = 3 * GAP
    tp_shifted6 = [p + np.array([x6, 0, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(tp_shifted6, tp_edge_list, color=[0.8, 0.5, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([x6, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.65, 0.1)))
        q_off = _IK_ROT_OFFSETS[ik_name]                             # wxyz
        R_off = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])  # → xyzw
        col5_rot = (R_AtoT[ik_name] * R.from_matrix(r_norm_tp[ik_name])).as_matrix()
        col6_rot = (R.from_matrix(col5_rot) * R_off).as_matrix()
        geoms += make_axis_arrows(pt, col6_rot, length=axis_len)

    # World-frame reference triad
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=[x1 - 0.3, -0.15, 0]))

    print("\n[Open3D]  6-column view")
    print(f"  COL 1 (blue,   x={x1:+.1f}): SMPL-X T-pose  — identity arrows                    ← TARGET")
    print(f"  COL 2 (red,    x={x2:+.1f}): MHR A-pose     — raw R_world (chaotic)")
    print(f"  COL 3 (yellow, x={x3:+.1f}): MHR A-pose     — R_rest⁻¹·R_world = I (normalized)")
    print(f"  COL 4 (green,  x={x4:+.1f}): MHR ~T-pose    — R_norm_tpose (residual before fix)")
    print(f"  COL 5 (purple, x={x5:+.1f}): MHR ~T-pose    — R_AtoT·R_norm = I                 ← matches COL 1")
    print(f"  COL 6 (orange, x={x6:+.1f}): MHR ~T-pose    — rot_offset·R_AtoT·R_norm          ← GMR IK target")
    print("  ✓ CHECK: COL 5 = COL 3 = COL 1  (all identity arrows)")
    print("  ✓ CHECK: COL 6 arrows all uniform = rot_offset (Z-up IK convention)")
    print("  Press Q to quit.\n")
    o3d.visualization.draw_geometries(
        geoms,
        window_name="COL1:SMPLX | COL2:MHR raw | COL3:norm | COL4:T-norm | COL5:T-corrected | COL6:+rot_offset",
        mesh_show_back_face=True,
        width=3600, height=1000,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = args.device

    smplx_path = args.smplx_path or find_smplx_path()
    if smplx_path is None:
        print("[ERROR] SMPL-X model path not found.  Pass --smplx_path.")
        sys.exit(1)

    print(f"[SMPLX] Loading from {smplx_path} ...")
    smplx_joints = load_smplx_tpose(smplx_path, device)

    mhr_model = load_mhr_model(device)
    pos_apose, R_rest, _ = get_mhr_apose(mhr_model, device)

    # Print incoming bone directions for sanity check
    print("\n── Incoming bone directions (IK-parent -> joint) ─────────────────────")
    print(f"  {'joint':<18}  {'MHR A-pose direction':<38}  SMPL-X T-pose direction")
    for ik_name, bp in _BONE_PAIRS.items():
        if bp is None:
            print(f"  {ik_name:<18}  (identity — skipped)")
            continue
        sx_par, sx_cur, mhr_par, mhr_cur = bp
        d_mhr = pos_apose[mhr_cur] - pos_apose[mhr_par]
        d_sx  = smplx_joints[sx_cur] - smplx_joints[sx_par]
        d_mhr = np.round(d_mhr / (np.linalg.norm(d_mhr) + 1e-9), 3)
        d_sx  = np.round(d_sx  / (np.linalg.norm(d_sx)  + 1e-9), 3)
        print(f"  {ik_name:<18}  {mhr_par}->{mhr_cur}: {str(d_mhr):<28}  "
              f"smplx[{sx_par}]->[{sx_cur}]: {d_sx}")

    # Optimise MHR into T-pose shape first — joint quaternions are used for
    # R_AtoT, and joint positions are used for the COL 4 skeleton shape.
    skel_tpose, pos_tpose = optimise_mhr_tpose(mhr_model, smplx_joints, device,
                                               iters=args.iters, lr=args.lr,
                                               reg=args.reg)

    # Compute R_AtoT from T-pose quaternions (exact method).
    R_AtoT = compute_r_ato_t(skel_tpose, R_rest)

    visualise(smplx_joints, pos_apose, R_rest, R_AtoT,
              axis_len=args.axis_len, skel_tpose=skel_tpose, pos_tpose=pos_tpose)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute R_{A->T} per IK joint (bone-direction method).")
    p.add_argument("--smplx_path", default=None)
    p.add_argument("--axis_len",   type=float, default=0.08)
    p.add_argument("--iters",      type=int,   default=1500,
                   help="T-pose optimisation iterations (position only, for viz)")
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--reg",        type=float, default=1e-3)
    p.add_argument("--device",     default="cpu")
    main(p.parse_args())
