"""
visualize_mhr_offsets3.py  --  single frame: MHR sequence vs SMPL-X reference

Three columns:
    COL 1  (LEFT,  BLUE)   -- SMPL-X skeleton at IK joints with world-frame axes
    COL 2  (CENTRE, RED)   -- MHR raw R_world arrows  (chaotic)
    COL 3  (RIGHT, GREEN)  -- MHR R_world @ R_corrector arrows  (world-aligned)

Coordinate note:
    fit3d / MHR sequence data is Z-up.
    For display all positions are remapped (x,y,z)->(y,z,x) so Y becomes height.
    Rotation axes get the same permutation: R_disp = P @ R @ P.T

Usage:
    conda run -n mhr_new python scripts_extra/visualize_mhr_offsets3.py \
        /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz \
        /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
        --frame_id 100
"""

import os, sys, argparse, json, threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui      as gui
import open3d.visualization.rendering as rendering
import torch
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.expanduser("~/MHR"))

from visualize_mhr_rot_offsets import (
    _JOINT_MAP, _MHR_SKEL_EDGES_NAMES,
    make_axis_arrows, build_lineset, sphere_at, load_mhr_model,
)

# ---- Hardcoded R_corrector (R_world_tpose.inv(), offsets2.py iters=1500) ----
# WXYZ (scalar-first)
_R_CORRECTOR_WXYZ = {
    "root":     [+0.999682, +0.008607, -0.022852, -0.006274],
    "l_upleg":  [+0.576042, +0.490021, -0.445904, +0.478774],
    "r_upleg":  [+0.475742, -0.579886, -0.481732, -0.453141],
    "l_lowleg": [+0.601068, +0.402296, -0.543664, +0.425799],
    "r_lowleg": [+0.391426, -0.601407, -0.433425, -0.545196],
    "l_ball":   [+0.771242, -0.266156, -0.522893, +0.246841],
    "r_ball":   [-0.129435, -0.704962, -0.228047, -0.658991],
    "c_spine3": [+0.458588, -0.546636, -0.471312, -0.518413],
    "l_uparm":  [+0.793685, -0.587507, +0.017141, +0.156863],
    "r_uparm":  [-0.582795, -0.808512, -0.081599, +0.000291],
    "l_lowarm": [+0.803007, -0.556798, +0.196657, -0.080508],
    "r_lowarm": [-0.557896, -0.795043, +0.156783, +0.179103],
}
_R_CORRECTOR = {
    name: R.from_quat([q[1], q[2], q[3], q[0]])
    for name, q in _R_CORRECTOR_WXYZ.items()
}

# Axis permutation matrix: world (x,y,z) z-up  ->  display (y,z,x) y-up
_P = np.array([[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]], dtype=np.float64)


def zup_to_disp(xyz):
    """Remap (x,y,z) z-up vector to (y,z,x) y-up display vector."""
    return xyz[[1, 2, 0]]


def rot_to_disp(mat3x3):
    """Transform a rotation matrix to the display frame: P @ R @ P.T"""
    return _P @ mat3x3 @ _P.T


# ---- SMPL-X parent chain for joints 0-21 ----------------------------------
# 0 pelvis, 1 l_hip, 2 r_hip, 3 spine1, 4 l_knee, 5 r_knee, 6 spine2,
# 7 l_ankle, 8 r_ankle, 9 spine3, 10 l_foot, 11 r_foot, 12 neck,
# 13 l_collar, 14 r_collar, 15 head, 16 l_shoulder, 17 r_shoulder,
# 18 l_elbow, 19 r_elbow, 20 l_wrist, 21 r_wrist
_SMPLX_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

# j25 index that best matches each sx_idx joint (wrists confirmed by SMPL-X
# forearm bone length matching j25 elbow→candidate distance ~0.265 m)
_SX_TO_J25 = {0:0, 1:1, 2:4, 4:2, 5:5, 7:17, 8:19, 9:8,
              16:11, 17:14, 18:12, 19:15, 20:13, 21:16}

# IK skeleton edges (sx_idx pairs) — includes forearms (18/19 → 20/21)
_SX_IK_EDGES = [
    (0,1),(0,2),(1,4),(2,5),(4,7),(5,8),
    (0,9),(9,16),(9,17),(16,18),(17,19),(18,20),(19,21),
]

# ---- SMPL-X model: T-pose joint positions + kintree for hand FK -----------
# Loaded once at import time from the neutral SMPL-X npz.
_SMPLX_NPZ    = np.load(
    os.path.join(os.path.dirname(__file__), "..", "assets", "body_models",
                 "smplx", "SMPLX_NEUTRAL.npz"), allow_pickle=True)
_J0_55         = (_SMPLX_NPZ["J_regressor"] @ _SMPLX_NPZ["v_template"])  # (55,3)
_SX_PARENTS_55 = _SMPLX_NPZ["kintree_table"][0]                            # (55,)
# Hand edges from kintree for joints 25–54 (skip 22-24 = jaw/eyes)
_SX_HAND_EDGES = [(int(_SX_PARENTS_55[j]), j) for j in range(25, 55)]

# ---- MHR finger edges ------------------------------------------------------
_MHR_FINGER_EDGES = [
    (40,42),(42,43),(43,44),(44,45),(45,46),
    (42,48),(48,49),(49,50),(42,52),(52,53),(53,54),
    (42,56),(56,57),(57,58),(42,60),(60,61),(61,62),(62,63),
    (76,78),(78,79),(79,80),(80,81),(81,82),
    (78,84),(84,85),(85,86),(78,88),(88,89),(89,90),
    (78,92),(92,93),(93,94),(78,96),(96,97),(97,98),(98,99),
]
_MHR_FINGER_JOINTS = sorted({j for e in _MHR_FINGER_EDGES for j in e})


# ---- Human JSON loader -----------------------------------------------------

def load_human_data(json_path, frame_id):
    """Returns (joints25_zup, smplx_world_rots, hand_pts_zup) for one frame.
    - joints25_zup    : (25,3) in metres, z-up world frame
    - smplx_world_rots: dict  sx_idx -> scipy Rotation  (world-frame, z-up)
    - hand_pts_zup    : dict  smplx_j (25-54) -> (3,) pos in z-up world frame
                        Hand positions anchored at j25 wrists via SMPL-X hand FK.
                        Empty dict if hand pose data unavailable.
    """
    d = json.load(open(json_path))
    seq_name = os.path.splitext(os.path.basename(json_path))[0]
    seq_root = os.path.dirname(os.path.dirname(json_path))

    # --- positions: from joints3d_25 ---
    if "joints3d_25" in d:
        j25    = np.array(d["joints3d_25"], dtype=np.float32)[frame_id]
        go_mat = None
        bp_mat = None
        lhp    = None
        rhp    = None
    else:
        j25_path = os.path.join(seq_root, "joints3d_25", seq_name + ".json")
        if not os.path.exists(j25_path):
            raise FileNotFoundError(f"joints3d_25 not found: {j25_path}")
        print(f"  [auto] joints3d_25: {j25_path}")
        j25    = np.array(json.load(open(j25_path))["joints3d_25"], dtype=np.float32)[frame_id]
        go_mat = np.array(d["global_orient"])[frame_id, 0]      # (3,3)
        bp_mat = np.array(d["body_pose"])[frame_id]              # (21,3,3)
        lhp    = np.array(d["left_hand_pose"])[frame_id]  if "left_hand_pose"  in d else None  # (15,3,3)
        rhp    = np.array(d["right_hand_pose"])[frame_id] if "right_hand_pose" in d else None  # (15,3,3)

    # --- world rotations from body kinematic chain (joints 0-21) ---
    smplx_world_rots = {}
    wrot = [None] * 55   # full 55-joint rotation array
    if go_mat is not None and bp_mat is not None:
        wrot[0] = R.from_matrix(go_mat)
        for j in range(1, 22):
            wrot[j] = wrot[_SX_PARENTS_55[j]] * R.from_matrix(bp_mat[j - 1])
        for sx_idx in _SX_TO_J25:
            if sx_idx <= 21:
                smplx_world_rots[sx_idx] = wrot[sx_idx]

    # --- hand world positions via SMPL-X hand FK ---
    # Finger FK: anchor G_p[20]=j25[wrist_l], G_p[21]=j25[wrist_r],
    # then chain:  G_p[j] = G_p[parent] + wrot[parent] @ (J0[j] - J0[parent])
    # J0 offsets are in SMPL-X canonical (Y-up) space; wrot[0]=global_orient
    # maps canonical→world, so the chain naturally produces world-frame offsets.
    hand_pts_zup = {}
    can_do_hands = (go_mat is not None and bp_mat is not None
                    and lhp is not None and rhp is not None)
    if can_do_hands:
        # Build world rotation for joints 22-54 using full kintree
        jaw_mat = np.array(d["jaw_pose"])[frame_id, 0]  if "jaw_pose"  in d else np.eye(3)
        le_mat  = np.array(d["leye_pose"])[frame_id, 0] if "leye_pose" in d else np.eye(3)
        re_mat  = np.array(d["reye_pose"])[frame_id, 0] if "reye_pose" in d else np.eye(3)
        extra_locs = [jaw_mat, le_mat, re_mat] + list(lhp) + list(rhp)  # 3+15+15=33
        for k, j in enumerate(range(22, 55)):
            p = int(_SX_PARENTS_55[j])
            wrot[j] = wrot[p] * R.from_matrix(extra_locs[k])

        # World positions for hand joints 25-54 (skip 22-24 = jaw/eyes)
        # Anchor wrists at j25 positions; also include wrists so edges connect.
        G_p = {20: j25[_SX_TO_J25[20]], 21: j25[_SX_TO_J25[21]]}  # z-up world metres
        hand_pts_zup[20] = G_p[20]   # l_wrist
        hand_pts_zup[21] = G_p[21]   # r_wrist
        for j in range(25, 55):
            p   = int(_SX_PARENTS_55[j])
            off = _J0_55[j] - _J0_55[p]   # offset in SMPL-X canonical (Y-up)
            G_p[j] = G_p[p] + wrot[p].as_matrix() @ off
            hand_pts_zup[j] = G_p[j]

    return j25, smplx_world_rots, hand_pts_zup


# ---- MHR helpers -----------------------------------------------------------

def _mhr_norm(skel_np, disp_h=1.0):
    """Root-centred + height-normalised positions in display (y-up) frame.
    skel_np[:,0:3] is in cm, z-up.
    Returns: root_zup_m (3,), scale, and a function pos(j)->(3,) display.
    """
    root_zup = skel_np[1,   :3] / 100.0
    head_zup = skel_np[113, :3] / 100.0
    h_m      = max(head_zup[2] - root_zup[2], 0.1)   # z is height
    scale    = disp_h / h_m
    def pos(j_idx):
        xyz = skel_np[j_idx, :3] / 100.0
        return (xyz - root_zup) * scale
    return root_zup, scale, pos


def _mhr_body_ls(skel_np, pos_fn, x_off, color):
    ik_names = [v[2] for v in _JOINT_MAP.values()]
    n_idx    = {n: i for i, n in enumerate(ik_names)}
    pts      = [pos_fn(mi) + np.array([x_off, 0, 0])
                for _, (_, mi, _) in _JOINT_MAP.items()]
    edges    = [(n_idx[a], n_idx[b])
                for a,b in _MHR_SKEL_EDGES_NAMES if a in n_idx and b in n_idx]
    return build_lineset(pts, edges, color=color)


def _mhr_finger_ls(skel_np, pos_fn, x_off, color):
    fp   = {j: pos_fn(j) for j in _MHR_FINGER_JOINTS}
    fi   = {j: k for k,j in enumerate(sorted(fp))}
    fpts = [fp[j] + np.array([x_off,0,0]) for j in sorted(fp)]
    fedge= [(fi[a],fi[b]) for a,b in _MHR_FINGER_EDGES if a in fp and b in fp]
    return build_lineset(fpts, fedge, color=color)


# ---- SMPL-X display helpers -----------------------------------------------

def _smplx_norm(j25_zup, disp_h=1.0):
    """Root-centred + normalised positions in display (y-up) frame.
    j25[0] = pelvis (root), j25[10] = head (for height).
    Returns: root_zup (3,), scale, pts_disp (25,3) in display frame.
    """
    root_zup = j25_zup[0].copy()
    head_zup = j25_zup[10].copy()
    h_m      = max(head_zup[2] - root_zup[2], 0.1)
    scale    = disp_h / h_m
    pts_disp = np.array([(j - root_zup) * scale for j in j25_zup])
    return root_zup, scale, pts_disp


def _smplx_ik_ls(pts_disp, x_off, color):
    """Skeleton lineset at the IK-mapped joints only (includes forearms)."""
    sx_list  = list(_SX_TO_J25.keys())
    sx_local = {sx: i for i, sx in enumerate(sx_list)}
    pts      = [pts_disp[_SX_TO_J25[sx]] + np.array([x_off,0,0]) for sx in sx_list]
    edges    = [(sx_local[a], sx_local[b])
                for a,b in _SX_IK_EDGES if a in sx_local and b in sx_local]
    return build_lineset(pts, edges, color=color)


def _smplx_hands_ls(hand_pts_zup, root_zup, scale, x_off, color):
    """Draw SMPL-X finger skeleton from hand_pts_zup (z-up world frame)."""
    if not hand_pts_zup:
        return None
    pts  = {j: (p - root_zup) * scale
            for j, p in hand_pts_zup.items()}
    j_list = sorted(pts)
    j_idx  = {j: i for i, j in enumerate(j_list)}
    pt_lst = [pts[j] + np.array([x_off, 0, 0]) for j in j_list]
    edges  = [(j_idx[a], j_idx[b]) for a, b in _SX_HAND_EDGES
              if a in pts and b in pts]
    return build_lineset(pt_lst, edges, color=color)


# ---- Main ------------------------------------------------------------------

def visualise(npz_path, human_path, frame_id, axis_len=0.10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev    = torch.device(device)
    GAP    = 2.5
    DISP_H = 1.0

    # Load MHR sequence
    data     = np.load(npz_path)
    n_frames = data["param_lbs_model_params"].shape[0]
    if not (0 <= frame_id < n_frames):
        raise ValueError(f"frame_id {frame_id} out of range [0, {n_frames-1}]")
    print(f"  MHR   : {os.path.basename(npz_path)}  ({n_frames} frames)  -> frame {frame_id}")

    # Load human data
    j25_zup, smplx_wrots, hand_pts = load_human_data(human_path, frame_id)
    print(f"  Human : {os.path.basename(human_path)}")

    # Run MHR FK
    mhr_model = load_mhr_model(device)
    with torch.no_grad():
        _, skel = mhr_model(
            torch.tensor(data["param_identity_coeffs"][frame_id][None], dtype=torch.float32).to(dev),
            torch.tensor(data["param_lbs_model_params"][frame_id][None], dtype=torch.float32).to(dev),
            torch.tensor(data["param_face_expr_coeffs"][frame_id][None], dtype=torch.float32).to(dev))
    skel_np = skel[0].cpu().numpy()   # (127,8)

    _, _, mhr_pos = _mhr_norm(skel_np, DISP_H)
    sx_root, sx_scale, sx_pts = _smplx_norm(j25_zup, DISP_H)

    # Build geometry
    geoms = []
    x1, x2, x3 = -GAP, 0.0, GAP

    # COL 1: SMPL-X IK joints + world-frame axes + hands (blue)
    geoms.append(_smplx_ik_ls(sx_pts, x1, [0.2, 0.5, 1.0]))
    hand_ls = _smplx_hands_ls(hand_pts, sx_root, sx_scale, x1, [0.3, 0.6, 1.0])
    if hand_ls is not None:
        geoms.append(hand_ls)
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        j25_idx = _SX_TO_J25.get(sx_idx)
        if j25_idx is None:
            continue
        pt = sx_pts[j25_idx] + np.array([x1, 0, 0])
        geoms.append(sphere_at(pt, radius=0.016, color=(0.2, 0.6, 1.0)))
        if sx_idx in smplx_wrots:
            geoms += make_axis_arrows(pt, smplx_wrots[sx_idx].as_matrix(), length=axis_len)
        else:
            geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)

    # COL 2: MHR raw R_world (red)
    geoms.append(_mhr_body_ls  (skel_np, mhr_pos, x2, [0.9, 0.2, 0.1]))
    geoms.append(_mhr_finger_ls(skel_np, mhr_pos, x2, [0.9, 0.2, 0.1]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt      = mhr_pos(mhr_idx) + np.array([x2, 0, 0])
        R_world = R.from_quat(skel_np[mhr_idx, 3:7])
        geoms.append(sphere_at(pt, radius=0.016, color=(1.0, 0.3, 0.15)))
        geoms += make_axis_arrows(pt, R_world.as_matrix(), length=axis_len)

    # COL 3: MHR corrected R_world @ R_corrector (green)
    geoms.append(_mhr_body_ls  (skel_np, mhr_pos, x3, [0.1, 0.7, 0.2]))
    geoms.append(_mhr_finger_ls(skel_np, mhr_pos, x3, [0.1, 0.7, 0.2]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt       = mhr_pos(mhr_idx) + np.array([x3, 0, 0])
        R_world  = R.from_quat(skel_np[mhr_idx, 3:7])
        R_corr   = (R_world * _R_CORRECTOR[ik_name]).as_matrix()
        geoms.append(sphere_at(pt, radius=0.016, color=(0.2, 0.85, 0.3)))
        geoms += make_axis_arrows(pt, R_corr, length=axis_len)

    # Reference triad
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=np.array([x1 - 0.5, -1.2, 0])))

    seq_name = os.path.splitext(os.path.basename(npz_path))[0]
    print("=" * 70)
    print(f"  Sequence : {seq_name}   frame {frame_id} / {n_frames-1}")
    print("=" * 70)
    print(f"  COL 1  x={x1:+.2f}  BLUE   SMPL-X IK joints + world-frame axes")
    print(f"  COL 2  x={x2:+.2f}  RED    MHR raw R_world (chaotic)")
    print(f"  COL 3  x={x3:+.2f}  GREEN  MHR R_world @ R_corrector (corrected)")
    print("=" * 70)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"offsets3 -- {seq_name}  frame {frame_id}",
        mesh_show_back_face=True,
        width=3600, height=1000,
    )


# ── Interactive sequence viewer ───────────────────────────────────────────────

def _build_frame_geoms(skel_np, j25_zup, smplx_wrots, hand_pts_zup=None, axis_len=0.10):
    """Build all Open3D geometry for one frame (shared by static + interactive)."""
    GAP    = 2.5
    DISP_H = 1.0
    if hand_pts_zup is None:
        hand_pts_zup = {}

    _, _, mhr_pos_fn   = _mhr_norm(skel_np, DISP_H)
    sx_root, sx_scale, sx_pts = _smplx_norm(j25_zup, DISP_H)

    geoms = []
    x1, x2, x3 = -GAP, 0.0, GAP

    # COL 1 — SMPL-X IK joints + world-frame axes + hands (blue)
    geoms.append(_smplx_ik_ls(sx_pts, x1, [0.2, 0.5, 1.0]))
    hand_ls = _smplx_hands_ls(hand_pts_zup, sx_root, sx_scale, x1, [0.3, 0.6, 1.0])
    if hand_ls is not None:
        geoms.append(hand_ls)
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        j25_idx = _SX_TO_J25.get(sx_idx)
        if j25_idx is None:
            continue
        pt = sx_pts[j25_idx] + np.array([x1, 0, 0])
        geoms.append(sphere_at(pt, radius=0.016, color=(0.2, 0.6, 1.0)))
        if sx_idx in smplx_wrots:
            geoms += make_axis_arrows(pt, smplx_wrots[sx_idx].as_matrix(), length=axis_len)
        else:
            geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)

    # COL 2 — MHR raw R_world (red)
    geoms.append(_mhr_body_ls  (skel_np, mhr_pos_fn, x2, [0.9, 0.2, 0.1]))
    geoms.append(_mhr_finger_ls(skel_np, mhr_pos_fn, x2, [0.9, 0.2, 0.1]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt      = mhr_pos_fn(mhr_idx) + np.array([x2, 0, 0])
        R_world = R.from_quat(skel_np[mhr_idx, 3:7])
        geoms.append(sphere_at(pt, radius=0.016, color=(1.0, 0.3, 0.15)))
        geoms += make_axis_arrows(pt, R_world.as_matrix(), length=axis_len)

    # COL 3 — MHR corrected R_world @ R_corrector (green)
    geoms.append(_mhr_body_ls  (skel_np, mhr_pos_fn, x3, [0.1, 0.7, 0.2]))
    geoms.append(_mhr_finger_ls(skel_np, mhr_pos_fn, x3, [0.1, 0.7, 0.2]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt      = mhr_pos_fn(mhr_idx) + np.array([x3, 0, 0])
        R_world = R.from_quat(skel_np[mhr_idx, 3:7])
        R_corr  = (R_world * _R_CORRECTOR[ik_name]).as_matrix()
        geoms.append(sphere_at(pt, radius=0.016, color=(0.2, 0.85, 0.3)))
        geoms += make_axis_arrows(pt, R_corr, length=axis_len)

    # Reference triad
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=np.array([x1 - 0.5, -1.2, 0])))

    return geoms


class SequenceApp:
    """Interactive frame slider viewer for an MHR sequence."""

    PANEL_W = 320

    def __init__(self, npz_path, human_path, axis_len=0.10):
        self.npz_path   = npz_path
        self.human_path = human_path
        self.axis_len   = axis_len
        self.mhr_model  = None
        self.dev        = None
        self.seq_data   = None      # dict of npz arrays
        self.j25_all    = None      # (N, 25, 3)  preloaded
        self.smplx_d    = None      # raw smplx JSON dict
        self.n_frames   = 0
        self.cur_frame  = 0
        self._geom_names = []

        self.app = gui.Application.instance
        self.app.initialize()
        self._build_window()
        threading.Thread(target=self._load, daemon=True).start()

    # ── Window layout ─────────────────────────────────────────────────────
    def _build_window(self):
        seq_name = os.path.splitext(os.path.basename(self.npz_path))[0]
        self.win = self.app.create_window(
            f"offsets3 — {seq_name}", 2000, 950)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.win.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])
        self._scene.scene.scene.enable_sun_light(False)

        panel = gui.Vert(6, gui.Margins(10, 10, 10, 10))
        self._panel = panel

        panel.add_child(gui.Label(os.path.splitext(os.path.basename(self.npz_path))[0]))
        panel.add_child(gui.Label(""))

        # Frame label
        self._frame_lbl = gui.Label("Frame: —  /  —")
        panel.add_child(self._frame_lbl)

        # Slider
        self._slider = gui.Slider(gui.Slider.INT)
        self._slider.set_limits(0, 1)
        self._slider.int_value = 0
        self._slider.set_on_value_changed(self._on_slider)
        panel.add_child(self._slider)

        panel.add_child(gui.Label(""))

        # Column legend
        for txt, col in [
            ("COL 1  BLUE   SMPL-X IK joints",   gui.Color(0.1, 0.3, 0.9)),
            ("COL 2  RED    MHR raw R_world",     gui.Color(0.8, 0.1, 0.05)),
            ("COL 3  GREEN  MHR corrected",       gui.Color(0.05, 0.55, 0.1)),
        ]:
            lbl = gui.Label(txt)
            lbl.text_color = col
            panel.add_child(lbl)

        panel.add_child(gui.Label(""))

        self._status = gui.Label("[Loading...]")
        self._status.text_color = gui.Color(0.7, 0.4, 0.0)
        panel.add_child(self._status)

        PW = self.PANEL_W
        def on_layout(ctx):
            r = self.win.content_rect
            self._scene.frame = gui.Rect(r.x, r.y, r.width - PW, r.height)
            self._panel.frame  = gui.Rect(r.x + r.width - PW, r.y, PW, r.height)
        self.win.set_on_layout(on_layout)
        self.win.add_child(self._scene)
        self.win.add_child(self._panel)

    # ── Background load ───────────────────────────────────────────────────
    def _load(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dev = torch.device(device)

            self.mhr_model = load_mhr_model(device)

            npz = np.load(self.npz_path)
            self.seq_data = {k: npz[k] for k in npz.files}
            self.n_frames = self.seq_data["param_lbs_model_params"].shape[0]

            # Preload all j25 positions
            j25_all, smplx_d = self._load_all_human()
            self.j25_all  = j25_all    # (N, 25, 3)
            self.smplx_d  = smplx_d    # raw dict or None

            gui.Application.instance.post_to_main_thread(self.win, self._on_loaded)
        except Exception as e:
            import traceback; traceback.print_exc()
            msg = str(e)
            gui.Application.instance.post_to_main_thread(
                self.win, lambda: setattr(self._status, "text", f"ERROR: {msg}"))

    def _load_all_human(self):
        """Return (j25_all (N,25,3), smplx_dict_or_None)."""
        d = json.load(open(self.human_path))
        seq_name = os.path.splitext(os.path.basename(self.human_path))[0]
        seq_root = os.path.dirname(os.path.dirname(self.human_path))

        if "joints3d_25" in d:
            return np.array(d["joints3d_25"], dtype=np.float32), None

        j25_path = os.path.join(seq_root, "joints3d_25", seq_name + ".json")
        if not os.path.exists(j25_path):
            raise FileNotFoundError(f"joints3d_25 not found: {j25_path}")
        j25_all = np.array(json.load(open(j25_path))["joints3d_25"], dtype=np.float32)
        return j25_all, d   # d is the smplx dict

    def _on_loaded(self):
        self._slider.set_limits(0, self.n_frames - 1)
        self._slider.int_value = 0
        self._status.text = "[Ready]"
        self._status.text_color = gui.Color(0.0, 0.5, 0.0)
        self._update_scene(0)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    # ── Slider callback ───────────────────────────────────────────────────
    def _on_slider(self, val):
        fid = int(val)
        self.cur_frame = fid
        self._frame_lbl.text = f"Frame: {fid}  /  {self.n_frames-1}"
        self._update_scene(fid)

    # ── Scene update ──────────────────────────────────────────────────────
    def _update_scene(self, frame_id):
        if self.mhr_model is None:
            return

        for name in self._geom_names:
            self._scene.scene.remove_geometry(name)
        self._geom_names.clear()

        # FK
        sd  = self.seq_data
        with torch.no_grad():
            _, skel = self.mhr_model(
                torch.tensor(sd["param_identity_coeffs"][frame_id][None], dtype=torch.float32).to(self.dev),
                torch.tensor(sd["param_lbs_model_params"][frame_id][None], dtype=torch.float32).to(self.dev),
                torch.tensor(sd["param_face_expr_coeffs"][frame_id][None], dtype=torch.float32).to(self.dev))
        skel_np = skel[0].cpu().numpy()

        j25_zup = self.j25_all[frame_id]

        # Compute smplx world rots + hand positions
        smplx_wrots = {}
        hand_pts_zup = {}
        if self.smplx_d is not None:
            go_mat = np.array(self.smplx_d["global_orient"])[frame_id, 0]
            bp_mat = np.array(self.smplx_d["body_pose"])[frame_id]
            lhp    = np.array(self.smplx_d["left_hand_pose"])[frame_id]  if "left_hand_pose"  in self.smplx_d else None
            rhp    = np.array(self.smplx_d["right_hand_pose"])[frame_id] if "right_hand_pose" in self.smplx_d else None
            wrot = [None] * 55
            wrot[0] = R.from_matrix(go_mat)
            for j in range(1, 22):
                wrot[j] = wrot[_SX_PARENTS_55[j]] * R.from_matrix(bp_mat[j - 1])
            for sx_idx in _SX_TO_J25:
                if sx_idx <= 21:
                    smplx_wrots[sx_idx] = wrot[sx_idx]
            # Hand FK
            if lhp is not None and rhp is not None:
                jaw = np.array(self.smplx_d["jaw_pose"])[frame_id, 0]  if "jaw_pose"  in self.smplx_d else np.eye(3)
                le  = np.array(self.smplx_d["leye_pose"])[frame_id, 0] if "leye_pose" in self.smplx_d else np.eye(3)
                re  = np.array(self.smplx_d["reye_pose"])[frame_id, 0] if "reye_pose" in self.smplx_d else np.eye(3)
                for k, j in enumerate(range(22, 55)):
                    extra = ([jaw, le, re] + list(lhp) + list(rhp))[k]
                    wrot[j] = wrot[int(_SX_PARENTS_55[j])] * R.from_matrix(extra)
                G_p = {20: j25_zup[_SX_TO_J25[20]], 21: j25_zup[_SX_TO_J25[21]]}
                hand_pts_zup[20] = G_p[20]   # l_wrist
                hand_pts_zup[21] = G_p[21]   # r_wrist
                for j in range(25, 55):
                    p = int(_SX_PARENTS_55[j])
                    G_p[j] = G_p[p] + wrot[p].as_matrix() @ (_J0_55[j] - _J0_55[p])
                    hand_pts_zup[j] = G_p[j]

        self._frame_lbl.text = f"Frame: {frame_id}  /  {self.n_frames-1}"

        for i, geom in enumerate(_build_frame_geoms(skel_np, j25_zup, smplx_wrots, hand_pts_zup, self.axis_len)):
            name = f"g{frame_id}_{i}"
            mat  = rendering.MaterialRecord()
            if isinstance(geom, o3d.geometry.LineSet):
                mat.shader     = "unlitLine"
                mat.line_width = 2.0
                cols = np.asarray(geom.colors)
                if len(cols):
                    c = cols[0]
                    mat.base_color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
            else:
                mat.shader = "defaultUnlit"
                cols = np.asarray(geom.vertex_colors)
                if not len(cols):
                    mat.base_color = (0.8, 0.8, 0.8, 1.0)
            self._scene.scene.add_geometry(name, geom, mat)
            self._geom_names.append(name)

        self.win.post_redraw()

    def run(self):
        self.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path",   type=str)
    parser.add_argument("human_path", type=str,
                        help="smplx or joints3d_25 JSON (smplx sibling auto-located)")
    parser.add_argument("--frame_id", type=int, default=None,
                        help="Frame to show (omit for interactive slider)")
    parser.add_argument("--axis_len", type=float, default=0.10)
    args = parser.parse_args()
    if args.frame_id is None:
        SequenceApp(args.npz_path, args.human_path, args.axis_len).run()
    else:
        visualise(args.npz_path, args.human_path, args.frame_id, args.axis_len)
