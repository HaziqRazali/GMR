"""
visualize_mhr_offsets2.py  —  simplified R_AtoT algorithm

Algorithm (plain):
    OFFLINE:
        R_world_tp  = FK output after IK(MHR positions → SMPL-X T-pose positions)
        R_corrector = R_world_tp.inv()    ← one matrix per joint, stored forever

    RUNTIME (every frame):
        R_target = R_corrector @ R_world_current

Five columns:
    COL 1  SMPL-X T-pose   — identity arrows     TARGET
    COL 2  MHR A-pose      — chaotic arrows       raw R_world at A-pose
    COL 3  MHR T-pose      — chaotic arrows       R_world_tp after IK (still not clean)
    COL 4  MHR T-pose      — world-aligned = I    R_corrector @ R_world_tp  (should = COL 1)
    COL 5  MHR A-pose      — R_corrector applied  R_corrector @ R_rest  (what A-pose looks like after correction)

Usage:
    conda run -n mhr_new python scripts_extra/visualize_mhr_offsets2.py
"""

import os, sys
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/MHR"))

# ── Reuse all constants from the original file ────────────────────────────────
from visualize_mhr_rot_offsets import (
    _JOINT_MAP,
    _IK_ROT_OFFSETS,
    _MHR_SKEL_EDGES_NAMES,
    make_axis_arrows,
    build_lineset,
    sphere_at,
    find_smplx_path,
    load_smplx_tpose,
    load_mhr_model,
    get_mhr_apose,
    optimise_mhr_tpose,
)


# ── Finger joint constants ───────────────────────────────────────────────────
# MHR skel_np indices — real joints only (no _proc / _null).
# r_lowarm=40, r_wrist=42; l_lowarm=76, l_wrist=78
_MHR_FINGER_EDGES_IDX = [
    (40, 42),                                    # r_lowarm → r_wrist
    (42,43),(43,44),(44,45),(45,46),             # r_pinky 0-3
    (42,48),(48,49),(49,50),                     # r_ring 1-3
    (42,52),(52,53),(53,54),                     # r_middle 1-3
    (42,56),(56,57),(57,58),                     # r_index 1-3
    (42,60),(60,61),(61,62),(62,63),             # r_thumb 0-3
    (76, 78),                                    # l_lowarm → l_wrist
    (78,79),(79,80),(80,81),(81,82),             # l_pinky 0-3
    (78,84),(84,85),(85,86),                     # l_ring 1-3
    (78,88),(88,89),(89,90),                     # l_middle 1-3
    (78,92),(92,93),(93,94),                     # l_index 1-3
    (78,96),(96,97),(97,98),(98,99),             # l_thumb 0-3
]
_MHR_FINGER_JOINT_INDICES = sorted({j for e in _MHR_FINGER_EDGES_IDX for j in e})

# SMPL-X standard layout with use_pca=False + flat_hand_mean:
#   elbows 18 (L) 19 (R),  wrists 20 (L) 21 (R),  fingers 25-54
_SMPLX_FINGER_EDGES = [
    (18, 20), (19, 21),            # elbows → wrists
    # left hand (wrist=20, joints 25–39)
    (20,25),(25,26),(26,27),       # l_index 1-3
    (20,28),(28,29),(29,30),       # l_middle 1-3
    (20,31),(31,32),(32,33),       # l_pinky 1-3
    (20,34),(34,35),(35,36),       # l_ring 1-3
    (20,37),(37,38),(38,39),       # l_thumb 1-3
    # right hand (wrist=21, joints 40–54)
    (21,40),(40,41),(41,42),       # r_index 1-3
    (21,43),(43,44),(44,45),       # r_middle 1-3
    (21,46),(46,47),(47,48),       # r_pinky 1-3
    (21,49),(49,50),(50,51),       # r_ring 1-3
    (21,52),(52,53),(53,54),       # r_thumb 1-3
]
_SMPLX_FINGER_JOINT_INDICES = sorted({j for e in _SMPLX_FINGER_EDGES for j in e})


# ── Geometry helpers (local, same as original) ────────────────────────────────

def _make_cloud(pos_dict, ik_names_ordered, disp_h=1.0):
    """Centred + height-normalised skeleton point list + edge list.
    Returns (pts_list, edges, root_p_m, scale) so callers can reuse the
    same normalisation for finger joints.
    """
    pts_arr  = np.array([pos_dict[n] for n in ik_names_ordered])
    root_p   = pos_dict["root"]
    pts_c    = pts_arr - root_p
    max_y    = pts_c[:, 1].max()
    scale    = disp_h / (max_y + 1e-9)
    pts_c    = pts_c * scale
    name_pt  = {n: pts_c[i] for i, n in enumerate(ik_names_ordered)}
    n_idx    = {n: i for i, n in enumerate(ik_names_ordered)}
    edges    = [(n_idx[a], n_idx[b])
                for a, b in _MHR_SKEL_EDGES_NAMES
                if a in n_idx and b in n_idx]
    return [name_pt[n] for n in ik_names_ordered], edges, root_p, scale


def _add_finger_geoms_mhr(geoms, skel_np, root_m, scale, x_offset, color):
    """Draw MHR finger skeleton. skel_np positions are in cm → convert to m."""
    pts = {j: (skel_np[j, :3] / 100.0 - root_m) * scale
           for j in _MHR_FINGER_JOINT_INDICES}
    sorted_j  = sorted(pts)
    idx       = {j: i for i, j in enumerate(sorted_j)}
    pts_list  = [pts[j] + np.array([x_offset, 0, 0]) for j in sorted_j]
    edges_loc = [(idx[a], idx[b]) for a, b in _MHR_FINGER_EDGES_IDX
                 if a in pts and b in pts]
    geoms.append(build_lineset(pts_list, edges_loc, color=color))


def _add_finger_geoms_smplx(geoms, joints, root_m, scale, x_offset, color):
    """Draw SMPL-X finger skeleton. Falls back gracefully if joints array too short."""
    n = len(joints)
    valid_j   = [j for j in _SMPLX_FINGER_JOINT_INDICES if j < n]
    pts       = {j: (joints[j] - root_m) * scale for j in valid_j}
    sorted_j  = sorted(pts)
    idx       = {j: i for i, j in enumerate(sorted_j)}
    pts_list  = [pts[j] + np.array([x_offset, 0, 0]) for j in sorted_j]
    edges_loc = [(idx[a], idx[b]) for a, b in _SMPLX_FINGER_EDGES
                 if a in pts and b in pts]
    geoms.append(build_lineset(pts_list, edges_loc, color=color))


# ── Core: new simplified R_AtoT computation ───────────────────────────────────

def compute_r_corrector(skel_tpose):
    """
    Simple algorithm:
        R_world_tp  = skel_tpose rotation  (world-frame, as stored by MHR FK)
        R_corrector = R_world_tp.inv()

    skel_tpose[j, 3:7] stores the world-frame quaternion output of MHR's FK.
    This is already the full world-frame rotation — R_rest is baked in.

    At runtime:  R_target = R_corrector @ R_world_current
    At T-pose:   R_corrector @ R_world_tp = I  (world-aligned arrows, by construction)

    Algebraically equivalent to the original decomposed form:
        R_world_tp.inv() == R_AtoT @ R_rest.inv()
    But stored as a single matrix — the runtime formula no longer needs a
    separate R_rest.inv() step.

    Returns dict  ik_name → scipy Rotation  (R_corrector per joint)
    Also prints the _R_ATO_T_WXYZ dict for copy-paste.

    NOTE: The printed values will differ from the original visualize_mhr_rot_offsets.py
    because the original stores only R_AtoT (the residual), whereas this stores
    the combined R_world_tp.inv() = R_AtoT @ R_rest.inv().  Both are correct;
    they just require different runtime formulas.
    """
    R_corrector = {}

    print("\n" + "═"*72)
    print("  R_corrector = R_world_tp.inv()  (simplified algorithm)")
    print("  skel_tpose stores world-frame rotations directly (R_rest baked in)")
    print("═"*72)
    print(f"  {'joint':<18}  {'angle':>6}    wxyz quaternion")
    print("  " + "-"*60)

    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        q_xyzw     = skel_tpose[mhr_idx, 3:7]   # world-frame directly
        R_world_tp = R.from_quat(q_xyzw)
        R_corr     = R_world_tp.inv()
        R_corrector[ik_name] = R_corr

        angle_deg = np.degrees(np.linalg.norm(R_corr.as_rotvec()))
        q = R_corr.as_quat(scalar_first=True)
        print(f"  {ik_name:<18}  {angle_deg:5.1f} deg   "
              f"[{q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}]")

    print("\n# ── _R_ATO_T_WXYZ (copy-paste) ──────────────────────────────────────")
    print("_R_ATO_T_WXYZ = {")
    for _, (_, _, ik_name) in _JOINT_MAP.items():
        q = R_corrector[ik_name].as_quat(scalar_first=True)
        print(f'    "{ik_name}": [{q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f}],')
    print("}")
    print("═"*72 + "\n")

    return R_corrector


# ── Main visualisation ────────────────────────────────────────────────────────

def visualise(axis_len=0.10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GAP    = 5
    DISP_H = 1.0

    ik_names = [ik_name for _, (_, _, ik_name) in _JOINT_MAP.items()]

    # ── Load SMPL-X T-pose ────────────────────────────────────────────────────
    smplx_path = find_smplx_path()
    assert smplx_path, "SMPL-X model not found"
    smplx_joints = load_smplx_tpose(smplx_path, device)    # (J, 3) metres

    # SMPL-X skeleton display
    sx_ik_indices = list(_JOINT_MAP.keys())
    sx_local      = {sx_idx: i for i, sx_idx in enumerate(sx_ik_indices)}
    sx_pts_raw    = np.array([smplx_joints[i] for i in sx_ik_indices])
    sx_root       = smplx_joints[0]
    sx_pts_c      = (sx_pts_raw - sx_root)
    sx_max_y      = sx_pts_c[:, 1].max()
    sx_scale      = DISP_H / (sx_max_y + 1e-9)
    sx_pts_c      = sx_pts_c * sx_scale
    sx_edge_list  = []
    _sx_ik_edges  = [
        (0,1),(0,2),(1,4),(2,5),(4,7),(5,8),
        (0,9),(9,16),(9,17),(16,18),(17,19),
    ]
    for a, b in _sx_ik_edges:
        if a in sx_local and b in sx_local:
            sx_edge_list.append((sx_local[a], sx_local[b]))
    sx_pts = {sx_idx: sx_pts_c[i] for i, sx_idx in enumerate(sx_ik_indices)}

    # ── Load MHR ──────────────────────────────────────────────────────────────
    mhr_model = load_mhr_model(device)
    dev       = torch.device(device)

    pos_apose, R_rest, skel_apose = get_mhr_apose(mhr_model, dev)

    skel_tpose, pos_tpose, _ = optimise_mhr_tpose(
        mhr_model, smplx_joints, dev, iters=1500)

    # ── Compute R_corrector = R_world_tp.inv() ────────────────────────────────
    R_corrector = compute_r_corrector(skel_tpose)

    # ── Build display skeletons ───────────────────────────────────────────────
    apose_pts, apose_edges, apose_root_m, apose_scale = _make_cloud(pos_apose, ik_names, DISP_H)
    tpose_pts, tpose_edges, tpose_root_m, tpose_scale = _make_cloud(pos_tpose, ik_names, DISP_H)

    # ── Geometry ──────────────────────────────────────────────────────────────
    geoms = []
    x1, x2, x3, x4, x5 = -2*GAP, -GAP, 0.0, GAP, 2*GAP

    # ── COL 1: SMPL-X T-pose  — identity arrows (the TARGET) ─────────────────
    pts1 = [sx_pts[sx_idx].copy() + np.array([x1, 0, 0]) for sx_idx in sx_ik_indices]
    geoms.append(build_lineset(pts1, sx_edge_list, color=[0.2, 0.5, 1.0]))
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pt = sx_pts[sx_idx].copy() + np.array([x1, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.6, 1.0)))
        geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)
    _add_finger_geoms_smplx(geoms, smplx_joints, sx_root, sx_scale, x1, [0.3, 0.6, 1.0])

    # ── COL 2: MHR A-pose  — raw R_world (chaotic) ───────────────────────────
    pts2 = [p + np.array([x2, 0, 0]) for p in apose_pts]
    geoms.append(build_lineset(pts2, apose_edges, color=[0.9, 0.2, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = apose_pts[i] + np.array([x2, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.3, 0.15)))
        geoms += make_axis_arrows(pt, R_rest[ik_name].as_matrix(), length=axis_len)
    _add_finger_geoms_mhr(geoms, skel_apose, apose_root_m, apose_scale, x2, [0.9, 0.2, 0.1])

    # ── COL 3: MHR T-pose  — raw R_world_tp (still chaotic) ─────────────────
    pts3 = [p + np.array([x3, 0, 0]) for p in tpose_pts]
    geoms.append(build_lineset(pts3, tpose_edges, color=[0.85, 0.55, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tpose_pts[i] + np.array([x3, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.65, 0.2)))
        q_xyzw     = skel_tpose[mhr_idx, 3:7]   # world-frame directly
        R_world_tp = R.from_quat(q_xyzw)
        geoms += make_axis_arrows(pt, R_world_tp.as_matrix(), length=axis_len)
    _add_finger_geoms_mhr(geoms, skel_tpose, tpose_root_m, tpose_scale, x3, [0.85, 0.55, 0.1])

    # ── COL 4: MHR T-pose  — R_world_tp @ R_corrector = I ───────────────────
    #   Should look identical to COL 1 (world-aligned arrows).
    #   By construction: R_corrector = R_world_tp.inv(), so the product = I exactly.
    #   Right-mult is the correct order: R_world @ R_corrector (not left-mult).
    pts4 = [p + np.array([x4, 0, 0]) for p in tpose_pts]
    geoms.append(build_lineset(pts4, tpose_edges, color=[0.1, 0.7, 0.2]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tpose_pts[i] + np.array([x4, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.85, 0.3)))
        q_xyzw     = skel_tpose[mhr_idx, 3:7]   # world-frame directly
        R_world_tp = R.from_quat(q_xyzw)
        R_aligned  = (R_world_tp * R_corrector[ik_name]).as_matrix()
        geoms += make_axis_arrows(pt, R_aligned, length=axis_len)
    _add_finger_geoms_mhr(geoms, skel_tpose, tpose_root_m, tpose_scale, x4, [0.1, 0.7, 0.2])

    # ── COL 5: MHR A-pose  — R_rest @ R_corrector = R_AtoT ──────────────────
    #   Right-mult: R_rest @ R_corrector = R_rest @ R_world_tp.inv() = R_AtoT
    #   R_AtoT is NOT small (up to ~54°) — it encodes the true bone orientation
    #   difference between MHR A-pose and T-pose.
    #   Arrows will be chaotic, NOT world-aligned — this is correct and expected.
    pts5 = [p + np.array([x5, 0, 0]) for p in apose_pts]
    geoms.append(build_lineset(pts5, apose_edges, color=[0.6, 0.2, 0.9]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = apose_pts[i] + np.array([x5, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.75, 0.3, 1.0)))
        R_a = (R_rest[ik_name] * R_corrector[ik_name]).as_matrix()
        geoms += make_axis_arrows(pt, R_a, length=axis_len)
    _add_finger_geoms_mhr(geoms, skel_apose, apose_root_m, apose_scale, x5, [0.6, 0.2, 0.9])

    # ── Reference triad ───────────────────────────────────────────────────────
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[-2*GAP - 0.3, -1.2, 0]))

    # ── Terminal legend ───────────────────────────────────────────────────────
    print("═"*72)
    print("  5-COLUMN VIEW  (algorithm:  R_corrector = R_world_tp.inv())")
    print("═"*72)
    print(f"  COL 1  x={x1:+.1f}  BLUE    SMPL-X T-pose  — identity arrows  (TARGET)")
    print(f"  COL 2  x={x2:+.1f}  RED     MHR A-pose     — raw R_world (chaotic)")
    print(f"  COL 3  x={x3:+.1f}  ORANGE  MHR T-pose     — R_world_tp after IK (still chaotic)")
    print(f"  COL 4  x={x4:+.1f}  GREEN   MHR T-pose     — R_world_tp @ R_corrector = I  (right-mult)")
    print(f"                             Should look identical to COL 1.")
    print(f"  COL 5  x={x5:+.1f}  PURPLE  MHR A-pose     — R_rest @ R_corrector = R_AtoT  (right-mult)")
    print(f"                             R_AtoT encodes A-pose vs T-pose difference.")
    print(f"                             Arrows will be chaotic — NOT world-aligned. This is correct.")
    print("═"*72)
    print()
    print("  CHECK: COL 4 arrows should match COL 1 arrows exactly.")
    print("         COL 4 != COL 1 means IK did not converge.")
    print("═"*72 + "\n")

    # ── Draw ──────────────────────────────────────────────────────────────────
    o3d.visualization.draw_geometries(
        geoms,
        window_name="MHR offsets2 — R_corrector = R_world_tp.inv()",
        mesh_show_back_face=True,
        width=5500, height=1000,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_len", type=float, default=0.10)
    args = parser.parse_args()
    visualise(axis_len=args.axis_len)
