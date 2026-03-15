"""
Visualize the MHR rotation correction pipeline and verify it on arbitrary poses.

Method  (T-pose quaternion — exact)
------------------------------------
  R_AtoT[j] = R_norm_tpose[j]⁻¹   where   R_norm_tpose = R_rest⁻¹ · R_world_tpose

Full correction chain used in IK:
  R_target = rot_offset · R_AtoT · R_rest⁻¹ · R_world

Six-column view (always):
  COL 1 — SMPL-X T-pose identity arrows         (target)
  COL 2 — MHR A-pose raw R_world                (chaotic)
  COL 3 — MHR A-pose R_norm = I                 (normalized)
  COL 4 — MHR ~T-pose R_norm_tpose              (residual before fix)
  COL 5 — MHR ~T-pose R_AtoT·R_norm = I        (matches COL 1)
  COL 6 — MHR ~T-pose rot_offset·R_AtoT·R_norm (GMR IK target)

Optional COL 7 — supply an MHR NPZ + frame to see the full correction on an
  actual motion frame.  Compare its arrows to the AFTER column in
  visualize_smpl_rot_offsets.py at the same frame for a cross-check.

Usage (mhr_new env):
# rest pose / T-pose (6 columns)
python /home/haziq/GMR/scripts_extra/visualize_mhr_rot_offsets.py

# add COL 7 with actual pose sanity check
python /home/haziq/GMR/scripts_extra/visualize_mhr_rot_offsets.py \
    --mhr_file /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz \
    --frame 100 --axis_len 0.10

conda run -n mhr_new python /home/haziq/GMR/scripts_extra/visualize_mhr_rot_offsets.py \
  --smplx_file /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
  --mhr_file /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz   \
  --frame 100 --axis_len 0.10

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


def _rotmat_to_rotvec(arr):
    shape = arr.shape[:-2]
    return R.from_matrix(arr.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


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


import json as _json_mod

def load_smplx_pose_frame(json_file, smplx_model_path, frame=0):
    """
    Load one frame from a Fit3D SMPL-X JSON and return per-IK-joint world
    rotations and positions, keyed by ik_name (matching _JOINT_MAP).

    Returns:
        positions : dict  ik_name → (3,)   world position (metres)
        world_rots: dict  ik_name → (3,3)  world rotation matrix
    """
    import smplx as smplx_lib
    from smplx.joint_names import JOINT_NAMES

    with open(json_file) as f:
        data = _json_mod.load(f)

    N = np.array(data["transl"]).shape[0]
    f_idx = min(frame, N - 1)
    print(f"[load_smplx] {N} frames, using frame {f_idx}  ({json_file})")

    transl        = np.array(data["transl"])[f_idx:f_idx+1]
    global_orient = np.array(data["global_orient"])[f_idx:f_idx+1]   # (1,1,3,3)
    body_pose_mat = np.array(data["body_pose"])[f_idx:f_idx+1]       # (1,21,3,3)
    betas_arr     = np.array(data["betas"])[f_idx:f_idx+1]
    lhand_mat     = np.array(data["left_hand_pose"])[f_idx:f_idx+1]
    rhand_mat     = np.array(data["right_hand_pose"])[f_idx:f_idx+1]
    jaw_mat       = np.array(data["jaw_pose"])[f_idx:f_idx+1]
    leye_mat      = np.array(data["leye_pose"])[f_idx:f_idx+1]
    reye_mat      = np.array(data["reye_pose"])[f_idx:f_idx+1]
    expr          = np.array(data["expression"])[f_idx:f_idx+1]

    root_orient = _rotmat_to_rotvec(global_orient[:, 0]).astype(np.float32)
    pose_body   = _rotmat_to_rotvec(body_pose_mat).reshape(1, -1).astype(np.float32)
    lhand_pose  = _rotmat_to_rotvec(lhand_mat).reshape(1, -1).astype(np.float32)
    rhand_pose  = _rotmat_to_rotvec(rhand_mat).reshape(1, -1).astype(np.float32)
    jaw_pose    = _rotmat_to_rotvec(jaw_mat[:, 0]).astype(np.float32)
    leye_pose   = _rotmat_to_rotvec(leye_mat[:, 0]).astype(np.float32)
    reye_pose   = _rotmat_to_rotvec(reye_mat[:, 0]).astype(np.float32)

    betas = np.mean(betas_arr, axis=0)
    betas_padded = np.pad(betas, (0, 6), mode="constant").astype(np.float32)

    body_model = smplx_lib.SMPLX(
        model_path=smplx_model_path, gender="neutral",
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

    joints     = out.joints[0].detach().numpy()                       # (J, 3)
    full_pose  = out.full_pose[0].reshape(-1, 3).detach().numpy()    # (J, 3) aa
    g_orient   = out.global_orient[0].detach().numpy().reshape(3)

    joint_names = JOINT_NAMES[:len(body_model.parents)]
    parents     = body_model.parents

    # Accumulate world-frame rotations via FK
    world_rots_list = []
    for i, jname in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(g_orient)
        else:
            rot = world_rots_list[parents[i]] * R.from_rotvec(full_pose[i])
        world_rots_list.append(rot)

    # Key everything by ik_name using _JOINT_MAP
    positions  = {}
    world_rots = {}
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        positions[ik_name]  = joints[sx_idx]                          # (3,) metres
        world_rots[ik_name] = world_rots_list[sx_idx].as_matrix()    # (3,3)
    return positions, world_rots


# ── MHR helpers ───────────────────────────────────────────────────────────────

def load_mhr_model(device):
    from mhr.mhr import MHR
    print("[MHR] Loading model ...")
    return MHR.from_files(device=torch.device(device), lod=1)


def load_mhr_npz_frame(mhr_model, mhr_file, frame=0, device="cpu"):
    """
    Load one frame from an MHR NPZ file.
    Returns skel_np (127, 8) and pos_frame dict (ik_name → metres).
    """
    dev = torch.device(device)
    data = np.load(mhr_file, allow_pickle=True)
    T    = data["param_lbs_model_params"].shape[0]
    f    = min(frame, T - 1)
    print(f"[load_mhr_npz] {T} frames, using frame {f}  ({mhr_file})")

    model_p = torch.tensor(data["param_lbs_model_params"][f:f+1], dtype=torch.float32).to(dev)
    shape_p = torch.tensor(data["param_identity_coeffs"][f:f+1],  dtype=torch.float32).to(dev)
    expr_p  = torch.tensor(data["param_face_expr_coeffs"][f:f+1], dtype=torch.float32).to(dev)

    with torch.no_grad():
        _, skel = mhr_model(shape_p, model_p, expr_p)
    skel_np = skel[0].cpu().numpy()   # (127, 8)

    pos_frame = {}
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        pos_frame[ik_name] = skel_np[mhr_idx, :3] / 100.0
    return skel_np, pos_frame


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
    return skel_np, pos_tpose, model_params.detach()


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

    # ── Sanity check: R_AtoT · R_rest⁻¹ should equal R_world_tp⁻¹ ─────────
    # These are the "combined corrector" values that visualize_mhr_offsets2.py
    # prints as R_corrector.  If both files are correct, the numbers match.
    print("\n── Sanity check: R_AtoT · R_rest⁻¹  (= R_world_tp⁻¹ = R_corrector) ──")
    print(f"  {'joint':<18}  {'angle':>6}    wxyz quaternion")
    print("  " + "-"*60)
    for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
        R_combined = R_AtoT[ik_name] * R_rest[ik_name].inv()
        angle_deg  = np.degrees(np.linalg.norm(R_combined.as_rotvec()))
        q          = R_combined.as_quat(scalar_first=True)
        print(f"  {ik_name:<18}  {angle_deg:5.1f} deg   "
              f"[{q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}]")
    print("  (these should match the R_corrector values in visualize_mhr_offsets2.py)")

    return R_AtoT


# ── Pose-comparison visualisation ────────────────────────────────────────────

def pose_compare_visualise(sx_positions, sx_world_rots, R_rest, R_AtoT,
                           skel_frame, pos_frame, axis_len=0.08,
                           frame_label=""):
    """
    Five-column per-frame comparison (used when --smplx_file + --mhr_file given).

    COL 1 (blue,  x=-2*GAP): SMPL-X frame N — rot_offset·R_world_sx
        What GMR uses as the IK orientation target from the SMPL-X side.

    COL 2 (red,   x=-GAP)  : MHR frame N — raw R_world (chaotic).

    COL 3 (yellow,x=0)     : MHR frame N — R_norm = R_rest⁻¹·R_world.

    COL 4 (purple,x=+GAP)  : MHR frame N — R_AtoT·R_norm.

    COL 5 (orange,x=+2*GAP): MHR frame N — rot_offset·R_AtoT·R_norm  ← final IK target.
        Should match COL 1 if the correction is working.
    """
    GAP    = 1.5
    DISP_H = 1.0

    ik_names_ordered = [ik_name for _, (_, _, ik_name) in _JOINT_MAP.items()]

    def _make_cloud(pos_dict):
        """Centred, height-normalised point list + edge list."""
        pts_arr  = np.array([pos_dict[n] for n in ik_names_ordered])
        root_p   = pos_dict["root"]
        pts_c    = pts_arr - root_p
        max_y    = pts_c[:, 1].max()
        h        = (max_y - 0.0) + 1e-9
        pts_c    = pts_c * (DISP_H / h)
        name_pt  = {n: pts_c[i] for i, n in enumerate(ik_names_ordered)}
        n_idx    = {n: i for i, n in enumerate(ik_names_ordered)}
        edges    = [(n_idx[a], n_idx[b])
                    for a, b in _MHR_SKEL_EDGES_NAMES
                    if a in n_idx and b in n_idx]
        return [name_pt[n] for n in ik_names_ordered], edges

    # SMPL-X skeleton edges (subset of IK joints)
    sx_ik_indices = list(_JOINT_MAP.keys())
    sx_local      = {sx_idx: i for i, sx_idx in enumerate(sx_ik_indices)}
    _sx_ik_edges  = [
        (0, 1),(0, 2),(1, 4),(2, 5),(4, 7),(5, 8),
        (0, 9),(9,16),(9,17),(16,18),(17,19),
    ]
    sx_edge_list  = [(sx_local[a], sx_local[b]) for a, b in _sx_ik_edges]

    sx_pts_arr = np.array([sx_positions[ik_name] for ik_name in ik_names_ordered])
    sx_root    = sx_positions["root"]
    sx_pts_c   = sx_pts_arr - sx_root
    sx_max_y   = sx_pts_c[:, 1].max()
    sx_h       = (sx_max_y - 0.0) + 1e-9
    sx_pts_c   = sx_pts_c * (DISP_H / sx_h)
    sx_name_pt = {n: sx_pts_c[i] for i, n in enumerate(ik_names_ordered)}

    mhr_ik_pts, mhr_edge_list = _make_cloud(pos_frame)
    fr_ik_pts,  fr_edge_list  = _make_cloud(pos_frame)  # same positions, different rots

    geoms = []
    x1, x2, x3, x4, x5 = -2*GAP, -GAP, 0.0, GAP, 2*GAP

    # ── COL 1:  SMPL-X frame N — rot_offset·R_world_sx ───────────────────────
    sx_pts_list = [sx_name_pt[n] + np.array([x1, 0, 0]) for n in ik_names_ordered]
    geoms.append(build_lineset(sx_pts_list, mhr_edge_list, color=[0.2, 0.5, 1.0]))
    col1_rots = {}
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = sx_name_pt[ik_name] + np.array([x1, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.6, 1.0)))
        q_off    = _IK_ROT_OFFSETS[ik_name]
        R_off    = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
        col1_rot = (R.from_matrix(sx_world_rots[ik_name]) * R_off).as_matrix()
        col1_rots[ik_name] = col1_rot
        geoms   += make_axis_arrows(pt, col1_rot, length=axis_len)

    # ── COL 2:  MHR frame N — raw R_world ──────────────────────────────────
    pts2 = [p + np.array([x2, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(pts2, mhr_edge_list, color=[0.9, 0.2, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x2, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.3, 0.15)))
        q_xyzw   = skel_frame[mhr_idx, 3:7]
        R_world  = R.from_quat(q_xyzw).as_matrix()
        geoms   += make_axis_arrows(pt, R_world, length=axis_len)

    # ── COL 3:  MHR frame N — R_norm = R_rest⁻¹·R_world ────────────────────
    pts3 = [p + np.array([x3, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(pts3, mhr_edge_list, color=[0.8, 0.75, 0.0]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x3, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.9, 0.1)))
        q_xyzw   = skel_frame[mhr_idx, 3:7]
        R_norm   = (R_rest[ik_name].inv() * R.from_quat(q_xyzw)).as_matrix()
        geoms   += make_axis_arrows(pt, R_norm, length=axis_len)

    # ── COL 4:  MHR frame N — R_AtoT·R_norm ─────────────────────────────
    pts4 = [p + np.array([x4, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(pts4, mhr_edge_list, color=[0.6, 0.2, 0.9]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x4, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.75, 0.3, 1.0)))
        q_xyzw   = skel_frame[mhr_idx, 3:7]
        R_norm   = R_rest[ik_name].inv() * R.from_quat(q_xyzw)
        col4_rot = (R_AtoT[ik_name] * R_norm).as_matrix()
        geoms   += make_axis_arrows(pt, col4_rot, length=axis_len)

    # ── COL 5:  MHR frame N — rot_offset·R_AtoT·R_norm  ← should match COL 1 ──
    pts5 = [p + np.array([x5, 0, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(pts5, mhr_edge_list, color=[0.8, 0.5, 0.1]))
    col5_rots = {}
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([x5, 0, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.65, 0.1)))
        q_xyzw   = skel_frame[mhr_idx, 3:7]
        R_norm   = R_rest[ik_name].inv() * R.from_quat(q_xyzw)
        q_off    = _IK_ROT_OFFSETS[ik_name]
        R_off    = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
        col5_rot = ((R_AtoT[ik_name] * R_norm) * R_off).as_matrix()
        col5_rots[ik_name] = col5_rot
        geoms   += make_axis_arrows(pt, col5_rot, length=axis_len)

    # World-frame reference triad (below feet)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=[x1 - 0.3, -1.3, 0]))

    def _print_summary():
        print(f"\n{'='*62}")
        print(f"  Pose-comparison  ({frame_label})")
        print(f"  COL 1 (blue,   x={x1:+.1f}): SMPL-X — rot_offset·R_world_sx   ← REFERENCE")
        print(f"  COL 2 (red,    x={x2:+.1f}): MHR   — raw R_world (chaotic)")
        print(f"  COL 3 (yellow, x={x3:+.1f}): MHR   — R_norm = R_rest⁻¹·R_world")
        print(f"  COL 4 (purple, x={x4:+.1f}): MHR   — R_AtoT·R_norm")
        print(f"  COL 5 (orange, x={x5:+.1f}): MHR   — rot_offset·R_AtoT·R_norm  ← final")
        print(f"{'='*62}")
        print(f"  {'joint':<14}  {'COL5 - COL1 error':>18}  {'ok?':>4}")
        print(f"  {'-'*14}  {'-'*18}  {'-'*4}")
        errs = []
        for ik_name in [ik for _, (_, _, ik) in _JOINT_MAP.items()]:
            R1 = col1_rots[ik_name]
            R5 = col5_rots[ik_name]
            # angle between R1 and R5
            diff = R1.T @ R5
            trace = np.clip(np.trace(diff), -1.0, 3.0)
            err_deg = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
            errs.append(err_deg)
            ok = "✓" if err_deg < 5.0 else ("~" if err_deg < 15.0 else "✗")
            print(f"  {ik_name:<14}  {err_deg:>15.2f} deg  {ok:>4}")
        print(f"  {'-'*14}  {'-'*18}")
        print(f"  {'mean error':<14}  {np.mean(errs):>15.2f} deg")
        print(f"  {'max  error':<14}  {np.max(errs):>15.2f} deg")
        print(f"{'='*62}")
        import sys; sys.stdout.flush()

    _print_summary()
    print("  Opening Open3D window — press Q to close ...\n")
    import sys; sys.stdout.flush()
    o3d.visualization.draw_geometries(
        geoms,
        window_name=(f"Pose compare {frame_label} | "
                     "COL1:SMPLX+off | COL2:MHR raw | COL3:MHR norm | "
                     "COL4:MHR+R_AtoT | COL5:MHR+full  ← match COL1?"),
        mesh_show_back_face=True,
        width=3000, height=1000,
    )
    print("\n  [Window closed]  Results were:")
    _print_summary()


# ── Open3D visualisation (T-pose pipeline) ────────────────────────────────────

def visualise(smplx_joints, pos_apose, R_rest, R_AtoT,
              axis_len=0.08, skel_tpose=None, pos_tpose=None,
              skel_frame=None, pos_frame=None, frame_label="",
              omit_col7=False):
    """
    Six-column layout (+ optional COL 7 when an NPZ frame is provided):

    COL 1 (blue,   x=-2*GAP): SMPL-X T-pose — identity arrows.  THE TARGET.
    COL 2 (red,    x=-GAP)  : MHR A-pose — raw R_world (chaotic).
    COL 3 (yellow, x=0)     : MHR A-pose — R_norm = I.
    COL 4 (green,  x=+GAP)  : MHR ~T-pose — R_norm_tpose (residual).
    COL 5 (purple, x=+2*GAP): MHR ~T-pose — R_AtoT·R_norm = I  ← matches COL 1.
    COL 6 (orange, x=+3*GAP): MHR ~T-pose — rot_offset·R_AtoT·R_norm  ← GMR IK target.
    COL 7 (cyan,   x=+4*GAP): [optional] Actual MHR pose — rot_offset·R_AtoT·R_norm_pose.
        Sanity check: compare with visualize_smpl_rot_offsets at the same frame.
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
        r_raw_tp  = {}   # raw world rots at T-pose before normalization (the missing intermediate)
        for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
            q_xyzw = skel_tpose[mhr_idx, 3:7]
            R_world_tp   = R.from_quat(q_xyzw)
            r_raw_tp[ik_name]  = R_world_tp.as_matrix()
            r_norm_tp[ik_name] = (R_rest[ik_name].inv() * R_world_tp).as_matrix()
    else:
        tp_ik_pts    = mhr_ik_pts
        tp_edge_list = mhr_edge_list
        r_norm_tp    = {ik_name: np.eye(3) for ik_name in ik_names_ordered}
        r_raw_tp     = {ik_name: R_rest[ik_name].as_matrix() for ik_name in ik_names_ordered}

    geoms = []

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0 (y = +3.2)  — OFFLINE ALGORITHM, one panel per equation step
    #
    #  S1  x=-2.5G  [A-pose skel]  arrows = R_rest           (raw bias)
    #  S2  x=-1.5G  [T-pose skel]  arrows = R_world_tpose    (IK output, still chaotic)
    #  S3  x=-0.5G  [A-pose skel]  arrows = R_rest^-1        (what the inverse looks like)
    #  S4  x=+0.5G  [T-pose skel]  arrows = R_rest^-1 · R_world_tpose = R_norm  (residual)
    #  S5  x=+1.5G  [T-pose skel]  arrows = R_AtoT · R_norm  (= I, corrected)
    #  S6  x=+2.5G  [T-pose skel]  arrows = R_AtoT · R_norm · R_offset  (final target)
    #
    # Open3D draw_geometries has no text API — see terminal legend printed below.
    # ══════════════════════════════════════════════════════════════════════════
    ROW0_Y = 3.2
    R0G = GAP        # re-use same GAP; panels at ±0.5/1.5/2.5 × GAP
    s1x = -2.5*R0G
    s2x = -1.5*R0G
    s3x = -0.5*R0G
    s4x = +0.5*R0G
    s5x = +1.5*R0G
    s6x = +2.5*R0G

    # S1 — R_rest  (A-pose skeleton, raw chaotic arrows = the bias frame)
    s1_pts = [p + np.array([s1x, ROW0_Y, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(s1_pts, mhr_edge_list, color=[0.9, 0.2, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([s1x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.3, 0.15)))
        geoms += make_axis_arrows(pt, R_rest[ik_name].as_matrix(), length=axis_len)

    # S2 — IK output at T-pose: R_world_tpose, still chaotic (the missing intermediate)
    s2_pts = [p + np.array([s2x, ROW0_Y, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(s2_pts, tp_edge_list, color=[0.85, 0.55, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([s2x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.65, 0.2)))
        geoms += make_axis_arrows(pt, r_raw_tp[ik_name], length=axis_len)

    # S3 — R_rest^-1  (A-pose skeleton; arrows are R_rest transposed — the "un-bias" tool)
    #      Visually: same joints, arrows now point in the directions that UNDO S1's chaos.
    s3_pts = [p + np.array([s3x, ROW0_Y, 0]) for p in mhr_ik_pts]
    geoms.append(build_lineset(s3_pts, mhr_edge_list, color=[0.2, 0.7, 0.8]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = mhr_ik_pts[i] + np.array([s3x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.3, 0.85, 0.95)))
        geoms += make_axis_arrows(pt, R_rest[ik_name].inv().as_matrix(), length=axis_len)

    # S4 — R_norm = R_rest^-1 · R_world_tpose  (T-pose skeleton; residual arrows)
    #      = S3 applied to S2: the IK-output chaos minus the rest-pose bias.
    s4_pts = [p + np.array([s4x, ROW0_Y, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(s4_pts, tp_edge_list, color=[0.1, 0.7, 0.2]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([s4x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.85, 0.3)))
        geoms += make_axis_arrows(pt, r_norm_tp[ik_name], length=axis_len)

    # S5 — R_AtoT · R_norm = I  (T-pose skeleton; arrows ≈ identity = world axes)
    s5_pts = [p + np.array([s5x, ROW0_Y, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(s5_pts, tp_edge_list, color=[0.6, 0.2, 0.9]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([s5x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(0.75, 0.3, 1.0)))
        s5_rot = (R_AtoT[ik_name] * R.from_matrix(r_norm_tp[ik_name])).as_matrix()
        geoms += make_axis_arrows(pt, s5_rot, length=axis_len)

    # S6 — R_AtoT · R_norm · R_offset  (T-pose skeleton; final GMR IK-target convention)
    s6_pts = [p + np.array([s6x, ROW0_Y, 0]) for p in tp_ik_pts]
    geoms.append(build_lineset(s6_pts, tp_edge_list, color=[0.8, 0.5, 0.1]))
    for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
        pt = tp_ik_pts[i] + np.array([s6x, ROW0_Y, 0])
        geoms.append(sphere_at(pt, radius=0.018, color=(1.0, 0.65, 0.1)))
        q_off  = _IK_ROT_OFFSETS[ik_name]
        R_off  = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
        s5_rot = (R_AtoT[ik_name] * R.from_matrix(r_norm_tp[ik_name])).as_matrix()
        s6_rot = (R.from_matrix(s5_rot) * R_off).as_matrix()
        geoms += make_axis_arrows(pt, s6_rot, length=axis_len)

    # Small reference triad for ROW 0
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=[s1x - 0.3, ROW0_Y - 1.0, 0]))

    # ── ROW 0 terminal legend (Open3D draw_geometries has no text API) ─────────
    print("\n" + "═"*80)
    print("  ROW 0 — OFFLINE ALGORITHM  (top row, y = +3.2)")
    print("  Goal: find the transform that makes MHR's arrows = I when MHR is at T-pose,")
    print("        so it matches SMPL-X's axis convention.")
    print("═"*80)
    print(f"  S1  x={s1x:+.1f}  RED")
    print(f"        arrows = R_rest  =  R_world_mhr(A-pose)")
    print(f"        MHR at A-pose. FK gives us world-frame rotation matrices.")
    print(f"        Arrows are chaotic because MHR's joints are not built aligned")
    print(f"        with world axes. This constant structural offset is called R_rest.")
    print()
    print(f"  S2  x={s2x:+.1f}  ORANGE")
    print(f"        arrows = R_world_tp  =  FK output after IK to T-pose")
    print(f"        We ran IK to push MHR's joint POSITIONS to match SMPL-X T-pose.")
    print(f"        The rotations came along passively — still chaotic. R_rest offset")
    print(f"        is still baked in on top of whatever T-pose motion happened.")
    print()
    print(f"  S3  x={s3x:+.1f}  CYAN")
    print(f"        arrows = R_rest^-1")
    print(f"        The inverse of S1. Does the opposite rotation of R_rest at every")
    print(f"        joint. It is the 'constant offset remover' we will use on S2.")
    print(f"        On its own it's still chaotic — it becomes useful when composed.")
    print()
    print(f"  S4  x={s4x:+.1f}  GREEN")
    print(f"        arrows = R_norm_tp  =  R_rest^-1 · R_world_tp")
    print(f"        S3 composed with S2: R_rest^-1 cancels the constant A-pose offset,")
    print(f"        leaving only the T-pose residual — how far MHR drifted from I")
    print(f"        when it ran IK. Close to I but not quite. (R_rest^-1 · R_rest = I,")
    print(f"        so at A-pose this would be exactly I — verified by COL 3 main row.)")
    print()
    print(f"  S5  x={s5x:+.1f}  PURPLE")
    print(f"        arrows = R_AtoT · R_norm_tp  =  R_norm_tp^-1 · R_norm_tp  =  I")
    print(f"        R_AtoT = R_norm_tp^-1 was built specifically to cancel the S4")
    print(f"        residual. Result is I — world-aligned arrows. This is the goal.")
    print(f"        R_AtoT is saved as a constant and used at runtime every frame.")
    print()
    print(f"  S6  x={s6x:+.1f}  YELLOW")
    print(f"        arrows = R_AtoT · R_norm_tp · R_offset")
    print(f"        Final step: R_offset fixes the axis convention mismatch between")
    print(f"        MHR and GMR's IK solver (Y-up vs Z-up etc). This is what GMR")
    print(f"        actually receives as its rotation target.")
    print()
    print("  RUNTIME (every frame):")
    print("    R_target = R_AtoT  ·  R_rest^-1  ·  R_world_current  ·  R_offset")
    print("               ^^^^^^     ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^")
    print("               cancel     remove         raw FK output for")
    print("               residual   constant       this frame")
    print("               offset     offset")
    print("═"*80 + "\n")

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

    # ── COL 7 (x = +4*GAP):  MHR A-pose — full correction rot_offset·R_AtoT ──
    # At A-pose R_norm = R_rest⁻¹·R_rest = I, so the full chain collapses to:
    #   rot_offset · R_AtoT · I  =  rot_offset · R_AtoT
    # Arrows should match COL 6 (same T-pose target) — confirms R_AtoT works
    # on the original A-pose skeleton, not just the optimised T-pose shape.
    #
    # Overlay: SMPL-X T-pose (white skeleton) with rot_offset arrows.
    # At SMPL-X T-pose R_world = I  →  IK target = rot_offset · I = rot_offset.
    # All three (COL 6 orange, COL 7 cyan, overlay white) should show identical
    # arrows — a three-way visual confirmation.
    if not omit_col7:
        x7 = 4 * GAP
        # ── MHR A-pose (cyan) ──────────────────────────────────────────────────
        mhr_shifted7 = [p + np.array([x7, 0, 0]) for p in mhr_ik_pts]
        geoms.append(build_lineset(mhr_shifted7, mhr_edge_list, color=[0.0, 0.8, 0.8]))
        for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
            pt = mhr_ik_pts[i] + np.array([x7, 0, 0])
            geoms.append(sphere_at(pt, radius=0.018, color=(0.0, 0.9, 0.9)))
            # At A-pose: R_norm = I  →  full chain = rot_offset · R_AtoT
            q_off    = _IK_ROT_OFFSETS[ik_name]
            R_off    = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
            col7_rot = (R_AtoT[ik_name] * R_off).as_matrix()
            geoms   += make_axis_arrows(pt, col7_rot, length=axis_len)
        # ── SMPL-X T-pose overlay (white) — rot_offset arrows ─────────────────
        sx_shifted7 = [sx_pts[idx].copy() + np.array([x7, 0, 0]) for idx in sx_ik_indices]
        geoms.append(build_lineset(sx_shifted7, sx_edge_list, color=[0.9, 0.9, 0.9]))
        for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
            pt = sx_pts[sx_idx].copy() + np.array([x7, 0, 0])
            geoms.append(sphere_at(pt, radius=0.012, color=(1.0, 1.0, 1.0)))
            q_off    = _IK_ROT_OFFSETS[ik_name]
            R_off    = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
            # SMPL-X T-pose: R_world = I  →  IK target = rot_offset · I = R_off
            geoms   += make_axis_arrows(pt, R_off.as_matrix(), length=axis_len)

    # ── ROW 2: direct comparison of COL 1 vs COL 7 (centred, side by side) ──
    # Placed below the main row so you can see COL 1 and COL 7 next to each other
    # without scrolling across 7 columns.  White SMPL-X overlay is included.
    if not omit_col7:
        ROW2_Y  = -3.2   # vertical shift below the main row
        r2_xA   = -GAP   # left  — copy of COL 1
        r2_xB   = +GAP   # right — copy of COL 7 (cyan MHR + white SMPL-X)

        # ── ROW2 left: SMPL-X T-pose + identity arrows (blue) ─────────────────
        sx_r2 = [sx_pts[idx].copy() + np.array([r2_xA, ROW2_Y, 0]) for idx in sx_ik_indices]
        geoms.append(build_lineset(sx_r2, sx_edge_list, color=[0.2, 0.5, 1.0]))
        for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
            pt = sx_pts[sx_idx].copy() + np.array([r2_xA, ROW2_Y, 0])
            geoms.append(sphere_at(pt, radius=0.018, color=(0.2, 0.6, 1.0)))
            geoms += make_axis_arrows(pt, np.eye(3), length=axis_len)

        # ── ROW2 right: MHR A-pose (cyan) + SMPL-X T-pose overlay (white) ─────
        mhr_r2 = [p + np.array([r2_xB, ROW2_Y, 0]) for p in mhr_ik_pts]
        geoms.append(build_lineset(mhr_r2, mhr_edge_list, color=[0.0, 0.8, 0.8]))
        for i, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
            pt = mhr_ik_pts[i] + np.array([r2_xB, ROW2_Y, 0])
            geoms.append(sphere_at(pt, radius=0.018, color=(0.0, 0.9, 0.9)))
            q_off    = _IK_ROT_OFFSETS[ik_name]
            R_off    = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
            col7_rot = (R_AtoT[ik_name] * R_off).as_matrix()
            geoms   += make_axis_arrows(pt, col7_rot, length=axis_len)

        sx_r2_white = [sx_pts[idx].copy() + np.array([r2_xB, ROW2_Y, 0]) for idx in sx_ik_indices]
        geoms.append(build_lineset(sx_r2_white, sx_edge_list, color=[0.9, 0.9, 0.9]))
        for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
            pt = sx_pts[sx_idx].copy() + np.array([r2_xB, ROW2_Y, 0])
            geoms.append(sphere_at(pt, radius=0.012, color=(1.0, 1.0, 1.0)))
            q_off = _IK_ROT_OFFSETS[ik_name]
            R_off = R.from_quat([q_off[1], q_off[2], q_off[3], q_off[0]])
            geoms += make_axis_arrows(pt, R_off.as_matrix(), length=axis_len)

        # Row-2 reference triad
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.12, origin=[r2_xA - 0.3, ROW2_Y - 1.0, 0]))

    # World-frame reference triad (below feet — main row)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.12, origin=[x1 - 0.3, -1.3, 0]))

    n_cols = 6 if omit_col7 else 7
    print(f"\n[Open3D]  {n_cols}-column view")
    print(f"  COL 1 (blue,   x={x1:+.1f}): SMPL-X T-pose  — identity arrows                    ← TARGET")
    print(f"  COL 2 (red,    x={x2:+.1f}): MHR A-pose     — raw R_world (chaotic)")
    print(f"  COL 3 (yellow, x={x3:+.1f}): MHR A-pose     — R_norm = I (normalized)")
    print(f"  COL 4 (green,  x={x4:+.1f}): MHR ~T-pose    — R_norm_tpose (residual before fix)")
    print(f"  COL 5 (purple, x={x5:+.1f}): MHR ~T-pose    — R_AtoT·R_norm = I                 ← matches COL 1")
    print(f"  COL 6 (orange, x={x6:+.1f}): MHR ~T-pose    — rot_offset·R_AtoT·R_norm          ← GMR IK target")
    if not omit_col7:
        print(f"  COL 7 (cyan,   x={x7:+.1f}): MHR A-pose     — rot_offset·R_AtoT  (R_norm=I)")
        print(f"        (white overlay x={x7:+.1f}): SMPL-X T-pose  — rot_offset only  ← should match cyan")
        print(f"  ROW 2 (below):  left=COL1 copy (blue)  right=COL7 copy (cyan+white)  ← direct comparison")
    print("  ✓ CHECK: COL 5 = COL 3 = COL 1  (all identity arrows)")
    if not omit_col7:
        print("  ✓ CHECK: COL 6 = COL 7 = white overlay  (all show rot_offset arrows)")
    print("  Press Q to quit.\n")
    win_title = ("COL1:SMPLX | COL2:MHR-A raw | COL3:MHR-A norm | COL4:MHR-T norm | "
                 "COL5:MHR-T corrected | COL6:MHR-T +rot_off"
                 + (" | COL7:MHR-A +rot_off" if not omit_col7 else ""))
    win_w = 3600 if omit_col7 else 4200
    # 3 rows (ROW0 +3.2, main 0, ROW2 -3.2) → taller for 7-col mode
    win_h = 1000 if omit_col7 else 2600
    o3d.visualization.draw_geometries(
        geoms,
        window_name=win_title,
        mesh_show_back_face=True,
        width=win_w, height=win_h,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    device = args.device

    smplx_path = args.smplx_path or find_smplx_path()
    if smplx_path is None:
        print("[ERROR] SMPL-X model path not found.  Pass --smplx_path.")
        sys.exit(1)

    mhr_model = load_mhr_model(device)
    pos_apose, R_rest, _ = get_mhr_apose(mhr_model, device)

    # ── Pose-comparison mode: both --smplx_file and --mhr_file supplied ────────
    if args.smplx_file is not None and args.mhr_file is not None:
        print("\n[MODE] Pose-comparison  "
              f"(smplx: {args.smplx_file}  mhr: {args.mhr_file}  frame: {args.frame})")

        # We still need R_AtoT — compute it via T-pose optimisation.
        print(f"\n[SMPLX] Loading T-pose from {smplx_path} ...")
        smplx_joints = load_smplx_tpose(smplx_path, device)
        skel_tpose, _, _ = optimise_mhr_tpose(mhr_model, smplx_joints, device,
                                           iters=args.iters, lr=args.lr,
                                           reg=args.reg)
        R_AtoT = compute_r_ato_t(skel_tpose, R_rest)

        # Load paired pose data.
        sx_positions, sx_world_rots = load_smplx_pose_frame(
            args.smplx_file, smplx_path, frame=args.frame
        )
        skel_frame, pos_frame = load_mhr_npz_frame(
            mhr_model, args.mhr_file, frame=args.frame, device=device
        )
        frame_label = f"frame {args.frame}"

        pose_compare_visualise(
            sx_positions, sx_world_rots, R_rest, R_AtoT,
            skel_frame, pos_frame,
            axis_len=args.axis_len, frame_label=frame_label,
        )
        return

    # ── T-pose pipeline mode (default) ────────────────────────────────────────
    print(f"[SMPLX] Loading from {smplx_path} ...")
    smplx_joints = load_smplx_tpose(smplx_path, device)

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
    skel_tpose, pos_tpose, _ = optimise_mhr_tpose(mhr_model, smplx_joints, device,
                                               iters=args.iters, lr=args.lr,
                                               reg=args.reg)

    # Compute R_AtoT from T-pose quaternions (exact method).
    R_AtoT = compute_r_ato_t(skel_tpose, R_rest)

    # Optionally load an arbitrary MHR NPZ frame for COL 7 sanity check.
    skel_frame = pos_frame = frame_label = None
    if args.mhr_file is not None:
        skel_frame, pos_frame = load_mhr_npz_frame(
            mhr_model, args.mhr_file, frame=args.frame, device=device
        )
        frame_label = f"frame {args.frame}"

    visualise(smplx_joints, pos_apose, R_rest, R_AtoT,
              axis_len=args.axis_len, skel_tpose=skel_tpose, pos_tpose=pos_tpose,
              skel_frame=skel_frame, pos_frame=pos_frame, frame_label=frame_label or "",
              omit_col7=(args.mhr_file is not None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize MHR rotation correction pipeline (6+1 columns).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # rest pose only (6 columns)\n"
            "  python visualize_mhr_rot_offsets.py\n\n"
            "  # add COL 7 with an actual pose for sanity check\n"
            "  python visualize_mhr_rot_offsets.py \\\n"
            "      --mhr_file /tmp/motion.npz --frame 42 --axis_len 0.10\n"
        ),
    )
    p.add_argument("--smplx_file", default=None,
                   help="Fit3D SMPL-X JSON for pose-comparison mode (requires --mhr_file).")
    p.add_argument("--mhr_file",   default=None,
                   help="MHR NPZ file: pose-comparison COL5 when used with --smplx_file, "
                        "else COL7 sanity-check in T-pose mode.")
    p.add_argument("--frame",      type=int, default=0,
                   help="Frame index within the NPZ file (default: 0).")
    p.add_argument("--smplx_path", default=None)
    p.add_argument("--axis_len",   type=float, default=0.08)
    p.add_argument("--iters",      type=int,   default=1500,
                   help="T-pose optimisation iterations (position only, for viz)")
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--reg",        type=float, default=1e-3)
    p.add_argument("--device",     default="cpu")
    main(p.parse_args())
