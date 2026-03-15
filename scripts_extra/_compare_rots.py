"""
Quick comparison: MHR world rots vs FIT3D SMPLX world rots at same frame.
Prints ZYX Euler angles for both sides to diagnose the source of offset.

Usage: conda run -n mhr_new python scripts_extra/_compare_rots.py --frame 100
"""
import sys, argparse
sys.path.insert(0, '/home/haziq/GMR')
sys.path.insert(0, '/home/haziq/MHR')
import numpy as np, torch, json
from scipy.spatial.transform import Rotation as R

# ── setup ────────────────────────────────────────────────────────────────────
MHR_NPZ   = "/home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz"
SMPLX_JSON= "/home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json"
SMPLX_PATH= "/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/smplx"

_MATCHED_PAIRS = [
    # (smplx_idx, smplx_name, mhr_idx, mhr_name,  ik_name)
    (0,  "pelvis",         1,  "root",    "root"),
    (1,  "left_hip",       2,  "l_upleg", "l_upleg"),
    (2,  "right_hip",     18,  "r_upleg", "r_upleg"),
    (4,  "left_knee",      3,  "l_lowleg","l_lowleg"),
    (5,  "right_knee",    19,  "r_lowleg","r_lowleg"),
    (9,  "spine3",        37,  "c_spine3","c_spine3"),
    (16, "left_shoulder", 75,  "l_uparm", "l_uparm"),
    (17, "right_shoulder",39,  "r_uparm", "r_uparm"),
    (18, "left_elbow",    76,  "l_lowarm","l_lowarm"),
    (19, "right_elbow",   40,  "r_lowarm","r_lowarm"),
]
_SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=100)
    args = p.parse_args()
    frame = args.frame

    # ── Load MHR ─────────────────────────────────────────────────────────────
    print("[MHR] loading model ...")
    from mhr.mhr import MHR
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
    data    = np.load(MHR_NPZ, allow_pickle=True)
    T       = data["param_lbs_model_params"].shape[0]
    f       = min(frame, T-1)
    mp      = torch.tensor(data["param_lbs_model_params"][f:f+1], dtype=torch.float32)
    sp      = torch.tensor(data["param_identity_coeffs"][f:f+1],  dtype=torch.float32)
    ep      = torch.tensor(data["param_face_expr_coeffs"][f:f+1], dtype=torch.float32)
    print(f"[MHR] model_params[0:9] at frame {f}: {mp[0,:9].numpy().round(3)}")
    with torch.no_grad():
        _, skel = mhr_model(sp, mp, ep)
    skel_np = skel[0].cpu().numpy()   # (127, 8)  global quats (xyzw)

    # ── Load SMPLX (Fit3D) ───────────────────────────────────────────────────
    print("[SMPLX] loading ...")
    import smplx as smplx_lib
    from smplx.joint_names import JOINT_NAMES

    def _rotmat_to_rotvec(arr):
        shape = arr.shape[:-2]
        return R.from_matrix(arr.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)
    with open(SMPLX_JSON) as fh:
        jsdata = json.load(fh)
    N = np.array(jsdata["transl"]).shape[0]
    fi = min(frame, N-1)
    root_orient = _rotmat_to_rotvec(np.array(jsdata["global_orient"])[fi:fi+1, 0]).astype(np.float32)  # (1,3)
    pose_body_m = np.array(jsdata["body_pose"])[fi:fi+1]  # (1,21,3,3) rot mats
    pose_body_aa= _rotmat_to_rotvec(pose_body_m).reshape(1,-1).astype(np.float32)  # (1,63)
    betas_arr   = np.array(jsdata["betas"])[fi:fi+1]
    betas       = np.pad(np.mean(betas_arr, axis=0), (0,6)).astype(np.float32)

    body_model = smplx_lib.SMPLX(model_path=SMPLX_PATH, gender="neutral", use_pca=False, num_betas=len(betas))
    parents    = body_model.parents
    joint_names= JOINT_NAMES[:len(parents)]

    # FK to get world rotations
    n_joints = 22  # body only
    local_rots = [None] * n_joints
    local_rots[0] = R.from_rotvec(root_orient[0])
    for i in range(1, n_joints):
        local_rots[i] = R.from_rotvec(pose_body_aa[0][(i-1)*3 : i*3])
    world_rots_sx = [None] * n_joints
    world_rots_sx[0] = local_rots[0]
    for i in range(1, n_joints):
        par = int(_SMPLX_PARENTS[i])
        world_rots_sx[i] = world_rots_sx[par] * local_rots[i]

    # ── Print comparison ──────────────────────────────────────────────────────
    W = 130
    print()
    print("=" * W)
    print(f"  Frame {frame}: MHR global (world) rotation  vs  Fit3D SMPLX FK world rotation")
    print(f"  {'joint pair':<26}  {'MHR ZYX (deg)':<36}  {'SMPLX ZYX (deg)':<36}  {'|diff|':>8}")
    print("-" * W)
    for sx_idx, sx_name, mhr_idx, mhr_name, ik_name in _MATCHED_PAIRS:
        mhr_R  = R.from_quat(skel_np[mhr_idx, 3:7])   # xyzw
        sx_R   = world_rots_sx[sx_idx]
        mhr_e  = mhr_R.as_euler("zyx", degrees=True)
        sx_e   = sx_R.as_euler("zyx", degrees=True)
        diff   = (sx_R * mhr_R.inv()).magnitude() * 180 / np.pi
        label  = f"{mhr_name} ↔ {sx_name}"
        print(f"  {label:<26}  MHR =[{mhr_e[0]:+7.1f},{mhr_e[1]:+7.1f},{mhr_e[2]:+7.1f}]"
              f"  SX  =[{sx_e[0]:+7.1f},{sx_e[1]:+7.1f},{sx_e[2]:+7.1f}]"
              f"  diff={diff:7.1f}°")
    print("=" * W)
    print()
    print("  NOTE: if diff is ~constant per joint → local-frame convention diff (fixable with R_AtoT)")
    print("        if diff varies/asymmetric      → independent global orientations don't match")
    print()

    # Also print raw model_params global rotation (indices 3:6)
    print(f"  MHR global rot axis-angle at frame {frame}: {mp[0,3:6].numpy()}")
    print(f"  SMPLX global_orient at frame {frame}: {root_orient[0]}")
    angle_diff = np.degrees((R.from_rotvec(mp[0,3:6].numpy()) * R.from_rotvec(root_orient[0]).inv()).magnitude())
    print(f"  Root global orientation diff (MHR vs SMPLX): {angle_diff:.2f} deg")

if __name__ == "__main__":
    main()
