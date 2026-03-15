"""Quick check: are MHR skel[:,3:7] local or world rotations?"""
import sys; sys.path.insert(0, '/home/haziq/GMR'); sys.path.insert(0, '/home/haziq/MHR')
import numpy as np, torch
from scipy.spatial.transform import Rotation as R

device = "cpu"
from mhr.mhr import MHR
mhr_model = MHR.from_files(device=torch.device(device), lod=1)
print("[MHR] loaded")

_JOINT_MAP = {
     0: ("root",            1, "root"),
     1: ("left_hip",        2, "l_upleg"),
     2: ("right_hip",      18, "r_upleg"),
     4: ("left_knee",       3, "l_lowleg"),
     9: ("c_spine3",       37, "c_spine3"),
    16: ("left_shoulder",  75, "l_uparm"),
    18: ("left_elbow",     76, "l_lowarm"),
}

with torch.no_grad():
    sp = torch.zeros(1,45,device=device)
    ep = torch.zeros(1,72,device=device)
    mp = torch.zeros(1,204,device=device)
    _, skel = mhr_model(sp, mp, ep)   # same call as get_mhr_apose
skel_np = skel[0].cpu().numpy()

print("\nA-pose (model_params=0) — column 3:7 quaternion (xyzw) and angle:")
for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
    q  = skel_np[mhr_idx, 3:7]          # xyzw
    Rq = R.from_quat(q)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rq.as_matrix()) - 1) / 2, -1, 1)))
    print(f"  {ik_name:<14}  q={np.round(q,4)}   angle_from_I={ang:.1f} deg")

# If LOCAL: expect near-zero angle (identity at rest pose)
# If WORLD: expect non-zero angle for arm joints (A-pose arms angled)
print("\n  NOTE: if LOCAL rots, angles should be ~0 for all joints at zero model_params.")
print("        if WORLD rots, arm angles should be substantial (arm tilted in A-pose).")

# Also check: what does the left should position look like?
print("\nLeft shoulder position (A-pose):", skel_np[75, :3], " cm")
print("Root position (A-pose):", skel_np[1, :3], " cm")
