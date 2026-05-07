"""
vis_mhr_and_robot_rest_pose.py

Visualise the MHR body model at A-pose (all model_params = 0) alongside
the robot at rest-pose (qpos = 0), with coordinate frames drawn at every
joint that is mapped in the IK config.

Run with the 'mhr' conda env (has MHR + mujoco + matplotlib):
    conda run -n mhr python /home/haziq/GMR/scripts/vis_mhr_and_robot_rest_pose.py
    conda run -n mhr python /home/haziq/GMR/scripts/vis_mhr_and_robot_rest_pose.py --robot booster_t1

Two interactive figure windows:

  Figure 1 — MHR A-pose  (two copies side by side)
               LEFT  copy → raw world-frame orientations from the MHR skeleton
                            quaternions (skel[j, 3:7], xyzw convention).
               RIGHT copy → same skeleton but frames AFTER the full normalisation:
                            R_norm = R_rest⁻¹ · R_world.  At A-pose R_world = R_rest,
                            so R_norm = I (identity).  This confirms A-pose is the
                            reference frame — all frames should point world-aligned.

  Figure 2 — Robot rest-pose (single copy)
               Actual body frames at qpos = 0 from MuJoCo.

Calibration goal: the RIGHT copy of Figure 1 (= identity frames) tells you
the frame you need to ROTATE THE IK TARGET TO before sending it to the robot.
The robot body frames in Figure 2 are what the IK solver sees as the "current"
configuration — they must be consistent with the IK target convention.

──────────────────────────────────────────────────────────────────────────────
HOW COORDINATE FRAMES ARE COMPUTED (Q&A)
──────────────────────────────────────────────────────────────────────────────
• SMPL-X:  The model runs Linear Blend Skinning (LBS) + FK internally.
  At each joint the library accumulates local rotation matrices along the
  kinematic tree.  At T-pose (all pose params = 0) each local rotation is I,
  so every world-frame rotation = I (all triads look identical/world-aligned).
  For a general pose the world rotation at joint j is:
      R_world[j] = R_world[parent(j)] · R_local[j]
  The GMR code reads back these world-frame quats from `out.full_pose` after a
  second FK pass (see load_smplx_pose_frame in the old reference script).

• MHR:  The model (via pymomentum) runs its own FK and outputs per-joint
  (position_cm, quaternion_xyzw) directly in the skeleton tensor skel[j, 0:7].
  The quaternion IS already the world-frame orientation — no further FK is
  needed.  At A-pose (model_params = 0) these are the "rest" quaternions
  R_rest[j].  For an arbitrary pose the FK gives the accumulated world-frame
  rotation just like SMPL-X.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.spatial.transform import Rotation as R

# Add MHR to path if not installed as package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MHR"))
sys.path.insert(0, "/home/haziq/MHR")

from mhr.mhr import MHR

# ── paths ──────────────────────────────────────────────────────────────────────
GMR_ROOT = Path(__file__).resolve().parent.parent

CONFIG_MAP = {
    "unitree_g1": GMR_ROOT / "general_motion_retargeting/ik_configs/smplx_to_g1.json",
    "booster_t1": GMR_ROOT / "general_motion_retargeting/ik_configs/smplx_to_t1.json",
    "booster_k1": GMR_ROOT / "general_motion_retargeting/ik_configs/smplx_to_k1.json",
}
XML_MAP = {
    "unitree_g1": GMR_ROOT / "assets/unitree_g1/g1_mocap_29dof.xml",
    "booster_t1": GMR_ROOT / "assets/booster_t1/T1_serial.xml",
    "booster_k1": GMR_ROOT / "assets/booster_k1/K1.xml",
}

MHR_MODEL_DIR = Path("/home/haziq/MHR")

FRAME_LEN = 5.0   # cm  (MHR positions are in cm)

# ── SMPL-X joint idx → (readable_name, MHR_joint_idx, ik_name) ────────────────
# From visualize_mhr_rot_offsets.py in GMR_old
_JOINT_MAP = {
     0: ("pelvis",          1,  "root"),
     1: ("left_hip",        2,  "l_upleg"),
     2: ("right_hip",      18,  "r_upleg"),
     4: ("left_knee",       3,  "l_lowleg"),
     5: ("right_knee",     19,  "r_lowleg"),
     7: ("left_ankle",      8,  "l_ball"),
     8: ("right_ankle",    24,  "r_ball"),
     9: ("spine3",         37,  "c_spine3"),
    16: ("left_shoulder",  75,  "l_uparm"),
    17: ("right_shoulder", 39,  "r_uparm"),
    18: ("left_elbow",     76,  "l_lowarm"),
    19: ("right_elbow",    40,  "r_lowarm"),
}

# MHR joint indices used in the stick-figure skeleton
# (from demo.py comments: root=1, L-leg 2-8, R-leg 18-24, spine 34-37,
#  R-arm 38-42, L-arm 74-78, head 110/112)
MHR_SKEL_EDGES = [
    # legs
    (1, 2),  (2, 3),  (3, 4),  (4, 8),    # root → L leg chain → ball
    (1, 18), (18, 19),(19,20), (20, 24),   # root → R leg chain → ball
    # spine
    (1, 34), (34, 35),(35, 36),(36, 37),   # root → spine0..3
    # R arm (clavicle=38, uparm=39, lowarm=40, wrist_twist=41, wrist=42)
    (37, 38),(38, 39),(39, 40),(40, 42),
    # L arm (clavicle=74, uparm=75, lowarm=76, wrist_twist=77, wrist=78)
    (37, 74),(74, 75),(75, 76),(76, 78),
    # head (neck=110, head=112)
    (37, 110),(110, 112),
]

# All unique joint indices used in the skeleton for extraction
_SKEL_JOINTS = sorted({j for edge in MHR_SKEL_EDGES for j in edge})
# Mapped joint indices (highlighted)
_MAPPED_MHR_IDX = {v[1] for v in _JOINT_MAP.values()}


# ── helpers ────────────────────────────────────────────────────────────────────

def draw_frame(ax, pos, rot_mat, length=FRAME_LEN, lw=2.0):
    """Draw X(red) Y(green) Z(blue) frame from pos."""
    for col, color in enumerate(["r", "g", "b"]):
        tip = pos + rot_mat[:, col] * length
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                color=color, linewidth=lw, zorder=5)
        ax.scatter(*tip, color=color, s=14, zorder=6)


def set_equal_3d(ax, pts, margin=5.0):
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    half   = max((hi - lo).max() / 2, 30.0) + margin
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)


# ── MHR A-pose ─────────────────────────────────────────────────────────────────

def get_mhr_apose(mhr_model_dir: Path):
    """
    Forward-pass MHR at A-pose (all params = 0).  Returns:
        pos_cm  : dict { mhr_joint_idx → (3,) float, cm }
        rots    : dict { mhr_joint_idx → (3,3) world-frame rot matrix }
                  Rotation from skel[j, 3:7] which is xyzw quaternion.
        skel_np : (127, 8) full skeleton array for drawing the stick figure
    """
    model = MHR.from_files(device=torch.device("cpu"), lod=1)
    with torch.no_grad():
        _, skel = model(
            torch.zeros(1, 45),   # identity_coeffs (shape)
            torch.zeros(1, 204),  # model_parameters (pose + scale)
            torch.zeros(1, 72),   # face_expr_coeffs
        )
    skel_np = skel[0].cpu().numpy()  # (127, 8): [:3]=pos_cm, [3:7]=quat_xyzw

    pos_cm = {}
    rots   = {}
    for j in set(_SKEL_JOINTS) | _MAPPED_MHR_IDX:
        pos_cm[j] = skel_np[j, :3]
        rots[j]   = R.from_quat(skel_np[j, 3:7]).as_matrix()  # xyzw

    return pos_cm, rots, skel_np


# ── Robot rest pose from MuJoCo ────────────────────────────────────────────────

def get_robot_rest(xml_path: Path):
    model = mj.MjModel.from_xml_path(str(xml_path))
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    parents, pos, rots = {}, {}, {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None or name == "world":
            continue
        pos[name]  = data.xpos[i].copy()
        rots[name] = R.from_quat(data.xquat[i], scalar_first=True).as_matrix()
        pid   = model.body_parentid[i]
        pname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, pid) if pid >= 0 else None
        parents[name] = pname if (pname and pname != "world") else None
    return pos, rots, parents


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_mhr_skeleton(ax, pos_cm, highlighted_idx,
                      color="steelblue", hl_color="orange", offset=None):
    off = np.array(offset) if offset is not None else np.zeros(3)
    for (a, b) in MHR_SKEL_EDGES:
        if a not in pos_cm or b not in pos_cm:
            continue
        p0 = pos_cm[a] + off
        p1 = pos_cm[b] + off
        c = hl_color if (a in highlighted_idx or b in highlighted_idx) else color
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=c, linewidth=1.8, alpha=0.9)
    for j in pos_cm:
        p = pos_cm[j] + off
        s = 30 if j in highlighted_idx else 12
        c = hl_color if j in highlighted_idx else color
        ax.scatter(*p, color=c, s=s, zorder=4)


def draw_robot_skeleton(ax, body_pos, parents, highlighted,
                        color="steelblue", hl_color="orange"):
    for bname, pname in parents.items():
        if pname is None or bname not in body_pos or pname not in body_pos:
            continue
        p0, p1 = body_pos[pname], body_pos[bname]
        c = hl_color if (bname in highlighted or pname in highlighted) else color
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=c, linewidth=1.8, alpha=0.9)
    for bname in body_pos:
        s = 30 if bname in highlighted else 12
        c = hl_color if bname in highlighted else color
        ax.scatter(*body_pos[bname], color=c, s=s, zorder=4)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1", choices=list(CONFIG_MAP))
    args = parser.parse_args()

    # ── load IK config (smplx_to_robot) ──────────────────────────────────────
    with open(CONFIG_MAP[args.robot]) as f:
        config = json.load(f)

    # Build: smplx_human_name → (robot_body_name, rot_offset_R)
    smplx_human_to_robot = {}
    for robot_body, entry in config["ik_match_table1"].items():
        human_name, pw, rw, _, rot_off_wxyz = entry
        rot_off = R.from_quat(rot_off_wxyz, scalar_first=True)
        smplx_human_to_robot[human_name] = (robot_body, rot_off)

    # Build: mhr_joint_idx → (readable_name, robot_body_name, rot_offset_R)
    # via _JOINT_MAP: smplx_idx → (readable, mhr_idx, ik_name)
    # and smplx_human_to_robot: readable → (robot_body, rot_off)
    mhr_to_robot = {}       # mhr_idx → (readable_name, robot_body_name, rot_off)
    for smplx_idx, (readable, mhr_idx, ik_name) in _JOINT_MAP.items():
        if readable in smplx_human_to_robot:
            robot_body, rot_off = smplx_human_to_robot[readable]
            mhr_to_robot[mhr_idx] = (readable, robot_body, rot_off)

    mapped_mhr_idx   = set(mhr_to_robot.keys())
    mapped_robot_set = {v[1] for v in mhr_to_robot.values()}

    # ── get data ──────────────────────────────────────────────────────────────
    print("Loading MHR A-pose ...")
    mhr_pos_cm, mhr_rots, _ = get_mhr_apose(MHR_MODEL_DIR)
    print("Loading robot rest pose ...")
    robot_pos, robot_rots, robot_parents = get_robot_rest(XML_MAP[args.robot])

    # Compute X offset for the RIGHT copy of the MHR figure
    mhr_xs = [mhr_pos_cm[j][0] for j in mhr_pos_cm]
    mhr_width = max(mhr_xs) - min(mhr_xs)
    MHR_OFFSET = np.array([mhr_width * 1.8 + 20.0, 0.0, 0.0])  # cm

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — MHR A-pose: two copies
    #   LEFT  → raw world-frame orientations (R_rest from skel quaternions)
    #   RIGHT → normalized: R_norm = R_rest⁻¹ · R_world = I at A-pose
    #           (shows that A-pose IS the rest/canonical frame — frames = I)
    # ══════════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(14, 9))
    fig1.canvas.manager.set_window_title("Figure 1 — MHR A-pose (raw vs normalised)")
    fig1.suptitle(
        "MHR — A-pose  (model_params = 0)\n"
        "LEFT: raw world-frame orientations from skeleton quaternions (R_rest)\n"
        "RIGHT (offset): normalised = R_rest⁻¹·R_world = I  (A-pose is the rest frame)\n"
        "Red=X   Green=Y   Blue=Z   |   Units: cm",
        fontsize=10,
    )
    ax1 = fig1.add_subplot(111, projection="3d")

    # LEFT — raw R_rest frames
    draw_mhr_skeleton(ax1, mhr_pos_cm, mapped_mhr_idx,
                      color="steelblue", hl_color="orange")
    for mhr_idx, (readable, robot_body, rot_off) in mhr_to_robot.items():
        if mhr_idx not in mhr_pos_cm:
            continue
        pos = mhr_pos_cm[mhr_idx]
        draw_frame(ax1, pos, mhr_rots[mhr_idx])
        ax1.text(pos[0], pos[1], pos[2] + FRAME_LEN * 1.3,
                 f"{readable}\n(MHR {mhr_idx})",
                 fontsize=5, ha="center", color="darkorange")
    ax1.text2D(0.18, 0.94, "LEFT: raw R_rest frames",
               transform=ax1.transAxes, fontsize=9, ha="center", color="navy",
               bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray"))

    # RIGHT — normalised = identity
    off = MHR_OFFSET
    draw_mhr_skeleton(ax1, mhr_pos_cm, mapped_mhr_idx,
                      color="cornflowerblue", hl_color="gold", offset=off)
    for mhr_idx, (readable, robot_body, rot_off) in mhr_to_robot.items():
        if mhr_idx not in mhr_pos_cm:
            continue
        pos = mhr_pos_cm[mhr_idx] + off
        # R_norm = R_rest⁻¹ · R_rest = I
        draw_frame(ax1, pos, np.eye(3))
        ax1.text(pos[0], pos[1], pos[2] + FRAME_LEN * 1.3,
                 f"{readable}\n(MHR {mhr_idx})",
                 fontsize=5, ha="center", color="goldenrod")
    ax1.text2D(0.82, 0.94, "RIGHT: normalised R_norm = I  (= rest frame)",
               transform=ax1.transAxes, fontsize=9, ha="center", color="navy",
               bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray"))

    all_mhr_pts = (
        list(mhr_pos_cm.values()) +
        [p + off for p in mhr_pos_cm.values()]
    )
    set_equal_3d(ax1, all_mhr_pts)
    ax1.set_xlabel("X (cm)"); ax1.set_ylabel("Y (cm)"); ax1.set_zlabel("Z (cm)")
    legend1 = [
        Patch(color="steelblue",      label="Left skeleton (non-mapped)"),
        Patch(color="orange",         label="Left mapped joint"),
        Patch(color="cornflowerblue", label="Right skeleton (non-mapped)"),
        Patch(color="gold",           label="Right mapped joint"),
        Patch(color="red",   label="X"), Patch(color="green", label="Y"),
        Patch(color="blue",  label="Z"),
    ]
    fig1.legend(handles=legend1, loc="lower center", ncol=7, fontsize=8)
    fig1.tight_layout(rect=[0, 0.06, 1, 1])
    out1 = Path(f"/tmp/rest_pose_mhr_{args.robot}.png")
    fig1.savefig(out1, dpi=130, bbox_inches="tight")
    print(f"Saved → {out1}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Robot qpos=0, single copy
    # ══════════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(9, 9))
    fig2.canvas.manager.set_window_title(f"Figure 2 — {args.robot} rest pose (qpos=0)")
    fig2.suptitle(
        f"{args.robot}  —  actual body frames at qpos = 0\n"
        "Compare with Figure 1 LEFT to assess how much MHR A-pose frames differ\n"
        "Red=X   Green=Y   Blue=Z",
        fontsize=10,
    )
    ax2 = fig2.add_subplot(111, projection="3d")
    draw_robot_skeleton(ax2, robot_pos, robot_parents, mapped_robot_set,
                        color="steelblue", hl_color="orange")
    for mhr_idx, (readable, robot_body, rot_off) in mhr_to_robot.items():
        if robot_body not in robot_pos:
            continue
        pos = robot_pos[robot_body]
        draw_frame(ax2, pos, robot_rots[robot_body], length=0.07, lw=2.2)
        ax2.text(pos[0], pos[1], pos[2] + 0.07 * 1.3,
                 f"{robot_body}\n({readable})",
                 fontsize=5, ha="center", color="darkorange")

    set_equal_3d(ax2, list(robot_pos.values()), margin=0.1)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    legend2 = [
        Patch(color="steelblue", label="Non-mapped"),
        Patch(color="orange",    label="Mapped joint"),
        Patch(color="red",  label="X"), Patch(color="green", label="Y"),
        Patch(color="blue", label="Z"),
    ]
    fig2.legend(handles=legend2, loc="lower center", ncol=5, fontsize=8)
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    out2 = Path(f"/tmp/rest_pose_robot_{args.robot}_mhr.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved → {out2}")

    plt.show()


if __name__ == "__main__":
    main()
