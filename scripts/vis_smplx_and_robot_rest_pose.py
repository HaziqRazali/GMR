"""
Visualize the SMPL-X body (T-pose) and the robot (rest pose / qpos=0) with
coordinate frames at every mapped joint.

Two separate interactive figure windows:

  Figure 1 — SMPL-X T-pose  (two copies side by side):
               LEFT  copy  -> raw T-pose frames (identity — what SMPL-X outputs)
               RIGHT copy  -> after rot_offset applied (= IK target sent to robot)

  Figure 2 — Robot rest pose (single copy):
               actual body frames at qpos=0

The goal of calibration: RIGHT copy of Figure 1 should align with Figure 2.

Usage:
    python scripts/vis_rest_pose_both.py
    python scripts/vis_rest_pose_both.py --robot booster_t1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import smplx
import torch
from matplotlib.patches import Patch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES

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
SMPLX_MODEL_DIR = GMR_ROOT / "assets/body_models"

FRAME_LEN = 0.07   # length of each coordinate axis arrow

# ── SMPL-X parent chain (joints 0-21 = body joints) ──────────────────────────
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12,
                  13, 14, 16, 17, 18, 19]

# ── helper: draw one triad ─────────────────────────────────────────────────────
def draw_frame(ax, pos, rot_mat, length=FRAME_LEN, lw=2.0, alpha=1.0):
    """Draw X(red) Y(green) Z(blue) axes from pos using rot_mat columns."""
    for col, color in enumerate(["r", "g", "b"]):
        tip = pos + rot_mat[:, col] * length
        ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                color=color, linewidth=lw, zorder=5, alpha=alpha)
        ax.scatter(*tip, color=color, s=14, zorder=6, alpha=alpha)


# ── SMPL-X T-pose ──────────────────────────────────────────────────────────────
def get_smplx_tpose(model_dir):
    body_model = smplx.create(
        str(model_dir), "smplx", gender="neutral", use_pca=False,
        num_betas=16,
    )
    with torch.no_grad():
        out = body_model(
            betas=torch.zeros(1, 16),
            expression=torch.zeros(1, 10),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            transl=torch.zeros(1, 3),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            return_full_pose=True,
        )
    joints = out.joints[0].numpy()           # (N_joints, 3)
    n = len(SMPLX_PARENTS)
    rots = [np.eye(3)] * n                   # all identity at T-pose
    return joints[:n], rots


# ── Robot rest pose from MuJoCo ────────────────────────────────────────────────
def get_robot_rest(xml_path):
    model = mj.MjModel.from_xml_path(str(xml_path))
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    parents = {}
    pos  = {}
    rots = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None or name == "world":
            continue
        pos[name]  = data.xpos[i].copy()
        rots[name] = R.from_quat(data.xquat[i], scalar_first=True).as_matrix()
        pid = model.body_parentid[i]
        pname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, pid) if pid >= 0 else None
        # Don't connect to "world" — that draws a spurious line from origin to pelvis
        parents[name] = pname if (pname and pname != "world") else None

    return pos, rots, parents


# ── draw skeleton (stick figure) ───────────────────────────────────────────────
def draw_robot_skeleton(ax, body_pos, parents, highlighted,
                        color="steelblue", hl_color="orange", offset=None):
    off = np.array(offset) if offset is not None else np.zeros(3)
    for bname, pname in parents.items():
        if pname is None or bname not in body_pos or pname not in body_pos:
            continue
        p0 = body_pos[pname] + off
        p1 = body_pos[bname] + off
        c = hl_color if (bname in highlighted or pname in highlighted) else color
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=c, linewidth=1.8, alpha=0.9)
    for bname in body_pos:
        p = body_pos[bname] + off
        s = 30 if bname in highlighted else 12
        c = hl_color if bname in highlighted else color
        ax.scatter(*p, color=c, s=s, zorder=4)


def draw_smplx_skeleton(ax, joints, highlighted_idx,
                        color="steelblue", hl_color="orange"):
    n = len(SMPLX_PARENTS)
    for i in range(n):
        p = SMPLX_PARENTS[i]
        if p < 0:
            continue
        p0 = joints[p]
        p1 = joints[i]
        c = hl_color if (i in highlighted_idx or p in highlighted_idx) else color
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=c, linewidth=1.8, alpha=0.9)
    for i in range(n):
        s = 30 if i in highlighted_idx else 12
        c = hl_color if i in highlighted_idx else color
        ax.scatter(*joints[i], color=c, s=s, zorder=4)


# ── axis limits helper ─────────────────────────────────────────────────────────
def set_equal_3d(ax, pts, margin=0.10):
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    half   = max((hi - lo).max() / 2, 0.3) + margin
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1", choices=list(CONFIG_MAP))
    args = parser.parse_args()

    # --- load config ---
    with open(CONFIG_MAP[args.robot]) as f:
        config = json.load(f)
    table = config["ik_match_table1"]

    pairs = {}   # robot_body_name -> (human_body_name, rot_offset_R)
    for robot_body, entry in table.items():
        human_body, pw, rw, _, rot_off_wxyz = entry
        rot_off = R.from_quat(rot_off_wxyz, scalar_first=True)
        pairs[robot_body] = (human_body, rot_off)

    mapped_human = {v[0] for v in pairs.values()}
    mapped_robot  = set(pairs.keys())

    # --- SMPL-X T-pose ---
    smplx_joints, smplx_rots = get_smplx_tpose(SMPLX_MODEL_DIR)
    smplx_name2idx = {name: i for i, name in enumerate(SMPLX_JOINT_NAMES)}
    mapped_smplx_idx = {smplx_name2idx[n] for n in mapped_human if n in smplx_name2idx}

    # --- Robot rest pose ---
    robot_pos, robot_rots, robot_parents = get_robot_rest(XML_MAP[args.robot])

    # Compute X offset to separate the two SMPL-X copies cleanly
    smpl_xs = smplx_joints[:22, 0]
    smpl_width = smpl_xs.max() - smpl_xs.min()
    SMPL_OFFSET = np.array([smpl_width * 1.8 + 0.3, 0.0, 0.0])

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — SMPL-X: two copies
    #   LEFT  (no offset) → raw T-pose frames (identity)
    #   RIGHT (+ SMPL_OFFSET) → after rot_offset applied (= IK target)
    # ══════════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(14, 9))
    fig1.canvas.manager.set_window_title("Figure 1 — SMPL-X T-pose: before vs after rot_offset")
    fig1.suptitle(
        "SMPL-X — T-pose\n"
        "LEFT: raw joint frames (identity)   |   RIGHT (offset): after rot_offset applied = IK target\n"
        "Goal of calibration: RIGHT should match the robot rest frames (Figure 2)\n"
        "Red=X   Green=Y   Blue=Z",
        fontsize=10,
    )
    ax1 = fig1.add_subplot(111, projection="3d")

    # ── LEFT copy: raw T-pose (identity frames) ───────────────────────────────
    draw_smplx_skeleton(ax1, smplx_joints, mapped_smplx_idx,
                        color="steelblue", hl_color="orange")
    for robot_body, (human_name, rot_off) in pairs.items():
        idx = smplx_name2idx.get(human_name)
        if idx is None:
            continue
        pos = smplx_joints[idx]
        draw_frame(ax1, pos, smplx_rots[idx])  # identity
        ax1.text(pos[0], pos[1], pos[2] + FRAME_LEN * 1.2, human_name,
                 fontsize=6, ha="center", color="darkorange")
    ax1.text2D(0.18, 0.94, "LEFT: raw T-pose frames (identity)",
               transform=ax1.transAxes, fontsize=9, ha="center", color="navy",
               bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray"))

    # ── RIGHT copy: after rot_offset applied ──────────────────────────────────
    right_joints = smplx_joints + SMPL_OFFSET  # same skeleton positions, shifted
    draw_smplx_skeleton(ax1, right_joints, mapped_smplx_idx,
                        color="cornflowerblue", hl_color="gold")
    for robot_body, (human_name, rot_off) in pairs.items():
        idx = smplx_name2idx.get(human_name)
        if idx is None:
            continue
        pos = right_joints[idx]
        # rot_offset applied to identity = rot_offset itself
        draw_frame(ax1, pos, rot_off.as_matrix())
        ax1.text(pos[0], pos[1], pos[2] + FRAME_LEN * 1.2, human_name,
                 fontsize=6, ha="center", color="goldenrod")
    ax1.text2D(0.82, 0.94, "RIGHT: after rot_offset (= IK target)",
               transform=ax1.transAxes, fontsize=9, ha="center", color="navy",
               bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray"))

    all_smpl_pts = list(smplx_joints[:22]) + list(right_joints[:22])
    set_equal_3d(ax1, all_smpl_pts)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
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
    out1 = Path(f"/tmp/rest_pose_smplx_{args.robot}.png")
    fig1.savefig(out1, dpi=130, bbox_inches="tight")
    print(f"Saved → {out1}")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Robot: single copy, actual qpos=0 body frames
    # ══════════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(9, 9))
    fig2.canvas.manager.set_window_title(f"Figure 2 — {args.robot} rest pose (qpos=0)")
    fig2.suptitle(
        f"{args.robot}  —  actual body frames at qpos=0\n"
        "Compare these frames with the RIGHT copy in Figure 1 to assess calibration\n"
        "Red=X   Green=Y   Blue=Z",
        fontsize=10,
    )
    ax2 = fig2.add_subplot(111, projection="3d")
    draw_robot_skeleton(ax2, robot_pos, robot_parents, mapped_robot,
                        color="steelblue", hl_color="orange")
    for robot_body, (human_body, rot_off) in pairs.items():
        if robot_body not in robot_pos:
            continue
        pos = robot_pos[robot_body]
        draw_frame(ax2, pos, robot_rots[robot_body], lw=2.2)
        ax2.text(pos[0], pos[1], pos[2] + FRAME_LEN * 1.2,
                 f"{robot_body}\n({human_body})",
                 fontsize=5, ha="center", color="darkorange")

    set_equal_3d(ax2, list(robot_pos.values()))
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    legend2 = [
        Patch(color="steelblue", label="Non-mapped"),
        Patch(color="orange",    label="Mapped joint"),
        Patch(color="red",   label="X"), Patch(color="green", label="Y"),
        Patch(color="blue",  label="Z"),
    ]
    fig2.legend(handles=legend2, loc="lower center", ncol=5, fontsize=8)
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    out2 = Path(f"/tmp/rest_pose_robot_{args.robot}.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved → {out2}")

    # Both windows are independent — scroll/zoom works on whichever is active
    plt.show()


if __name__ == "__main__":
    main()
