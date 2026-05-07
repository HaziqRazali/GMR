"""
Visualize SMPL-X joint coordinate frames vs robot body frames at rest pose.

For each mapped (SMPL-X joint → robot body) pair, draws 3 triads:
  ─── Solid   : SMPL-X T-pose frame  (always world-aligned / identity rotation)
  ─── Dashed  : SMPL-X frame × rot_offset  (what the IK sends as target)
  ··· Dotted  : Robot body frame at MuJoCo rest  (qpos = 0)

If dashed ≈ dotted, the calibration is correct for that joint at rest.
Δ° printed in the subplot title = angular error between dashed and dotted.

Usage (run in GMR conda env):
    python scripts/vis_frame_alignment.py --robot unitree_g1
    python scripts/vis_frame_alignment.py --robot booster_t1
    python scripts/vis_frame_alignment.py --robot unitree_g1 --table ik_match_table2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

# ── repo paths ------------------------------------------------------------------
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

ARROW_LEN = 0.06   # metres

# Colors for each triad (RGB for X/Y/Z within each triad)
# Triad 0: SMPL-X T-pose     → muted gray-ish
# Triad 1: × rot_offset      → vivid (red/green/blue)
# Triad 2: robot at rest     → warm (orange/lime/cyan)
TRIAD_STYLES = [
    dict(colors=["#888888", "#555555", "#333333"], lw=1.5, ls="-",  label="SMPL-X T-pose"),
    dict(colors=["#e63232", "#22aa22", "#2244ff"], lw=2.5, ls="-",  label="× rot_offset (IK target)"),
    dict(colors=["#e63232", "#22aa22", "#2244ff"], lw=2.5, ls="--", label="Robot at rest"),
]

# Separate the three triads spatially so they don't overlap.
# We spread them along the world X axis.
TRIAD_OFFSETS = [
    np.array([-ARROW_LEN * 1.6, 0, 0]),
    np.array([0.0,               0, 0]),
    np.array([ ARROW_LEN * 1.6, 0, 0]),
]


# ── helpers --------------------------------------------------------------------

def draw_triad(ax, origin, rot_mat, triad_idx):
    """Draw X/Y/Z axes from `origin` using rotation matrix `rot_mat`."""
    style = TRIAD_STYLES[triad_idx]
    off   = TRIAD_OFFSETS[triad_idx]
    o = origin + off
    l = ARROW_LEN
    for col, color in enumerate(style["colors"]):
        end = o + rot_mat[:, col] * l
        ax.plot(
            [o[0], end[0]], [o[1], end[1]], [o[2], end[2]],
            color=color, linewidth=style["lw"], linestyle=style["ls"],
        )
        # arrowhead dot at the tip
        ax.scatter(*end, color=color, s=18, zorder=5)


def set_axes_equal(ax, center, half=None):
    if half is None:
        half = ARROW_LEN * 3.2   # wider to fit the three offset triads
    cx, cy, cz = center
    ax.set_xlim([cx - half, cx + half])
    ax.set_ylim([cy - half / 2, cy + half / 2])
    ax.set_zlim([cz - half / 2, cz + half / 2])


def get_robot_body_frames(xml_path: Path) -> dict:
    """Return {body_name: (world_pos, world_rot_matrix)} at zero qpos."""
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    frames = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name is None:
            continue
        pos = data.xpos[i].copy()
        rot = R.from_quat(data.xquat[i], scalar_first=True).as_matrix()
        frames[name] = (pos, rot)
    return frames


# ── main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1", choices=list(CONFIG_MAP))
    parser.add_argument(
        "--table", default="ik_match_table1",
        choices=["ik_match_table1", "ik_match_table2"],
    )
    parser.add_argument(
        "--only_arms", action="store_true",
        help="Only show shoulder / elbow / wrist rows (skip legs/spine).",
    )
    args = parser.parse_args()

    with open(CONFIG_MAP[args.robot]) as f:
        config = json.load(f)
    table = config[args.table]
    robot_frames = get_robot_body_frames(XML_MAP[args.robot])

    # ── collect pairs with nonzero rotation weight ────────────────────────────
    pairs = []
    arm_keywords = {"shoulder", "elbow", "wrist", "hand", "AL", "AR"}
    for robot_body, entry in table.items():
        human_body, pos_w, rot_w, pos_off, rot_off_wxyz = entry
        if rot_w == 0:
            continue
        if args.only_arms and not any(k.lower() in robot_body.lower() for k in arm_keywords):
            continue
        rot_off = R.from_quat(rot_off_wxyz, scalar_first=True)
        pairs.append((robot_body, human_body, rot_off))

    if not pairs:
        print("No pairs found. Try removing --only_arms.")
        return

    # ── layout ────────────────────────────────────────────────────────────────
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))
    fig.suptitle(
        f"Frame alignment — {args.robot}  ({args.table})\n"
        "LEFT triad (gray) = SMPL-X T-pose  │  "
        "MIDDLE triad (solid) = SMPL-X × rot_offset (IK target at T-pose)  │  "
        "RIGHT triad (dashed) = Robot body at rest  (qpos=0)\n"
        "Δ° = rotation from IK-target-at-T-pose to robot-rest  "
        "(tells you how far robot moves from rest to match human T-pose)",
        fontsize=9,
    )

    for idx, (robot_body, human_body, rot_off) in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        if robot_body not in robot_frames:
            ax.set_title(f"{robot_body}\n(missing in XML)")
            continue

        robot_pos, robot_rot = robot_frames[robot_body]
        o = robot_pos  # draw all three triads at the same world position

        # 1) SMPL-X T-pose frame → identity
        smplx_rot = np.eye(3)
        draw_triad(ax, o, smplx_rot, triad_idx=0)

        # 2) SMPL-X × rot_offset  (IK target at T-pose)
        target_rot = (R.from_matrix(smplx_rot) * rot_off).as_matrix()
        draw_triad(ax, o, target_rot, triad_idx=1)

        # 3) Robot body frame at rest
        draw_triad(ax, o, robot_rot, triad_idx=2)

        # angular error between target and robot-rest
        diff = R.from_matrix(target_rot) * R.from_matrix(robot_rot).inv()
        angle_deg = np.degrees(diff.magnitude())

        # colour-code title by severity
        color = "green" if angle_deg < 15 else ("orange" if angle_deg < 45 else "red")
        ax.set_title(
            f"{robot_body}\n→ {human_body}       Δ = {angle_deg:.1f}°",
            fontsize=8, color=color,
        )

        set_axes_equal(ax, o)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_items = []
    axis_labels = ["X", "Y", "Z"]
    for ti, style in enumerate(TRIAD_STYLES):
        for col, (color, axis) in enumerate(zip(style["colors"], axis_labels)):
            legend_items.append(
                Line2D([0], [0], color=color, lw=style["lw"], linestyle=style["ls"],
                       label=f"{style['label']} — {axis}")
            )
    fig.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = Path(f"/tmp/frame_alignment_{args.robot}_{args.table}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
