#!/usr/bin/env python3
"""
Generate SMPL-X retargeting JSON config from a URDF file.

Usage:
    python generate_smplx_retarget.py <urdf_file> [options]

Options:
    -o / --output <path>         Output JSON (default: <urdf_stem>.json)
    --robot-root <link>          Override detected root link name
    --ground-height <float>      Default 0.0
    --human-height <float>       Default 1.8
    --human-scale <float>        Scale factor applied to all joints (default 1.0)
    --profile <name>             Weight profile: 'default' or 'h1' (default: 'default')
    --no-spine                   Exclude spine3 even if detected
    --no-hip-anatomy             Use pelvis frame for hips (ignore 15.75 deg SMPL-X offset)
    --approx-arms                Hardcoded arm quats ignoring FK tilt (TienKung style)
    --no-wrists                  Exclude wrist links even if detected

── How quaternions are computed ──────────────────────────────────────────────
For each matched robot link, the correction quaternion is:

    Q_correction = R_smplx_joint  @  R_world_link^-1

where R_smplx_joint is the canonical SMPL-X rest-pose frame and
R_world_link is obtained from forward kinematics at the zero pose.

This means any structural offset in the URDF chain (e.g. a tilted shoulder
joint) is automatically absorbed into the quaternion — no manual tuning needed.

── Canonical SMPL-X frames (recovered from 3 reference robots) ───────────────
Robot convention assumed: Z-up, X-forward, Y-left.

  Body / legs / head / spine:
      SMPL-X X = world Z (up)
      SMPL-X Y = world X (forward)
      SMPL-X Z = world Y (left)

  Hip (body frame rotated -15.7506 deg around body-local Y):
      Encodes the pelvis->hip anatomical offset in the SMPL-X skeleton.
      Use --no-hip-anatomy to disable (uses body frame instead).

  Left shoulder :  [[0,0,-1],[0,1,0],[1,0,0]]
  Left  elbow/wrist: identity
  Right shoulder:  [[0,0,1],[0,-1,0],[1,0,0]]
  Right elbow/wrist: [[-1,0,0],[0,-1,0],[0,0,1]]

── Weight profiles ──────────────────────────────────────────────────────────
'default' (OpenLoong / TienKung):
  table2 pelvis=100,  feet=[100,50], spine=[0,10], shoulders=[10,5]

'h1':
  table2 pelvis=10,   feet=[50,10],  spine=[10,0], shoulders=[50,5]
"""

import xml.etree.ElementTree as ET
import json
import argparse
import sys
import re
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


# ── Canonical SMPL-X rest-pose frames ─────────────────────────────────────────
def _make_R_smplx():
    body = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=float)
    Ry   = Rotation.from_euler('y', -15.7506085, degrees=True).as_matrix()
    hip  = body @ Ry
    L_sh = np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=float)
    R_sh = np.array([[0,0,1],[0,-1,0],[1,0,0]], dtype=float)
    R_el = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=float)
    return {
        "pelvis":         body,      "head":          body,      "spine3":        body,
        "left_hip":       hip,       "left_knee":     body,      "left_foot":     body,
        "right_hip":      hip,       "right_knee":    body,      "right_foot":    body,
        "left_shoulder":  L_sh,      "left_elbow":    np.eye(3), "left_wrist":    np.eye(3),
        "right_shoulder": R_sh,      "right_elbow":   R_el,      "right_wrist":   R_el,
    }

R_SMPLX = _make_R_smplx()


# ── IK weight/iteration profiles ──────────────────────────────────────────────
#
# TABLE 1 is the same for all profiles:
#   pos_weight = 100 for root+feet endpoints, 0 otherwise; iter = 10
#
# TABLE 2 differs between profiles (observed across reference robots).
# Feet quaternion sign is negated in table2 (same rotation, solver preference).

_T1 = {sj: [0, 10] for sj in R_SMPLX}
_T1.update({"pelvis": [100, 10], "left_foot": [100, 10], "right_foot": [100, 10]})

_T2_DEFAULT = {   # OpenLoong / TienKung profile
    "pelvis":         [100, 5],
    "head":           [10,  5],
    "spine3":         [0,   10],
    "left_hip":       [10,  5],  "right_hip":       [10,  5],
    "left_knee":      [10,  5],  "right_knee":      [10,  5],
    "left_foot":      [100, 50], "right_foot":      [100, 50],
    "left_shoulder":  [10,  5],  "right_shoulder":  [10,  5],
    "left_elbow":     [10,  5],  "right_elbow":     [10,  5],
    "left_wrist":     [10,  5],  "right_wrist":     [10,  5],
}

_T2_H1 = {        # H1 profile
    "pelvis":         [10,  5],
    "head":           [10,  5],
    "spine3":         [10,  0],  # 0 iterations = effectively disabled
    "left_hip":       [10,  5],  "right_hip":       [10,  5],
    "left_knee":      [10,  5],  "right_knee":      [10,  5],
    "left_foot":      [50,  10], "right_foot":      [50,  10],
    "left_shoulder":  [50,  5],  "right_shoulder":  [50,  5],
    "left_elbow":     [10,  5],  "right_elbow":     [10,  5],
    "left_wrist":     [10,  5],  "right_wrist":     [10,  5],
}

PROFILES = {"default": _T2_DEFAULT, "h1": _T2_H1}

FLIP_SIGN_TABLE2 = {"left_foot", "right_foot"}


# ── Body-part link patterns ────────────────────────────────────────────────────
# Priority-ordered, case-insensitive. First match wins.
# "roll" preferred over "yaw" for shoulders: last structural link before
# the arm segment, consistent with all three reference JSONs.
BODY_PART_PATTERNS = {
    "pelvis":  [r"^pelvis$", r"base_link", r"waist_yaw", r"^torso$"],
    # Left leg
    "left_hip":  [r"left_hip_roll",  r"hip_roll_l",   r"hip_l_roll",  r"l_hip_roll"],
    "left_knee": [r"left_knee",      r"knee_pitch_l",  r"knee_l_pitch", r"l_knee"],
    "left_foot": [r"left_ankle_roll", r"ankle_roll_l", r"ankle_l_roll",
                  r"left_ankle",      r"ankle_l",
                  r"left_toe",        r"toe_l",
                  r"left_foot",       r"foot_l"],
    # Right leg
    "right_hip":  [r"right_hip_roll", r"hip_roll_r",   r"hip_r_roll",  r"r_hip_roll"],
    "right_knee": [r"right_knee",     r"knee_pitch_r",  r"knee_r_pitch", r"r_knee"],
    "right_foot": [r"right_ankle_roll", r"ankle_roll_r", r"ankle_r_roll",
                   r"right_ankle",      r"ankle_r",
                   r"right_toe",        r"toe_r",
                   r"right_foot",       r"foot_r"],
    # Spine / head (after legs, before arms — matches reference JSON ordering)
    "spine3":  [r"^torso_link$", r"torso_link", r"waist_pitch", r"waist_roll",
                r"chest", r"spine"],
    "head":    [r"head(?:_yaw|_pitch)?_link$", r"head_link", r"^head$"],
    # Left arm
    # Shoulder patterns exclude "pitch" — that joint is always the proximal
    # attachment (forward/back swing) and never the SMPL-X shoulder target.
    # Among all non-pitch matches, match_links_to_smplx picks the MOST PROXIMAL
    # one in the kinematic chain (see WALK_TO_PROXIMAL), which correctly handles
    # both roll-first robots (TienKung, H1) and yaw-first robots.
    "left_shoulder": [r"left_shoulder_roll",  r"shoulder_roll_l(?:_|$)",  r"shoulder_l_roll",
                      r"left_shoulder_yaw",   r"shoulder_yaw_l(?:_|$)",   r"l_shoulder_yaw",
                      r"arm_l_0[12]"],
    "left_elbow":    [r"left_elbow",  r"elbow_pitch_l", r"elbow_l", r"l_elbow",
                      r"arm_l_0[34]"],
    "left_wrist":    [r"left_wrist_yaw", r"wrist_yaw_l",
                      r"left_wrist",     r"wrist_l",      r"l_wrist",
                      r"left_hand_link", r"hand_l_link",
                      r"arm_l_0[67]"],
    # Right arm (same logic as left)
    "right_shoulder": [r"right_shoulder_roll", r"shoulder_roll_r(?:_|$)",  r"shoulder_r_roll",
                       r"right_shoulder_yaw",  r"shoulder_yaw_r(?:_|$)",   r"r_shoulder_yaw",
                       r"arm_r_0[12]"],
    "right_elbow":    [r"right_elbow",  r"elbow_pitch_r", r"elbow_r", r"r_elbow",
                       r"arm_r_0[34]"],
    "right_wrist":    [r"right_wrist_yaw", r"wrist_yaw_r",
                       r"right_wrist",     r"wrist_r",      r"r_wrist",
                       r"right_hand_link", r"hand_r_link",
                       r"arm_r_0[67]"],
}

# Shoulder joints: use the MOST PROXIMAL matched link (closest to torso).
# Mirror of WALK_TO_LEAF for feet (most distal).
# Correctly picks roll on TienKung/H1 (roll proximal to yaw),
# and yaw on robots where yaw precedes roll in the arm chain.
WALK_TO_PROXIMAL = {"left_shoulder", "right_shoulder"}


# ── URDF parsing & FK ─────────────────────────────────────────────────────────

def parse_urdf(urdf_path):
    """
    Return (robot_root, all_links, fk, children_of) where:
      fk(link)           -> 4×4 world transform at zero pose
      children_of(link)  -> list of direct child link names
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    all_links = [l.get("name") for l in root.findall("link")]

    joints = {}
    has_parent = set()
    parent_to_children = {}
    for j in root.findall("joint"):
        p = j.find("parent"); c = j.find("child"); o = j.find("origin")
        if p is None or c is None:
            continue
        rpy = [float(x) for x in o.get("rpy","0 0 0").split()] if o is not None else [0,0,0]
        xyz = [float(x) for x in o.get("xyz","0 0 0").split()] if o is not None else [0,0,0]
        child = c.get("link")
        pname = p.get("link")
        joints[child] = {"parent": pname, "rpy": rpy, "xyz": xyz}
        has_parent.add(child)
        parent_to_children.setdefault(pname, []).append(child)

    root_links = [l for l in all_links if l not in has_parent]
    robot_root = root_links[0] if root_links else all_links[0]

    # joints_parent: link -> parent link name (for depth computation)
    joints_parent = {child: j["parent"] for child, j in joints.items()}
    # link_local_rpy: link -> own joint rpy (not accumulated)
    link_local_rpy = {child: j["rpy"] for child, j in joints.items()}

    _cache = {}
    def fk(link):
        if link in _cache:
            return _cache[link]
        if link not in joints:
            T = np.eye(4)
        else:
            j = joints[link]
            R = Rotation.from_euler('xyz', j['rpy']).as_matrix()
            T = np.eye(4); T[:3,:3] = R; T[:3,3] = j['xyz']
            T = fk(j['parent']) @ T
        _cache[link] = T
        return T

    def children_of(link):
        return parent_to_children.get(link, [])

    return robot_root, all_links, fk, children_of, joints_parent, link_local_rpy


# ── Link → SMPL-X matching ────────────────────────────────────────────────────

# Foot joints use the MOST DISTAL link in the chain (leaf node).
# After pattern matching finds any link in the foot/ankle region, we walk
# forward through the kinematic chain to reach the actual leaf.
# This correctly selects toe_link when it exists, or ankle_roll when it doesn't,
# without needing to know the naming convention of each robot.
WALK_TO_LEAF = {"left_foot", "right_foot"}


def _walk_to_leaf(link, children_of):
    """Follow the single-child kinematic chain from link until a leaf is reached."""
    while True:
        children = children_of(link)
        if len(children) == 1:
            link = children[0]
        else:
            return link


def _walk_to_proximal(sj, candidates, joints_parent, fk):
    """
    Among all candidate links for a shoulder joint, pick the one whose FK
    world-frame rotation is closest to identity (minimum tilt from zero pose).

    Rationale: the reference JSONs consistently use the shoulder link whose
    frame is aligned with the world at rest — this is the joint where the
    canonical SMPL-X shoulder quaternion can be applied cleanly.  When two
    links have equal tilt (both effectively zero, e.g. H1 roll and yaw),
    the most proximal one (smallest kinematic depth) is chosen as a tiebreaker.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as _R

    def tilt(link):
        R = fk(link)[:3, :3]
        return _R.from_matrix(R).magnitude()   # radians; 0 = identity

    def depth(link):
        d, l = 0, link
        while l in joints_parent:
            l = joints_parent[l]
            d += 1
        return d

    best = min(candidates, key=lambda l: (round(tilt(l), 6), depth(l)))
    return best


def match_links_to_smplx(all_links, children_of, joints_parent, fk):
    mapping = {}
    used = set()
    for sj, patterns in BODY_PART_PATTERNS.items():

        # For shoulder: collect ALL pattern matches first, then pick most proximal
        if sj in WALK_TO_PROXIMAL:
            seen = set()
            candidates = []
            for pat in patterns:
                for link in all_links:
                    if link in used or link in seen:
                        continue
                    if re.search(pat, link, re.IGNORECASE):
                        candidates.append(link)
                        seen.add(link)
            if candidates:
                found = _walk_to_proximal(sj, candidates, joints_parent, fk)
                if len(candidates) > 1:
                    print(f"  [INFO] {sj}: chose {found} (most proximal of {candidates})",
                          file=sys.stderr)
            else:
                found = None
        else:
            found = None
            for pat in patterns:
                for link in all_links:
                    if link in used:
                        continue
                    if re.search(pat, link, re.IGNORECASE):
                        found = link
                        break
                if found:
                    break

            # For foot joints: walk forward to the true leaf of the leg chain
            if found is not None and sj in WALK_TO_LEAF:
                leaf = _walk_to_leaf(found, children_of)
                if leaf != found:
                    print(f"  [INFO] {sj}: walking {found} → {leaf} (leaf node)",
                          file=sys.stderr)
                found = leaf

        mapping[sj] = found
        if found:
            used.add(found)
    return mapping


# ── Quaternion computation ────────────────────────────────────────────────────

_APPROX_ARM_QUATS = {
    "left_shoulder":  [0.70710678,  0.0, -0.70710678, 0.0],
    "left_elbow":     [0.70710678,  0.0, -0.70710678, 0.0],
    "left_wrist":     [0.70710678,  0.0, -0.70710678, 0.0],
    "right_shoulder": [0.0,  0.70710678,  0.0,  0.70710678],
    "right_elbow":    [0.0,  0.70710678,  0.0,  0.70710678],
    "right_wrist":    [0.0,  0.70710678,  0.0,  0.70710678],
}

_ARM_JOINTS = {"left_shoulder","left_elbow","left_wrist",
               "right_shoulder","right_elbow","right_wrist"}

def compute_correction_quat(smplx_joint, R_world,
                            no_hip_anatomy=False, approx_arms=False):
    """
    Returns Q_correction = R_smplx @ R_world^T  as  [w, x, y, z].
    Canonical form: w >= 0.
    """
    if approx_arms and smplx_joint in _ARM_JOINTS:
        return list(_APPROX_ARM_QUATS[smplx_joint])

    R_smplx = R_SMPLX[smplx_joint].copy()
    if no_hip_anatomy and smplx_joint in ("left_hip", "right_hip"):
        R_smplx = R_SMPLX["pelvis"].copy()

    R_corr = R_smplx @ R_world.T
    q = Rotation.from_matrix(R_corr).as_quat()   # scipy: [x, y, z, w]
    if q[3] < 0:
        q = -q
    return [float(round(q[3], 10)),   # w
            float(round(q[0], 10)),   # x
            float(round(q[1], 10)),   # y
            float(round(q[2], 10))]   # z



def _snap_quat(q, tol=5e-4):
    """Snap near-axis-aligned quaternions to exact values.
    Eliminates floating-point residuals from cancel-out joint chains
    (e.g. pitch +0.28 -> roll -0.28 -> residual ~1e-5).
    """
    import itertools
    # All 24 axis-aligned unit quaternions (w,x,y,z)
    candidates = []
    for signs in itertools.product([0, 1, -1], repeat=4):
        n = sum(s*s for s in signs)
        if abs(n - 1) < 1e-9:
            candidates.append(signs)
    best, best_dist = q, float('inf')
    for c in candidates:
        d = sum((q[i]-c[i])**2 for i in range(4))
        if d < best_dist:
            best_dist = d; best = c
    if best_dist < tol**2:
        return list(best)
    return q



def detect_profile(all_links):
    """
    Auto-detect the IK weight profile from URDF link names.

    Heuristic: robots with dexterous hands (links containing finger/thumb/palm
    keywords) are manipulation robots → h1 profile (higher shoulder weight,
    relaxed foot weight).  Robots without hands are locomotion-dominant →
    default profile (strong foot anchor).

    Returns "h1" or "default".
    """
    finger_keywords = ["finger", "thumb"]
    finger_links = [l for l in all_links
                    if any(kw in l.lower() for kw in finger_keywords)]
    if finger_links:
        print(f"  [INFO] {len(finger_links)} finger links detected → profile=h1",
              file=sys.stderr)
        return "h1"
    return "default"


# ── JSON builder ──────────────────────────────────────────────────────────────

def build_json(urdf_path,
               robot_root_override=None,
               ground_height=0.0,
               human_height=1.8,
               human_scale=1.0,
               arm_scale=None,
               profile=None,
               use_spine=True,
               use_wrists=True,
               use_head=True,
               no_hip_anatomy=False,
               approx_arms=False):

    robot_root, all_links, fk, children_of, joints_parent, _link_local_rpy = parse_urdf(urdf_path)
    if robot_root_override:
        robot_root = robot_root_override

    smplx_to_robot = match_links_to_smplx(all_links, children_of, joints_parent, fk)

    if smplx_to_robot.get("pelvis") is None:
        smplx_to_robot["pelvis"] = robot_root

    excluded = set()
    if not use_spine:
        excluded.add("spine3")
    if not use_wrists:
        excluded.update({"left_wrist", "right_wrist"})
    if not use_head:
        excluded.add("head")

    # head is optional — many robots don't have a head link in retargeting
    OPTIONAL_JOINTS = {"head"}

    for sj, rl in smplx_to_robot.items():
        if rl is None and sj not in excluded and sj not in OPTIONAL_JOINTS:
            print(f"  [WARN] No robot link for '{sj}' — skipping.", file=sys.stderr)

    # Auto-detect profile from URDF structure if not explicitly provided
    if profile is None:
        profile = detect_profile(all_links)
        print(f"  [INFO] Auto-detected profile: {profile}", file=sys.stderr)
    t2_weights = PROFILES.get(profile, _T2_DEFAULT)

    # Canonical output order: pelvis → legs → spine → arms
    OUTPUT_ORDER = [
        "pelvis",
        "left_hip", "left_knee", "left_foot",
        "right_hip", "right_knee", "right_foot",
        "spine3",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "head",
    ]

    # Resolve arm_scale: defaults to body scale if not provided
    _arm_scale = arm_scale if arm_scale is not None else human_scale
    ARM_JOINTS = {"left_shoulder", "right_shoulder",
                  "left_elbow",    "right_elbow",
                  "left_wrist",    "right_wrist"}

    # Human scale table — arm joints use _arm_scale, everything else human_scale
    hs_table = {}
    for sj in OUTPUT_ORDER:
        if sj in excluded:
            continue
        if smplx_to_robot.get(sj):
            hs_table[sj] = _arm_scale if sj in ARM_JOINTS else human_scale

    # Foot positional offset: if the matched foot link is a toe link,
    # apply a ±0.02 m lateral (Y) offset so the IK targets the ankle
    # center rather than the displaced toe link origin.
    # Sign convention: +Y = left side, −Y = right side (robot body frame).
    def _foot_offset(sj, robot_link):
        if robot_link is None:
            return [0.0, 0.0, 0.0]
        if "toe" in robot_link.lower():
            y = 0.02 if "left" in sj else -0.02
            print(f"  [INFO] {sj}: toe link detected → applying offset [0, {y}, 0]",
                  file=sys.stderr)
            return [0.0, y, 0.0]
        return [0.0, 0.0, 0.0]

    def make_table(table_num):
        table = {}
        t_weights = _T1 if table_num == 1 else t2_weights
        for sj in OUTPUT_ORDER:
            robot_link = smplx_to_robot.get(sj)
            if sj in excluded or robot_link is None:
                continue
            # Anatomy joints (legs, spine) use R=I: their SMPL-X correction
            # quaternions are defined in body frame and must not be influenced
            # by the robot's zero-pose link orientations (structural offsets
            # in the URDF that position joints for natural stance).
            # Arm joints use the full FK so their canonical frames are correct.
            ANATOMY_JOINTS = {"left_hip", "right_hip",
                              "left_knee", "right_knee",
                              "left_foot", "right_foot",
                              "spine3", "pelvis"}
            if sj in ANATOMY_JOINTS:
                R_world = np.eye(3)
            else:
                R_world = fk(robot_link)[:3, :3]
            quat = compute_correction_quat(sj, R_world, no_hip_anatomy, approx_arms)
            if table_num == 2 and sj in FLIP_SIGN_TABLE2:
                quat = [-quat[0], -quat[1], -quat[2], -quat[3]]
            quat = _snap_quat(quat)
            pos_w, iters = t_weights[sj]
            # Foot offset only applies in table1 (IK position target refinement)
            offset = _foot_offset(sj, robot_link) if (sj in {"left_foot", "right_foot"} and table_num == 1) else [0.0, 0.0, 0.0]
            table[robot_link] = [sj, pos_w, iters, offset, quat]
        return table

    return {
        "robot_root_name":         robot_root,
        "human_root_name":         "pelvis",
        "ground_height":           float(ground_height),
        "human_height_assumption": human_height,
        "use_ik_match_table1":     True,
        "use_ik_match_table2":     True,
        "human_scale_table":       hs_table,
        "ik_match_table1":         make_table(1),
        "ik_match_table2":         make_table(2),
    }


# ── Clean float output ────────────────────────────────────────────────────────

def _clean(obj):
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    if isinstance(obj, float):
        r = round(obj, 10)
        return int(r) if r == int(r) else r
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urdf")
    parser.add_argument("-o", "--output")
    parser.add_argument("--robot-root")
    parser.add_argument("--ground-height",  type=float,  default=0.0)
    parser.add_argument("--human-height",   type=float,  default=1.8)
    parser.add_argument("--arm-scale",      type=float,  default=None,
                        help="Scale for arm joints in human_scale_table. "
                             "Defaults to --human-scale. Use when robot arms are "
                             "proportionally shorter/longer than its legs vs SMPL-X.")
    parser.add_argument("--human-scale",    type=float,  default=1.0,
                        help="Scale factor for all human_scale_table entries (default 1.0)")
    parser.add_argument("--profile",        default=None,
                        choices=["default", "h1"],
                        help="IK weight profile: 'default' (OpenLoong/TienKung) or 'h1'")
    parser.add_argument("--no-spine",       action="store_true")
    parser.add_argument("--no-wrists",      action="store_true",
                        help="Exclude wrist links (use when robot has no wrist DOF in retargeting)")
    parser.add_argument("--no-head",        action="store_true",
                        help="Exclude head link from tables and human_scale_table")
    parser.add_argument("--no-hip-anatomy", action="store_true",
                        help="Use pelvis frame for hips (ignores SMPL-X 15.75 deg hip offset)")
    parser.add_argument("--approx-arms",    action="store_true",
                        help="Hardcoded arm quaternions ignoring FK tilt (TienKung style)")
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"ERROR: not found: {urdf_path}", file=sys.stderr); sys.exit(1)

    output_path = Path(args.output) if args.output else urdf_path.with_suffix(".json")

    print(f"Parsing : {urdf_path}")
    _display_profile = args.profile if args.profile is not None else "auto"
    print(f"Profile : {_display_profile}  |  scale={args.human_scale}  |  "
          f"ground={args.ground_height}  |  hip_anatomy={'yes' if not args.no_hip_anatomy else 'no'}")

    config = build_json(
        urdf_path,
        robot_root_override=args.robot_root,
        ground_height=args.ground_height,
        human_height=args.human_height,
        human_scale=args.human_scale,
        arm_scale=args.arm_scale,
        profile=args.profile,
        use_spine=not args.no_spine,
        use_wrists=not args.no_wrists,
        use_head=not args.no_head,
        no_hip_anatomy=args.no_hip_anatomy,
        approx_arms=args.approx_arms,
    )

    print("\nDetected SMPL-X → Robot link mapping:")
    for robot_link, entry in config["ik_match_table1"].items():
        print(f"  {entry[0]:20s}  →  {robot_link}")

    with open(output_path, "w") as f:
        json.dump(_clean(config), f, indent=4)
    print(f"\nWrote: {output_path}")
    return config, output_path


if __name__ == "__main__":
    main()
