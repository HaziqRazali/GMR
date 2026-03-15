"""
visualize_smpl_interactive.py
Interactive viewer: SMPL-X pose with corrected arrows (R_corrector @ R_world).

SMPL-X T-pose has identity world rotations, so R_corrector = I for all joints.
The arrows show FK world-frame rotations as you pose the character.

Left panel  — +/- buttons for each IK-mapped joint axis (5 deg per click).
3D panel    — skeleton + RGB triads at every IK joint.
              Green arrows = world-aligned at T-pose (all zeros).

Usage:
    conda run -n mhr_new python scripts_extra/visualize_smpl_interactive.py
"""

import os, sys, math, threading
import numpy as np
import open3d as o3d
import open3d.visualization.gui      as gui
import open3d.visualization.rendering as rendering
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/MHR"))

from visualize_mhr_rot_offsets import (
    _JOINT_MAP,
    make_axis_arrows,
    build_lineset,
    sphere_at,
    find_smplx_path,
)

# ── SMPL-X kinematic tree (joints 0–21) ───────────────────────────────────
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

SMPLX_BODY_EDGES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),
]

# ── Flat params layout: [global_orient(3) | body_pose(63)] ────────────────
# flat_idx(joint j) = 3 + (j-1)*3   for j >= 1
#                   = 0              for j == 0  (global_orient)
N_PARAMS = 66   # 3 + 63

# Exactly the 12 joints in _JOINT_MAP (IK-mapped joints only)
JOINTS_PARAMS = [
    ("root",      [("rx",  0), ("ry",  1), ("rz",  2)]),
    ("l_upleg",   [("rx",  3), ("ry",  4), ("rz",  5)]),
    ("r_upleg",   [("rx",  6), ("ry",  7), ("rz",  8)]),
    ("l_lowleg",  [("rx", 12), ("ry", 13), ("rz", 14)]),
    ("r_lowleg",  [("rx", 15), ("ry", 16), ("rz", 17)]),
    ("l_ball",    [("rx", 21), ("ry", 22), ("rz", 23)]),
    ("r_ball",    [("rx", 24), ("ry", 25), ("rz", 26)]),
    ("c_spine3",  [("rx", 27), ("ry", 28), ("rz", 29)]),
    ("l_uparm",   [("rx", 48), ("ry", 49), ("rz", 50)]),
    ("r_uparm",   [("rx", 51), ("ry", 52), ("rz", 53)]),
    ("l_lowarm",  [("rx", 54), ("ry", 55), ("rz", 56)]),
    ("r_lowarm",  [("rx", 57), ("ry", 58), ("rz", 59)]),
]

STEP     = math.radians(5)   # radians per button click
AXIS_LEN = 0.10
DISP_H   = 1.0               # normalised display height (metres)

# SMPL-X IK joint indices (smplx_idx → ik_name)
_SMPLX_IK = {sx: v[2] for sx, v in _JOINT_MAP.items()}

# ── SMPL-X finger edges ────────────────────────────────────────────────────
_SMPLX_FINGER_EDGES = [
    (18,20),(19,21),
    (20,25),(25,26),(26,27),
    (20,28),(28,29),(29,30),
    (20,31),(31,32),(32,33),
    (20,34),(34,35),(35,36),
    (20,37),(37,38),(38,39),
    (21,40),(40,41),(41,42),
    (21,43),(43,44),(44,45),
    (21,46),(46,47),(47,48),
    (21,49),(49,50),(50,51),
    (21,52),(52,53),(53,54),
]
_SMPLX_FINGER_JOINTS = sorted({j for e in _SMPLX_FINGER_EDGES for j in e})


def _flat_idx(joint_idx):
    """Map SMPL-X joint index to flat params array index (first component)."""
    if joint_idx == 0:
        return 0
    return 3 + (joint_idx - 1) * 3


def compute_smplx_world_rots(params):
    """params: (66,) → list of 22 world-frame Rotation objects (joints 0–21)."""
    local_rots = [R.from_rotvec(params[0:3])]  # joint 0 = global_orient
    for j in range(1, 22):
        aa = params[3 + (j-1)*3 : 3 + j*3]
        local_rots.append(R.from_rotvec(aa))

    world_rots = [None] * 22
    world_rots[0] = local_rots[0]
    for j in range(1, 22):
        p = SMPLX_PARENTS[j]
        world_rots[j] = world_rots[p] * local_rots[j]
    return world_rots


class App:
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev    = torch.device(self.device)
        self.params = np.zeros(N_PARAMS, dtype=np.float32)

        self.smplx_model = None

        self._geom_names = []
        self._val_labels = {}   # param_idx → gui.Label

        self._build_window()
        self._load_in_background()

    # ── Window / layout ────────────────────────────────────────────────────
    def _build_window(self):
        self.win = self.app.create_window(
            "SMPL-X Interactive — world-frame arrows", 1700, 950)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.win.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])
        self._scene.scene.scene.enable_sun_light(False)

        scroll = gui.ScrollableVert(4, gui.Margins(8, 8, 8, 8))
        self._panel = scroll

        title = gui.Label("IK-mapped joint controls")
        title.text_color = gui.Color(0.1, 0.1, 0.6)
        scroll.add_child(title)
        scroll.add_child(gui.Label(f"Step = {math.degrees(STEP):.0f}° per click  (axis-angle)"))
        scroll.add_child(gui.Label(""))

        for joint_name, axes in JOINTS_PARAMS:
            hdr = gui.Label(f"── {joint_name}")
            hdr.text_color = gui.Color(0.0, 0.4, 0.0)
            scroll.add_child(hdr)

            for ax_name, param_idx in axes:
                row = gui.Horiz(4)
                row.add_child(gui.Label(f"  {ax_name:<6}"))

                btn_m = gui.Button(" − ")
                btn_p = gui.Button(" + ")
                btn_m.vertical_padding_em = 0
                btn_p.vertical_padding_em = 0

                def _cb(idx, sign):
                    def _do():
                        self.params[idx] += sign * STEP
                        self._update_scene()
                    return _do
                btn_m.set_on_clicked(_cb(param_idx, -1))
                btn_p.set_on_clicked(_cb(param_idx, +1))

                row.add_child(btn_m)
                row.add_child(btn_p)

                val_lbl = gui.Label("  +0.0°")
                val_lbl.text_color = gui.Color(0.5, 0.1, 0.1)
                row.add_child(val_lbl)

                self._val_labels[param_idx] = val_lbl
                scroll.add_child(row)

            scroll.add_child(gui.Label(""))

        btn_reset = gui.Button("Reset to T-pose")
        btn_reset.set_on_clicked(self._on_reset)
        scroll.add_child(btn_reset)
        scroll.add_child(gui.Label(""))

        self._status = gui.Label("[Loading SMPL-X model — please wait...]")
        self._status.text_color = gui.Color(0.7, 0.3, 0.0)
        scroll.add_child(self._status)

        PANEL_W = 300
        def on_layout(ctx):
            r = self.win.content_rect
            self._scene.frame = gui.Rect(r.x, r.y, r.width - PANEL_W, r.height)
            self._panel.frame  = gui.Rect(r.x + r.width - PANEL_W, r.y, PANEL_W, r.height)

        self.win.set_on_layout(on_layout)
        self.win.add_child(self._scene)
        self.win.add_child(self._panel)

    # ── Loading ────────────────────────────────────────────────────────────
    def _load_in_background(self):
        def _load():
            try:
                import smplx
                smplx_path = find_smplx_path()
                assert smplx_path, "SMPL-X model not found — check find_smplx_path()"
                self.smplx_model = smplx.SMPLX(
                    model_path=smplx_path, gender="neutral",
                    use_pca=False, num_betas=10, num_expression_coeffs=10,
                ).to(self.dev)
                self.smplx_model.eval()
                gui.Application.instance.post_to_main_thread(
                    self.win, self._on_loaded)
            except Exception as e:
                import traceback; traceback.print_exc()
                msg = str(e)
                gui.Application.instance.post_to_main_thread(
                    self.win,
                    lambda: setattr(self._status, 'text', f"ERROR: {msg}"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self):
        self._status.text = "[Ready]  T-pose = all zeros.  Use +/− to pose joints."
        self._status.text_color = gui.Color(0.0, 0.5, 0.0)
        self._update_scene()
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    # ── Reset ──────────────────────────────────────────────────────────────
    def _on_reset(self):
        self.params[:] = 0.0
        for pidx, lbl in self._val_labels.items():
            lbl.text = "  +0.0°"
        self._update_scene()

    # ── Scene update ───────────────────────────────────────────────────────
    def _update_scene(self):
        if self.smplx_model is None:
            return

        for name in self._geom_names:
            self._scene.scene.remove_geometry(name)
        self._geom_names.clear()

        # ── Run SMPL-X forward ────────────────────────────────────────────
        go  = torch.tensor(self.params[0:3],  dtype=torch.float32, device=self.dev).unsqueeze(0)
        bp  = torch.tensor(self.params[3:66], dtype=torch.float32, device=self.dev).unsqueeze(0)
        with torch.no_grad():
            out = self.smplx_model(
                betas=torch.zeros(1, 10, device=self.dev),
                global_orient=go,
                body_pose=bp,
                left_hand_pose=torch.zeros(1, 45, device=self.dev),
                right_hand_pose=torch.zeros(1, 45, device=self.dev),
                jaw_pose=torch.zeros(1, 3, device=self.dev),
                leye_pose=torch.zeros(1, 3, device=self.dev),
                reye_pose=torch.zeros(1, 3, device=self.dev),
                expression=torch.zeros(1, 10, device=self.dev),
            )
        pos = out.joints[0].cpu().numpy()   # (J, 3) metres

        # ── Normalise display ─────────────────────────────────────────────
        root_m = pos[0].copy()
        head_m = pos[15]
        h_m    = max(head_m[1] - root_m[1], 0.1)
        sc     = DISP_H / h_m

        def P(j):
            return (pos[j] - root_m) * sc

        # ── Update value labels ───────────────────────────────────────────
        for pidx, lbl in self._val_labels.items():
            deg = math.degrees(float(self.params[pidx]))
            lbl.text = f"  {deg:+5.1f}°"

        # ── FK world rotations ────────────────────────────────────────────
        world_rots = compute_smplx_world_rots(self.params)

        # ── Body skeleton (all 22 joints) ─────────────────────────────────
        body_pts  = [P(j) for j in range(22)]
        body_idx  = {j: j for j in range(22)}
        self._add("body", build_lineset(body_pts, SMPLX_BODY_EDGES, [0.25, 0.55, 1.0]))

        # ── Finger skeleton ───────────────────────────────────────────────
        fi_joints = {j: P(j) for j in _SMPLX_FINGER_JOINTS if j < len(pos)}
        fi_idx    = {j: k for k, j in enumerate(sorted(fi_joints))}
        fi_pts    = [fi_joints[j] for j in sorted(fi_joints)]
        fi_edges  = [(fi_idx[a], fi_idx[b]) for a, b in _SMPLX_FINGER_EDGES
                     if a in fi_joints and b in fi_joints]
        if fi_pts:
            self._add("fingers", build_lineset(fi_pts, fi_edges, [0.5, 0.7, 1.0]))

        # ── World-frame arrows at IK joints ───────────────────────────────
        for gi, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
            pt      = P(sx_idx)
            R_world = world_rots[sx_idx]
            # R_corrector = I (T-pose world rots are identity), so corrected = R_world
            R_mat   = R_world.as_matrix()

            self._add(f"sph{gi}", sphere_at(pt, radius=0.016, color=(0.2, 0.8, 0.3)))
            for ai, arr in enumerate(make_axis_arrows(pt, R_mat, length=AXIS_LEN)):
                self._add(f"arr{gi}_{ai}", arr)

        # ── World reference triad ─────────────────────────────────────────
        triad = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.12, origin=np.array([-0.6, -0.55, 0.0]))
        self._add("triad", triad)

        self.win.post_redraw()

    def _add(self, name, geom):
        mat = rendering.MaterialRecord()
        if isinstance(geom, o3d.geometry.LineSet):
            mat.shader = "unlitLine"
            mat.line_width = 2.0
            cols = np.asarray(geom.colors)
            if len(cols):
                c = cols[0]
                mat.base_color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
        else:
            mat.shader = "defaultUnlit"
            cols = np.asarray(geom.vertex_colors)
            if len(cols) == 0:
                mat.base_color = (0.8, 0.8, 0.8, 1.0)
        self._scene.scene.add_geometry(name, geom, mat)
        self._geom_names.append(name)

    def run(self):
        self.app.run()


if __name__ == "__main__":
    App().run()
