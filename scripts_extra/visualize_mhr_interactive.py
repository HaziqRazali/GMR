"""
visualize_mhr_interactive.py
Interactive viewer: MHR pose with corrected arrows (R_corrector @ R_world).

Left panel  — +/- buttons for each upper-body joint axis (5 deg per click).
3D panel    — skeleton + corrected RGB triads at every IK joint.
              Green arrows = correct, world-aligned at T-pose.

Usage:
    conda run -n mhr_new python scripts_extra/visualize_mhr_interactive.py
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

# ── Upper body parameter indices ───────────────────────────────────────────
# (from model.character_torch.parameter_transform.parameter_names)
# Exactly the 12 joints in _JOINT_MAP (IK-mapped joints only)
JOINTS_PARAMS = [
    ("root",      [("rx",     3), ("ry",    4), ("rz",    5)]),
    ("l_upleg",   [("twist", 59), ("ry",   60), ("rz",   61)]),
    ("r_upleg",   [("twist", 50), ("ry",   51), ("rz",   52)]),
    ("l_lowleg",  [("twist", 63)]),
    ("r_lowleg",  [("twist", 54)]),
    ("l_ball",    [("bend",  67)]),
    ("r_ball",    [("bend",  58)]),
    ("c_spine3",  [("rx",    21), ("ry",   22), ("rz",   23)]),
    ("l_uparm",   [("twist", 43), ("ry",   44), ("rz",   45)]),
    ("r_uparm",   [("twist", 33), ("ry",   34), ("rz",   35)]),
    ("l_lowarm",  [("twist", 47)]),
    ("r_lowarm",  [("twist", 37)]),
]

STEP     = math.radians(5)   # radians per button click
AXIS_LEN = 0.10
DISP_H   = 1.0               # normalised display height (metres)
N_PARAMS = 204               # model_params input size (NOT len(parameter_names)=321)

# ── MHR finger edges ───────────────────────────────────────────────────────
_MHR_FINGER_EDGES = [
    (40,42),
    (42,43),(43,44),(44,45),(45,46),
    (42,48),(48,49),(49,50),
    (42,52),(52,53),(53,54),
    (42,56),(56,57),(57,58),
    (42,60),(60,61),(61,62),(62,63),
    (76,78),
    (78,79),(79,80),(80,81),(81,82),
    (78,84),(84,85),(85,86),
    (78,88),(88,89),(89,90),
    (78,92),(92,93),(93,94),
    (78,96),(96,97),(97,98),(98,99),
]
_MHR_FINGER_JOINTS = sorted({j for e in _MHR_FINGER_EDGES for j in e})


class App:
    def __init__(self):
        self.app    = gui.Application.instance
        self.app.initialize()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev    = torch.device(self.device)
        self.params = np.zeros(N_PARAMS, dtype=np.float32)
        self._tpose_params = np.zeros(N_PARAMS, dtype=np.float32)  # set after IK

        # Set at load time
        self.mhr_model   = None
        self.shape_p     = None
        self.expr_p      = None
        self.R_corrector = {}

        self._geom_names  = []
        self._val_labels  = {}   # param_idx → gui.Label

        self._build_window()
        self._load_in_background()

    # ── Window / layout ────────────────────────────────────────────────────
    def _build_window(self):
        self.win = self.app.create_window(
            "MHR Interactive — corrected arrows", 1700, 950)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.win.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])

        # Set a nice default lighting
        self._scene.scene.scene.enable_sun_light(False)

        scroll = gui.ScrollableVert(4, gui.Margins(8, 8, 8, 8))
        self._panel = scroll

        title = gui.Label("Upper-body joint controls")
        title.text_color = gui.Color(0.1, 0.1, 0.6)
        scroll.add_child(title)
        scroll.add_child(gui.Label(f"Step = {math.degrees(STEP):.0f}° per click"))
        scroll.add_child(gui.Label(""))

        # +/- button rows
        for joint_name, axes in JOINTS_PARAMS:
            hdr = gui.Label(f"── {joint_name}")
            hdr.text_color = gui.Color(0.0, 0.4, 0.0)
            scroll.add_child(hdr)

            for ax_name, param_idx in axes:
                row = gui.Horiz(4)
                lbl = gui.Label(f"  {ax_name:<6}")
                row.add_child(lbl)

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

            scroll.add_child(gui.Label(""))  # spacer

        # Reset button
        btn_reset = gui.Button("Reset to T-pose")
        btn_reset.set_on_clicked(self._on_reset)
        scroll.add_child(btn_reset)
        scroll.add_child(gui.Label(""))

        # Status
        self._status = gui.Label("[Loading models — please wait...]")
        self._status.text_color = gui.Color(0.7, 0.3, 0.0)
        scroll.add_child(self._status)

        # Layout
        PANEL_W = 300
        def on_layout(ctx):
            r = self.win.content_rect
            self._scene.frame = gui.Rect(r.x, r.y, r.width - PANEL_W, r.height)
            self._panel.frame  = gui.Rect(r.x + r.width - PANEL_W, r.y, PANEL_W, r.height)

        self.win.set_on_layout(on_layout)
        self.win.add_child(self._scene)
        self.win.add_child(self._panel)

    # ── Loading (background thread) ────────────────────────────────────────
    def _load_in_background(self):
        def _load():
            try:
                self.mhr_model = load_mhr_model(self.device)
                _, R_rest, _   = get_mhr_apose(self.mhr_model, self.dev)
                self.R_rest    = R_rest

                self.shape_p = torch.zeros(1, 45,  device=self.dev)
                self.expr_p  = torch.zeros(1, 72,  device=self.dev)

                smplx_path   = find_smplx_path()
                assert smplx_path, "SMPL-X model not found"
                smplx_joints = load_smplx_tpose(smplx_path, self.device)

                skel_tpose, _, tpose_model_p = optimise_mhr_tpose(
                    self.mhr_model, smplx_joints, self.dev, iters=500)

                self._tpose_params    = tpose_model_p.cpu().numpy().flatten().astype(np.float32)
                self.params[:]        = self._tpose_params

                R_corr = {}
                for sx_idx, (name, mhr_idx, ik_name) in _JOINT_MAP.items():
                    q = skel_tpose[mhr_idx, 3:7]
                    R_corr[ik_name] = R.from_quat(q).inv()
                self.R_corrector = R_corr

                gui.Application.instance.post_to_main_thread(
                    self.win, self._on_loaded)
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = str(e)
                gui.Application.instance.post_to_main_thread(
                    self.win,
                    lambda: setattr(self._status, 'text', f"ERROR: {msg}"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self):
        self._status.text = "[Ready]  Use + / − buttons to rotate joints."
        self._status.text_color = gui.Color(0.0, 0.5, 0.0)
        self._update_scene()
        # Fit camera to scene
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    # ── Reset ──────────────────────────────────────────────────────────────
    def _on_reset(self):
        self.params[:] = self._tpose_params
        for pidx, lbl in self._val_labels.items():
            deg = math.degrees(float(self.params[pidx]))
            lbl.text = f"  {deg:+5.1f}°"
        self._update_scene()

    # ── Scene update ───────────────────────────────────────────────────────
    def _update_scene(self):
        if self.mhr_model is None:
            return

        # ── Remove old geometry ────────────────────────────────────────────
        for name in self._geom_names:
            self._scene.scene.remove_geometry(name)
        self._geom_names.clear()

        # ── Run FK ────────────────────────────────────────────────────────
        model_p = torch.tensor(
            self.params[np.newaxis], dtype=torch.float32).to(self.dev)
        with torch.no_grad():
            _, skel = self.mhr_model(self.shape_p, model_p, self.expr_p)
        skel_np = skel[0].cpu().numpy()   # (127, 8)

        # Normalise display (root at origin, DISP_H tall)
        root_m = skel_np[1,   :3] / 100.0
        head_m = skel_np[113, :3] / 100.0
        h_m    = max(head_m[1] - root_m[1], 0.1)
        sc     = DISP_H / h_m

        def P(j):
            return (skel_np[j, :3] / 100.0 - root_m) * sc

        # ── Update value labels ────────────────────────────────────────────
        for pidx, lbl in self._val_labels.items():
            deg = math.degrees(float(self.params[pidx]))
            lbl.text = f"  {deg:+5.1f}°"

        # ── Body skeleton ─────────────────────────────────────────────────
        ik_names = [v[2] for v in _JOINT_MAP.values()]
        pos_d    = {ik_name: P(mhr_idx)
                    for _, (_, mhr_idx, ik_name) in _JOINT_MAP.items()}
        n_idx    = {n: i for i, n in enumerate(ik_names)}
        pts_body = [pos_d[n] for n in ik_names]
        edges_b  = [(n_idx[a], n_idx[b])
                    for a, b in _MHR_SKEL_EDGES_NAMES
                    if a in n_idx and b in n_idx]
        self._add("body", build_lineset(pts_body, edges_b, [0.25, 0.55, 1.0]))

        # ── Finger skeleton ───────────────────────────────────────────────
        fp    = {j: P(j) for j in _MHR_FINGER_JOINTS}
        fi    = {j: k for k, j in enumerate(sorted(fp))}
        flist = [fp[j] for j in sorted(fp)]
        fedge = [(fi[a], fi[b]) for a, b in _MHR_FINGER_EDGES
                 if a in fp and b in fp]
        self._add("fingers", build_lineset(flist, fedge, [0.5, 0.7, 1.0]))

        # ── Corrected triads at IK joints ─────────────────────────────────
        for gi, (sx_idx, (name, mhr_idx, ik_name)) in enumerate(_JOINT_MAP.items()):
            if ik_name not in self.R_corrector:
                continue
            pt      = pos_d[ik_name]
            R_world = R.from_quat(skel_np[mhr_idx, 3:7])
            R_corr  = (R_world * self.R_corrector[ik_name]).as_matrix()

            self._add(f"sph{gi}", sphere_at(pt, radius=0.016, color=(0.2, 0.8, 0.3)))
            for ai, arr in enumerate(make_axis_arrows(pt, R_corr, length=AXIS_LEN)):
                self._add(f"arr{gi}_{ai}", arr)

        # ── World-frame reference triad ────────────────────────────────────
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
