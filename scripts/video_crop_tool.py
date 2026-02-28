#!/usr/bin/env python3

"""

python scripts/video_crop_tool.py \
/home/haziq/datasets/mocap/data/fit3d/train/s04/videos/65906101/warmup_9.mp4:109 \
/home/haziq/GMR/videos/unitree_g1_warmup_9_generated.mp4:109 \
/home/haziq/GMR/videos/unitree_g1_warmup_9_true.mp4:109 \
/home/haziq/GMR/videos/tienkung_warmup_9_generated.mp4:109 \
/home/haziq/GMR/videos/tienkung_warmup_9_true.mp4:110 \

python scripts/video_crop_tool.py \
/home/haziq/datasets/mocap/data/humaneva/train/S1/videos/C1/Box_1.avi:162 \
/home/haziq/GMR/videos/unitree_g1_Box_1_generated.mp4:162 \
/home/haziq/GMR/videos/unitree_g1_Box_1_true.mp4:162 \
/home/haziq/GMR/videos/tienkung_Box_1_generated.mp4:162 \
/home/haziq/GMR/videos/tienkung_Box_1_true.mp4:163 \

python scripts/video_crop_tool.py \
/home/haziq/datasets/mocap/data/fit3d/train/s03/videos/60457274/mule_kick.mp4:204 \
/home/haziq/GMR/videos/unitree_g1_mule_kick_generated.mp4:204 \
/home/haziq/GMR/videos/unitree_g1_mule_kick_true.mp4:204 \
/home/haziq/GMR/videos/tienkung_mule_kick_generated.mp4:204 \
/home/haziq/GMR/videos/tienkung_mule_kick_true.mp4:205 \

python scripts/video_crop_tool.py \
/home/haziq/datasets/mocap/data/fit3d/train/s03/videos/60457274/dumbbell_biceps_curls.mp4:109 \
/home/haziq/GMR/videos/unitree_g1_bicep_curl_generated.mp4:154 \
/home/haziq/GMR/videos/unitree_g1_bicep_curl_true.mp4:154 \
/home/haziq/GMR/videos/tienkung_bicep_curl_generated.mp4:154 \
/home/haziq/GMR/videos/tienkung_bicep_curl_true.mp4:155 \

python scripts/video_crop_tool.py \
/home/haziq/datasets/motion-x++/data/video/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1.mp4:22 \
/home/haziq/GMR/videos/unitree_g1_open_generated.mp4:22 \
/home/haziq/GMR/videos/unitree_g1_open_true.mp4:22 \
/home/haziq/GMR/videos/tienkung_open_generated.mp4:22 \
/home/haziq/GMR/videos/tienkung_open_true.mp4:23 \

Usage:
    python scripts/video_crop_tool.py <video1> [<video2> ...]

    Optionally append :<frame_number> to any video path to jump directly to
    that frame without showing a slider:

        python scripts/video_crop_tool.py video1.mp4:42 video2.mp4:150

    Mix is allowed — videos without a frame number still show the slider.

Controls:
    q          - Switch to TOP-LEFT corner selection mode
    w          - Switch to BOTTOM-RIGHT corner selection mode
    Left click - Place the currently selected corner (TL or BR)
                 On 2nd+ videos: place a box with the same size as the last crop
    Enter      - Crop the current frame and save as <video_stem>.jpg
    ESC        - Skip current video
    Arrow keys - Step one frame forward/backward (only when slider is shown)
"""

import cv2
import json
import numpy as np
import os
import sys
from pathlib import Path

WINDOW_NAME = "Video Crop Tool"
MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 900
BBOX_STATE_FILE = Path(".") / ".video_crop_box.json"


def compute_scale(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    return scale


def scale_up(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def _parse_video_args(args: list[str]) -> list[tuple[str, int | None]]:
    """Parse 'path/video.mp4[:frame]' entries into (path, frame_or_None) tuples."""
    result = []
    for arg in args:
        # Split on the LAST colon to avoid breaking Windows drive letters
        # but on Linux a plain colon after the path is unambiguous.
        if ":" in arg:
            # Try to parse the part after the last colon as an integer
            head, _, tail = arg.rpartition(":")
            try:
                frame = int(tail)
                result.append((head, frame))
                continue
            except ValueError:
                pass  # not a frame number, treat whole string as path
        result.append((arg, None))
    return result


class CropTool:
    def __init__(self, video_files: list[str]):
        self.video_files = _parse_video_args(video_files)

        # Bounding box in **original frame** coordinates
        self.pt1 = None   # (x, y) top-left
        self.pt2 = None   # (x, y) bottom-right
        self.selecting = "TL"  # current active corner: "TL" or "BR"

        # Remembered box size from the last successful crop
        self.saved_w: int | None = None
        self.saved_h: int | None = None

        self._frame: np.ndarray | None = None
        self._scale: float = 1.0
        self._cap: cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._cur_idx: int = 0
        self._first_video = True

        # When True, left-click sets individual corners (q/w mode) instead of
        # placing the saved-size box.  Reset to False after a saved-box click.
        self._manual_corner_mode: bool = False

        # Middle-click drag state (display coords)
        self._drag_origin: tuple[int, int] | None = None  # mouse pos when drag started
        self._drag_pt1: tuple[int, int] | None = None     # box pt1 when drag started
        self._drag_pt2: tuple[int, int] | None = None     # box pt2 when drag started

        # Stash loaded points so they survive the per-video reset
        self._preloaded_pt1 = None
        self._preloaded_pt2 = None

        self._load_box_state()

    # ------------------------------------------------------------------
    # Box state persistence
    # ------------------------------------------------------------------

    def _save_box_state(self):
        """Persist current box corners and saved size to BBOX_STATE_FILE."""
        data = {
            "pt1": list(self.pt1) if self.pt1 else None,
            "pt2": list(self.pt2) if self.pt2 else None,
            "saved_w": self.saved_w,
            "saved_h": self.saved_h,
        }
        BBOX_STATE_FILE.write_text(json.dumps(data, indent=2))
        print(f"[s] Box state saved → {BBOX_STATE_FILE}")

    def _load_box_state(self):
        """Load box state from BBOX_STATE_FILE if it exists."""
        if not BBOX_STATE_FILE.exists():
            return
        try:
            data = json.loads(BBOX_STATE_FILE.read_text())
            if data.get("pt1"):
                self.pt1 = tuple(data["pt1"])
                self._preloaded_pt1 = self.pt1
            if data.get("pt2"):
                self.pt2 = tuple(data["pt2"])
                self._preloaded_pt2 = self.pt2
            if data.get("saved_w") is not None:
                self.saved_w = data["saved_w"]
                self.saved_h = data["saved_h"]
                self._first_video = False  # treat saved size as already known
            print(f"[auto] Loaded box state from {BBOX_STATE_FILE}  "
                  f"(size {self.saved_w}×{self.saved_h}, "
                  f"TL={self.pt1}, BR={self.pt2})")
        except Exception as e:
            print(f"[WARN] Could not load box state: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _orig_to_disp(self, pt):
        return (int(pt[0] * self._scale), int(pt[1] * self._scale))

    def _disp_to_orig(self, x, y):
        return (int(x / self._scale), int(y / self._scale))

    def _clamp_to_frame(self, x, y):
        h, w = self._frame.shape[:2]
        return (max(0, min(x, w - 1)), max(0, min(y, h - 1)))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _refresh(self):
        if self._frame is None:
            return

        disp = scale_up(self._frame, self._scale).copy()
        dh, dw = disp.shape[:2]

        # Draw bounding box / points
        if self.pt1 and self.pt2:
            cv2.rectangle(disp, self._orig_to_disp(self.pt1),
                          self._orig_to_disp(self.pt2), (0, 255, 0), 2)
            # Corner indicators
            cv2.circle(disp, self._orig_to_disp(self.pt1), 6, (0, 200, 255), -1)
            cv2.circle(disp, self._orig_to_disp(self.pt2), 6, (255, 100, 0), -1)
        elif self.pt1:
            cv2.circle(disp, self._orig_to_disp(self.pt1), 6, (0, 200, 255), -1)
        elif self.pt2:
            cv2.circle(disp, self._orig_to_disp(self.pt2), 6, (255, 100, 0), -1)

        # HUD
        if self._first_video or self.saved_w is None:
            mode_str = (
                f"[q] TOP-LEFT  [w] BOTTOM-RIGHT   active: "
                f"{'TOP-LEFT' if self.selecting == 'TL' else 'BOTTOM-RIGHT'}"
            )
        else:
            mode_str = (
                f"Left-click = place box ({self.saved_w}x{self.saved_h})  "
                f"|  [q] adjust TL  [w] adjust BR"
            )

        overlay_lines = [
            mode_str,
            "[ENTER] crop & save    [ESC] skip    [+/-] resize box    [s] save box    [← →] step",
            "[middle-drag] move box",
        ]
        if self.pt1:
            overlay_lines.append(f"TL: {self.pt1}")
        if self.pt2:
            overlay_lines.append(f"BR: {self.pt2}")
        if self.pt1 and self.pt2:
            bw = abs(self.pt2[0] - self.pt1[0])
            bh = abs(self.pt2[1] - self.pt1[1])
            ar = bw / bh if bh else float("nan")
            overlay_lines.append(f"Size: {bw}×{bh}  AR={ar:.4f}")

        for i, line in enumerate(overlay_lines):
            y = 28 + i * 28
            cv2.putText(disp, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(disp, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1,
                        cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, disp)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_trackbar(self, val):
        self._cur_idx = val
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
            self._refresh()

    def _on_mouse(self, event, x, y, flags, param):
        # ---- Middle-button drag: move the whole box -------------------------
        if event == cv2.EVENT_MBUTTONDOWN:
            if self.pt1 and self.pt2:
                self._drag_origin = (x, y)
                self._drag_pt1 = self.pt1
                self._drag_pt2 = self.pt2
            return

        if event == cv2.EVENT_MOUSEMOVE and self._drag_origin is not None:
            if self._frame is None:
                return
            fh, fw = self._frame.shape[:2]
            dx = int((x - self._drag_origin[0]) / self._scale)
            dy = int((y - self._drag_origin[1]) / self._scale)
            bw = self._drag_pt2[0] - self._drag_pt1[0]
            bh = self._drag_pt2[1] - self._drag_pt1[1]
            nx1 = max(0, min(self._drag_pt1[0] + dx, fw - 1 - bw))
            ny1 = max(0, min(self._drag_pt1[1] + dy, fh - 1 - bh))
            self.pt1 = (nx1, ny1)
            self.pt2 = (nx1 + bw, ny1 + bh)
            self._refresh()
            return

        if event == cv2.EVENT_MBUTTONUP:
            self._drag_origin = None
            self._drag_pt1 = None
            self._drag_pt2 = None
            return

        # ---- Left-click: place corner or saved-size box ---------------------
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        ox, oy = self._disp_to_orig(x, y)
        ox, oy = self._clamp_to_frame(ox, oy)

        if not self._manual_corner_mode and not self._first_video and self.saved_w is not None:
            # Place saved-size box with click = top-left
            self.pt1 = (ox, oy)
            h, w = self._frame.shape[:2]
            self.pt2 = (
                min(ox + self.saved_w, w - 1),
                min(oy + self.saved_h, h - 1),
            )
        else:
            # Manual corner placement (q/w mode)
            if self.selecting == "TL":
                self.pt1 = (ox, oy)
            else:
                self.pt2 = (ox, oy)

        self._refresh()

    # ------------------------------------------------------------------
    # Box scaling
    # ------------------------------------------------------------------

    def _fit_box_to_frame(self):
        """Ensure the current box lies fully inside the frame.

        Two conditions can require a fix:
          1. The box is larger than the frame in at least one dimension
             → scale it down (keeping aspect ratio) around its centre.
          2. The box is smaller than the frame but shifted out of bounds
             → translate it back in (no resize).

        In both cases the final coordinates are hard-clamped to
        [0, fw-1] × [0, fh-1] to absorb any integer-rounding overshoot.
        """
        if self.pt1 is None or self.pt2 is None or self._frame is None:
            return
        fh, fw = self._frame.shape[:2]
        bw = float(self.pt2[0] - self.pt1[0])
        bh = float(self.pt2[1] - self.pt1[1])
        if bw <= 0 or bh <= 0:
            return

        cx = (self.pt1[0] + self.pt2[0]) / 2.0
        cy = (self.pt1[1] + self.pt2[1]) / 2.0

        # --- Step 1: scale down if the box is too large ---
        scale_w = (fw - 1) / bw if bw > fw - 1 else 1.0
        scale_h = (fh - 1) / bh if bh > fh - 1 else 1.0
        factor = min(scale_w, scale_h)
        if factor < 1.0:
            bw *= factor
            bh *= factor
            print(f"[auto] Box scaled by {factor:.3f} to fit {fw}×{fh} frame  "
                  f"→ new size {int(bw)}×{int(bh)}")

        # --- Step 2: translate centre so box stays inside frame ---
        cx = max(bw / 2.0, min(cx, (fw - 1) - bw / 2.0))
        cy = max(bh / 2.0, min(cy, (fh - 1) - bh / 2.0))

        # Hard clamp to guard against integer-rounding overshoot
        x1 = max(0, int(round(cx - bw / 2.0)))
        y1 = max(0, int(round(cy - bh / 2.0)))
        x2 = min(fw - 1, int(round(cx + bw / 2.0)))
        y2 = min(fh - 1, int(round(cy + bh / 2.0)))

        self.pt1 = (x1, y1)
        self.pt2 = (x2, y2)
        # Update remembered size to match this frame's fitted box
        self.saved_w = x2 - x1
        self.saved_h = y2 - y1

    def _scale_box(self, factor: float):
        """Scale the current bounding box by *factor* around its centre.

        The aspect ratio is always preserved. If scaling up would push any
        edge outside the frame, the factor is clamped uniformly so the box
        just touches the nearest frame boundary without distortion.
        """
        if self.pt1 is None or self.pt2 is None or self._frame is None:
            return
        fh, fw = self._frame.shape[:2]
        cx = (self.pt1[0] + self.pt2[0]) / 2.0
        cy = (self.pt1[1] + self.pt2[1]) / 2.0
        orig_hw = (self.pt2[0] - self.pt1[0]) / 2.0
        orig_hh = (self.pt2[1] - self.pt1[1]) / 2.0
        if orig_hw <= 0 or orig_hh <= 0:
            return

        hw = orig_hw * factor
        hh = orig_hh * factor

        # Maximum half-extents allowed by the frame around this centre
        max_hw = min(cx, (fw - 1) - cx)
        max_hh = min(cy, (fh - 1) - cy)

        # If either dimension would overflow, reduce the factor uniformly (AR preserved)
        if hw > max_hw or hh > max_hh:
            limiting = min(max_hw / orig_hw, max_hh / orig_hh)
            hw = orig_hw * limiting
            hh = orig_hh * limiting

        # Ensure at least 4 px in each dimension
        hw = max(hw, 2.0)
        hh = max(hh, 2.0)

        x1 = max(0, int(round(cx - hw)))
        y1 = max(0, int(round(cy - hh)))
        x2 = min(fw - 1, int(round(cx + hw)))
        y2 = min(fh - 1, int(round(cy + hh)))
        self.pt1 = (x1, y1)
        self.pt2 = (x2, y2)
        self._refresh()

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------

    def _load_frame(self, idx: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
        return ret

    def process_video(self, video_path: str, fixed_frame: int | None = None) -> bool:
        """Returns True if a crop was saved, False if skipped.

        Args:
            video_path:  Path to the video file.
            fixed_frame: If given, jump directly to this frame index and do
                         *not* show a slider. Arrow-key stepping is also
                         disabled in that mode.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {video_path}")
            return False
        self._cap = cap

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._total_frames = max(total, 1)

        # Determine starting frame
        start_frame = 0
        if fixed_frame is not None:
            start_frame = max(0, min(fixed_frame, self._total_frames - 1))
        self._cur_idx = start_frame

        self._scale = 1.0  # placeholder; set after first frame read

        if not self._load_frame(start_frame):
            print(f"[WARN] Cannot read frame {start_frame}: {video_path}")
            cap.release()
            return False

        self._scale = compute_scale(self._frame)

        # Restore preloaded box on first video; otherwise carry the box over
        if self._preloaded_pt1 is not None:
            self.pt1 = self._preloaded_pt1
            self.pt2 = self._preloaded_pt2
            self._preloaded_pt1 = None
            self._preloaded_pt2 = None
        # pt1/pt2 are intentionally NOT reset so the box persists across videos
        # Auto-scale the box if it doesn't fit the new frame resolution
        self._fit_box_to_frame()
        # Reset to saved-box placement mode for each new video
        self._manual_corner_mode = False
        if self._first_video:
            self.selecting = "TL"

        video_stem = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"Video : {video_stem}")
        print(f"Frames: {self._total_frames}  |  Size: {self._frame.shape[1]}x{self._frame.shape[0]}")
        if fixed_frame is not None:
            print(f"Frame : {start_frame} (fixed — no slider)")
        if self.saved_w:
            print(f"Saved box size: {self.saved_w}x{self.saved_h}")
        if fixed_frame is not None:
            print("Controls: q=TL  w=BR  click=place  ENTER=crop  ESC=skip")
        else:
            print("Controls: q=TL  w=BR  click=place  ENTER=crop  ESC=skip  ←→=step")

        # Recreate window to reset (optional) trackbar
        cv2.destroyWindow(WINDOW_NAME)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        show_trackbar = fixed_frame is None and self._total_frames > 1
        if show_trackbar:
            cv2.createTrackbar("Frame", WINDOW_NAME, start_frame,
                               self._total_frames - 1, self._on_trackbar)

        self._refresh()
        saved = False

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                self.selecting = "TL"
                self._manual_corner_mode = True
                print("Mode → set TOP-LEFT (manual)")
                self._refresh()

            elif key == ord('w'):
                self.selecting = "BR"
                self._manual_corner_mode = True
                print("Mode → set BOTTOM-RIGHT (manual)")
                self._refresh()

            elif key == 13 or key == 10:  # Enter
                if self.pt1 and self.pt2:
                    self._crop_and_save(video_stem)
                    self._save_box_state()  # auto-save on successful crop
                    saved = True
                    break
                else:
                    print("[!] Set both TL and BR points first (click or q/w + click)")

            elif key == ord('s'):  # manual save box state
                if self.pt1 and self.pt2:
                    self._save_box_state()
                else:
                    print("[!] No box to save yet")

            elif key == 27:  # ESC
                print(f"Skipped: {video_stem}")
                break

            elif key in (ord('+'), ord('=')):  # scale up
                self._scale_box(1.05)

            elif key == ord('-'):  # scale down
                self._scale_box(1.0 / 1.05)

            elif show_trackbar and key == 2:  # left arrow
                new_idx = max(0, self._cur_idx - 1)
                cv2.setTrackbarPos("Frame", WINDOW_NAME, new_idx)

            elif show_trackbar and key == 3:  # right arrow
                new_idx = min(self._total_frames - 1, self._cur_idx + 1)
                cv2.setTrackbarPos("Frame", WINDOW_NAME, new_idx)

            # Check if window was closed
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed — exiting.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        cap.release()
        self._first_video = False
        return saved

    # ------------------------------------------------------------------
    # Crop & save
    # ------------------------------------------------------------------

    def _crop_and_save(self, stem: str):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        self.saved_w = x2 - x1
        self.saved_h = y2 - y1

        crop = self._frame[y1:y2, x1:x2]
        out_path = os.path.join(os.getcwd(), f"{stem}.jpg")
        cv2.imwrite(out_path, crop)
        ar = self.saved_w / self.saved_h if self.saved_h else float("nan")
        print(f"[✓] Saved  → {out_path}")
        print(f"     box   : ({x1},{y1}) → ({x2},{y2})  size: {self.saved_w}×{self.saved_h}  AR={ar:.4f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        for video_path, fixed_frame in self.video_files:
            if not os.path.isfile(video_path):
                print(f"[WARN] File not found: {video_path}")
                continue
            self.process_video(video_path, fixed_frame)

        cv2.destroyAllWindows()
        print("\nAll videos processed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_VIDEOS = [
    "/home/haziq/GMR/videos/tienkung_warmup_9.mp4",
    "/home/haziq/GMR/videos/tienkung_Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1_new.mp4",
]

if __name__ == "__main__":
    videos = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_VIDEOS
    if not videos:
        print("Usage: python scripts/video_crop_tool.py <video1.mp4[:frame]> [video2.mp4[:frame] ...]")
        print("  e.g. python scripts/video_crop_tool.py clip1.mp4:42 clip2.mp4")
        sys.exit(1)
    CropTool(videos).run()
