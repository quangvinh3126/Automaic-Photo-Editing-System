import json
import math
import os
import threading
from dataclasses import dataclass, asdict, field, fields

MPL_DIR = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)

import cv2
import imageio.v2 as imageio
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    RAWPY_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

APP_TITLE = "Photo Editor Pro X Rebuilt"
PREVIEW_MAX_DIM = 1800
FAST_PREVIEW_MAX_DIM = 1100
UNDO_LIMIT = 30
HIST_W = 180
HIST_H = 112
SIDE_CAR_SUFFIX = ".photoeditorx.json"
PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
STATE_DIR = os.path.join(os.path.dirname(__file__), ".photo_editor_state")
SESSION_FILE = os.path.join(STATE_DIR, "last_session.json")
SESSION_CACHE_DIR = os.path.join(STATE_DIR, "session_cache")

APP_BG = "#eef6f2"
TOPBAR_BG = "#f7fffb"
STATUS_BG = "#e3f1eb"
PANEL_BG = "#ffffff"
PANEL_ALT = "#ecf7f1"
PANEL_SECTION = "#f5fbf8"
PREVIEW_BG = "#21323d"
BORDER = "#c8ddd4"
TEXT_PRIMARY = "#17322a"
TEXT_MUTED = "#678277"
TEXT_INVERTED = "#f8fffd"
ACCENT_TEAL = "#19b38c"
ACCENT_TEAL_ACTIVE = "#109776"
ACCENT_BLUE = "#5c85ff"
ACCENT_BLUE_ACTIVE = "#456de4"
ACCENT_CORAL = "#ff8a6b"
ACCENT_CORAL_ACTIVE = "#ef7050"
ACCENT_GOLD = "#efc15d"
ACCENT_GOLD_ACTIVE = "#dfae45"
BUTTON_BG = "#e7f4ee"
BUTTON_ACTIVE = "#d5ece2"
BUTTON_TEXT = "#1a3a31"
SUCCESS_BG = "#5fc08d"
SUCCESS_ACTIVE = "#4ba877"
WARN_BG = "#f0bf62"
WARN_ACTIVE = "#dfa947"
DANGER_BG = "#de756f"
DANGER_ACTIVE = "#cb615a"

SUPPORTED_EXTS = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.raw *.dng *.nef *.cr2 *.arw"), ("All files", "*.*")]
SAVE_EXTS = [("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")]
FRAME_RATIOS = {"Original": None, "1:1": (1, 1), "4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9), "5:4": (5, 4), "A4 Portrait": (210, 297), "A4 Landscape": (297, 210), "ID 3x4": (3, 4), "ID 4x6": (4, 6)}
HSL_BANDS = [
    ("Red", 0),
    ("Orange", 15),
    ("Yellow", 30),
    ("Green", 60),
    ("Aqua", 90),
    ("Blue", 120),
    ("Purple", 140),
    ("Magenta", 165),
]

os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(SESSION_CACHE_DIR, exist_ok=True)


@dataclass
class EditParams:
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    gamma: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    clarity: float = 0.0
    texture: float = 0.0
    sharpness: float = 0.0
    sharp_radius: float = 1.5
    denoise_luma: float = 0.0
    denoise_color: float = 0.0
    dehaze: float = 0.0
    lens_distortion: float = 0.0
    chroma_fix: float = 0.0
    vignette: float = 0.0
    fade: float = 0.0
    grain: float = 0.0
    curve_shadow: float = 0.0
    curve_darks: float = 0.0
    curve_mids: float = 0.0
    curve_lights: float = 0.0
    curve_highlights: float = 0.0
    subject_light: float = 0.0
    background_blur: float = 0.0
    background_desat: float = 0.0
    body_slim: float = 0.0
    face_slim: float = 0.0
    skin_smooth: float = 0.0
    skin_whiten: float = 0.0
    acne_remove: float = 0.0
    eye_brighten: float = 0.0
    teeth_whiten: float = 0.0
    under_eye_soften: float = 0.0
    lip_enhance: float = 0.0
    skin_tone_balance: float = 0.0
    hsl_red_hue: float = 0.0
    hsl_red_sat: float = 0.0
    hsl_red_lum: float = 0.0
    hsl_orange_hue: float = 0.0
    hsl_orange_sat: float = 0.0
    hsl_orange_lum: float = 0.0
    hsl_yellow_hue: float = 0.0
    hsl_yellow_sat: float = 0.0
    hsl_yellow_lum: float = 0.0
    hsl_green_hue: float = 0.0
    hsl_green_sat: float = 0.0
    hsl_green_lum: float = 0.0
    hsl_aqua_hue: float = 0.0
    hsl_aqua_sat: float = 0.0
    hsl_aqua_lum: float = 0.0
    hsl_blue_hue: float = 0.0
    hsl_blue_sat: float = 0.0
    hsl_blue_lum: float = 0.0
    hsl_purple_hue: float = 0.0
    hsl_purple_sat: float = 0.0
    hsl_purple_lum: float = 0.0
    hsl_magenta_hue: float = 0.0
    hsl_magenta_sat: float = 0.0
    hsl_magenta_lum: float = 0.0


@dataclass
class ExportOptions:
    image_format: str = "Keep"
    jpeg_quality: int = 95
    long_edge: int = 0
    suffix: str = "_edited"
    name_pattern: str = "{name}_edited"
    preserve_metadata: bool = True
    output_sharpen_profile: str = "Custom"
    output_sharpen: float = 0.0


@dataclass
class LocalAdjustmentStroke:
    mask_type: str = "brush"
    effect: str = "Lighten"
    amount: float = 25.0
    radius_norm: float = 0.035
    softness: float = 0.65
    range_low: float = 0.0
    range_high: float = 1.0
    label: str = ""
    points: list = field(default_factory=list)


@dataclass
class BatchImageRecord:
    path: str
    edit_params: EditParams
    frame_name: str = "Original"
    modified_base_bgr: np.ndarray | None = None
    thumbnail_rgb: np.ndarray | None = None
    base_revision: int = 0
    local_adjustments: list = field(default_factory=list)


class PreviewCanvas(tk.Canvas):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self.configure(bg=PREVIEW_BG, highlightthickness=0)
        self.base_pil = None
        self.tk_img = None
        self.base_w = 1
        self.base_h = 1
        self.offset_x = 0
        self.offset_y = 0
        self.drag_last = None
        self.drag_mode = None
        self.drawn_bbox = (0, 0, 1, 1)
        self.grid_enabled = tk.BooleanVar(value=True)
        self.bind("<Configure>", lambda _e: self.app.request_preview_refresh())
        self.bind("<ButtonPress-1>", self._start_pan)
        self.bind("<B1-Motion>", self._pan)
        self.bind("<ButtonRelease-1>", self._end_drag)
        self.bind("<ButtonPress-3>", self._secondary_action)
        self.bind("<MouseWheel>", self._zoom_windows)
        self.bind("<Button-4>", lambda e: self._zoom(1.12, e.x, e.y))
        self.bind("<Button-5>", lambda e: self._zoom(1 / 1.12, e.x, e.y))
        self.bind("<Double-Button-1>", lambda _e: self.app.set_zoom_100() if self.app.zoom_mode == "fit" else self.app.set_zoom_fit())

    def _start_pan(self, event):
        if self.app.crop_mode:
            self.drag_mode = "crop"
            self.app.start_crop_drag(event.x, event.y)
            return
        if self.app.radial_mode:
            self.drag_mode = "radial"
            self.app.start_local_gradient("radial", event.x, event.y)
            return
        if self.app.linear_mode:
            self.drag_mode = "linear"
            self.app.start_local_gradient("linear", event.x, event.y)
            return
        if self.app.brush_mode:
            self.drag_mode = "brush"
            self.app.start_local_brush(event.x, event.y)
            return
        if self.app.clone_mode:
            self.drag_mode = "clone"
            self.app.apply_clone_from_canvas(event.x, event.y)
            return
        if self.app.heal_mode:
            self.drag_mode = "heal"
            self.app.heal_from_canvas(event.x, event.y)
            return
        if self.app.preview_mode_var.get() == "Split" and self.is_inside_image(event.x, event.y):
            self.drag_mode = "split"
            self.app.update_split_from_canvas(event.x)
            return
        self.drag_mode = "pan"
        self.drag_last = (event.x, event.y)

    def _pan(self, event):
        if self.drag_mode == "crop":
            self.app.update_crop_drag(event.x, event.y)
            return
        if self.drag_mode == "brush":
            self.app.update_local_brush(event.x, event.y)
            return
        if self.drag_mode in {"radial", "linear"}:
            self.app.update_local_gradient(event.x, event.y)
            return
        if self.drag_mode == "split":
            self.app.update_split_from_canvas(event.x)
            return
        if self.drag_mode == "clone":
            return
        if self.drag_mode != "pan":
            return
        if self.drag_last is None:
            return
        dx, dy = event.x - self.drag_last[0], event.y - self.drag_last[1]
        self.drag_last = (event.x, event.y)
        self.offset_x += dx
        self.offset_y += dy
        self.app.draw_preview()

    def _end_drag(self, _event):
        if self.drag_mode == "crop":
            self.app.finish_crop_drag()
        elif self.drag_mode == "brush":
            self.app.finish_local_brush()
        elif self.drag_mode in {"radial", "linear"}:
            self.app.finish_local_gradient()
        self.drag_last = None
        self.drag_mode = None

    def _secondary_action(self, event):
        if self.app.clone_mode:
            self.app.set_clone_source_from_canvas(event.x, event.y)
            return
        if self.app.brush_mode or self.app.radial_mode or self.app.linear_mode:
            self.app.clear_last_local_adjustment()
            return

    def _zoom(self, factor, x=None, y=None):
        self.app.zoom_factor = max(0.05, min(8.0, self.app.zoom_factor * factor))
        self.app.zoom_mode = "manual"
        if x is not None and y is not None:
            self.offset_x += (self.winfo_width() / 2 - x) * 0.08
            self.offset_y += (self.winfo_height() / 2 - y) * 0.08
        self.app.update_zoom_label()
        self.app.draw_preview()

    def _zoom_windows(self, event):
        self._zoom(1.12 if event.delta > 0 else 1 / 1.12, event.x, event.y)

    def reset_pan(self):
        self.offset_x = 0
        self.offset_y = 0

    def set_base_image(self, rgb_img):
        self.base_h, self.base_w = rgb_img.shape[:2]
        self.base_pil = Image.fromarray(rgb_img)

    def is_inside_image(self, x, y):
        bx, by, bw, bh = self.drawn_bbox
        return bx <= x <= bx + bw and by <= y <= by + bh

    def image_fraction_from_canvas(self, x, y):
        bx, by, bw, bh = self.drawn_bbox
        if bw <= 0 or bh <= 0:
            return None
        fx = np.clip((x - bx) / bw, 0.0, 1.0)
        fy = np.clip((y - by) / bh, 0.0, 1.0)
        return float(fx), float(fy)

    def draw_image(self):
        self.delete("all")
        cw, ch = max(self.winfo_width(), 100), max(self.winfo_height(), 100)
        self.create_rectangle(0, 0, cw, ch, fill=PREVIEW_BG, outline="")
        if self.base_pil is None:
            self.drawn_bbox = (0, 0, 1, 1)
            self.create_text(cw // 2, ch // 2, text="Load images to start editing", fill="#d8e8f2", font=("Segoe UI", 16, "bold"))
            return
        scale = min((cw - 20) / self.base_w, (ch - 20) / self.base_h) if self.app.zoom_mode == "fit" else self.app.zoom_factor
        scale = max(scale, 0.02)
        if self.app.zoom_mode == "fit":
            self.app.zoom_factor = scale
            self.app.update_zoom_label(refresh=False)
        draw_w, draw_h = max(1, int(self.base_w * scale)), max(1, int(self.base_h * scale))
        resample = Image.Resampling.BILINEAR if scale > 1 else Image.Resampling.LANCZOS
        self.tk_img = ImageTk.PhotoImage(self.base_pil.resize((draw_w, draw_h), resample))
        x = (cw - draw_w) // 2 + int(self.offset_x)
        y = (ch - draw_h) // 2 + int(self.offset_y)
        self.drawn_bbox = (x, y, draw_w, draw_h)
        self.create_image(x, y, anchor="nw", image=self.tk_img)
        if self.grid_enabled.get() and draw_w >= 60 and draw_h >= 60:
            for xx in [x + draw_w / 3, x + 2 * draw_w / 3]:
                self.create_line(xx, y, xx, y + draw_h, fill="#8aa6b7", dash=(2, 4))
            for yy in [y + draw_h / 3, y + 2 * draw_h / 3]:
                self.create_line(x, yy, x + draw_w, yy, fill="#8aa6b7", dash=(2, 4))
        if self.app.crop_rect_norm is not None:
            x0 = x + self.app.crop_rect_norm[0] * draw_w
            y0 = y + self.app.crop_rect_norm[1] * draw_h
            x1 = x + self.app.crop_rect_norm[2] * draw_w
            y1 = y + self.app.crop_rect_norm[3] * draw_h
            self.create_rectangle(x, y, x + draw_w, y0, fill="#000000", outline="", stipple="gray50")
            self.create_rectangle(x, y1, x + draw_w, y + draw_h, fill="#000000", outline="", stipple="gray50")
            self.create_rectangle(x, y0, x0, y1, fill="#000000", outline="", stipple="gray50")
            self.create_rectangle(x1, y0, x + draw_w, y1, fill="#000000", outline="", stipple="gray50")
            self.create_rectangle(x0, y0, x1, y1, outline=ACCENT_GOLD, width=2)
        if self.app.live_local_stroke is not None and self.app.live_local_stroke.points:
            self._draw_stroke_overlay(self.app.live_local_stroke, x, y, draw_w, draw_h, outline=ACCENT_CORAL)
        elif 0 <= self.app.local_adjustment_selection < len(self.app.local_adjustments):
            self._draw_stroke_overlay(self.app.local_adjustments[self.app.local_adjustment_selection], x, y, draw_w, draw_h, outline=ACCENT_BLUE)
        if self.app.clone_source_norm is not None:
            sx = x + self.app.clone_source_norm[0] * draw_w
            sy = y + self.app.clone_source_norm[1] * draw_h
            self.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline=ACCENT_BLUE, width=2)
            self.create_line(sx - 10, sy, sx + 10, sy, fill="#ffffff", width=1)
            self.create_line(sx, sy - 10, sx, sy + 10, fill="#ffffff", width=1)
        if self.app.preview_mode_var.get() == "Split":
            split_x = x + draw_w * self.app.split_position
            self.create_line(split_x, y, split_x, y + draw_h, fill="#ffffff", width=2)
            self.create_line(split_x, y, split_x, y + draw_h, fill=ACCENT_BLUE, dash=(5, 4))
        self.create_rectangle(x, y, x + draw_w, y + draw_h, outline="#c3d7e5", width=1)

    def _draw_stroke_overlay(self, stroke, x, y, draw_w, draw_h, outline):
        if not stroke.points:
            return
        mask_type = getattr(stroke, "mask_type", "brush")
        if mask_type == "radial" and len(stroke.points) >= 2:
            cx = x + stroke.points[0][0] * draw_w
            cy = y + stroke.points[0][1] * draw_h
            ex = x + stroke.points[1][0] * draw_w
            ey = y + stroke.points[1][1] * draw_h
            radius = max(6.0, math.hypot(ex - cx, ey - cy))
            self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=outline, width=2)
            self.create_line(cx, cy, ex, ey, fill=outline, dash=(3, 3))
            return
        if mask_type == "linear" and len(stroke.points) >= 2:
            x0 = x + stroke.points[0][0] * draw_w
            y0 = y + stroke.points[0][1] * draw_h
            x1 = x + stroke.points[1][0] * draw_w
            y1 = y + stroke.points[1][1] * draw_h
            self.create_line(x0, y0, x1, y1, fill=outline, width=3)
            self.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill=outline, outline="")
            self.create_oval(x1 - 4, y1 - 4, x1 + 4, y1 + 4, fill=outline, outline="")
            return
        points = []
        for px, py in stroke.points:
            points.extend([x + px * draw_w, y + py * draw_h])
        radius = max(4.0, stroke.radius_norm * max(draw_w, draw_h))
        if len(points) >= 4:
            self.create_line(*points, fill=outline, width=max(2, int(radius * 0.35)), smooth=True)
        lx = x + stroke.points[-1][0] * draw_w
        ly = y + stroke.points[-1][1] * draw_h
        self.create_oval(lx - radius, ly - radius, lx + radius, ly + radius, outline=outline, width=2)


class PhotoEditorProX:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x900")
        self.root.minsize(1220, 760)
        self.path = None
        self.original_bgr = None
        self.current_bgr = None
        self.full_res_bgr = None
        self.preview_cache = {}
        self.curve_cache = {}
        self.local_mask_cache = {}
        self.subject_cache = {}
        self.mesh_cache = {}
        self.feature_cache = {}
        self.lens_cache = {}
        self.face_cache = {}
        self.body_cache = {}
        self.source_cache = {}
        self.batch_items = []
        self.current_index = -1
        self.thumbnail_widgets = []
        self.batch_export_thread = None
        self.batch_export_running = False
        self.draw_job = None
        self.preview_render_job = None
        self.full_preview_job = None
        self.render_generation = 0
        self.current_full_generation = -1
        self.current_source_stamp = 0
        self.render_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.render_event = threading.Event()
        self.pending_render_request = None
        self.pending_analysis_request = None
        self.render_result_queue = []
        self.zoom_factor = 1.0
        self.zoom_mode = "fit"
        self.split_position = 0.5
        self.undo_stack = []
        self.redo_stack = []
        self.history_items = []
        self.edit_params = EditParams()
        self.local_adjustments = []
        self.export_options = ExportOptions()
        self._building_ui = False
        self._updating_sliders = False
        self._updating_hsl = False
        self.session_save_job = None
        self.stroke_undo_state = None
        self.curve_drag_field = None
        self.preview_mode_var = tk.StringVar(value="After")
        self.frame_var = tk.StringVar(value="Original")
        self.grid_var = tk.BooleanVar(value=True)
        self.clipping_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.info_var = tk.StringVar(value="No image")
        self.zoom_var = tk.StringVar(value="Fit")
        self.batch_title_var = tk.StringVar(value="No image loaded")
        self.batch_progress_var = tk.StringVar(value="0 items")
        self.crop_mode_var = tk.BooleanVar(value=False)
        self.heal_mode_var = tk.BooleanVar(value=False)
        self.brush_mode_var = tk.BooleanVar(value=False)
        self.clone_mode_var = tk.BooleanVar(value=False)
        self.radial_mode_var = tk.BooleanVar(value=False)
        self.linear_mode_var = tk.BooleanVar(value=False)
        self.heal_radius_var = tk.IntVar(value=28)
        self.brush_radius_var = tk.IntVar(value=42)
        self.brush_strength_var = tk.IntVar(value=32)
        self.brush_effect_var = tk.StringVar(value="Lighten")
        self.local_luma_min_var = tk.IntVar(value=0)
        self.local_luma_max_var = tk.IntVar(value=100)
        self.straighten_var = tk.DoubleVar(value=0.0)
        self.hsl_band_var = tk.StringVar(value=HSL_BANDS[0][0])
        self.clone_source_var = tk.StringVar(value="No clone source")
        self.copied_settings = None
        self.crop_mode = False
        self.heal_mode = False
        self.brush_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.crop_rect_norm = None
        self.crop_drag_anchor = None
        self.live_local_stroke = None
        self.clone_source_norm = None
        self.current_scene_profile = "Unknown"
        self.local_adjustment_selection = -1
        self.analysis_request_id = 0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.mp_face_detector = None
        self.mp_pose = None
        self.mp_face_mesh = None
        self.mp_selfie_segmentation = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.55)
            except Exception:
                self.mp_face_detector = None
            try:
                self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0, min_detection_confidence=0.45)
            except Exception:
                self.mp_pose = None
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.45)
            except Exception:
                self.mp_face_mesh = None
            try:
                self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
            except Exception:
                self.mp_selfie_segmentation = None
        self.render_worker = threading.Thread(target=self._render_worker_loop, daemon=True)
        self.render_worker.start()
        self._setup_theme()
        self._create_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self._poll_render_results)
        self.root.after(200, self.restore_last_session_if_available)
        self._set_status_ready()

    def _setup_theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Fresh.TNotebook", background=PANEL_BG, borderwidth=0, tabmargins=(4, 4, 4, 0))
        style.configure("Fresh.TNotebook.Tab", background=PANEL_ALT, foreground=TEXT_PRIMARY, padding=(12, 7), font=("Segoe UI", 10, "bold"))
        style.map("Fresh.TNotebook.Tab", background=[("selected", ACCENT_TEAL), ("active", "#d8f3ea")], foreground=[("selected", TEXT_INVERTED), ("active", TEXT_PRIMARY)])
        style.configure("Fresh.TCombobox", fieldbackground=PANEL_SECTION, background=PANEL_SECTION, foreground=TEXT_PRIMARY, arrowcolor=ACCENT_BLUE, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)

    def _toolbar_button(self, parent, text, cmd, width=9, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE):
        return tk.Button(parent, text=text, command=cmd, width=width, bg=bg, fg=fg, activebackground=activebackground, activeforeground=fg, bd=0, relief="flat", font=("Segoe UI", 10, "bold"), padx=10, pady=5, highlightthickness=0, cursor="hand2")

    def _create_ui(self):
        self._building_ui = True
        self.root.configure(bg=APP_BG)
        topbar = tk.Frame(self.root, bg=TOPBAR_BG, height=60, highlightbackground=BORDER, highlightthickness=1)
        topbar.pack(side="top", fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="Photo Editor Pro X", bg=TOPBAR_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 14, "bold")).pack(side="left", padx=(12, 10))
        self._toolbar_button(topbar, "Open Files", self.load_image, width=10, bg=ACCENT_TEAL, fg=TEXT_INVERTED, activebackground=ACCENT_TEAL_ACTIVE).pack(side="left", padx=(0, 6), pady=10)
        self._toolbar_button(topbar, "Open Folder", self.load_folder, width=11, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Save Current", self.save_image, width=11).pack(side="left", padx=(14, 6), pady=10)
        self._toolbar_button(topbar, "Export Batch", self.export_batch_dialog, width=11, bg=ACCENT_CORAL, fg=TEXT_INVERTED, activebackground=ACCENT_CORAL_ACTIVE).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Copy", self.copy_current_settings, width=7).pack(side="left", padx=(14, 6), pady=10)
        self._toolbar_button(topbar, "Paste", self.paste_settings_to_current, width=7).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Variant", self.duplicate_current_variant, width=8).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Sync All", self.sync_current_to_all, width=8).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Undo", self.undo, width=7).pack(side="left", padx=(14, 6), pady=10)
        self._toolbar_button(topbar, "Redo", self.redo, width=7).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Reset", self.reset_image, width=7, bg=DANGER_BG, fg=TEXT_INVERTED, activebackground=DANGER_ACTIVE).pack(side="left", padx=6, pady=10)
        self._toolbar_button(topbar, "Auto", self.auto_enhance, width=7, bg=WARN_BG, fg=TEXT_PRIMARY, activebackground=WARN_ACTIVE).pack(side="left", padx=6, pady=10)

        tk.Label(topbar, textvariable=self.zoom_var, bg=TOPBAR_BG, fg=TEXT_MUTED, font=("Segoe UI", 10, "bold")).pack(side="right", padx=(8, 14))
        self._toolbar_button(topbar, "100%", self.set_zoom_100, width=6).pack(side="right", padx=4, pady=10)
        self._toolbar_button(topbar, "Fit", self.set_zoom_fit, width=6).pack(side="right", padx=4, pady=10)
        tk.Checkbutton(topbar, text="Clipping", variable=self.clipping_var, command=lambda: self.request_preview_refresh(immediate=True), bg=TOPBAR_BG, fg=TEXT_PRIMARY, selectcolor=PANEL_ALT, activebackground=TOPBAR_BG, activeforeground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold")).pack(side="right", padx=6)
        tk.Checkbutton(topbar, text="Grid", variable=self.grid_var, command=self._toggle_grid, bg=TOPBAR_BG, fg=TEXT_PRIMARY, selectcolor=PANEL_ALT, activebackground=TOPBAR_BG, activeforeground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold")).pack(side="right", padx=8)
        preview_choice = ttk.Combobox(topbar, textvariable=self.preview_mode_var, values=["After", "Before", "Split"], state="readonly", width=10, style="Fresh.TCombobox")
        preview_choice.pack(side="right", padx=(6, 10), pady=10)
        preview_choice.bind("<<ComboboxSelected>>", lambda _e: self.on_preview_mode_change())

        body = tk.PanedWindow(self.root, orient="horizontal", sashrelief="flat", bg=APP_BG, sashwidth=8)
        body.pack(fill="both", expand=True)
        left_panel, center_panel, right_panel = tk.Frame(body, bg=PANEL_BG, width=255, highlightbackground=BORDER, highlightthickness=1), tk.Frame(body, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1), tk.Frame(body, bg=PANEL_BG, width=390, highlightbackground=BORDER, highlightthickness=1)
        body.add(left_panel, minsize=230)
        body.add(center_panel, minsize=600)
        body.add(right_panel, minsize=340)
        self._build_left_panel(left_panel)
        self._build_center_panel(center_panel)
        self._build_right_panel(right_panel)

        status = tk.Frame(self.root, bg=STATUS_BG, height=30, highlightbackground=BORDER, highlightthickness=1)
        status.pack(side="bottom", fill="x")
        status.pack_propagate(False)
        tk.Label(status, textvariable=self.status_var, bg=STATUS_BG, fg=TEXT_PRIMARY, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left", fill="x", expand=True, padx=10)
        tk.Label(status, textvariable=self.batch_progress_var, bg=STATUS_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side="left", padx=8)
        tk.Label(status, textvariable=self.info_var, bg=STATUS_BG, fg=ACCENT_BLUE, anchor="e", font=("Segoe UI", 9, "bold")).pack(side="right", padx=10)
        self._building_ui = False

    def _build_left_panel(self, parent):
        tk.Label(parent, text="Analysis", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 8))
        info_box = tk.LabelFrame(parent, text="Image Info", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        info_box.pack(fill="x", padx=10, pady=8)
        self.file_info_text = tk.Text(info_box, height=7, bg=PANEL_SECTION, fg=TEXT_PRIMARY, bd=0, wrap="word", font=("Segoe UI", 9))
        self.file_info_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.file_info_text.insert("1.0", "No image loaded")
        self.file_info_text.config(state="disabled")
        hist_box = tk.LabelFrame(parent, text="Histogram", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        hist_box.pack(fill="x", padx=10, pady=8)
        self.hist_canvas = tk.Canvas(hist_box, width=HIST_W, height=HIST_H, bg=PANEL_SECTION, highlightthickness=0)
        self.hist_canvas.pack(fill="x", padx=6, pady=6)
        preset_box = tk.LabelFrame(parent, text="Quick Presets", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        preset_box.pack(fill="x", padx=10, pady=8)
        presets = [("Natural", self.preset_natural), ("Vivid", self.preset_vivid), ("Portrait", self.preset_portrait), ("Landscape", self.preset_landscape), ("B&W", self.preset_bw), ("Beauty", self.preset_beauty)]
        for i, (name, cmd) in enumerate(presets):
            accent_bg = ACCENT_TEAL if name in {"Natural", "Portrait"} else ACCENT_BLUE if name in {"Vivid", "Landscape"} else ACCENT_CORAL if name == "Beauty" else BUTTON_BG
            accent_fg = TEXT_INVERTED if accent_bg != BUTTON_BG else BUTTON_TEXT
            active_bg = ACCENT_TEAL_ACTIVE if accent_bg == ACCENT_TEAL else ACCENT_BLUE_ACTIVE if accent_bg == ACCENT_BLUE else ACCENT_CORAL_ACTIVE if accent_bg == ACCENT_CORAL else BUTTON_ACTIVE
            tk.Button(preset_box, text=name, command=cmd, bg=accent_bg, fg=accent_fg, activebackground=active_bg, activeforeground=accent_fg, bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2").grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        preset_box.grid_columnconfigure(0, weight=1)
        preset_box.grid_columnconfigure(1, weight=1)
        storage_box = tk.LabelFrame(parent, text="Preset / Sidecar", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        storage_box.pack(fill="x", padx=10, pady=8)
        storage_actions = [
            ("Save Preset", self.save_preset_dialog, ACCENT_BLUE, TEXT_INVERTED, ACCENT_BLUE_ACTIVE),
            ("Load Preset", self.load_preset_dialog, BUTTON_BG, BUTTON_TEXT, BUTTON_ACTIVE),
            ("Save Sidecar", self.save_current_sidecar, ACCENT_TEAL, TEXT_INVERTED, ACCENT_TEAL_ACTIVE),
            ("Load Sidecar", self.load_current_sidecar, BUTTON_BG, BUTTON_TEXT, BUTTON_ACTIVE),
        ]
        for i, (name, cmd, bg, fg, active) in enumerate(storage_actions):
            tk.Button(storage_box, text=name, command=cmd, bg=bg, fg=fg, activebackground=active, activeforeground=fg, bd=0, font=("Segoe UI", 9, "bold"), cursor="hand2").grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        storage_box.grid_columnconfigure(0, weight=1)
        storage_box.grid_columnconfigure(1, weight=1)
        history_box = tk.LabelFrame(parent, text="History", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        history_box.pack(fill="both", expand=True, padx=10, pady=8)
        self.history_list = tk.Listbox(history_box, height=8, bg=PANEL_SECTION, fg=TEXT_PRIMARY, bd=0, activestyle="none", font=("Segoe UI", 9), highlightthickness=0, selectbackground="#cfe9de", selectforeground=TEXT_PRIMARY)
        self.history_list.pack(fill="both", expand=True, padx=6, pady=6)
        tips_box = tk.LabelFrame(parent, text="Lightroom-Like Shortcuts", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        tips_box.pack(fill="x", padx=10, pady=8)
        tips = "Ctrl+Shift+C  Copy settings\nCtrl+Shift+V  Paste settings\nLeft / Right  Previous / Next image\nCtrl+Alt+S  Export batch\nCtrl+Shift+O  Open folder\nB / R / G / H / V  Brush / Radial / Linear / Heal / Clone"
        tk.Label(tips_box, text=tips, justify="left", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=8, pady=8)

    def _build_center_panel(self, parent):
        header = tk.Frame(parent, bg=PANEL_ALT)
        header.pack(fill="x", padx=10, pady=(10, 4))
        nav = tk.Frame(header, bg=PANEL_ALT)
        nav.pack(side="left")
        self._toolbar_button(nav, "< Prev", self.show_previous_image, width=8).pack(side="left", padx=(0, 6))
        tk.Label(nav, textvariable=self.batch_title_var, bg=PANEL_ALT, fg=TEXT_PRIMARY, font=("Segoe UI", 13, "bold")).pack(side="left")
        self._toolbar_button(nav, "Next >", self.show_next_image, width=8).pack(side="left", padx=(6, 0))
        tk.Label(header, text="Wheel: zoom | Drag: pan | Double click: Fit/100%", bg=PANEL_ALT, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side="right")
        self.preview_canvas = PreviewCanvas(parent, self)
        self.preview_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        actionbar = tk.Frame(parent, bg=PANEL_ALT)
        actionbar.pack(fill="x", padx=10, pady=(0, 6))
        self._toolbar_button(actionbar, "Copy Settings", self.copy_current_settings, width=13).pack(side="left", padx=(0, 6))
        self._toolbar_button(actionbar, "Paste Settings", self.paste_settings_to_current, width=13).pack(side="left")
        self._toolbar_button(actionbar, "Sync Current To All", self.sync_current_to_all, width=16, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=(6, 6))
        self._toolbar_button(actionbar, "Export Batch", self.export_batch_dialog, width=12, bg=ACCENT_CORAL, fg=TEXT_INVERTED, activebackground=ACCENT_CORAL_ACTIVE).pack(side="left")
        self._toolbar_button(actionbar, "Crop", self.toggle_crop_mode, width=8, bg=ACCENT_GOLD, fg=TEXT_PRIMARY, activebackground=ACCENT_GOLD_ACTIVE).pack(side="left", padx=(6, 4))
        self._toolbar_button(actionbar, "Apply Crop", self.apply_interactive_crop, width=10, bg=ACCENT_TEAL, fg=TEXT_INVERTED, activebackground=ACCENT_TEAL_ACTIVE).pack(side="left", padx=4)
        self._toolbar_button(actionbar, "Spot Heal", self.toggle_heal_mode, width=9, bg=WARN_BG, fg=TEXT_PRIMARY, activebackground=WARN_ACTIVE).pack(side="left", padx=4)
        self._toolbar_button(actionbar, "Brush", self.toggle_brush_mode, width=8, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=4)
        self._toolbar_button(actionbar, "Radial", self.toggle_radial_mode, width=8, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=4)
        self._toolbar_button(actionbar, "Linear", self.toggle_linear_mode, width=8, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=4)
        self._toolbar_button(actionbar, "Clone", self.toggle_clone_mode, width=8, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE).pack(side="left", padx=4)
        filmstrip_box = tk.Frame(parent, bg=PANEL_BG, height=152, highlightbackground=BORDER, highlightthickness=1)
        filmstrip_box.pack(fill="x", padx=10, pady=(0, 10))
        filmstrip_box.pack_propagate(False)
        self.filmstrip_canvas = tk.Canvas(filmstrip_box, bg=PANEL_SECTION, highlightthickness=0, height=124)
        self.filmstrip_scroll = tk.Scrollbar(filmstrip_box, orient="horizontal", command=self.filmstrip_canvas.xview)
        self.filmstrip_canvas.configure(xscrollcommand=self.filmstrip_scroll.set)
        self.filmstrip_scroll.pack(side="bottom", fill="x")
        self.filmstrip_canvas.pack(side="top", fill="both", expand=True)
        self.filmstrip_inner = tk.Frame(self.filmstrip_canvas, bg=PANEL_SECTION)
        self.filmstrip_window = self.filmstrip_canvas.create_window((0, 0), window=self.filmstrip_inner, anchor="nw")
        self.filmstrip_inner.bind("<Configure>", lambda _e: self.filmstrip_canvas.configure(scrollregion=self.filmstrip_canvas.bbox("all")))
        self.filmstrip_canvas.bind("<Configure>", lambda e: self.filmstrip_canvas.itemconfigure(self.filmstrip_window, height=e.height))

    def _build_right_panel(self, parent):
        tk.Label(parent, text="Edit Controls", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        crop_box = tk.LabelFrame(parent, text="Crop / Frame", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        crop_box.pack(fill="x", padx=10, pady=6)
        ttk.Combobox(crop_box, textvariable=self.frame_var, values=list(FRAME_RATIOS.keys()), state="readonly", style="Fresh.TCombobox").grid(row=0, column=0, padx=6, pady=8, sticky="ew")
        tk.Button(crop_box, text="Apply Frame", command=self.apply_standard_frame, bg=ACCENT_BLUE, fg=TEXT_INVERTED, activebackground=ACCENT_BLUE_ACTIVE, activeforeground=TEXT_INVERTED, bd=0, cursor="hand2").grid(row=0, column=1, padx=6, pady=8)
        tk.Label(crop_box, text="Straighten", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", padx=6, pady=(0, 2))
        straighten = tk.Scale(crop_box, from_=-15, to=15, resolution=0.1, orient="horizontal", variable=self.straighten_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8))
        straighten.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 8))
        tk.Button(crop_box, text="Apply Straighten", command=self.apply_straighten, bg=ACCENT_GOLD, fg=TEXT_PRIMARY, activebackground=ACCENT_GOLD_ACTIVE, activeforeground=TEXT_PRIMARY, bd=0, cursor="hand2").grid(row=2, column=1, padx=6, pady=(0, 8))
        tk.Label(crop_box, text="Heal Radius", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=3, column=0, sticky="w", padx=6, pady=(0, 2))
        tk.Scale(crop_box, from_=8, to=80, orient="horizontal", variable=self.heal_radius_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8)).grid(row=4, column=0, sticky="ew", padx=6, pady=(0, 8))
        tk.Button(crop_box, text="Cancel Modes", command=self.reset_interaction_modes, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").grid(row=4, column=1, padx=6, pady=(0, 8))
        crop_box.grid_columnconfigure(0, weight=1)
        qa_box = tk.LabelFrame(parent, text="Quick Actions", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        qa_box.pack(fill="x", padx=10, pady=6)
        actions = [("Gray", self.to_gray), ("CLAHE", self.apply_clahe_once), ("Sharpen+", self.apply_sharpen_once), ("Denoise", self.apply_denoise_once), ("Rotate L", lambda: self.rotate_image(-90)), ("Rotate R", lambda: self.rotate_image(90)), ("Flip H", self.flip_horizontal), ("Flip V", self.flip_vertical)]
        for i, (name, cmd) in enumerate(actions):
            tk.Button(qa_box, text=name, command=cmd, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        qa_box.grid_columnconfigure(0, weight=1)
        qa_box.grid_columnconfigure(1, weight=1)
        auto_box = tk.LabelFrame(parent, text="Auto Retouch", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        auto_box.pack(fill="x", padx=10, pady=6)
        auto_actions = [("Auto Face", self.auto_face_refine), ("Auto Skin", self.auto_skin_refine), ("Auto Body", self.auto_body_refine), ("Auto Beauty+", self.auto_beauty_plus)]
        for i, (name, cmd) in enumerate(auto_actions):
            tone_bg = ACCENT_TEAL if "Beauty" not in name else ACCENT_CORAL
            tone_active = ACCENT_TEAL_ACTIVE if tone_bg == ACCENT_TEAL else ACCENT_CORAL_ACTIVE
            tk.Button(auto_box, text=name, command=cmd, bg=tone_bg, fg=TEXT_INVERTED, activebackground=tone_active, activeforeground=TEXT_INVERTED, bd=0, cursor="hand2").grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        auto_box.grid_columnconfigure(0, weight=1)
        auto_box.grid_columnconfigure(1, weight=1)
        ai_box = tk.LabelFrame(parent, text="AI Assist", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        ai_box.pack(fill="x", padx=10, pady=(0, 6))
        ai_actions = [
            ("AI Scene", self.ai_scene_match),
            ("AI Portrait", self.ai_portrait_studio),
            ("AI Group", self.ai_group_retouch),
            ("AI Subject", self.ai_subject_pop),
        ]
        for i, (name, cmd) in enumerate(ai_actions):
            tone_bg = ACCENT_GOLD if name in {"AI Scene", "AI Subject"} else ACCENT_BLUE
            tone_fg = TEXT_PRIMARY if tone_bg == ACCENT_GOLD else TEXT_INVERTED
            tone_active = ACCENT_GOLD_ACTIVE if tone_bg == ACCENT_GOLD else ACCENT_BLUE_ACTIVE
            tk.Button(ai_box, text=name, command=cmd, bg=tone_bg, fg=tone_fg, activebackground=tone_active, activeforeground=tone_fg, bd=0, cursor="hand2").grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        ai_box.grid_columnconfigure(0, weight=1)
        ai_box.grid_columnconfigure(1, weight=1)
        notebook = ttk.Notebook(parent, style="Fresh.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=8)
        self.sliders = {}
        slider_tabs = {
            "Basic": [("Exposure", "exposure", -100, 100), ("Brightness", "brightness", -100, 100), ("Contrast", "contrast", -100, 100), ("Highlights", "highlights", -100, 100), ("Shadows", "shadows", -100, 100), ("Whites", "whites", -100, 100), ("Blacks", "blacks", -100, 100), ("Gamma", "gamma", -100, 100)],
            "Color": [("Temperature", "temperature", -100, 100), ("Tint", "tint", -100, 100), ("Saturation", "saturation", -100, 100), ("Vibrance", "vibrance", -100, 100)],
            "Curve": [("Shadow Point", "curve_shadow", -100, 100), ("Darks Point", "curve_darks", -100, 100), ("Mid Point", "curve_mids", -100, 100), ("Lights Point", "curve_lights", -100, 100), ("Highlights Point", "curve_highlights", -100, 100)],
            "Lens": [("Lens Distortion", "lens_distortion", -50, 50), ("Chromatic Fix", "chroma_fix", 0, 100)],
            "Detail": [("Clarity", "clarity", -100, 100), ("Texture", "texture", -100, 100), ("Sharpness", "sharpness", 0, 150), ("Sharp Radius", "sharp_radius", 5, 40), ("Denoise Luma", "denoise_luma", 0, 100), ("Denoise Color", "denoise_color", 0, 100), ("Dehaze", "dehaze", 0, 100)],
            "Mask": [("Subject Light", "subject_light", -100, 100), ("Background Blur", "background_blur", 0, 100), ("Background Desat", "background_desat", 0, 100)],
            "Effects": [("Vignette", "vignette", 0, 100), ("Fade", "fade", 0, 100), ("Grain", "grain", 0, 100)],
            "Beauty": [("Body Slim", "body_slim", 0, 100), ("Face Slim", "face_slim", 0, 100), ("Skin Smooth", "skin_smooth", 0, 100), ("Skin Whiten", "skin_whiten", 0, 100), ("Acne Remove", "acne_remove", 0, 100), ("Eye Brighten", "eye_brighten", 0, 100), ("Teeth Whiten", "teeth_whiten", 0, 100), ("Under-Eye Soften", "under_eye_soften", 0, 100), ("Lip Enhance", "lip_enhance", 0, 100)],
        }
        for tab_name, specs in slider_tabs.items():
            frame = tk.Frame(notebook, bg=PANEL_BG)
            notebook.add(frame, text=tab_name)
            row_offset = 0
            if tab_name == "Curve":
                self.curve_canvas = tk.Canvas(frame, width=240, height=110, bg=PANEL_SECTION, highlightthickness=0)
                self.curve_canvas.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))
                self.curve_canvas.bind("<ButtonPress-1>", self.start_curve_drag)
                self.curve_canvas.bind("<B1-Motion>", self.drag_curve_point)
                self.curve_canvas.bind("<ButtonRelease-1>", self.finish_curve_drag)
                row_offset = 1
            for row, (label, field_name, a, b) in enumerate(specs, start=row_offset):
                tk.Label(frame, text=label, bg=PANEL_BG, fg=TEXT_PRIMARY, anchor="w", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=8, pady=(8, 2))
                value_lbl = tk.Label(frame, text="0", width=5, bg=PANEL_BG, fg=ACCENT_BLUE, font=("Segoe UI", 9, "bold"))
                value_lbl.grid(row=row, column=2, sticky="e", padx=(0, 8))
                scale = tk.Scale(frame, from_=a, to=b, orient="horizontal", length=210, command=lambda v, f=field_name: self.on_slider_change(f, v), bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8))
                scale.set(getattr(self.edit_params, field_name))
                scale.grid(row=row, column=1, sticky="ew", padx=8, pady=(6, 4))
                self.sliders[field_name] = (scale, value_lbl)
            frame.grid_columnconfigure(1, weight=1)
        hsl_frame = tk.Frame(notebook, bg=PANEL_BG)
        notebook.add(hsl_frame, text="HSL")
        tk.Label(hsl_frame, text="Band", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(10, 4))
        band_combo = ttk.Combobox(hsl_frame, textvariable=self.hsl_band_var, values=[name for name, _center in HSL_BANDS], state="readonly", style="Fresh.TCombobox")
        band_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=(10, 4))
        band_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_hsl_band_ui())
        tk.Button(hsl_frame, text="Reset Band", command=self.reset_current_hsl_band, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").grid(row=0, column=2, padx=8, pady=(10, 4))
        self.hsl_sliders = {}
        for row, (label, component) in enumerate([("Hue Shift", "hue"), ("Saturation", "sat"), ("Luminance", "lum")], start=1):
            tk.Label(hsl_frame, text=label, bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=8, pady=(8, 2))
            value_lbl = tk.Label(hsl_frame, text="0", width=5, bg=PANEL_BG, fg=ACCENT_BLUE, font=("Segoe UI", 9, "bold"))
            value_lbl.grid(row=row, column=2, sticky="e", padx=(0, 8))
            scale = tk.Scale(hsl_frame, from_=-100, to=100, orient="horizontal", length=210, command=lambda v, c=component: self.on_hsl_slider_change(c, v), bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8))
            scale.grid(row=row, column=1, sticky="ew", padx=8, pady=(6, 4))
            self.hsl_sliders[component] = (scale, value_lbl)
        hsl_frame.grid_columnconfigure(1, weight=1)
        local_box = tk.LabelFrame(parent, text="Local Adjust / Clone", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        local_box.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(local_box, text="Local Effect", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ttk.Combobox(local_box, textvariable=self.brush_effect_var, values=["Lighten", "Darken", "Dodge", "Burn", "Color Pop", "Soften", "Texture"], state="readonly", style="Fresh.TCombobox").grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 2))
        tk.Label(local_box, text="Brush Radius", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", padx=8, pady=(8, 2))
        tk.Scale(local_box, from_=10, to=140, orient="horizontal", variable=self.brush_radius_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8)).grid(row=1, column=1, sticky="ew", padx=8, pady=(6, 4))
        tk.Label(local_box, text="Brush Strength", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", padx=8, pady=(8, 2))
        tk.Scale(local_box, from_=5, to=100, orient="horizontal", variable=self.brush_strength_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8)).grid(row=2, column=1, sticky="ew", padx=8, pady=(6, 4))
        tk.Label(local_box, text="Luma Range", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=3, column=0, sticky="w", padx=8, pady=(8, 2))
        range_row = tk.Frame(local_box, bg=PANEL_BG)
        range_row.grid(row=3, column=1, sticky="ew", padx=8, pady=(6, 4))
        tk.Scale(range_row, from_=0, to=100, orient="horizontal", variable=self.local_luma_min_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8), length=100).pack(side="left", fill="x", expand=True)
        tk.Scale(range_row, from_=0, to=100, orient="horizontal", variable=self.local_luma_max_var, bg=PANEL_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor="#cde6db", activebackground=ACCENT_TEAL, bd=0, sliderrelief="flat", font=("Segoe UI", 8), length=100).pack(side="left", fill="x", expand=True, padx=(6, 0))
        tk.Label(local_box, textvariable=self.clone_source_var, bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 8, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 6))
        local_actions = tk.Frame(local_box, bg=PANEL_BG)
        local_actions.grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        tk.Button(local_actions, text="Clear Local", command=self.clear_last_local_adjustment, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(local_actions, text="Delete Selected", command=self.delete_selected_local_adjustment, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(local_actions, text="Reset Clone Src", command=self.clear_clone_source, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").pack(side="left")
        stack_box = tk.LabelFrame(local_box, text="Local Stack", bg=PANEL_BG, fg=TEXT_PRIMARY, bd=1, highlightbackground=BORDER, highlightcolor=BORDER)
        stack_box.grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.local_stack_list = tk.Listbox(stack_box, height=4, bg=PANEL_SECTION, fg=TEXT_PRIMARY, bd=0, activestyle="none", font=("Segoe UI", 8), highlightthickness=0, selectbackground="#cfe9de", selectforeground=TEXT_PRIMARY)
        self.local_stack_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.local_stack_list.bind("<<ListboxSelect>>", lambda _e: self.select_local_adjustment())
        local_box.grid_columnconfigure(1, weight=1)
        bottom_buttons = tk.Frame(parent, bg=PANEL_BG)
        bottom_buttons.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(bottom_buttons, text="Apply Current as Base", command=self.commit_current_to_base, bg=SUCCESS_BG, fg=TEXT_INVERTED, activebackground=SUCCESS_ACTIVE, activeforeground=TEXT_INVERTED, bd=0, cursor="hand2", font=("Segoe UI", 10, "bold")).pack(side="left", fill="x", expand=True, padx=4, pady=4)
        tk.Button(bottom_buttons, text="Reset Sliders", command=self.reset_sliders_only, bg=DANGER_BG, fg=TEXT_INVERTED, activebackground=DANGER_ACTIVE, activeforeground=TEXT_INVERTED, bd=0, cursor="hand2", font=("Segoe UI", 10, "bold")).pack(side="left", fill="x", expand=True, padx=4, pady=4)
        self._sync_params_to_sliders()
        self._sync_hsl_band_ui()
        self.refresh_local_adjustment_stack()
        self.update_curve_graph()

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda _e: self.load_image())
        self.root.bind("<Control-Shift-O>", lambda _e: self.load_folder())
        self.root.bind("<Control-s>", lambda _e: self.save_image())
        self.root.bind("<Control-Shift-S>", lambda _e: self.save_preset_dialog())
        self.root.bind("<Control-Alt-s>", lambda _e: self.export_batch_dialog())
        self.root.bind("<Control-Alt-S>", lambda _e: self.export_batch_dialog())
        self.root.bind("<Control-Shift-C>", lambda _e: self.copy_current_settings())
        self.root.bind("<Control-Shift-V>", lambda _e: self.paste_settings_to_current())
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-y>", lambda _e: self.redo())
        self.root.bind("<Left>", lambda _e: self.show_previous_image())
        self.root.bind("<Right>", lambda _e: self.show_next_image())
        self.root.bind("0", lambda _e: self.set_zoom_fit())
        self.root.bind("1", lambda _e: self.set_zoom_100())
        self.root.bind("c", lambda _e: self.toggle_crop_mode())
        self.root.bind("h", lambda _e: self.toggle_heal_mode())
        self.root.bind("b", lambda _e: self.toggle_brush_mode())
        self.root.bind("r", lambda _e: self.toggle_radial_mode())
        self.root.bind("g", lambda _e: self.toggle_linear_mode())
        self.root.bind("v", lambda _e: self.toggle_clone_mode())

    def _toggle_grid(self):
        self.preview_canvas.grid_enabled.set(self.grid_var.get())
        self.draw_preview()

    def on_preview_mode_change(self):
        self.request_preview_refresh(immediate=True)
        self.schedule_session_save()

    def add_history_entry(self, text):
        self.history_items.append(text)
        if len(self.history_items) > 120:
            self.history_items = self.history_items[-120:]
        if hasattr(self, "history_list"):
            self.history_list.delete(0, "end")
            for item in reversed(self.history_items[-40:]):
                self.history_list.insert("end", item)

    def reset_interaction_modes(self):
        self.crop_mode = False
        self.heal_mode = False
        self.brush_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.crop_mode_var.set(False)
        self.heal_mode_var.set(False)
        self.brush_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_rect_norm = None
        self.crop_drag_anchor = None
        self.live_local_stroke = None
        self.stroke_undo_state = None
        self.request_preview_refresh(immediate=True)

    def toggle_crop_mode(self):
        self.heal_mode = False
        self.heal_mode_var.set(False)
        self.brush_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.brush_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_mode = not self.crop_mode
        self.crop_mode_var.set(self.crop_mode)
        if self.crop_mode:
            self.crop_rect_norm = None
            self.add_history_entry("Crop mode enabled")
            self.status_var.set("Crop mode: drag on preview to define a crop box")
        else:
            self.crop_rect_norm = None
            self.status_var.set("Crop mode disabled")
        self.request_preview_refresh(immediate=True)

    def toggle_heal_mode(self):
        self.crop_mode = False
        self.crop_mode_var.set(False)
        self.brush_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.brush_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_rect_norm = None
        self.heal_mode = not self.heal_mode
        self.heal_mode_var.set(self.heal_mode)
        self.status_var.set("Spot heal: click on the preview to heal small defects" if self.heal_mode else "Spot heal disabled")
        if self.heal_mode:
            self.add_history_entry("Spot heal mode enabled")
        self.request_preview_refresh(immediate=True)

    def toggle_brush_mode(self):
        self.crop_mode = False
        self.heal_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.crop_mode_var.set(False)
        self.heal_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_rect_norm = None
        self.brush_mode = not self.brush_mode
        self.brush_mode_var.set(self.brush_mode)
        self.live_local_stroke = None
        if self.brush_mode:
            self.status_var.set("Local brush: drag on preview. Right-click removes the last local stroke.")
            self.add_history_entry("Local brush mode enabled")
        else:
            self.status_var.set("Local brush mode disabled")
        self.request_preview_refresh(immediate=True)

    def toggle_clone_mode(self):
        self.crop_mode = False
        self.heal_mode = False
        self.brush_mode = False
        self.radial_mode = False
        self.linear_mode = False
        self.crop_mode_var.set(False)
        self.heal_mode_var.set(False)
        self.brush_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_rect_norm = None
        self.live_local_stroke = None
        self.clone_mode = not self.clone_mode
        self.clone_mode_var.set(self.clone_mode)
        if self.clone_mode:
            self.status_var.set("Clone mode: right-click to set source, left-click to stamp.")
            self.add_history_entry("Clone mode enabled")
        else:
            self.status_var.set("Clone mode disabled")
        self.request_preview_refresh(immediate=True)

    def toggle_radial_mode(self):
        self.crop_mode = False
        self.heal_mode = False
        self.brush_mode = False
        self.clone_mode = False
        self.linear_mode = False
        self.crop_mode_var.set(False)
        self.heal_mode_var.set(False)
        self.brush_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.linear_mode_var.set(False)
        self.crop_rect_norm = None
        self.live_local_stroke = None
        self.radial_mode = not self.radial_mode
        self.radial_mode_var.set(self.radial_mode)
        self.status_var.set("Radial local filter: drag from center outward." if self.radial_mode else "Radial local filter disabled")
        if self.radial_mode:
            self.add_history_entry("Radial local filter mode enabled")
        self.request_preview_refresh(immediate=True)

    def toggle_linear_mode(self):
        self.crop_mode = False
        self.heal_mode = False
        self.brush_mode = False
        self.clone_mode = False
        self.radial_mode = False
        self.crop_mode_var.set(False)
        self.heal_mode_var.set(False)
        self.brush_mode_var.set(False)
        self.clone_mode_var.set(False)
        self.radial_mode_var.set(False)
        self.crop_rect_norm = None
        self.live_local_stroke = None
        self.linear_mode = not self.linear_mode
        self.linear_mode_var.set(self.linear_mode)
        self.status_var.set("Linear local filter: drag to define the gradient axis." if self.linear_mode else "Linear local filter disabled")
        if self.linear_mode:
            self.add_history_entry("Linear local filter mode enabled")
        self.request_preview_refresh(immediate=True)

    def duplicate_current_variant(self):
        record = self.get_current_record()
        if record is None:
            return
        self._persist_current_record_state()
        record = self.get_current_record()
        duplicate = BatchImageRecord(
            path=record.path,
            edit_params=EditParams(**asdict(record.edit_params)),
            frame_name=record.frame_name,
            modified_base_bgr=None if record.modified_base_bgr is None else record.modified_base_bgr.copy(),
            thumbnail_rgb=None if record.thumbnail_rgb is None else record.thumbnail_rgb.copy(),
            base_revision=record.base_revision,
            local_adjustments=self.deserialize_local_adjustments(self.serialize_local_adjustments(record.local_adjustments)),
        )
        insert_at = self.current_index + 1
        self.batch_items.insert(insert_at, duplicate)
        self.refresh_filmstrip()
        self.open_record(insert_at)
        self.add_history_entry(f"Created variant for {os.path.basename(record.path)}")
        self.schedule_session_save()

    def local_adjustments_signature(self):
        return tuple(
            (
                getattr(stroke, "mask_type", "brush"),
                stroke.effect,
                round(float(stroke.amount), 3),
                round(float(stroke.radius_norm), 5),
                round(float(stroke.softness), 3),
                round(float(getattr(stroke, "range_low", 0.0)), 3),
                round(float(getattr(stroke, "range_high", 1.0)), 3),
                str(getattr(stroke, "label", "")),
                tuple((round(float(px), 4), round(float(py), 4)) for px, py in stroke.points),
            )
            for stroke in self.local_adjustments
        )

    def clone_local_adjustments(self):
        return [
            LocalAdjustmentStroke(
                mask_type=str(getattr(stroke, "mask_type", "brush")),
                effect=stroke.effect,
                amount=float(stroke.amount),
                radius_norm=float(stroke.radius_norm),
                softness=float(stroke.softness),
                range_low=float(getattr(stroke, "range_low", 0.0)),
                range_high=float(getattr(stroke, "range_high", 1.0)),
                label=str(getattr(stroke, "label", "")),
                points=[(float(px), float(py)) for px, py in stroke.points],
            )
            for stroke in self.local_adjustments
        ]

    def serialize_local_adjustments(self, strokes=None):
        strokes = self.local_adjustments if strokes is None else strokes
        return [asdict(stroke) for stroke in strokes]

    def deserialize_local_adjustments(self, payload):
        strokes = []
        if not payload:
            return strokes
        for item in payload:
            try:
                points = [(float(pt[0]), float(pt[1])) for pt in item.get("points", []) if len(pt) >= 2]
                strokes.append(
                    LocalAdjustmentStroke(
                        mask_type=str(item.get("mask_type", "brush")),
                        effect=str(item.get("effect", "Lighten")),
                        amount=float(item.get("amount", 25.0)),
                        radius_norm=float(item.get("radius_norm", 0.035)),
                        softness=float(item.get("softness", 0.65)),
                        range_low=float(item.get("range_low", 0.0)),
                        range_high=float(item.get("range_high", 1.0)),
                        label=str(item.get("label", "")),
                        points=points,
                    )
                )
            except Exception:
                continue
        return strokes

    def push_undo_snapshot(self, state):
        if state is None:
            return
        self.undo_stack.append(state)
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def create_state_snapshot(self, resolved_current=None):
        if self.original_bgr is None:
            return None
        if resolved_current is None:
            resolved_current = self.ensure_current_full_res(update_preview=False)
        if resolved_current is None:
            return None
        return {
            "original_bgr": self.original_bgr.copy(),
            "current_bgr": resolved_current.copy(),
            "params": asdict(self.edit_params),
            "frame": self.frame_var.get(),
            "path": self.path,
            "local_adjustments": self.serialize_local_adjustments(),
        }

    def clear_clone_source(self):
        self.clone_source_norm = None
        self.clone_source_var.set("No clone source")
        self.request_preview_refresh(immediate=True)

    def describe_local_adjustment(self, stroke, index=None):
        index_prefix = f"{index + 1}. " if index is not None else ""
        mask_name = getattr(stroke, "mask_type", "brush").title()
        label = stroke.label.strip() if getattr(stroke, "label", "") else f"{mask_name} {stroke.effect}"
        return f"{index_prefix}{label} ({int(round(stroke.amount))})"

    def refresh_local_adjustment_stack(self):
        if not hasattr(self, "local_stack_list"):
            return
        self.local_stack_list.delete(0, "end")
        for index, stroke in enumerate(self.local_adjustments):
            self.local_stack_list.insert("end", self.describe_local_adjustment(stroke, index=index))
        if self.local_adjustments:
            selection = int(np.clip(self.local_adjustment_selection, 0, len(self.local_adjustments) - 1))
            self.local_adjustment_selection = selection
            self.local_stack_list.selection_set(selection)
        else:
            self.local_adjustment_selection = -1

    def select_local_adjustment(self):
        if not hasattr(self, "local_stack_list"):
            return
        selection = self.local_stack_list.curselection()
        self.local_adjustment_selection = selection[0] if selection else -1
        if 0 <= self.local_adjustment_selection < len(self.local_adjustments):
            self.status_var.set(self.describe_local_adjustment(self.local_adjustments[self.local_adjustment_selection]))
        self.request_preview_refresh(immediate=True)

    def delete_selected_local_adjustment(self):
        if not self.local_adjustments:
            return
        self.select_local_adjustment()
        index = self.local_adjustment_selection if self.local_adjustment_selection >= 0 else len(self.local_adjustments) - 1
        snapshot = self.create_state_snapshot()
        if snapshot is not None:
            self.push_undo_snapshot(snapshot)
        removed = self.local_adjustments.pop(index)
        self.local_adjustment_selection = min(index, len(self.local_adjustments) - 1)
        self._persist_current_record_state()
        self.refresh_local_adjustment_stack()
        self.refresh_filmstrip()
        self.add_history_entry(f"Deleted local layer: {self.describe_local_adjustment(removed)}")
        self.schedule_render(immediate=True)
        self.schedule_session_save()

    def build_live_brush_stroke(self, mask_type="brush"):
        radius_norm = max(0.003, float(self.brush_radius_var.get()) / max(self.preview_canvas.base_w, self.preview_canvas.base_h, 1))
        range_low = min(float(self.local_luma_min_var.get()), float(self.local_luma_max_var.get())) / 100.0
        range_high = max(float(self.local_luma_min_var.get()), float(self.local_luma_max_var.get())) / 100.0
        return LocalAdjustmentStroke(
            mask_type=mask_type,
            effect=self.brush_effect_var.get(),
            amount=float(self.brush_strength_var.get()),
            radius_norm=radius_norm,
            softness=0.68,
            range_low=range_low,
            range_high=range_high,
            label=f"{mask_type.title()} {self.brush_effect_var.get()}",
            points=[],
        )

    def start_local_brush(self, x, y):
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        if self.stroke_undo_state is None:
            self.stroke_undo_state = self.create_state_snapshot()
        self.live_local_stroke = self.build_live_brush_stroke("brush")
        self.live_local_stroke.points.append(frac)
        self.schedule_render(immediate=False)
        self.request_preview_refresh(immediate=True)

    def start_local_gradient(self, mask_type, x, y):
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        if self.stroke_undo_state is None:
            self.stroke_undo_state = self.create_state_snapshot()
        self.live_local_stroke = self.build_live_brush_stroke(mask_type)
        self.live_local_stroke.points = [frac, frac]
        self.schedule_render(immediate=False)
        self.request_preview_refresh(immediate=True)

    def update_local_brush(self, x, y):
        if self.live_local_stroke is None:
            return
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        if not self.live_local_stroke.points or math.dist(frac, self.live_local_stroke.points[-1]) >= max(0.002, self.live_local_stroke.radius_norm * 0.18):
            self.live_local_stroke.points.append(frac)
            self.schedule_render(immediate=False)
            self.request_preview_refresh(immediate=True)

    def update_local_gradient(self, x, y):
        if self.live_local_stroke is None:
            return
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        if not self.live_local_stroke.points:
            self.live_local_stroke.points = [frac, frac]
        elif len(self.live_local_stroke.points) == 1:
            self.live_local_stroke.points.append(frac)
        else:
            self.live_local_stroke.points[1] = frac
        self.schedule_render(immediate=False)
        self.request_preview_refresh(immediate=True)

    def finish_local_brush(self):
        if self.live_local_stroke is None:
            return
        if len(self.live_local_stroke.points) >= 1:
            if self.stroke_undo_state is not None:
                self.push_undo_snapshot(self.stroke_undo_state)
            self.local_adjustments.append(self.live_local_stroke)
            self._persist_current_record_state()
            self.local_adjustment_selection = len(self.local_adjustments) - 1
            self.refresh_local_adjustment_stack()
            self.refresh_filmstrip()
            self.add_history_entry(f"Added local layer: {self.describe_local_adjustment(self.live_local_stroke)}")
            self.schedule_session_save()
        self.live_local_stroke = None
        self.stroke_undo_state = None
        self.schedule_render(immediate=True)
        self.request_preview_refresh(immediate=True)

    def finish_local_gradient(self):
        if self.live_local_stroke is None:
            return
        if len(self.live_local_stroke.points) >= 2 and math.dist(self.live_local_stroke.points[0], self.live_local_stroke.points[1]) >= 0.01:
            if self.stroke_undo_state is not None:
                self.push_undo_snapshot(self.stroke_undo_state)
            self.local_adjustments.append(self.live_local_stroke)
            self._persist_current_record_state()
            self.local_adjustment_selection = len(self.local_adjustments) - 1
            self.refresh_local_adjustment_stack()
            self.refresh_filmstrip()
            self.add_history_entry(f"Added local layer: {self.describe_local_adjustment(self.live_local_stroke)}")
            self.schedule_session_save()
        self.live_local_stroke = None
        self.stroke_undo_state = None
        self.schedule_render(immediate=True)
        self.request_preview_refresh(immediate=True)

    def clear_last_local_adjustment(self):
        if self.brush_mode and self.live_local_stroke is not None:
            self.live_local_stroke = None
            self.stroke_undo_state = None
            self.schedule_render(immediate=True)
            self.request_preview_refresh(immediate=True)
            return
        if not self.local_adjustments:
            return
        snapshot = self.create_state_snapshot()
        if snapshot is not None:
            self.push_undo_snapshot(snapshot)
        removed = self.local_adjustments.pop()
        self.local_adjustment_selection = len(self.local_adjustments) - 1
        self._persist_current_record_state()
        self.refresh_local_adjustment_stack()
        self.refresh_filmstrip()
        self.add_history_entry(f"Removed local layer: {self.describe_local_adjustment(removed)}")
        self.schedule_render(immediate=True)
        self.schedule_session_save()

    def set_clone_source_from_canvas(self, x, y):
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        self.clone_source_norm = frac
        self.clone_source_var.set(f"Clone source: {frac[0]:.2f}, {frac[1]:.2f}")
        self.status_var.set("Clone source set. Left-click to stamp.")
        self.request_preview_refresh(immediate=True)

    def apply_clone_patch(self, bgr, source_pt, dest_pt, radius):
        if bgr is None:
            return None
        out = bgr.copy()
        h, w = out.shape[:2]
        sx, sy = source_pt
        dx, dy = dest_pt
        radius = max(4, int(radius))
        x0s, y0s = max(0, sx - radius), max(0, sy - radius)
        x1s, y1s = min(w, sx + radius), min(h, sy + radius)
        source_patch = out[y0s:y1s, x0s:x1s]
        if source_patch.size == 0:
            return out
        x0d = int(np.clip(dx - source_patch.shape[1] // 2, 0, max(0, w - source_patch.shape[1])))
        y0d = int(np.clip(dy - source_patch.shape[0] // 2, 0, max(0, h - source_patch.shape[0])))
        x1d = x0d + source_patch.shape[1]
        y1d = y0d + source_patch.shape[0]
        if x1d > w or y1d > h:
            return out
        dest_patch = out[y0d:y1d, x0d:x1d].astype(np.float32)
        source_patch_f = source_patch.astype(np.float32)
        mask = np.zeros(source_patch.shape[:2], dtype=np.float32)
        cv2.circle(mask, (source_patch.shape[1] // 2, source_patch.shape[0] // 2), max(2, min(source_patch.shape[:2]) // 2 - 1), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, radius * 0.22))
        blended = dest_patch * (1.0 - mask[..., None]) + source_patch_f * mask[..., None]
        out[y0d:y1d, x0d:x1d] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def apply_clone_from_canvas(self, x, y):
        if self.clone_source_norm is None:
            self.status_var.set("Clone mode: right-click to define the source point first.")
            return
        src = self.ensure_current_full_res()
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if src is None or frac is None:
            return
        h, w = src.shape[:2]
        source_pt = (int(round(self.clone_source_norm[0] * w)), int(round(self.clone_source_norm[1] * h)))
        dest_pt = (int(round(frac[0] * w)), int(round(frac[1] * h)))
        radius = max(8, int(round(self.heal_radius_var.get() * (w / max(self.preview_canvas.drawn_bbox[2], 1)))))
        snapshot = self.create_state_snapshot(resolved_current=src)
        if snapshot is not None:
            self.push_undo_snapshot(snapshot)
        cloned = self.apply_clone_patch(src, source_pt, dest_pt, radius)
        self._replace_with_new_base(cloned, f"Cloned patch to {dest_pt[0]}, {dest_pt[1]}")

    def start_crop_drag(self, x, y):
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        self.crop_drag_anchor = frac
        self.crop_rect_norm = [frac[0], frac[1], frac[0], frac[1]]
        self.request_preview_refresh(immediate=True)

    def update_crop_drag(self, x, y):
        if self.crop_drag_anchor is None:
            return
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if frac is None:
            return
        x0, y0 = self.crop_drag_anchor
        self.crop_rect_norm = [min(x0, frac[0]), min(y0, frac[1]), max(x0, frac[0]), max(y0, frac[1])]
        self.request_preview_refresh(immediate=True)

    def finish_crop_drag(self):
        if self.crop_rect_norm is None:
            return
        x0, y0, x1, y1 = self.crop_rect_norm
        if (x1 - x0) < 0.02 or (y1 - y0) < 0.02:
            self.crop_rect_norm = None
        self.request_preview_refresh(immediate=True)

    def apply_interactive_crop(self):
        src = self.ensure_current_full_res()
        if src is None or self.crop_rect_norm is None:
            return
        x0, y0, x1, y1 = self.crop_rect_norm
        h, w = src.shape[:2]
        ix0, iy0 = int(np.clip(round(x0 * w), 0, w - 1)), int(np.clip(round(y0 * h), 0, h - 1))
        ix1, iy1 = int(np.clip(round(x1 * w), ix0 + 1, w)), int(np.clip(round(y1 * h), iy0 + 1, h))
        self.push_undo(resolved_current=src)
        self._replace_with_new_base(src[iy0:iy1, ix0:ix1], "Interactive crop applied")
        self.reset_interaction_modes()
        self.set_zoom_fit(reset_pan=True)

    def update_split_from_canvas(self, x):
        frac = self.preview_canvas.image_fraction_from_canvas(x, self.preview_canvas.drawn_bbox[1] + 1)
        if frac is None:
            return
        self.split_position = float(np.clip(frac[0], 0.05, 0.95))
        self.request_preview_refresh(immediate=True)
        self.schedule_session_save()

    def apply_spot_heal(self, bgr, cx, cy, radius):
        out = bgr.copy()
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(radius * 0.35, 1.0))
        mask = np.where(mask > 16, 255, 0).astype(np.uint8)
        if not np.any(mask):
            return out
        healed = cv2.inpaint(out, mask, max(2, int(radius * 0.28)), cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur((mask.astype(np.float32) / 255.0), (0, 0), max(radius * 0.25, 1.0))
        return np.clip(out.astype(np.float32) * (1.0 - alpha[..., None]) + healed.astype(np.float32) * alpha[..., None], 0, 255).astype(np.uint8)

    def heal_from_canvas(self, x, y):
        src = self.ensure_current_full_res()
        frac = self.preview_canvas.image_fraction_from_canvas(x, y)
        if src is None or frac is None:
            return
        h, w = src.shape[:2]
        cx, cy = int(round(frac[0] * w)), int(round(frac[1] * h))
        draw_w = max(self.preview_canvas.drawn_bbox[2], 1)
        scale = w / draw_w
        radius = max(6, int(round(self.heal_radius_var.get() * scale)))
        self.push_undo(resolved_current=src)
        healed = self.apply_spot_heal(src, cx, cy, radius)
        self._replace_with_new_base(healed, f"Spot healed at {cx}, {cy}")

    def apply_straighten(self):
        src = self.ensure_current_full_res()
        angle = float(self.straighten_var.get())
        if src is None or abs(angle) < 0.05:
            return
        self.push_undo(resolved_current=src)
        h, w = src.shape[:2]
        center = (w * 0.5, h * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(src, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        mask = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_rows = np.where(np.all(mask > 0, axis=1))[0]
        valid_cols = np.where(np.all(mask > 0, axis=0))[0]
        if valid_rows.size > 0 and valid_cols.size > 0:
            rotated = rotated[valid_rows[0]:valid_rows[-1] + 1, valid_cols[0]:valid_cols[-1] + 1]
        else:
            points = cv2.findNonZero(mask)
            if points is not None:
                x, y, bw, bh = cv2.boundingRect(points)
                rotated = rotated[y:y + bh, x:x + bw]
        self._replace_with_new_base(rotated, f"Straightened by {angle:.1f} degrees")
        self.straighten_var.set(0.0)
        self.set_zoom_fit(reset_pan=True)

    def clear_processing_cache(self):
        with self.render_lock:
            self.preview_cache.clear()
            self.curve_cache.clear()
            self.local_mask_cache.clear()
            self.face_cache.clear()
            self.body_cache.clear()
            self.subject_cache.clear()
            self.mesh_cache.clear()
            self.feature_cache.clear()

    def _mark_source_changed(self):
        self.current_source_stamp += 1

    def _set_full_resolution_state(self, bgr, source_changed=False):
        if source_changed:
            self._mark_source_changed()
        self.render_generation += 1
        self.current_bgr = bgr.copy()
        self.full_res_bgr = bgr.copy()
        self.current_full_generation = self.render_generation

    def is_supported_path(self, path):
        return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".raw", ".dng", ".nef", ".cr2", ".arw"}

    def read_thumbnail_image(self, path, max_dim=140):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".raw", ".dng", ".nef", ".cr2", ".arw"} and RAWPY_AVAILABLE:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, auto_bright_thr=0.01, no_auto_bright=False, output_bps=8, gamma=(1, 1), user_flip=0, half_size=True)
            thumb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            thumb_bgr = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
            if thumb_bgr is None:
                thumb_bgr = self.read_image(path)
        return cv2.cvtColor(self.downscale_for_preview(thumb_bgr, max_dim), cv2.COLOR_BGR2RGB)

    def create_batch_record(self, path):
        thumb_rgb = None
        try:
            thumb_rgb = self.read_thumbnail_image(path)
        except Exception:
            thumb_rgb = np.full((90, 120, 3), 220, dtype=np.uint8)
        record = BatchImageRecord(path=path, edit_params=EditParams(), thumbnail_rgb=thumb_rgb)
        self._maybe_load_sidecar_into_record(record)
        return record

    def make_thumbnail_from_bgr(self, bgr):
        return cv2.cvtColor(self.downscale_for_preview(bgr, 140), cv2.COLOR_BGR2RGB)

    def get_current_record(self):
        if 0 <= self.current_index < len(self.batch_items):
            return self.batch_items[self.current_index]
        return None

    def get_variant_label(self, index):
        if not (0 <= index < len(self.batch_items)):
            return ""
        path = self.batch_items[index].path
        seen = 0
        for idx, record in enumerate(self.batch_items):
            if record.path == path:
                seen += 1
            if idx == index:
                break
        return "" if seen <= 1 else f" V{seen}"

    def _persist_current_record_state(self):
        record = self.get_current_record()
        if record is None:
            return
        record.edit_params = EditParams(**asdict(self.edit_params))
        record.frame_name = self.frame_var.get()
        record.local_adjustments = self.deserialize_local_adjustments(self.serialize_local_adjustments())

    def _update_current_record_base(self, base_bgr):
        record = self.get_current_record()
        if record is None:
            return
        record.modified_base_bgr = base_bgr.copy()
        record.base_revision += 1
        record.thumbnail_rgb = self.make_thumbnail_from_bgr(base_bgr)

    def _load_record_base_image(self, record):
        if record.modified_base_bgr is not None:
            return record.modified_base_bgr.copy()
        cached = self.source_cache.get(record.path)
        if cached is not None:
            return cached.copy()
        img = self.read_image(record.path)
        self.source_cache[record.path] = img.copy()
        while len(self.source_cache) > 4:
            self.source_cache.pop(next(iter(self.source_cache)))
        return img

    def load_batch_paths(self, paths):
        normalized = []
        seen = set()
        for path in paths:
            full = os.path.abspath(path)
            if os.path.isfile(full) and self.is_supported_path(full) and full not in seen:
                seen.add(full)
                normalized.append(full)
        if not normalized:
            return
        self.status_var.set("Building filmstrip...")
        self.root.update_idletasks()
        records = []
        for idx, path in enumerate(sorted(normalized), start=1):
            records.append(self.create_batch_record(path))
            if idx % 8 == 0 or idx == len(normalized):
                self.batch_progress_var.set(f"Loading thumbs {idx}/{len(normalized)}")
                self.root.update_idletasks()
        self.batch_items = records
        self.current_index = -1
        self.source_cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_processing_cache()
        self.refresh_filmstrip()
        self.open_record(0)
        self.batch_progress_var.set(f"{len(self.batch_items)} items")
        self.add_history_entry(f"Loaded {len(self.batch_items)} image(s)")
        self.schedule_session_save()

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = [os.path.join(folder, name) for name in sorted(os.listdir(folder)) if self.is_supported_path(os.path.join(folder, name))]
        if not paths:
            messagebox.showinfo("Info", "No supported images found in this folder.")
            return
        try:
            self.load_batch_paths(paths)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open folder:\n{exc}")

    def build_sidecar_path(self, image_path):
        stem, _ext = os.path.splitext(image_path)
        return f"{stem}{SIDE_CAR_SUFFIX}"

    def edit_params_from_dict(self, payload):
        allowed = {item.name for item in fields(EditParams)}
        clean = {}
        for key, value in payload.items():
            if key in allowed:
                try:
                    clean[key] = float(value)
                except Exception:
                    continue
        return EditParams(**clean)

    def _maybe_load_sidecar_into_record(self, record):
        sidecar_path = self.build_sidecar_path(record.path)
        if not os.path.isfile(sidecar_path):
            return
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record.edit_params = self.edit_params_from_dict(payload.get("edit_params", {}))
            record.frame_name = payload.get("frame_name", "Original")
            record.local_adjustments = self.deserialize_local_adjustments(payload.get("local_adjustments", []))
        except Exception:
            return

    def save_preset_dialog(self):
        if self.original_bgr is None:
            return
        path = filedialog.asksaveasfilename(initialdir=PRESET_DIR, defaultextension=".json", filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        payload = {"app": APP_TITLE, "edit_params": asdict(self.edit_params), "saved_from": self.path}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.status_var.set(f"Preset saved: {os.path.basename(path)}")
        self.add_history_entry(f"Saved preset {os.path.basename(path)}")

    def apply_edit_params(self, params, description, local_adjustments=None):
        if self.original_bgr is None:
            return
        self.push_undo()
        self.edit_params = EditParams(**asdict(params))
        if local_adjustments is not None:
            self.local_adjustments = self.deserialize_local_adjustments(self.serialize_local_adjustments(local_adjustments))
        self.local_adjustment_selection = len(self.local_adjustments) - 1
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.clear_processing_cache()
        self._persist_current_record_state()
        self.refresh_filmstrip()
        self.schedule_render(immediate=True)
        self.add_history_entry(description)
        self.schedule_session_save()

    def load_preset_dialog(self):
        path = filedialog.askopenfilename(initialdir=PRESET_DIR, filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.apply_edit_params(self.edit_params_from_dict(payload.get("edit_params", {})), f"Loaded preset {os.path.basename(path)}")
            self.status_var.set(f"Preset loaded: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Preset", f"Could not load preset:\n{exc}")

    def save_current_sidecar(self):
        record = self.get_current_record()
        if record is None or self.path is None:
            return
        self._persist_current_record_state()
        sidecar_path = self.build_sidecar_path(record.path)
        payload = {
            "app": APP_TITLE,
            "image_path": record.path,
            "frame_name": record.frame_name,
            "edit_params": asdict(record.edit_params),
            "local_adjustments": self.serialize_local_adjustments(record.local_adjustments),
        }
        with open(sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.status_var.set(f"Sidecar saved: {os.path.basename(sidecar_path)}")
        self.add_history_entry(f"Saved sidecar for {os.path.basename(record.path)}")

    def load_current_sidecar(self):
        record = self.get_current_record()
        if record is None:
            return
        sidecar_path = self.build_sidecar_path(record.path)
        if not os.path.isfile(sidecar_path):
            messagebox.showinfo("Sidecar", "No sidecar file found for the current image.")
            return
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            params = self.edit_params_from_dict(payload.get("edit_params", {}))
            record.edit_params = params
            record.frame_name = payload.get("frame_name", "Original")
            record.local_adjustments = self.deserialize_local_adjustments(payload.get("local_adjustments", []))
            self.frame_var.set(record.frame_name)
            self.apply_edit_params(params, f"Loaded sidecar for {os.path.basename(record.path)}", local_adjustments=record.local_adjustments)
            self.status_var.set(f"Sidecar loaded: {os.path.basename(sidecar_path)}")
        except Exception as exc:
            messagebox.showerror("Sidecar", f"Could not load sidecar:\n{exc}")

    def schedule_session_save(self):
        if self._building_ui:
            return
        if self.session_save_job is not None:
            self.root.after_cancel(self.session_save_job)
        self.session_save_job = self.root.after(800, self.save_session_state)

    def save_session_state(self):
        self.session_save_job = None
        if not self.batch_items:
            return
        try:
            self._persist_current_record_state()
            for name in os.listdir(SESSION_CACHE_DIR):
                path = os.path.join(SESSION_CACHE_DIR, name)
                if os.path.isfile(path):
                    os.remove(path)
            records = []
            for index, record in enumerate(self.batch_items):
                modified_cache = None
                if record.modified_base_bgr is not None:
                    safe_name = f"{index:03d}_{os.path.splitext(os.path.basename(record.path))[0]}_{record.base_revision}.png"
                    modified_cache = os.path.join(SESSION_CACHE_DIR, safe_name)
                    cv2.imwrite(modified_cache, record.modified_base_bgr)
                records.append({
                    "path": record.path,
                    "edit_params": asdict(record.edit_params),
                    "frame_name": record.frame_name,
                    "base_revision": record.base_revision,
                    "modified_base_cache": modified_cache,
                    "local_adjustments": self.serialize_local_adjustments(record.local_adjustments),
                })
            payload = {
                "app": APP_TITLE,
                "current_index": self.current_index,
                "preview_mode": self.preview_mode_var.get(),
                "split_position": self.split_position,
                "records": records,
            }
            with open(SESSION_FILE, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            return

    def restore_last_session_if_available(self):
        if self.batch_items or not os.path.isfile(SESSION_FILE):
            return
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        records_payload = payload.get("records") or []
        if not records_payload:
            return
        if not messagebox.askyesno("Restore Session", f"Restore the previous session with {len(records_payload)} image(s)?"):
            return
        records = []
        for item in records_payload:
            path = item.get("path")
            if not path or not os.path.isfile(path):
                continue
            try:
                thumb = self.read_thumbnail_image(path)
            except Exception:
                thumb = np.full((90, 120, 3), 220, dtype=np.uint8)
            record = BatchImageRecord(
                path=path,
                edit_params=self.edit_params_from_dict(item.get("edit_params", {})),
                frame_name=item.get("frame_name", "Original"),
                thumbnail_rgb=thumb,
                base_revision=int(item.get("base_revision", 0)),
                local_adjustments=self.deserialize_local_adjustments(item.get("local_adjustments", [])),
            )
            modified_cache = item.get("modified_base_cache")
            if modified_cache and os.path.isfile(modified_cache):
                record.modified_base_bgr = cv2.imread(modified_cache, cv2.IMREAD_COLOR)
            records.append(record)
        if not records:
            return
        self.batch_items = records
        self.current_index = -1
        self.source_cache.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_processing_cache()
        self.preview_mode_var.set(payload.get("preview_mode", "After"))
        self.split_position = float(np.clip(payload.get("split_position", 0.5), 0.05, 0.95))
        self.refresh_filmstrip()
        restore_index = int(np.clip(payload.get("current_index", 0), 0, len(records) - 1))
        self.open_record(restore_index)
        self.add_history_entry("Restored previous session")

    def on_close(self):
        self.save_session_state()
        for model in (self.mp_face_detector, self.mp_pose, self.mp_face_mesh, self.mp_selfie_segmentation):
            if hasattr(model, "close"):
                try:
                    model.close()
                except Exception:
                    pass
        self.root.destroy()

    def show_export_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Batch Export")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=PANEL_BG)
        vars_map = {
            "output_dir": tk.StringVar(value=os.path.dirname(self.path) if self.path else os.getcwd()),
            "image_format": tk.StringVar(value=self.export_options.image_format),
            "jpeg_quality": tk.IntVar(value=self.export_options.jpeg_quality),
            "long_edge": tk.IntVar(value=self.export_options.long_edge),
            "suffix": tk.StringVar(value=self.export_options.suffix),
            "name_pattern": tk.StringVar(value=self.export_options.name_pattern),
            "preserve_metadata": tk.BooleanVar(value=self.export_options.preserve_metadata),
            "output_sharpen_profile": tk.StringVar(value=self.export_options.output_sharpen_profile),
            "output_sharpen": tk.DoubleVar(value=self.export_options.output_sharpen),
        }
        rows = [("Output Folder", "output_dir"), ("Format", "image_format"), ("JPEG Quality", "jpeg_quality"), ("Long Edge (0 keep)", "long_edge"), ("Suffix", "suffix"), ("Pattern", "name_pattern"), ("Sharpen Profile", "output_sharpen_profile"), ("Output Sharpen", "output_sharpen")]
        for row, (label, key) in enumerate(rows):
            tk.Label(dialog, text=label, bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=10, pady=6)
            if key == "image_format":
                ttk.Combobox(dialog, textvariable=vars_map[key], values=["Keep", "JPEG", "PNG", "TIFF"], state="readonly", style="Fresh.TCombobox").grid(row=row, column=1, sticky="ew", padx=10, pady=6)
            elif key == "output_sharpen_profile":
                ttk.Combobox(dialog, textvariable=vars_map[key], values=["Off", "Screen", "Print", "Custom"], state="readonly", style="Fresh.TCombobox").grid(row=row, column=1, sticky="ew", padx=10, pady=6)
            elif key == "output_dir":
                entry = tk.Entry(dialog, textvariable=vars_map[key], bg=PANEL_SECTION, fg=TEXT_PRIMARY, bd=0)
                entry.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
                tk.Button(dialog, text="Browse", command=lambda: vars_map["output_dir"].set(filedialog.askdirectory() or vars_map["output_dir"].get()), bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2").grid(row=row, column=2, padx=(0, 10), pady=6)
            else:
                tk.Entry(dialog, textvariable=vars_map[key], bg=PANEL_SECTION, fg=TEXT_PRIMARY, bd=0).grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        tk.Checkbutton(dialog, text="Preserve Metadata When Possible", variable=vars_map["preserve_metadata"], bg=PANEL_BG, fg=TEXT_PRIMARY, selectcolor=PANEL_ALT, activebackground=PANEL_BG, activeforeground=TEXT_PRIMARY, font=("Segoe UI", 9, "bold")).grid(row=len(rows), column=0, columnspan=2, sticky="w", padx=10, pady=(6, 0))
        tk.Label(dialog, text="Pattern tokens: {name}, {suffix}, {index}, {ext}", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 8)).grid(row=len(rows) + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))

        def submit():
            output_dir = vars_map["output_dir"].get().strip()
            if not output_dir or not os.path.isdir(output_dir):
                messagebox.showwarning("Export", "Choose a valid output directory.", parent=dialog)
                return
            try:
                options = ExportOptions(
                    image_format=vars_map["image_format"].get(),
                    jpeg_quality=int(vars_map["jpeg_quality"].get()),
                    long_edge=int(vars_map["long_edge"].get()),
                    suffix=vars_map["suffix"].get(),
                    name_pattern=vars_map["name_pattern"].get(),
                    preserve_metadata=bool(vars_map["preserve_metadata"].get()),
                    output_sharpen_profile=vars_map["output_sharpen_profile"].get(),
                    output_sharpen=float(vars_map["output_sharpen"].get()),
                )
            except Exception:
                messagebox.showwarning("Export", "Some export fields are invalid.", parent=dialog)
                return
            dialog.destroy()
            self.start_batch_export(output_dir, options)

        buttons = tk.Frame(dialog, bg=PANEL_BG)
        buttons.grid(row=len(rows) + 2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        tk.Button(buttons, text="Start Export", command=submit, bg=ACCENT_CORAL, fg=TEXT_INVERTED, activebackground=ACCENT_CORAL_ACTIVE, activeforeground=TEXT_INVERTED, bd=0, cursor="hand2", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 6))
        tk.Button(buttons, text="Cancel", command=dialog.destroy, bg=BUTTON_BG, fg=BUTTON_TEXT, activebackground=BUTTON_ACTIVE, activeforeground=BUTTON_TEXT, bd=0, cursor="hand2", font=("Segoe UI", 10, "bold")).pack(side="left")
        dialog.grid_columnconfigure(1, weight=1)

    def resize_for_export(self, image, long_edge):
        if long_edge is None or long_edge <= 0:
            return image
        h, w = image.shape[:2]
        scale = min(1.0, float(long_edge) / max(h, w))
        if scale >= 0.999:
            return image
        return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    def extract_source_metadata(self, source_path):
        try:
            with Image.open(source_path) as img:
                return img.info.get("exif")
        except Exception:
            return None

    def read_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".raw", ".dng", ".nef", ".cr2", ".arw"}:
            if not RAWPY_AVAILABLE:
                raise ImportError("rawpy is not installed")
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, auto_bright_thr=0.01, no_auto_bright=False, output_bps=8, gamma=(1, 1), user_flip=0)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        data = cv2.imread(path, cv2.IMREAD_COLOR)
        if data is not None:
            return data
        arr = imageio.imread(path)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.integer):
                max_value = np.iinfo(arr.dtype).max
                arr = np.clip(arr.astype(np.float32) / max(max_value, 1) * 255.0, 0, 255).astype(np.uint8)
            else:
                scale = 255.0 if float(np.nanmax(arr)) <= 1.2 else 1.0
                arr = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        raise ValueError("Unsupported image format")

    def load_image(self):
        paths = filedialog.askopenfilenames(filetypes=SUPPORTED_EXTS)
        if not paths:
            return
        try:
            self.load_batch_paths(paths)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image set:\n{exc}")

    def open_record(self, index):
        if not (0 <= index < len(self.batch_items)):
            return
        if index != self.current_index:
            self.undo_stack.clear()
            self.redo_stack.clear()
        self._persist_current_record_state()
        self.reset_interaction_modes()
        self.clear_clone_source()
        record = self.batch_items[index]
        try:
            base_bgr = self._load_record_base_image(record)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return
        self.current_index = index
        self.path = record.path
        self.original_bgr = base_bgr.copy()
        self.refresh_scene_profile(base_bgr)
        self.edit_params = EditParams(**asdict(record.edit_params))
        self.local_adjustments = self.deserialize_local_adjustments(self.serialize_local_adjustments(record.local_adjustments))
        self.local_adjustment_selection = len(self.local_adjustments) - 1
        self.frame_var.set(record.frame_name or "Original")
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.preview_canvas.reset_pan()
        self.clear_processing_cache()
        self._set_full_resolution_state(base_bgr, source_changed=True)
        self.update_file_info()
        self.update_histogram(self.current_bgr)
        self.schedule_render(immediate=True)
        self.queue_analysis_precompute(base_bgr.copy(), self.current_source_stamp)
        self.batch_title_var.set(f"{os.path.basename(record.path)}{self.get_variant_label(index)} ({index + 1} / {len(self.batch_items)})")
        self.batch_progress_var.set(f"{len(self.batch_items)} items")
        self.refresh_filmstrip()
        self.status_var.set(f"Loaded: {os.path.basename(record.path)}")
        self.add_history_entry(f"Opened {os.path.basename(record.path)}")
        self.schedule_session_save()

    def show_previous_image(self):
        if self.current_index > 0:
            self.open_record(self.current_index - 1)

    def show_next_image(self):
        if 0 <= self.current_index < len(self.batch_items) - 1:
            self.open_record(self.current_index + 1)

    def refresh_filmstrip(self):
        if not hasattr(self, "filmstrip_inner"):
            return
        for child in self.filmstrip_inner.winfo_children():
            child.destroy()
        self.thumbnail_widgets = []
        for index, record in enumerate(self.batch_items):
            selected = index == self.current_index
            frame_bg = ACCENT_TEAL if selected else PANEL_SECTION
            text_fg = TEXT_INVERTED if selected else TEXT_PRIMARY
            frame = tk.Frame(self.filmstrip_inner, bg=frame_bg, bd=0, padx=4, pady=4, cursor="hand2", highlightbackground=BORDER, highlightthickness=1)
            frame.pack(side="left", padx=4, pady=6)
            if record.thumbnail_rgb is None:
                try:
                    record.thumbnail_rgb = self.read_thumbnail_image(record.path)
                except Exception:
                    record.thumbnail_rgb = np.full((90, 120, 3), 220, dtype=np.uint8)
            thumb = Image.fromarray(record.thumbnail_rgb)
            photo = ImageTk.PhotoImage(thumb)
            image_label = tk.Label(frame, image=photo, bg=frame_bg, cursor="hand2")
            image_label.image = photo
            image_label.pack()
            name = os.path.basename(record.path)
            display_name = f"{name}{self.get_variant_label(index)}"
            short_name = display_name if len(display_name) <= 18 else f"{display_name[:15]}..."
            tk.Label(frame, text=short_name, bg=frame_bg, fg=text_fg, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(pady=(6, 0))
            badge_text = "Edited" if record.edit_params != EditParams() or record.modified_base_bgr is not None or bool(record.local_adjustments) else f"#{index + 1}"
            badge_color = "#d9fff2" if selected else TEXT_MUTED
            tk.Label(frame, text=badge_text, bg=frame_bg, fg=badge_color, font=("Segoe UI", 8), cursor="hand2").pack()
            for widget in frame.winfo_children() + [frame]:
                widget.bind("<Button-1>", lambda _e, idx=index: self.open_record(idx))
            self.thumbnail_widgets.append(frame)
        if hasattr(self, "filmstrip_canvas"):
            self.filmstrip_canvas.update_idletasks()
            self.filmstrip_canvas.configure(scrollregion=self.filmstrip_canvas.bbox("all"))

    def copy_current_settings(self):
        if self.original_bgr is None:
            return
        self._persist_current_record_state()
        self.copied_settings = EditParams(**asdict(self.edit_params))
        self.status_var.set("Copied current settings")
        self.add_history_entry("Copied current settings")

    def paste_settings_to_current(self):
        if self.copied_settings is None:
            messagebox.showinfo("Info", "No copied settings yet. Use Ctrl+Shift+C first.")
            return
        if self.original_bgr is None:
            return
        current = self.ensure_current_full_res(update_preview=False)
        if current is None:
            return
        self.push_undo(resolved_current=current)
        self.edit_params = EditParams(**asdict(self.copied_settings))
        self._sync_params_to_sliders()
        self.clear_processing_cache()
        self._persist_current_record_state()
        self.schedule_render(immediate=True)
        self.refresh_filmstrip()
        self.status_var.set("Pasted copied settings")
        self.add_history_entry("Pasted settings to current image")
        self.schedule_session_save()

    def sync_current_to_all(self):
        if not self.batch_items:
            return
        self._persist_current_record_state()
        template_params = EditParams(**asdict(self.edit_params))
        count = 0
        for index, record in enumerate(self.batch_items):
            if index == self.current_index:
                continue
            record.edit_params = EditParams(**asdict(template_params))
            count += 1
        self.batch_progress_var.set(f"{len(self.batch_items)} items | synced")
        self.refresh_filmstrip()
        self.status_var.set(f"Synced current settings to {count} other images")
        self.add_history_entry(f"Synced current settings to {count} image(s)")
        self.schedule_session_save()

    def save_image(self):
        out_bgr = self.ensure_current_full_res()
        if out_bgr is None:
            messagebox.showwarning("Info", "No image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=SAVE_EXTS)
        if not path:
            return
        self.write_processed_image(path, out_bgr, self.path, ExportOptions(image_format="Keep", jpeg_quality=95, preserve_metadata=True))
        self.status_var.set(f"Saved: {os.path.basename(path)}")
        self.add_history_entry(f"Saved current image to {os.path.basename(path)}")

    def export_batch_dialog(self):
        if self.batch_export_running:
            messagebox.showinfo("Info", "Batch export is already running.")
            return
        if not self.batch_items:
            messagebox.showinfo("Info", "Load images before exporting batch.")
            return
        self.show_export_dialog()

    def start_batch_export(self, output_dir, options):
        self._persist_current_record_state()
        snapshots = [
            BatchImageRecord(
                path=record.path,
                edit_params=EditParams(**asdict(record.edit_params)),
                frame_name=record.frame_name,
                modified_base_bgr=None if record.modified_base_bgr is None else record.modified_base_bgr.copy(),
                thumbnail_rgb=None,
                base_revision=record.base_revision,
                local_adjustments=self.deserialize_local_adjustments(self.serialize_local_adjustments(record.local_adjustments)),
            )
            for record in self.batch_items
        ]
        self.export_options = options
        self.batch_export_running = True
        self.batch_progress_var.set(f"Exporting 0/{len(snapshots)}")
        self.status_var.set("Batch export started...")
        self.batch_export_thread = threading.Thread(target=self._batch_export_worker, args=(output_dir, snapshots, options), daemon=True)
        self.batch_export_thread.start()
        self.add_history_entry(f"Started batch export to {output_dir}")

    def build_export_path(self, source_path, output_dir, options, index=None):
        stem, ext = os.path.splitext(os.path.basename(source_path))
        ext = ext.lower()
        desired_ext = {
            "Keep": ext if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} else ".jpg",
            "JPEG": ".jpg",
            "PNG": ".png",
            "TIFF": ".tif",
        }.get(options.image_format, ".jpg")
        pattern = options.name_pattern.strip() or "{name}_edited"
        try:
            filename = pattern.format(name=stem, suffix=options.suffix, index=index if index is not None else "", ext=desired_ext.lstrip("."))
        except Exception:
            filename = f"{stem}{options.suffix}"
        if not filename.lower().endswith(desired_ext):
            filename = f"{filename}{desired_ext}"
        return os.path.join(output_dir, filename)

    def write_processed_image(self, path, image, source_path=None, options=None):
        options = options or ExportOptions()
        image = self.resize_for_export(image, options.long_edge)
        sharpen_profile = getattr(options, "output_sharpen_profile", "Custom")
        sharpen_amount = float(options.output_sharpen)
        if sharpen_profile == "Off":
            sharpen_amount = 0.0
        elif sharpen_profile == "Screen":
            sharpen_amount = max(sharpen_amount, 16.0)
        elif sharpen_profile == "Print":
            sharpen_amount = max(sharpen_amount, 28.0)
        if sharpen_amount > 0:
            image = self.apply_sharpen_advanced(image, sharpen_amount, 0.95 if sharpen_profile == "Screen" else 1.15)
        ext = os.path.splitext(path)[1].lower()
        if options.preserve_metadata and source_path and os.path.isfile(source_path):
            exif = self.extract_source_metadata(source_path)
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            save_kwargs = {}
            if exif:
                save_kwargs["exif"] = exif
            if ext in {".jpg", ".jpeg"}:
                save_kwargs.update({"quality": int(np.clip(options.jpeg_quality, 70, 100)), "subsampling": 0})
                pil_img.save(path, format="JPEG", **save_kwargs)
                return
            if ext in {".tif", ".tiff"}:
                pil_img.save(path, format="TIFF", **save_kwargs)
                return
            if ext == ".png":
                pil_img.save(path, format="PNG")
                return
        if ext in {".jpg", ".jpeg"}:
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(options.jpeg_quality, 70, 100))])
        elif ext == ".png":
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        else:
            cv2.imwrite(path, image)

    def _batch_export_worker(self, output_dir, snapshots, options):
        failures = []
        total = len(snapshots)
        for index, record in enumerate(snapshots, start=1):
            try:
                base = record.modified_base_bgr.copy() if record.modified_base_bgr is not None else self.read_image(record.path)
                source_stamp = f"batch:{record.path}:{record.base_revision}"
                out = self.apply_pipeline(
                    base,
                    record.edit_params,
                    preview_mode=False,
                    source_bgr=base,
                    source_stamp=source_stamp,
                    local_adjustments=record.local_adjustments,
                )
                export_path = self.build_export_path(record.path, output_dir, options, index=index)
                self.write_processed_image(export_path, out, record.path, options)
            except Exception as exc:
                failures.append(f"{os.path.basename(record.path)}: {exc}")
            self.root.after(0, lambda i=index, t=total: self.batch_progress_var.set(f"Exporting {i}/{t}"))

        def finish():
            self.batch_export_running = False
            if failures:
                self.status_var.set(f"Batch export finished with {len(failures)} errors")
                messagebox.showwarning("Batch export", "\n".join(failures[:8]))
                self.add_history_entry(f"Batch export finished with {len(failures)} error(s)")
            else:
                self.status_var.set(f"Batch export finished: {total} images")
                self.add_history_entry(f"Batch exported {total} image(s)")
            self.batch_progress_var.set(f"{total} items | export done")

        self.root.after(0, finish)

    def push_undo(self, resolved_current=None):
        if self.original_bgr is None:
            return
        if resolved_current is None:
            resolved_current = self.ensure_current_full_res(update_preview=False)
        if resolved_current is None:
            return
        self.push_undo_snapshot(self.create_state_snapshot(resolved_current=resolved_current))

    def _restore_state(self, state):
        self.path = state.get("path")
        self.original_bgr = state["original_bgr"].copy()
        self.refresh_scene_profile(self.original_bgr)
        self.current_bgr = state["current_bgr"].copy()
        self.full_res_bgr = state["current_bgr"].copy()
        self.edit_params = EditParams(**state["params"])
        self.local_adjustments = self.deserialize_local_adjustments(state.get("local_adjustments", []))
        self.local_adjustment_selection = len(self.local_adjustments) - 1
        self.frame_var.set(state.get("frame", "Original"))
        self.clear_processing_cache()
        self._set_full_resolution_state(self.current_bgr, source_changed=True)
        self._update_current_record_base(self.original_bgr)
        self._persist_current_record_state()
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.update_file_info()
        self.update_histogram(self.current_bgr)
        self.request_preview_refresh(immediate=True)
        self.refresh_filmstrip()
        self.queue_analysis_precompute(self.original_bgr.copy(), self.current_source_stamp)

    def undo(self):
        if not self.undo_stack:
            return
        current = self.ensure_current_full_res(update_preview=False)
        if current is None:
            return
        self.redo_stack.append(self.create_state_snapshot(resolved_current=current))
        self._restore_state(self.undo_stack.pop())
        self.status_var.set("Undo")
        self.add_history_entry("Undo")
        self.schedule_session_save()

    def redo(self):
        if not self.redo_stack:
            return
        current = self.ensure_current_full_res(update_preview=False)
        if current is None:
            return
        self.undo_stack.append(self.create_state_snapshot(resolved_current=current))
        self._restore_state(self.redo_stack.pop())
        self.status_var.set("Redo")
        self.add_history_entry("Redo")
        self.schedule_session_save()

    def commit_current_to_base(self):
        current = self.ensure_current_full_res()
        if current is None:
            return
        self.push_undo(resolved_current=current)
        self.original_bgr = current.copy()
        self.refresh_scene_profile(current)
        self.edit_params = EditParams()
        self.local_adjustments = []
        self.local_adjustment_selection = -1
        self.frame_var.set("Original")
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.clear_processing_cache()
        self._set_full_resolution_state(current, source_changed=True)
        self._update_current_record_base(self.original_bgr)
        self._persist_current_record_state()
        self.update_histogram(self.current_bgr)
        self.update_file_info()
        self.request_preview_refresh(immediate=True)
        self.refresh_filmstrip()
        self.queue_analysis_precompute(current.copy(), self.current_source_stamp)
        self.status_var.set("Current image applied as base")
        self.add_history_entry("Committed current render as new base")
        self.schedule_session_save()

    def reset_image(self):
        if self.original_bgr is None:
            return
        self.push_undo()
        self.edit_params = EditParams()
        self.local_adjustments = []
        self.local_adjustment_selection = -1
        self.frame_var.set("Original")
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.clear_processing_cache()
        self._set_full_resolution_state(self.original_bgr, source_changed=False)
        self.refresh_scene_profile(self.original_bgr)
        self.update_histogram(self.current_bgr)
        self.update_file_info()
        self._persist_current_record_state()
        self.refresh_filmstrip()
        self.request_preview_refresh(immediate=True)
        self.status_var.set("Image reset")
        self.add_history_entry("Reset all adjustments")
        self.schedule_session_save()

    def reset_sliders_only(self):
        if self.original_bgr is None:
            return
        self.edit_params = EditParams()
        self._sync_params_to_sliders()
        self._persist_current_record_state()
        self.refresh_local_adjustment_stack()
        self.clear_processing_cache()
        self.refresh_filmstrip()
        self.schedule_render(immediate=True)
        self.add_history_entry("Reset sliders only")
        self.schedule_session_save()

    def request_preview_refresh(self, immediate=False):
        if immediate:
            self.draw_preview()
            return
        if self.draw_job is not None:
            self.root.after_cancel(self.draw_job)
        self.draw_job = self.root.after(16, self.draw_preview)

    def draw_preview(self):
        self.draw_job = None
        rgb = self.get_preview_rgb()
        if rgb is None:
            self.preview_canvas.base_pil = None
            self.preview_canvas.draw_image()
            return
        self.preview_canvas.set_base_image(rgb)
        self.preview_canvas.draw_image()

    def get_preview_rgb(self):
        if self.original_bgr is None:
            return None
        mode = self.preview_mode_var.get()
        after = self.current_bgr if self.current_bgr is not None else self.original_bgr
        if mode == "Before":
            composed = self.original_bgr
        elif mode == "Split":
            h, w = min(self.original_bgr.shape[0], after.shape[0]), min(self.original_bgr.shape[1], after.shape[1])
            before = cv2.resize(self.original_bgr, (w, h), interpolation=cv2.INTER_AREA)
            after_r = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)
            split_x = int(np.clip(round(w * self.split_position), 1, w - 1))
            composed = after_r.copy()
            composed[:, :split_x] = before[:, :split_x]
            cv2.line(composed, (split_x, 0), (split_x, h - 1), (255, 255, 255), 2)
        else:
            composed = after
        if self.clipping_var.get():
            composed = composed.copy()
            highlight_mask = np.min(composed, axis=2) >= 248
            shadow_mask = np.max(composed, axis=2) <= 7
            composed[highlight_mask] = (40, 40, 255)
            composed[shadow_mask] = (255, 90, 30)
        return cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)

    def set_zoom_fit(self, reset_pan=True):
        self.zoom_mode = "fit"
        if reset_pan:
            self.preview_canvas.reset_pan()
        self.update_zoom_label()
        self.draw_preview()

    def set_zoom_100(self):
        self.zoom_mode = "manual"
        self.zoom_factor = 1.0
        self.update_zoom_label()
        self.draw_preview()

    def update_zoom_label(self, refresh=True):
        self.zoom_var.set("Fit" if self.zoom_mode == "fit" else f"{int(self.zoom_factor * 100)}%")
        if refresh:
            self.root.update_idletasks()

    def get_active_local_adjustments(self):
        strokes = self.clone_local_adjustments()
        if self.live_local_stroke is not None and self.live_local_stroke.points:
            strokes.append(
                LocalAdjustmentStroke(
                    mask_type=self.live_local_stroke.mask_type,
                    effect=self.live_local_stroke.effect,
                    amount=self.live_local_stroke.amount,
                    radius_norm=self.live_local_stroke.radius_norm,
                    softness=self.live_local_stroke.softness,
                    range_low=self.live_local_stroke.range_low,
                    range_high=self.live_local_stroke.range_high,
                    label=self.live_local_stroke.label,
                    points=[(float(px), float(py)) for px, py in self.live_local_stroke.points],
                )
            )
        return strokes

    def has_non_destructive_adjustments(self):
        return self.edit_params != EditParams() or bool(self.local_adjustments) or (self.live_local_stroke is not None and bool(self.live_local_stroke.points))

    def schedule_render(self, immediate=False):
        if self.original_bgr is None or self._updating_sliders:
            return
        if self.preview_render_job is not None:
            self.root.after_cancel(self.preview_render_job)
        if self.full_preview_job is not None:
            self.root.after_cancel(self.full_preview_job)
        if not self.has_non_destructive_adjustments():
            self.clear_processing_cache()
            self._set_full_resolution_state(self.original_bgr, source_changed=False)
            self.update_histogram(self.current_bgr)
            self.update_file_info(current_only=True)
            self.request_preview_refresh(immediate=True)
            self.status_var.set("Render complete")
            return
        generation = self.render_generation + 1
        self.render_generation = generation
        if immediate:
            self._queue_render_request(generation, full_quality=False)
            self.full_preview_job = self.root.after(55, lambda gen=generation: self._queue_render_request(gen, full_quality=True))
        else:
            self.preview_render_job = self.root.after(35, lambda gen=generation: self._queue_render_request(gen, full_quality=False))
            self.full_preview_job = self.root.after(220, lambda gen=generation: self._queue_render_request(gen, full_quality=True))

    def _queue_render_request(self, generation, full_quality=False):
        if self.original_bgr is None or generation != self.render_generation:
            return
        if full_quality:
            self.full_preview_job = None
        else:
            self.preview_render_job = None
        request = {
            "kind": "render",
            "generation": generation,
            "full_quality": full_quality,
            "params": EditParams(**asdict(self.edit_params)),
            "source": self.original_bgr.copy(),
            "source_stamp": self.current_source_stamp,
            "local_adjustments": self.get_active_local_adjustments(),
        }
        with self.render_lock:
            self.pending_render_request = request
        self.render_event.set()
        self.status_var.set("Rendering full quality..." if full_quality else "Rendering preview...")

    def queue_analysis_precompute(self, source_bgr, source_stamp):
        self.analysis_request_id += 1
        request = {
            "kind": "analysis",
            "analysis_id": self.analysis_request_id,
            "source": source_bgr,
            "source_stamp": source_stamp,
            "preview_shape": self.downscale_for_preview(source_bgr, FAST_PREVIEW_MAX_DIM).shape,
            "full_shape": source_bgr.shape,
        }
        with self.render_lock:
            self.pending_analysis_request = request
        self.render_event.set()

    def _render_worker_loop(self):
        while True:
            self.render_event.wait()
            with self.render_lock:
                request = self.pending_render_request
                analysis_request = self.pending_analysis_request if request is None else None
                if request is not None:
                    self.pending_render_request = None
                elif analysis_request is not None:
                    self.pending_analysis_request = None
                else:
                    analysis_request = None
                if self.pending_render_request is None and self.pending_analysis_request is None:
                    self.render_event.clear()
            request = request or analysis_request
            if request is None:
                continue
            try:
                if request.get("kind") == "analysis":
                    self._run_analysis_precompute(request)
                    continue
                out = self._compute_render_request(request)
            except Exception:
                continue
            with self.render_lock:
                self.render_result_queue.append((request, out))

    def _poll_render_results(self):
        with self.render_lock:
            results = list(self.render_result_queue)
            self.render_result_queue.clear()
        for request, out in results:
            self._apply_render_result(request, out)
        self.root.after(30, self._poll_render_results)

    def _compute_render_request(self, request):
        params = request["params"]
        source = request["source"]
        full_quality = request["full_quality"]
        key = (
            request["source_stamp"],
            tuple(asdict(params).values()),
            tuple(
                (
                    getattr(stroke, "mask_type", "brush"),
                    stroke.effect,
                    round(stroke.amount, 3),
                    round(stroke.radius_norm, 5),
                    round(stroke.softness, 3),
                    round(getattr(stroke, "range_low", 0.0), 3),
                    round(getattr(stroke, "range_high", 1.0), 3),
                    str(getattr(stroke, "label", "")),
                    tuple((round(px, 4), round(py, 4)) for px, py in stroke.points),
                )
                for stroke in request.get("local_adjustments", [])
            ),
            source.shape[:2],
            full_quality,
        )
        with self.render_lock:
            cached = self.preview_cache.get(key)
        if cached is not None:
            return cached.copy()
        with self.processing_lock:
            working = self.downscale_for_preview(source, PREVIEW_MAX_DIM if full_quality else FAST_PREVIEW_MAX_DIM)
            out = self.apply_pipeline(
                working,
                params,
                preview_mode=not full_quality,
                source_bgr=source,
                source_stamp=request["source_stamp"],
                local_adjustments=request.get("local_adjustments"),
            )
            if full_quality and working.shape[:2] != source.shape[:2]:
                out = self.apply_pipeline(
                    source,
                    params,
                    preview_mode=False,
                    source_bgr=source,
                    source_stamp=request["source_stamp"],
                    local_adjustments=request.get("local_adjustments"),
                )
        with self.render_lock:
            self.preview_cache[key] = out.copy()
            if len(self.preview_cache) > 8:
                self.preview_cache.pop(next(iter(self.preview_cache)))
        return out

    def _apply_render_result(self, request, out):
        if self.original_bgr is None or request["generation"] != self.render_generation:
            return
        if not request["full_quality"] and self.current_full_generation == request["generation"]:
            return
        self.current_bgr = out.copy()
        if request["full_quality"]:
            self.full_res_bgr = out.copy()
            self.current_full_generation = request["generation"]
        self.update_histogram(self.current_bgr)
        self.update_file_info(current_only=True)
        self.request_preview_refresh(immediate=True)
        self.status_var.set("Render complete" if request["full_quality"] else "Preview updated")

    def ensure_current_full_res(self, update_preview=True):
        if self.original_bgr is None:
            return None
        if self.current_full_generation == self.render_generation and self.full_res_bgr is not None:
            if update_preview and (self.current_bgr is None or self.current_bgr.shape[:2] != self.full_res_bgr.shape[:2]):
                self.current_bgr = self.full_res_bgr.copy()
                self.update_histogram(self.current_bgr)
                self.update_file_info(current_only=True)
                self.request_preview_refresh(immediate=True)
            return self.full_res_bgr.copy()
        request = {
            "kind": "render",
            "generation": self.render_generation,
            "full_quality": True,
            "params": EditParams(**asdict(self.edit_params)),
            "source": self.original_bgr.copy(),
            "source_stamp": self.current_source_stamp,
            "local_adjustments": self.get_active_local_adjustments(),
        }
        self.status_var.set("Rendering full quality...")
        self.root.update_idletasks()
        out = self._compute_render_request(request)
        if request["generation"] == self.render_generation:
            self.full_res_bgr = out.copy()
            self.current_full_generation = request["generation"]
            if update_preview:
                self.current_bgr = out.copy()
                self.update_histogram(self.current_bgr)
                self.update_file_info(current_only=True)
                self.request_preview_refresh(immediate=True)
        return out.copy()

    def _run_analysis_precompute(self, request):
        source = request["source"]
        source_stamp = request["source_stamp"]
        preview_shape = request["preview_shape"]
        full_shape = request["full_shape"]
        self.get_detected_faces_from_source(source, source_stamp, preview_shape, preview_mode=True)
        self.get_body_region_from_source(source, source_stamp, preview_shape, preview_mode=True)
        self.get_subject_mask_from_source(source, source_stamp, preview_shape, preview_mode=True)
        self.get_face_feature_masks_from_source(source, source_stamp, preview_shape, preview_mode=True)
        if max(full_shape[:2]) <= 4200:
            self.get_detected_faces_from_source(source, source_stamp, full_shape, preview_mode=False)
            self.get_body_region_from_source(source, source_stamp, full_shape, preview_mode=False)
            self.get_subject_mask_from_source(source, source_stamp, full_shape, preview_mode=False)
            self.get_face_feature_masks_from_source(source, source_stamp, full_shape, preview_mode=False)

    def downscale_for_preview(self, img, max_dim):
        h, w = img.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale >= 0.999:
            return img.copy()
        return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    @staticmethod
    def bgr8_to_float(bgr):
        return bgr.astype(np.float32) / 255.0

    @staticmethod
    def float_to_bgr8(img):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def apply_pipeline(self, bgr8, params, preview_mode=False, source_bgr=None, source_stamp=None, local_adjustments=None):
        source_bgr = self.original_bgr if source_bgr is None else source_bgr
        source_stamp = self.current_source_stamp if source_stamp is None else source_stamp
        working = self.apply_lens_controls(bgr8, params)
        img = self.bgr8_to_float(working)
        img = self.apply_basic_tone(img, params)
        img = self.apply_tone_curve(img, params)
        img = self.apply_color_controls(img, params)
        img = self.apply_hsl_mixer(img, params)
        img = self.apply_detail_controls(img, params, preview_mode=preview_mode)
        img = self.apply_beauty_controls(img, params, preview_mode=preview_mode, source_bgr=source_bgr, source_stamp=source_stamp)
        img = self.apply_subject_background_controls(img, params, preview_mode=preview_mode, source_bgr=source_bgr, source_stamp=source_stamp)
        img = self.apply_local_adjustments(img, local_adjustments or [], preview_mode=preview_mode)
        img = self.apply_effects(img, params)
        return self.float_to_bgr8(img)

    def apply_basic_tone(self, img, p):
        if p.exposure != 0:
            img *= 2 ** (p.exposure / 50.0)
        if p.brightness != 0:
            img += (p.brightness / 100.0) * 0.35
        if p.contrast != 0:
            img = (img - 0.5) * (1.0 + p.contrast / 100.0) + 0.5
        if p.gamma != 0:
            gamma = max(0.2, 1.0 + (p.gamma / 100.0) * 0.8)
            img = np.power(np.clip(img, 0, 1), 1.0 / gamma)
        img = np.clip(img, 0, 1)
        luma = self.luminance(img)
        if p.shadows != 0:
            img += (p.shadows / 100.0) * np.clip(1.0 - luma * 1.6, 0, 1)[..., None] * 0.25
        if p.highlights != 0:
            img -= (p.highlights / 100.0) * np.clip((luma - 0.35) / 0.65, 0, 1)[..., None] * 0.22
        if p.whites != 0:
            img += (p.whites / 100.0) * np.clip((luma - 0.55) / 0.45, 0, 1)[..., None] * 0.18
        if p.blacks != 0:
            img -= (p.blacks / 100.0) * np.clip((0.45 - luma) / 0.45, 0, 1)[..., None] * 0.18
        if p.exposure > 0 or p.whites > 0 or p.brightness > 0:
            rolloff = np.clip((self.luminance(img) - 0.78) / 0.22, 0, 1)[..., None]
            strength = np.clip((max(p.exposure, 0.0) * 0.004) + (max(p.whites, 0.0) * 0.003) + (max(p.brightness, 0.0) * 0.002), 0, 0.28)
            compressed = 1.0 - np.power(1.0 - np.clip(img, 0, 1), 1.0 / (1.0 + strength * 2.2))
            img = img * (1.0 - rolloff * strength) + compressed * (rolloff * strength)
        return np.clip(img, 0, 1)

    def build_tone_curve_lut(self, p):
        key = ("curve", p.curve_shadow, p.curve_darks, p.curve_mids, p.curve_lights, p.curve_highlights)
        cached = self.curve_cache.get(key)
        if cached is not None:
            return cached
        x = np.array([0, 32, 80, 128, 176, 224, 255], dtype=np.float32)
        y = np.array([
            0,
            32 + p.curve_shadow * 0.55,
            80 + p.curve_darks * 0.70,
            128 + p.curve_mids * 0.80,
            176 + p.curve_lights * 0.70,
            224 + p.curve_highlights * 0.55,
            255,
        ], dtype=np.float32)
        y = np.maximum.accumulate(np.clip(y, 0, 255))
        lut = np.interp(np.arange(256, dtype=np.float32), x, y).clip(0, 255).astype(np.uint8)
        self.curve_cache[key] = lut
        return lut

    def apply_tone_curve(self, img, p):
        if p.curve_shadow == 0 and p.curve_darks == 0 and p.curve_mids == 0 and p.curve_lights == 0 and p.curve_highlights == 0:
            return img
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.LUT(l, self.build_tone_curve_lut(p))
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    def apply_color_controls(self, img, p):
        if p.temperature != 0 or p.tint != 0:
            lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 2] = np.clip(lab[..., 2] + p.temperature * 0.62, 0, 255)
            lab[..., 1] = np.clip(lab[..., 1] + p.tint * 0.58, 0, 255)
            img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
        if p.saturation != 0 or p.vibrance != 0:
            hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            if p.saturation != 0:
                s *= 1.0 + p.saturation / 100.0
            if p.vibrance != 0:
                s += (1.0 - np.clip(s / 255.0, 0, 1)) * (p.vibrance / 100.0) * 65.0
            img = cv2.cvtColor(cv2.merge([h, np.clip(s, 0, 255), v]).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        return np.clip(img, 0, 1)

    def apply_hsl_mixer(self, img, p):
        adjustments = []
        for band_name, center in HSL_BANDS:
            prefix = band_name.lower()
            hue_value = getattr(p, f"hsl_{prefix}_hue")
            sat_value = getattr(p, f"hsl_{prefix}_sat")
            lum_value = getattr(p, f"hsl_{prefix}_lum")
            if hue_value != 0 or sat_value != 0 or lum_value != 0:
                adjustments.append((center, hue_value, sat_value, lum_value))
        if not adjustments:
            return img
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        for center, hue_value, sat_value, lum_value in adjustments:
            delta = np.abs(h - center)
            delta = np.minimum(delta, 180.0 - delta)
            weight = np.clip(1.0 - delta / 24.0, 0.0, 1.0)
            if hue_value != 0:
                h = np.mod(h + weight * (hue_value / 100.0) * 10.0, 180.0)
            if sat_value != 0:
                s *= 1.0 + weight * (sat_value / 100.0) * 0.72
            if lum_value != 0:
                v += weight * (lum_value / 100.0) * 40.0
        hsv = cv2.merge([np.mod(h, 180.0), np.clip(s, 0, 255), np.clip(v, 0, 255)]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    def build_lens_map(self, shape, distortion):
        key = (shape[:2], round(distortion, 3))
        cached = self.lens_cache.get(key)
        if cached is not None:
            return cached
        h, w = shape[:2]
        fx = fy = float(max(w, h))
        cx, cy = w * 0.5, h * 0.5
        camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([-distortion / 100.0 * 0.26, 0.0, 0.0, 0.0], dtype=np.float32)
        maps = cv2.initUndistortRectifyMap(camera, dist_coeffs, None, camera, (w, h), cv2.CV_32FC1)
        self.lens_cache[key] = maps
        if len(self.lens_cache) > 12:
            self.lens_cache.pop(next(iter(self.lens_cache)))
        return maps

    def apply_chromatic_fix(self, bgr8, amount):
        if amount <= 0:
            return bgr8
        h, w = bgr8.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        cx, cy = w * 0.5, h * 0.5
        dx = (gx - cx) / max(cx, 1.0)
        dy = (gy - cy) / max(cy, 1.0)
        r2 = np.clip(dx * dx + dy * dy, 0.0, 1.0)
        shift = (amount / 100.0) * 2.2 * r2
        map_x_r = np.clip(gx - dx * shift * 4.0, 0, w - 1).astype(np.float32)
        map_y_r = np.clip(gy - dy * shift * 4.0, 0, h - 1).astype(np.float32)
        map_x_b = np.clip(gx + dx * shift * 4.0, 0, w - 1).astype(np.float32)
        map_y_b = np.clip(gy + dy * shift * 4.0, 0, h - 1).astype(np.float32)
        b, g, r = cv2.split(bgr8)
        r = cv2.remap(r, map_x_r, map_y_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        b = cv2.remap(b, map_x_b, map_y_b, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return cv2.merge([b, g, r])

    def apply_lens_controls(self, bgr8, p):
        out = bgr8
        if p.lens_distortion != 0:
            map_x, map_y = self.build_lens_map(out.shape, p.lens_distortion)
            out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if p.chroma_fix > 0:
            out = self.apply_chromatic_fix(out, p.chroma_fix)
        return out

    def apply_detail_controls(self, img, p, preview_mode=False):
        bgr8 = self.float_to_bgr8(img)
        if p.dehaze > 0:
            bgr8 = self.dehaze_fast(bgr8, amount=p.dehaze / 100.0)
        if p.denoise_luma > 0 or p.denoise_color > 0:
            bgr8 = self.apply_shadow_aware_denoise(bgr8, p.denoise_luma, p.denoise_color, preview_mode=preview_mode)
        if p.clarity != 0:
            bgr8 = self.apply_clarity(bgr8, p.clarity)
        if p.texture != 0:
            bgr8 = self.apply_texture(bgr8, p.texture)
        if p.sharpness > 0:
            bgr8 = self.apply_sharpen_advanced(bgr8, p.sharpness, max(0.6, p.sharp_radius / 10.0))
        return self.bgr8_to_float(bgr8)

    def apply_beauty_controls(self, img, p, preview_mode=False, source_bgr=None, source_stamp=None):
        if (
            p.body_slim <= 0
            and p.face_slim <= 0
            and p.skin_smooth <= 0
            and p.skin_whiten <= 0
            and p.acne_remove <= 0
            and p.eye_brighten <= 0
            and p.teeth_whiten <= 0
            and p.under_eye_soften <= 0
            and p.lip_enhance <= 0
            and p.skin_tone_balance <= 0
        ):
            return img
        bgr8 = self.float_to_bgr8(img)
        source_bgr = self.original_bgr if source_bgr is None else source_bgr
        source_stamp = self.current_source_stamp if source_stamp is None else source_stamp
        if p.body_slim > 0:
            body_region = self.get_body_region_from_source(source_bgr, source_stamp, bgr8.shape[:2], preview_mode=preview_mode)
            if body_region is not None:
                bgr8 = self.apply_body_slim(bgr8, body_region, p.body_slim)
        faces = self.get_detected_faces_from_source(source_bgr, source_stamp, bgr8.shape[:2], preview_mode=preview_mode)
        if not faces:
            return self.bgr8_to_float(bgr8)
        if p.face_slim > 0:
            bgr8 = self.apply_face_slim(bgr8, faces, p.face_slim)
        if p.skin_smooth > 0 or p.skin_whiten > 0 or p.acne_remove > 0:
            bgr8 = self.apply_skin_beauty(bgr8, faces, p.skin_smooth, p.skin_whiten, p.acne_remove, preview_mode)
        if p.skin_tone_balance > 0:
            bgr8 = self.apply_skin_tone_balance(bgr8, faces, p.skin_tone_balance, preview_mode)
        if p.eye_brighten > 0 or p.teeth_whiten > 0 or p.under_eye_soften > 0 or p.lip_enhance > 0:
            bgr8 = self.apply_face_feature_enhancements(bgr8, p, source_bgr, source_stamp, preview_mode)
        return self.bgr8_to_float(bgr8)

    def apply_subject_background_controls(self, img, p, preview_mode=False, source_bgr=None, source_stamp=None):
        if p.subject_light == 0 and p.background_blur == 0 and p.background_desat == 0:
            return img
        source_bgr = self.original_bgr if source_bgr is None else source_bgr
        source_stamp = self.current_source_stamp if source_stamp is None else source_stamp
        mask = self.get_subject_mask_from_source(source_bgr, source_stamp, self.float_to_bgr8(img).shape[:2], preview_mode=preview_mode)
        if mask is None:
            return img
        subject = np.clip(mask.astype(np.float32), 0, 1)[..., None]
        background = 1.0 - subject
        out = self.float_to_bgr8(img).astype(np.float32)
        if p.background_blur > 0:
            sigma = max(1.0, (3.0 + p.background_blur * 0.18) * (0.55 if preview_mode else 1.0))
            blurred = cv2.GaussianBlur(out, (0, 0), sigma)
            alpha = background * np.clip(p.background_blur / 100.0, 0.0, 1.0)
            out = out * (1.0 - alpha) + blurred * alpha
        if p.background_desat > 0:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= 1.0 - background[..., 0] * (p.background_desat / 100.0) * 0.92
            out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        if p.subject_light != 0:
            lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 0] = np.clip(lab[..., 0] + subject[..., 0] * p.subject_light * 0.55, 0, 255)
            out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        return np.clip(out / 255.0, 0, 1)

    def apply_effects(self, img, p):
        if p.vignette > 0:
            img = self.apply_vignette(img, p.vignette)
        if p.fade > 0:
            fade = p.fade / 100.0
            img = img * (1.0 - fade * 0.35) + 0.08 * fade
        if p.grain > 0:
            img = self.apply_grain(img, p.grain)
        return np.clip(img, 0, 1)

    def polygon_mask(self, shape, points, blur_sigma=0.0):
        mask = np.zeros(shape[:2], dtype=np.float32)
        if len(points) < 3:
            return mask
        polygon = np.round(np.array(points, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1.0)
        if blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        return np.clip(mask, 0, 1)

    def luminance(self, img):
        b, g, r = cv2.split(img)
        return 0.114 * b + 0.587 * g + 0.299 * r

    def apply_clarity(self, bgr8, amount):
        lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        blur = cv2.GaussianBlur(l, (0, 0), 5)
        detail = cv2.addWeighted(l, 1.0, blur, -1.0, 128)
        l2 = cv2.addWeighted(l, 1.0, detail, (amount / 100.0) * 0.5, -64 * (amount / 100.0))
        return cv2.cvtColor(cv2.merge([np.clip(l2, 0, 255).astype(np.uint8), a, b]), cv2.COLOR_LAB2BGR)

    def apply_texture(self, bgr8, amount):
        blur = cv2.GaussianBlur(bgr8, (0, 0), 1.2)
        detail = cv2.addWeighted(bgr8, 1.4, blur, -0.4, 0)
        alpha = amount / 100.0
        return cv2.addWeighted(bgr8, 1 - alpha * 0.6, detail, alpha * 0.6, 0)

    def apply_sharpen_advanced(self, bgr8, amount, radius=1.5):
        blur = cv2.GaussianBlur(bgr8, (0, 0), radius)
        alpha = amount / 100.0 * 1.6
        sharpened = np.clip(cv2.addWeighted(bgr8, 1.0 + alpha, blur, -alpha, 0), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge_mask = np.clip(cv2.GaussianBlur(np.abs(edge), (0, 0), max(0.8, radius * 0.75)) / 16.0, 0.0, 1.0)
        shadow_protect = np.clip((gray / 255.0 - 0.08) / 0.32, 0.0, 1.0)
        alpha_mask = np.clip(edge_mask * shadow_protect, 0.0, 1.0)[..., None]
        blended = bgr8.astype(np.float32) * (1.0 - alpha_mask) + sharpened.astype(np.float32) * alpha_mask
        return np.clip(blended, 0, 255).astype(np.uint8)

    def apply_shadow_aware_denoise(self, bgr8, luma_amount, color_amount, preview_mode=False):
        if luma_amount <= 0 and color_amount <= 0:
            return bgr8
        h_luma = max(0.0, luma_amount * (0.38 if preview_mode else 0.68))
        h_color = max(0.0, color_amount * (0.38 if preview_mode else 0.68))
        denoised = cv2.fastNlMeansDenoisingColored(bgr8, None, h_luma, h_color, 7, 21)
        luma = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shadow_mask = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)
        edge = np.abs(cv2.Laplacian((luma * 255).astype(np.uint8), cv2.CV_32F, ksize=3))
        edge_gate = 1.0 - np.clip(cv2.GaussianBlur(edge, (0, 0), 1.0) / 22.0, 0.0, 1.0)
        alpha = np.clip(shadow_mask * edge_gate * (max(luma_amount, color_amount) / 100.0) * 1.15, 0.0, 1.0)[..., None]
        return np.clip(bgr8.astype(np.float32) * (1.0 - alpha) + denoised.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def render_local_stroke_mask(self, shape, stroke):
        key = (
            shape[:2],
            getattr(stroke, "mask_type", "brush"),
            stroke.effect,
            round(stroke.amount, 3),
            round(stroke.radius_norm, 5),
            round(stroke.softness, 3),
            round(getattr(stroke, "range_low", 0.0), 3),
            round(getattr(stroke, "range_high", 1.0), 3),
            str(getattr(stroke, "label", "")),
            tuple((round(px, 4), round(py, 4)) for px, py in stroke.points),
        )
        cached = self.local_mask_cache.get(key)
        if cached is not None:
            return cached.copy()
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if not stroke.points:
            return mask
        mask_type = getattr(stroke, "mask_type", "brush")
        radius = max(2.0, stroke.radius_norm * max(h, w))
        pixel_points = [(float(px * w), float(py * h)) for px, py in stroke.points]
        if mask_type == "radial" and len(pixel_points) >= 2:
            cx, cy = pixel_points[0]
            ex, ey = pixel_points[1]
            radius = max(6.0, math.hypot(ex - cx, ey - cy))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            feather = max(2.0, radius * max(0.18, stroke.softness))
            mask = np.clip(1.0 - (dist - (radius - feather)) / max(feather, 1.0), 0.0, 1.0)
        elif mask_type == "linear" and len(pixel_points) >= 2:
            x0, y0 = pixel_points[0]
            x1, y1 = pixel_points[1]
            dx, dy = x1 - x0, y1 - y0
            length = max(math.hypot(dx, dy), 1.0)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            proj = ((xx - x0) * dx + (yy - y0) * dy) / (length * length)
            feather = max(0.08, stroke.softness * 0.35)
            mask = np.clip((proj - (0.5 - feather)) / max(feather * 2.0, 1e-4), 0.0, 1.0)
        else:
            pixel_points_i = [(int(round(px)), int(round(py))) for px, py in pixel_points]
            for point in pixel_points_i:
                cv2.circle(mask, point, int(round(radius)), 1.0, -1)
            if len(pixel_points_i) >= 2:
                cv2.polylines(mask, [np.array(pixel_points_i, dtype=np.int32)], False, 1.0, thickness=max(1, int(round(radius * 2.0))), lineType=cv2.LINE_AA)
            sigma = max(1.0, radius * max(0.15, stroke.softness))
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        self.local_mask_cache[key] = mask.copy()
        if len(self.local_mask_cache) > 40:
            self.local_mask_cache.pop(next(iter(self.local_mask_cache)))
        return mask

    def apply_local_adjustments(self, img, strokes, preview_mode=False):
        if not strokes:
            return img
        out = img.copy()
        base_luma = self.luminance(out)
        for stroke in strokes:
            mask = self.render_local_stroke_mask(out.shape, stroke)
            if float(mask.max()) <= 0.001:
                continue
            range_low = float(np.clip(getattr(stroke, "range_low", 0.0), 0.0, 1.0))
            range_high = float(np.clip(getattr(stroke, "range_high", 1.0), 0.0, 1.0))
            if range_high < range_low:
                range_low, range_high = range_high, range_low
            if range_low > 0.0 or range_high < 1.0:
                feather = max(0.025, getattr(stroke, "softness", 0.65) * 0.08)
                lower = np.clip((base_luma - (range_low - feather)) / max(feather, 1e-4), 0.0, 1.0)
                upper = np.clip(((range_high + feather) - base_luma) / max(feather, 1e-4), 0.0, 1.0)
                mask = mask * lower * upper
            weight = np.clip(mask * (abs(stroke.amount) / 100.0), 0.0, 1.0)[..., None]
            effect = (stroke.effect or "Lighten").lower()
            if effect == "lighten":
                lifted = np.clip(out * (1.0 + 0.45 * (stroke.amount / 100.0)) + 0.10 * (stroke.amount / 100.0), 0, 1)
                out = out * (1.0 - weight) + lifted * weight
            elif effect == "darken":
                lowered = np.clip(out * (1.0 - 0.42 * (stroke.amount / 100.0)), 0, 1)
                out = out * (1.0 - weight) + lowered * weight
            elif effect == "dodge":
                lifted = np.clip(out + (1.0 - out) * (stroke.amount / 100.0) * 0.42, 0, 1)
                out = out * (1.0 - weight) + lifted * weight
            elif effect == "burn":
                lowered = np.clip(out * (1.0 - (stroke.amount / 100.0) * 0.50), 0, 1)
                out = out * (1.0 - weight) + lowered * weight
            elif effect == "color pop":
                hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] + mask * stroke.amount * 1.65, 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] + mask * stroke.amount * 0.48, 0, 255)
                popped = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
                out = out * (1.0 - weight) + popped * weight
            elif effect == "soften":
                softened = cv2.bilateralFilter((out * 255).astype(np.uint8), 0, 10 + stroke.amount * (0.16 if preview_mode else 0.22), 7 + stroke.amount * 0.08).astype(np.float32) / 255.0
                out = out * (1.0 - weight * 0.92) + softened * (weight * 0.92)
            elif effect == "texture":
                textured = self.bgr8_to_float(self.apply_sharpen_advanced((out * 255).astype(np.uint8), max(12.0, stroke.amount * 1.2), 1.0))
                out = out * (1.0 - weight * 0.78) + textured * (weight * 0.78)
        return np.clip(out, 0, 1)

    def apply_vignette(self, img, value):
        rows, cols = img.shape[:2]
        mask = cv2.getGaussianKernel(rows, max(rows * 0.55, 1)) * cv2.getGaussianKernel(cols, max(cols * 0.55, 1)).T
        mask = 1.0 - (value / 100.0) * (1.0 - mask / mask.max())
        return np.clip(img * mask[..., None], 0, 1)

    def apply_grain(self, img, value):
        noise = np.random.normal(0, 0.04 * (value / 100.0), img.shape[:2]).astype(np.float32)
        return np.clip(img + (noise * (0.7 + 0.3 * (1.0 - self.luminance(img))))[..., None], 0, 1)

    def dehaze_fast(self, bgr8, amount=0.25):
        img = bgr8.astype(np.float32) / 255.0
        dark = cv2.erode(np.min(img, axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        airlight = max(np.percentile(np.max(img, axis=2), 99.5), 0.6)
        transmission = np.clip(1 - amount * dark, 0.35, 1.0)
        return np.clip(((img - airlight) / transmission[..., None] + airlight) * 255, 0, 255).astype(np.uint8)

    def get_subject_mask_from_source(self, source_bgr, source_stamp, target_shape, preview_mode=False):
        if source_bgr is None:
            return None
        target_h, target_w = target_shape[:2]
        key = ("subject", source_stamp, target_w, target_h, preview_mode)
        cached = self.subject_cache.get(key)
        if cached is not None:
            return cached.copy()
        detect_src = self.downscale_for_preview(source_bgr, min(max(target_w, target_h), 1280 if not preview_mode else 900))
        mask = None
        if self.mp_selfie_segmentation is not None:
            try:
                result = self.mp_selfie_segmentation.process(cv2.cvtColor(detect_src, cv2.COLOR_BGR2RGB))
                if result is not None and getattr(result, "segmentation_mask", None) is not None:
                    mask = np.clip((result.segmentation_mask.astype(np.float32) - 0.28) / 0.50, 0.0, 1.0)
                    mask = cv2.GaussianBlur(mask, (0, 0), 3.2 if preview_mode else 4.0)
            except Exception:
                mask = None
        if mask is None or float(mask.max()) < 0.08:
            region = self.detect_body_region(detect_src)
            if region is None:
                faces = self.detect_faces(detect_src)
                region = self.estimate_body_region_from_face(detect_src.shape[:2], faces[0] if faces else None)
            if region is not None:
                mask = np.zeros(detect_src.shape[:2], dtype=np.float32)
                cx = int(round(region["cx"]))
                cy = int(round((region["shoulder_y"] + region["hip_y"]) * 0.5))
                axes = (
                    max(20, int(round((region["x1"] - region["x0"]) * 0.58))),
                    max(30, int(round((region["y1"] - region["y0"]) * 0.60))),
                )
                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), max(6.0, axes[0] * 0.12))
            else:
                h, w = detect_src.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(mask, (w // 2, h // 2), (max(20, int(w * 0.28)), max(30, int(h * 0.38))), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), max(10.0, min(h, w) * 0.06))
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        self.subject_cache[key] = mask.copy()
        return mask

    def get_face_mesh_from_source(self, source_bgr, source_stamp, target_shape, preview_mode=False):
        if source_bgr is None or self.mp_face_mesh is None:
            return []
        target_h, target_w = target_shape[:2]
        key = ("mesh", source_stamp, target_w, target_h, preview_mode)
        cached = self.mesh_cache.get(key)
        if cached is not None:
            return [face.copy() for face in cached]
        detect_src = self.downscale_for_preview(source_bgr, min(max(target_w, target_h), 1100 if not preview_mode else 800))
        try:
            result = self.mp_face_mesh.process(cv2.cvtColor(detect_src, cv2.COLOR_BGR2RGB))
        except Exception:
            return []
        if not result or not getattr(result, "multi_face_landmarks", None):
            return []
        scale_x, scale_y = target_w / detect_src.shape[1], target_h / detect_src.shape[0]
        faces = []
        for face_landmarks in result.multi_face_landmarks:
            points = np.array([(lm.x * detect_src.shape[1] * scale_x, lm.y * detect_src.shape[0] * scale_y) for lm in face_landmarks.landmark], dtype=np.float32)
            faces.append(points)
        self.mesh_cache[key] = [face.copy() for face in faces]
        return faces

    def get_face_feature_masks_from_source(self, source_bgr, source_stamp, target_shape, preview_mode=False):
        target_h, target_w = target_shape[:2]
        key = ("feature_masks", source_stamp, target_w, target_h, preview_mode)
        cached = self.feature_cache.get(key)
        if cached is not None:
            return {name: mask.copy() for name, mask in cached.items()}
        masks = {name: np.zeros((target_h, target_w), dtype=np.float32) for name in ("eyes", "under_eye", "lips", "mouth")}
        mesh_faces = self.get_face_mesh_from_source(source_bgr, source_stamp, target_shape, preview_mode=preview_mode)
        if mesh_faces:
            for points in mesh_faces:
                left_eye = points[[33, 160, 158, 133, 153, 144]]
                right_eye = points[[362, 385, 387, 263, 373, 380]]
                lips = points[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]]
                mouth_inner = points[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]]
                for eye_points in (left_eye, right_eye):
                    masks["eyes"] = np.maximum(masks["eyes"], self.polygon_mask((target_h, target_w), eye_points, blur_sigma=2.0 if preview_mode else 2.8))
                    ex, ey, ew, eh = cv2.boundingRect(np.round(eye_points).astype(np.int32))
                    under_eye = np.zeros((target_h, target_w), dtype=np.float32)
                    cv2.ellipse(under_eye, (int(ex + ew * 0.5), int(ey + eh * 1.4)), (max(4, int(ew * 0.55)), max(4, int(eh * 0.42))), 0, 0, 360, 1.0, -1)
                    masks["under_eye"] = np.maximum(masks["under_eye"], cv2.GaussianBlur(under_eye, (0, 0), 2.0 if preview_mode else 2.8))
                masks["lips"] = np.maximum(masks["lips"], self.polygon_mask((target_h, target_w), lips, blur_sigma=2.0 if preview_mode else 3.0))
                masks["mouth"] = np.maximum(masks["mouth"], self.polygon_mask((target_h, target_w), mouth_inner, blur_sigma=1.4 if preview_mode else 2.1))
        else:
            for x, y, w, h in self.get_detected_faces_from_source(source_bgr, source_stamp, target_shape, preview_mode=preview_mode):
                left_eye = np.zeros((target_h, target_w), dtype=np.float32)
                right_eye = np.zeros((target_h, target_w), dtype=np.float32)
                lips = np.zeros((target_h, target_w), dtype=np.float32)
                mouth = np.zeros((target_h, target_w), dtype=np.float32)
                under_eye = np.zeros((target_h, target_w), dtype=np.float32)
                cv2.ellipse(left_eye, (int(x + w * 0.34), int(y + h * 0.42)), (max(4, int(w * 0.09)), max(3, int(h * 0.05))), 0, 0, 360, 1.0, -1)
                cv2.ellipse(right_eye, (int(x + w * 0.66), int(y + h * 0.42)), (max(4, int(w * 0.09)), max(3, int(h * 0.05))), 0, 0, 360, 1.0, -1)
                cv2.ellipse(under_eye, (int(x + w * 0.34), int(y + h * 0.52)), (max(5, int(w * 0.12)), max(4, int(h * 0.06))), 0, 0, 360, 1.0, -1)
                cv2.ellipse(under_eye, (int(x + w * 0.66), int(y + h * 0.52)), (max(5, int(w * 0.12)), max(4, int(h * 0.06))), 0, 0, 360, 1.0, -1)
                cv2.ellipse(lips, (int(x + w * 0.50), int(y + h * 0.78)), (max(6, int(w * 0.18)), max(3, int(h * 0.06))), 0, 0, 360, 1.0, -1)
                cv2.ellipse(mouth, (int(x + w * 0.50), int(y + h * 0.75)), (max(4, int(w * 0.13)), max(2, int(h * 0.04))), 0, 0, 360, 1.0, -1)
                masks["eyes"] = np.maximum(masks["eyes"], cv2.GaussianBlur(left_eye + right_eye, (0, 0), 2.0))
                masks["under_eye"] = np.maximum(masks["under_eye"], cv2.GaussianBlur(under_eye, (0, 0), 2.5))
                masks["lips"] = np.maximum(masks["lips"], cv2.GaussianBlur(lips, (0, 0), 2.6))
                masks["mouth"] = np.maximum(masks["mouth"], cv2.GaussianBlur(mouth, (0, 0), 2.0))
        self.feature_cache[key] = {name: np.clip(mask, 0, 1).astype(np.float32) for name, mask in masks.items()}
        return {name: mask.copy() for name, mask in self.feature_cache[key].items()}

    def apply_face_feature_enhancements(self, bgr8, p, source_bgr, source_stamp, preview_mode=False):
        masks = self.get_face_feature_masks_from_source(source_bgr, source_stamp, bgr8.shape[:2], preview_mode=preview_mode)
        out = bgr8.astype(np.float32)
        if p.under_eye_soften > 0 and float(masks["under_eye"].max()) > 0:
            smooth = cv2.bilateralFilter(np.clip(out, 0, 255).astype(np.uint8), 0, 12 + p.under_eye_soften * 0.18, 7 + p.under_eye_soften * 0.10).astype(np.float32)
            alpha = masks["under_eye"][..., None] * (p.under_eye_soften / 100.0) * 0.58
            out = out * (1.0 - alpha) + smooth * alpha
        if p.eye_brighten > 0 and float(masks["eyes"].max()) > 0:
            lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 0] = np.clip(lab[..., 0] + masks["eyes"] * p.eye_brighten * 0.70, 0, 255)
            neutral = masks["eyes"] * (0.05 + p.eye_brighten / 100.0 * 0.12)
            lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * (1.0 - neutral)
            lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * (1.0 - neutral)
            out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        if p.lip_enhance > 0 and float(masks["lips"].max()) > 0:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] + masks["lips"] * p.lip_enhance * 0.95, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + masks["lips"] * p.lip_enhance * 0.30, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        if p.teeth_whiten > 0 and float(masks["mouth"].max()) > 0:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            teeth_mask = masks["mouth"] * ((hsv[..., 2] > 92).astype(np.float32)) * ((hsv[..., 1] < 88).astype(np.float32))
            if float(teeth_mask.max()) > 0:
                lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                strength = p.teeth_whiten / 100.0
                lab[..., 0] = np.clip(lab[..., 0] + teeth_mask * (22.0 + strength * 12.0), 0, 255)
                lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * (1.0 - teeth_mask * (0.12 + strength * 0.18))
                lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * (1.0 - teeth_mask * (0.16 + strength * 0.18))
                out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_detected_faces(self, target_shape, preview_mode=False):
        return self.get_detected_faces_from_source(self.original_bgr, self.current_source_stamp, target_shape, preview_mode=preview_mode)

    def get_detected_faces_from_source(self, source_bgr, source_stamp, target_shape, preview_mode=False):
        if source_bgr is None:
            return []
        target_h, target_w = target_shape[:2]
        key = (source_stamp, target_w, target_h, preview_mode)
        if key in self.face_cache:
            return list(self.face_cache[key])
        detect_src = self.downscale_for_preview(source_bgr, min(max(target_w, target_h), 1280 if not preview_mode else 900))
        faces = self.detect_faces(detect_src)
        scale_x, scale_y = target_w / detect_src.shape[1], target_h / detect_src.shape[0]
        out = [(int(round(x * scale_x)), int(round(y * scale_y)), int(round(w * scale_x)), int(round(h * scale_y))) for x, y, w, h in faces]
        self.face_cache[key] = out
        return list(out)

    def detect_faces(self, bgr8):
        faces = self.detect_faces_mediapipe(bgr8)
        return faces if faces else self.detect_faces_haar(bgr8)

    def detect_faces_mediapipe(self, bgr8):
        if self.mp_face_detector is None:
            return []
        try:
            result = self.mp_face_detector.process(cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB))
        except Exception:
            return []
        if not result or not result.detections:
            return []
        h, w = bgr8.shape[:2]
        faces = []
        for d in result.detections:
            bbox = d.location_data.relative_bounding_box
            x, y = int(max(0, bbox.xmin * w)), int(max(0, bbox.ymin * h))
            bw, bh = int(min(w - x, bbox.width * w)), int(min(h - y, bbox.height * h))
            if bw > 20 and bh > 20:
                faces.append((x, y, bw, bh))
        return faces

    def detect_faces_haar(self, bgr8):
        if self.face_cascade.empty():
            return []
        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return list(self.face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(max(48, min(gray.shape[:2]) // 8),) * 2))

    def detect_pose_landmarks(self, bgr8):
        if self.mp_pose is None:
            return None
        try:
            result = self.mp_pose.process(cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB))
        except Exception:
            return None
        if not result or not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    def get_body_region_from_source(self, source_bgr, source_stamp, target_shape, preview_mode=False):
        if source_bgr is None:
            return None
        target_h, target_w = target_shape[:2]
        key = ("body", source_stamp, target_w, target_h, preview_mode)
        if key in self.body_cache:
            return self.body_cache[key]
        detect_src = self.downscale_for_preview(source_bgr, min(max(target_w, target_h), 1280 if not preview_mode else 900))
        region = self.detect_body_region(detect_src)
        if region is None:
            faces = self.detect_faces(detect_src)
            region = self.estimate_body_region_from_face(detect_src.shape[:2], faces[0] if faces else None)
        if region is not None:
            scale_x, scale_y = target_w / detect_src.shape[1], target_h / detect_src.shape[0]
            region = {
                "x0": int(round(region["x0"] * scale_x)),
                "y0": int(round(region["y0"] * scale_y)),
                "x1": int(round(region["x1"] * scale_x)),
                "y1": int(round(region["y1"] * scale_y)),
                "cx": float(region["cx"] * scale_x),
                "shoulder_y": float(region["shoulder_y"] * scale_y),
                "waist_y": float(region["waist_y"] * scale_y),
                "hip_y": float(region["hip_y"] * scale_y),
            }
        self.body_cache[key] = region
        return region

    def detect_body_region(self, bgr8):
        landmarks = self.detect_pose_landmarks(bgr8)
        if landmarks is None:
            return None
        pose_ids = mp.solutions.pose.PoseLandmark if MEDIAPIPE_AVAILABLE else None
        if pose_ids is None:
            return None

        def pick(name):
            landmark = landmarks[getattr(pose_ids, name).value]
            if landmark.visibility < 0.35:
                return None
            return landmark.x * bgr8.shape[1], landmark.y * bgr8.shape[0]

        left_shoulder = pick("LEFT_SHOULDER")
        right_shoulder = pick("RIGHT_SHOULDER")
        left_hip = pick("LEFT_HIP")
        right_hip = pick("RIGHT_HIP")
        if None in (left_shoulder, right_shoulder, left_hip, right_hip):
            return None
        left_knee = pick("LEFT_KNEE")
        right_knee = pick("RIGHT_KNEE")
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
        hip_y = (left_hip[1] + right_hip[1]) * 0.5
        knee_y = (left_knee[1] + right_knee[1]) * 0.5 if left_knee and right_knee else hip_y + (hip_y - shoulder_y) * 0.95
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])
        hip_span = abs(right_hip[0] - left_hip[0])
        torso_h = max(knee_y - shoulder_y, hip_y - shoulder_y, 40.0)
        region_w = max(shoulder_span, hip_span) * 1.75
        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) * 0.25
        x0 = max(0.0, center_x - region_w * 0.5)
        x1 = min(float(bgr8.shape[1]), center_x + region_w * 0.5)
        y0 = max(0.0, shoulder_y - torso_h * 0.12)
        y1 = min(float(bgr8.shape[0]), knee_y + torso_h * 0.06)
        if x1 - x0 < 24 or y1 - y0 < 48:
            return None
        return {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "cx": center_x,
            "shoulder_y": shoulder_y,
            "waist_y": shoulder_y + (hip_y - shoulder_y) * 0.55,
            "hip_y": hip_y,
        }

    def estimate_body_region_from_face(self, image_shape, face):
        if face is None:
            return None
        img_h, img_w = image_shape[:2]
        x, y, w, h = face
        center_x = x + w * 0.5
        shoulder_y = y + h * 1.45
        hip_y = y + h * 3.85
        x0 = max(0.0, center_x - w * 1.7)
        x1 = min(float(img_w), center_x + w * 1.7)
        y0 = max(0.0, y + h * 0.95)
        y1 = min(float(img_h), y + h * 6.2)
        if x1 - x0 < 24 or y1 - y0 < 48:
            return None
        return {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "cx": center_x,
            "shoulder_y": shoulder_y,
            "waist_y": y + h * 2.75,
            "hip_y": hip_y,
        }

    def apply_body_slim(self, bgr8, region, amount):
        if region is None:
            return bgr8
        out = bgr8.copy()
        img_h, img_w = out.shape[:2]
        x0 = max(0, int(region["x0"]))
        y0 = max(0, int(region["y0"]))
        x1 = min(img_w, int(region["x1"]))
        y1 = min(img_h, int(region["y1"]))
        if x1 - x0 < 24 or y1 - y0 < 48:
            return out
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            return out
        rh, rw = roi.shape[:2]
        gx, gy = np.meshgrid(np.arange(rw, dtype=np.float32), np.arange(rh, dtype=np.float32))
        cx = region["cx"] - x0
        half_w = max((x1 - x0) * 0.5, 1.0)
        waist_y = region["waist_y"] - y0
        shoulder_y = region["shoulder_y"] - y0
        hip_y = region["hip_y"] - y0
        nx = (gx - cx) / half_w
        side = np.clip((np.abs(nx) - 0.08) / 0.92, 0.0, 1.0)
        vertical = np.exp(-np.square((gy - waist_y) / max((y1 - y0) * 0.24, 1.0)))
        vertical += 0.32 * np.exp(-np.square((gy - ((shoulder_y + hip_y) * 0.5)) / max((y1 - y0) * 0.42, 1.0)))
        gate_top = np.clip((gy - max(0.0, shoulder_y - (y1 - y0) * 0.08)) / max((hip_y - shoulder_y) * 0.22, 1.0), 0.0, 1.0)
        gate_bottom = np.clip((float(rh - 1) - gy) / max((rh - max(hip_y, 0.0)) * 0.8 + 1.0, 1.0), 0.0, 1.0)
        weight = np.power(np.clip(side * vertical * gate_top * gate_bottom, 0.0, 1.0), 1.18)
        strength = np.clip(amount / 100.0, 0.0, 1.0)
        shift = np.sign(gx - cx) * ((x1 - x0) * (0.045 + 0.075 * strength)) * strength * weight
        warped = cv2.remap(roi, np.clip(gx + shift, 0, rw - 1).astype(np.float32), gy.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        mask = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), max(2.0, (x1 - x0) * 0.045))
        blended = roi.astype(np.float32) * (1.0 - mask[..., None]) + warped.astype(np.float32) * mask[..., None]
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def apply_face_slim(self, bgr8, faces, amount):
        out = bgr8.copy()
        strength = np.clip(amount / 100.0, 0.0, 1.0)
        img_h, img_w = out.shape[:2]
        for x, y, w, h in faces:
            x0, y0 = max(0, int(x - w * 0.18)), max(0, int(y - h * 0.06))
            x1, y1 = min(img_w, int(x + w * 1.18)), min(img_h, int(y + h * 1.06))
            roi = out[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            rh, rw = roi.shape[:2]
            gx, gy = np.meshgrid(np.arange(rw, dtype=np.float32), np.arange(rh, dtype=np.float32))
            cx, cy = (x + w * 0.5) - x0, (y + h * 0.58) - y0
            nx, ny = (gx - cx) / max(w * 0.54, 1.0), (gy - cy) / max(h * 0.72, 1.0)
            radial = np.clip(1.0 - (nx * nx + ny * ny), 0.0, 1.0)
            side_band = np.clip((np.abs(nx) - 0.12) / 0.88, 0.0, 1.0)
            cheek_vertical = np.clip(1.0 - np.abs((gy - (y + h * 0.63 - y0)) / max(h * 0.34, 1.0)), 0.0, 1.0)
            weight = np.power(radial * side_band * cheek_vertical, 1.15)
            shift = np.sign(gx - cx) * (w * 0.115 * strength) * weight
            warped = cv2.remap(roi, np.clip(gx + shift, 0, rw - 1).astype(np.float32), gy.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            mask = cv2.GaussianBlur(radial.astype(np.float32), (0, 0), max(1.2, w * 0.05))
            blended = roi.astype(np.float32) * (1.0 - mask[..., None]) + warped.astype(np.float32) * mask[..., None]
            out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def apply_skin_beauty(self, bgr8, faces, smooth_amount, whiten_amount, acne_remove, preview_mode=False):
        mask = np.zeros(bgr8.shape[:2], dtype=np.float32)
        for face in faces:
            mask = np.maximum(mask, self.build_skin_mask(bgr8, face))
        if mask.max() <= 0:
            return bgr8
        points = cv2.findNonZero(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
        if points is None:
            return bgr8
        x, y, w, h = cv2.boundingRect(points)
        roi_u8, roi_mask = bgr8[y:y + h, x:x + w], mask[y:y + h, x:x + w].astype(np.float32)
        roi_mask_3, out_roi = roi_mask[..., None], roi_u8.astype(np.float32)
        if acne_remove > 0:
            out_roi = self.apply_acne_removal(out_roi.astype(np.uint8), roi_mask, acne_remove, preview_mode)
        if smooth_amount > 0:
            smooth_source = np.clip(out_roi, 0, 255).astype(np.uint8)
            smooth = cv2.bilateralFilter(smooth_source, 0, 14 + smooth_amount * (0.36 if preview_mode else 0.52), 5 + smooth_amount * 0.14).astype(np.float32)
            detail = np.clip(out_roi - cv2.GaussianBlur(out_roi, (0, 0), 1.0 if preview_mode else 1.25), -18.0, 18.0)
            refined = smooth * 0.84 + out_roi * 0.16 + detail * 0.22
            alpha = np.clip(smooth_amount / 100.0, 0.0, 1.0) * 0.82
            out_roi = out_roi * (1.0 - roi_mask_3 * alpha) + refined * (roi_mask_3 * alpha)
        if whiten_amount > 0:
            strength = np.clip(whiten_amount / 100.0, 0.0, 1.0)
            lab = cv2.cvtColor(np.clip(out_roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lift = (14.0 + 26.0 * (1.0 - lab[..., 0] / 255.0)) * strength
            lab[..., 0] = np.clip(lab[..., 0] + lift * roi_mask, 0, 255)
            neutral = roi_mask * (0.05 + strength * 0.12)
            lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * (1.0 - neutral)
            lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * (1.0 - neutral * 1.1)
            whitened = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            out_roi = out_roi * (1.0 - roi_mask_3 * (0.45 + strength * 0.22)) + whitened * (roi_mask_3 * (0.45 + strength * 0.22))
        out = bgr8.astype(np.float32)
        out[y:y + h, x:x + w] = np.clip(out_roi, 0, 255)
        return out.astype(np.uint8)

    def apply_skin_tone_balance(self, bgr8, faces, amount, preview_mode=False):
        strength = np.clip(amount / 100.0, 0.0, 1.0)
        if strength <= 0:
            return bgr8
        mask = np.zeros(bgr8.shape[:2], dtype=np.float32)
        for face in faces:
            mask = np.maximum(mask, self.build_skin_mask(bgr8, face))
        if float(mask.max()) <= 0.01:
            return bgr8
        points = cv2.findNonZero(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
        if points is None:
            return bgr8
        x, y, w, h = cv2.boundingRect(points)
        roi_u8 = bgr8[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w].astype(np.float32)
        if roi_u8.size == 0:
            return bgr8
        selector = roi_mask > 0.08
        if not np.any(selector):
            return bgr8
        lab = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan, a_chan, b_chan = cv2.split(lab)
        mean_a = float(np.mean(a_chan[selector]))
        mean_b = float(np.mean(b_chan[selector]))
        smooth_a = cv2.GaussianBlur(a_chan, (0, 0), 2.4 if preview_mode else 3.2)
        smooth_b = cv2.GaussianBlur(b_chan, (0, 0), 2.4 if preview_mode else 3.2)
        target_a = 128.0 + (mean_a - 128.0) * (1.0 - 0.72 * strength)
        target_b = 128.0 + (mean_b - 128.0) * (1.0 - 0.34 * strength) + 3.0 * strength
        chroma_blend = roi_mask * (0.22 + 0.42 * strength)
        lab[..., 1] = a_chan * (1.0 - chroma_blend) + (smooth_a * 0.56 + target_a * 0.44) * chroma_blend
        lab[..., 2] = b_chan * (1.0 - chroma_blend) + (smooth_b * 0.56 + target_b * 0.44) * chroma_blend
        lift = (4.0 + 10.0 * (1.0 - l_chan / 255.0)) * strength
        lab[..., 0] = np.clip(l_chan + lift * roi_mask, 0, 255)
        balanced = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        alpha = roi_mask[..., None] * (0.34 + 0.34 * strength)
        out = bgr8.astype(np.float32)
        out[y:y + h, x:x + w] = out[y:y + h, x:x + w] * (1.0 - alpha) + balanced * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_acne_removal(self, roi_u8, roi_mask, amount, preview_mode=False):
        strength = np.clip(amount / 100.0, 0.0, 1.0)
        if strength <= 0 or roi_u8.size == 0:
            return roi_u8.astype(np.float32)
        lab = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2LAB)
        l = lab[..., 0]
        k_size = 5 if preview_mode else 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        blackhat = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)
        local_blur = cv2.GaussianBlur(l, (0, 0), 1.2 if preview_mode else 1.6)
        spot_score = np.maximum(blackhat, tophat)
        residual = cv2.absdiff(l, local_blur)
        threshold = 11.0 - strength * 4.0
        raw_mask = np.where((spot_score > threshold) | (residual > threshold + 2.0), 255, 0).astype(np.uint8)
        skin_mask = cv2.erode(np.clip(roi_mask * 255.0, 0, 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        blemish_mask = cv2.bitwise_and(raw_mask, skin_mask)
        blemish_mask = cv2.medianBlur(blemish_mask, 3)
        blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        max_area = max(18, int(min(roi_u8.shape[:2]) ** 2 * (0.00018 + strength * 0.00028)))
        blemish_mask = self.keep_small_connected_components(blemish_mask, max_area=max_area)
        if not np.any(blemish_mask):
            return roi_u8.astype(np.float32)
        inpaint_radius = 1 + int(round(strength * 2.0))
        healed = cv2.inpaint(roi_u8, blemish_mask, inpaint_radius, cv2.INPAINT_TELEA).astype(np.float32)
        alpha = (0.34 + strength * 0.42) * (blemish_mask.astype(np.float32) / 255.0)
        return roi_u8.astype(np.float32) * (1.0 - alpha[..., None]) + healed * alpha[..., None]

    def keep_small_connected_components(self, mask, max_area):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask
        out = np.zeros_like(mask)
        for idx in range(1, count):
            area = stats[idx, cv2.CC_STAT_AREA]
            if 2 <= area <= max_area:
                out[labels == idx] = 255
        return out

    def build_skin_mask(self, bgr8, face):
        img_h, img_w = bgr8.shape[:2]
        x, y, w, h = face
        x0, y0 = max(0, int(x - w * 0.08)), max(0, int(y - h * 0.03))
        x1, y1 = min(img_w, int(x + w * 1.08)), min(img_h, int(y + h * 1.05))
        roi = bgr8[y0:y1, x0:x1]
        if roi.size == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        skin_ycrcb = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb), np.array([0, 132, 76], dtype=np.uint8), np.array([255, 180, 136], dtype=np.uint8))
        skin_hsv = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), np.array([0, 18, 46], dtype=np.uint8), np.array([24, 210, 255], dtype=np.uint8))
        skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)
        ellipse = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(ellipse, (roi.shape[1] // 2, int(roi.shape[0] * 0.53)), (max(1, int(roi.shape[1] * 0.42)), max(1, int(roi.shape[0] * 0.50))), 0, 0, 360, 255, -1)
        skin = cv2.bitwise_and(skin, ellipse)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.morphologyEx(cv2.morphologyEx(cv2.medianBlur(skin, 5), cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        skin = cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (0, 0), max(2.0, min(w, h) * 0.045))
        full_mask = np.zeros((img_h, img_w), dtype=np.float32)
        full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], skin)
        return np.clip(full_mask, 0, 1)

    def on_slider_change(self, field_name, value):
        if self._building_ui or self._updating_sliders:
            return
        self.sliders[field_name][1].config(text=str(int(float(value))))
        self._params_from_sliders()
        self._persist_current_record_state()
        if field_name.startswith("curve_"):
            self.update_curve_graph()
        self.status_var.set("Updating parameters...")
        self.schedule_render(immediate=False)
        self.schedule_session_save()

    def _params_from_sliders(self):
        params_dict = asdict(self.edit_params)
        for field_name, (scale, _value_label) in self.sliders.items():
            params_dict[field_name] = float(scale.get())
        self.edit_params = EditParams(**params_dict)
        self.preview_cache.clear()

    def _sync_params_to_sliders(self):
        self._updating_sliders = True
        for field_name, (scale, value_label) in self.sliders.items():
            value = getattr(self.edit_params, field_name)
            scale.set(value)
            value_label.config(text=str(int(value)))
        self._updating_sliders = False
        self._sync_hsl_band_ui()
        self.update_curve_graph()

    def _current_hsl_field(self, component):
        return f"hsl_{self.hsl_band_var.get().lower()}_{component}"

    def on_hsl_slider_change(self, component, value):
        if self._building_ui or self._updating_hsl:
            return
        field_name = self._current_hsl_field(component)
        setattr(self.edit_params, field_name, float(value))
        self._persist_current_record_state()
        self.hsl_sliders[component][1].config(text=str(int(float(value))))
        self.preview_cache.clear()
        self.status_var.set("Updating HSL mixer...")
        self.schedule_render(immediate=False)
        self.schedule_session_save()

    def _sync_hsl_band_ui(self):
        if not hasattr(self, "hsl_sliders"):
            return
        self._updating_hsl = True
        for component, (scale, value_label) in self.hsl_sliders.items():
            value = getattr(self.edit_params, self._current_hsl_field(component))
            scale.set(value)
            value_label.config(text=str(int(value)))
        self._updating_hsl = False

    def reset_current_hsl_band(self):
        for component in ("hue", "sat", "lum"):
            setattr(self.edit_params, self._current_hsl_field(component), 0.0)
        self._sync_hsl_band_ui()
        self.preview_cache.clear()
        self.schedule_render(immediate=True)
        self.schedule_session_save()

    def update_curve_graph(self):
        if not hasattr(self, "curve_canvas"):
            return
        canvas = self.curve_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 240)
        h = max(canvas.winfo_height(), 110)
        canvas.create_rectangle(0, 0, w, h, fill=PANEL_SECTION, outline="")
        margin = 10
        plot_w = w - margin * 2
        plot_h = h - margin * 2
        for idx in range(1, 4):
            x = margin + idx * plot_w / 4
            y = margin + idx * plot_h / 4
            canvas.create_line(x, margin, x, margin + plot_h, fill=BORDER)
            canvas.create_line(margin, y, margin + plot_w, y, fill=BORDER)
        canvas.create_line(margin, margin + plot_h, margin + plot_w, margin, fill="#9ab0c1", dash=(4, 4))
        lut = self.build_tone_curve_lut(self.edit_params)
        points = []
        for idx, value in enumerate(lut):
            px = margin + idx / 255.0 * plot_w
            py = margin + plot_h - (value / 255.0) * plot_h
            points.extend([px, py])
        canvas.create_line(*points, fill=ACCENT_CORAL, width=2, smooth=True)
        for field_name, x_px, y_px in self.get_curve_control_positions(plot_w, plot_h, margin):
            active = field_name == self.curve_drag_field
            canvas.create_oval(x_px - 5, y_px - 5, x_px + 5, y_px + 5, fill=ACCENT_GOLD if active else ACCENT_BLUE, outline="#ffffff", width=1)
        canvas.create_rectangle(margin, margin, margin + plot_w, margin + plot_h, outline=BORDER)

    def get_curve_field_order(self):
        return ["curve_shadow", "curve_darks", "curve_mids", "curve_lights", "curve_highlights"]

    def get_curve_control_positions(self, plot_w, plot_h, margin):
        anchors_x = [32, 80, 128, 176, 224]
        base_y = {
            "curve_shadow": 32 + self.edit_params.curve_shadow * 0.55,
            "curve_darks": 80 + self.edit_params.curve_darks * 0.70,
            "curve_mids": 128 + self.edit_params.curve_mids * 0.80,
            "curve_lights": 176 + self.edit_params.curve_lights * 0.70,
            "curve_highlights": 224 + self.edit_params.curve_highlights * 0.55,
        }
        positions = []
        for field_name, x_anchor in zip(self.get_curve_field_order(), anchors_x):
            y_anchor = np.clip(base_y[field_name], 0, 255)
            x_px = margin + (x_anchor / 255.0) * plot_w
            y_px = margin + plot_h - (y_anchor / 255.0) * plot_h
            positions.append((field_name, x_px, y_px))
        return positions

    def start_curve_drag(self, event):
        if not hasattr(self, "curve_canvas"):
            return
        canvas = self.curve_canvas
        w = max(canvas.winfo_width(), 240)
        h = max(canvas.winfo_height(), 110)
        margin = 10
        plot_w = w - margin * 2
        plot_h = h - margin * 2
        nearest = None
        nearest_distance = 1e9
        for field_name, x_px, y_px in self.get_curve_control_positions(plot_w, plot_h, margin):
            distance = math.hypot(event.x - x_px, event.y - y_px)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = field_name
        if nearest is not None and nearest_distance <= 22:
            self.curve_drag_field = nearest
            self.drag_curve_point(event)

    def drag_curve_point(self, event):
        if self.curve_drag_field is None:
            return
        canvas = self.curve_canvas
        w = max(canvas.winfo_width(), 240)
        h = max(canvas.winfo_height(), 110)
        margin = 10
        plot_h = h - margin * 2
        normalized_y = np.clip((margin + plot_h - event.y) / max(plot_h, 1), 0.0, 1.0)
        target_y = normalized_y * 255.0
        base_map = {
            "curve_shadow": (32.0, 0.55),
            "curve_darks": (80.0, 0.70),
            "curve_mids": (128.0, 0.80),
            "curve_lights": (176.0, 0.70),
            "curve_highlights": (224.0, 0.55),
        }
        base, scale = base_map[self.curve_drag_field]
        slider_value = np.clip((target_y - base) / max(scale, 0.001), -100.0, 100.0)
        self.sliders[self.curve_drag_field][0].set(float(slider_value))
        self.on_slider_change(self.curve_drag_field, slider_value)

    def finish_curve_drag(self, _event=None):
        self.curve_drag_field = None
        self.update_curve_graph()

    def analyze_auto_photo(self, bgr8):
        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
        b_mean = float(np.mean(bgr8[..., 0]))
        g_mean = float(np.mean(bgr8[..., 1]))
        r_mean = float(np.mean(bgr8[..., 2]))
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dynamic_range = float(np.percentile(gray, 96) - np.percentile(gray, 4))
        saturation = float(np.mean(hsv[..., 1]) / 255.0)
        faces = self.get_detected_faces_from_source(bgr8, self.current_source_stamp, bgr8.shape[:2], preview_mode=False)
        image_area = float(max(1, bgr8.shape[0] * bgr8.shape[1]))
        face_areas = sorted([(w * h) / image_area for _x, _y, w, h in faces], reverse=True)
        face_ratio = float(min(0.25, sum(face_areas[:2]))) if face_areas else 0.0
        portrait_strength = float(np.clip(face_ratio / 0.12, 0.0, 1.0))
        if faces:
            portrait_strength = max(0.28, portrait_strength)
        return {
            "brightness": brightness,
            "contrast": contrast,
            "dynamic_range": dynamic_range,
            "saturation": saturation,
            "b_mean": b_mean,
            "g_mean": g_mean,
            "r_mean": r_mean,
            "faces": faces,
            "portrait_strength": portrait_strength,
        }

    def classify_scene_profile(self, scene):
        if len(scene["faces"]) >= 2 and scene["portrait_strength"] >= 0.18:
            return "Group Portrait"
        if scene["portrait_strength"] >= 0.28:
            return "Portrait"
        if scene["brightness"] < 0.24:
            return "Low Light"
        if scene["saturation"] > 0.46 and scene["dynamic_range"] > 0.55:
            return "Landscape"
        if scene["saturation"] < 0.20 and scene["contrast"] < 0.14:
            return "Soft Indoor"
        if scene["dynamic_range"] > 0.72:
            return "High Contrast"
        return "General"

    def refresh_scene_profile(self, source_bgr=None):
        source_bgr = self.original_bgr if source_bgr is None else source_bgr
        if source_bgr is None:
            self.current_scene_profile = "Unknown"
            return
        try:
            self.current_scene_profile = self.classify_scene_profile(self.analyze_auto_photo(source_bgr))
        except Exception:
            self.current_scene_profile = "Unknown"

    def build_soft_auto_enhance_params(self, bgr8):
        scene = self.analyze_auto_photo(bgr8)
        portrait = scene["portrait_strength"]
        brightness = scene["brightness"]
        contrast = scene["contrast"]
        saturation = scene["saturation"]
        dynamic_range = scene["dynamic_range"]
        temperature = np.clip((scene["b_mean"] - scene["r_mean"]) * 0.12, -8.0, 8.0)
        tint = np.clip((scene["g_mean"] - (scene["r_mean"] + scene["b_mean"]) * 0.5) * 0.14, -6.0, 6.0)
        return EditParams(
            exposure=float(np.clip((0.53 - brightness) * 22.0, -8.0, 8.0)),
            brightness=float(np.clip((0.56 - brightness) * 18.0, -5.0, 7.0)),
            contrast=float(np.clip((0.15 - contrast) * 48.0 - portrait * 2.0 - max(0.0, dynamic_range - 0.62) * 18.0, -10.0, 2.5)),
            highlights=float(np.clip(-6.0 - portrait * 10.0 - max(0.0, contrast - 0.20) * 45.0, -20.0, -4.0)),
            shadows=float(np.clip(5.0 + portrait * 8.0 + max(0.0, 0.52 - brightness) * 18.0, 3.0, 16.0)),
            whites=float(np.clip(-2.0 - portrait * 4.0 - max(0.0, dynamic_range - 0.72) * 16.0, -10.0, 1.0)),
            blacks=float(np.clip(1.0 + max(0.0, 0.15 - contrast) * 20.0, 0.0, 4.0)),
            temperature=float(temperature),
            tint=float(tint),
            saturation=float(-3.0 if saturation > 0.42 else -1.0 if saturation > 0.34 else 0.0),
            vibrance=float(np.clip(5.0 - max(0.0, saturation - 0.32) * 12.0, 1.0, 6.0)),
            clarity=float(-3.0 - portrait * 7.0),
            texture=float(-2.0 - portrait * 9.0),
            sharpness=float(6.0 if portrait > 0.15 else 10.0),
            sharp_radius=float(12.0 if portrait > 0.15 else 14.0),
            denoise_luma=float(6.0 + portrait * 10.0),
            denoise_color=float(4.0 + portrait * 8.0),
            curve_mids=float(2.0 + portrait * 2.0),
            curve_lights=float(-3.0 - portrait * 4.0),
            curve_highlights=float(-5.0 - portrait * 5.0),
            fade=float(1.0 + portrait * 2.0),
            subject_light=float(portrait * 4.0),
            background_desat=float(portrait * 3.0),
            skin_smooth=float(10.0 + portrait * 16.0 if portrait > 0 else 0.0),
            skin_whiten=float(4.0 + portrait * 8.0 if portrait > 0 else 0.0),
            acne_remove=float(8.0 + portrait * 10.0 if portrait > 0 else 0.0),
            skin_tone_balance=float(6.0 + portrait * 10.0 if portrait > 0 else 0.0),
            eye_brighten=float(2.0 + portrait * 5.0 if portrait > 0 else 0.0),
            teeth_whiten=float(1.0 + portrait * 4.0 if portrait > 0.25 else 0.0),
            under_eye_soften=float(3.0 + portrait * 7.0 if portrait > 0 else 0.0),
            lip_enhance=float(1.0 + portrait * 3.0 if portrait > 0.20 else 0.0),
        )

    def build_auto_retouch_values(self, mode):
        if self.original_bgr is None:
            return {}
        scene = self.analyze_auto_photo(self.original_bgr)
        portrait = scene["portrait_strength"]
        if mode == "face":
            return {
                "highlights": -8 - portrait * 4.0,
                "shadows": 6 + portrait * 4.0,
                "contrast": -2 - portrait * 2.0,
                "clarity": -4 - portrait * 4.0,
                "texture": -6 - portrait * 6.0,
                "denoise_luma": 6 + portrait * 6.0,
                "face_slim": 8 + portrait * 8.0 if portrait > 0 else 0.0,
                "skin_smooth": 14 + portrait * 12.0 if portrait > 0 else 0.0,
                "skin_whiten": 5 + portrait * 5.0 if portrait > 0 else 0.0,
                "acne_remove": 10 + portrait * 8.0 if portrait > 0 else 0.0,
                "skin_tone_balance": 8 + portrait * 8.0 if portrait > 0 else 0.0,
                "under_eye_soften": 4 + portrait * 8.0 if portrait > 0 else 0.0,
                "eye_brighten": 3 + portrait * 5.0 if portrait > 0 else 0.0,
            }
        if mode == "skin":
            return {
                "contrast": -2 - portrait * 2.0,
                "clarity": -5 - portrait * 4.0,
                "texture": -10 - portrait * 8.0,
                "denoise_luma": 8 + portrait * 8.0,
                "denoise_color": 5 + portrait * 6.0,
                "skin_smooth": 22 + portrait * 14.0 if portrait > 0 else 0.0,
                "skin_whiten": 7 + portrait * 7.0 if portrait > 0 else 0.0,
                "acne_remove": 14 + portrait * 12.0 if portrait > 0 else 0.0,
                "skin_tone_balance": 12 + portrait * 10.0 if portrait > 0 else 0.0,
                "under_eye_soften": 5 + portrait * 8.0 if portrait > 0 else 0.0,
            }
        if mode == "body":
            return {
                "contrast": -1.0,
                "texture": -2.0,
                "sharpness": 6.0,
                "subject_light": 3.0 + portrait * 3.0,
                "background_desat": portrait * 3.0,
                "body_slim": 10.0 + portrait * 10.0 if portrait > 0 else 0.0,
            }
        return {
            "highlights": -10 - portrait * 4.0,
            "shadows": 8 + portrait * 4.0,
            "contrast": -3 - portrait * 2.0,
            "clarity": -6 - portrait * 5.0,
            "texture": -10 - portrait * 8.0,
            "denoise_luma": 8 + portrait * 8.0,
            "denoise_color": 5 + portrait * 6.0,
            "subject_light": 4 + portrait * 3.0,
            "background_desat": portrait * 4.0,
            "body_slim": 8 + portrait * 8.0 if portrait > 0 else 0.0,
            "face_slim": 10 + portrait * 8.0 if portrait > 0 else 0.0,
            "skin_smooth": 24 + portrait * 14.0 if portrait > 0 else 0.0,
            "skin_whiten": 8 + portrait * 7.0 if portrait > 0 else 0.0,
            "acne_remove": 14 + portrait * 12.0 if portrait > 0 else 0.0,
            "skin_tone_balance": 14 + portrait * 10.0 if portrait > 0 else 0.0,
            "eye_brighten": 4 + portrait * 5.0 if portrait > 0 else 0.0,
            "teeth_whiten": 2 + portrait * 4.0 if portrait > 0.25 else 0.0,
            "under_eye_soften": 6 + portrait * 8.0 if portrait > 0 else 0.0,
            "lip_enhance": 2 + portrait * 3.0 if portrait > 0.20 else 0.0,
            "vibrance": 3.0,
        }

    def build_ai_mode_values(self, mode):
        if self.original_bgr is None:
            return {}
        scene = self.analyze_auto_photo(self.original_bgr)
        profile = self.classify_scene_profile(scene)
        portrait = scene["portrait_strength"]
        faces_count = len(scene["faces"])
        base = asdict(self.build_soft_auto_enhance_params(self.original_bgr))
        if mode == "scene":
            if profile == "Group Portrait":
                base.update({
                    "contrast": -4.0,
                    "clarity": -6.0,
                    "texture": -8.0,
                    "subject_light": 8.0,
                    "background_blur": 10.0,
                    "background_desat": 6.0,
                    "skin_smooth": 20.0,
                    "skin_whiten": 7.0,
                    "skin_tone_balance": 18.0,
                    "under_eye_soften": 7.0,
                    "eye_brighten": 4.0,
                })
            elif profile == "Portrait":
                base.update({
                    "contrast": -4.0,
                    "clarity": -7.0,
                    "texture": -10.0,
                    "subject_light": 10.0,
                    "background_blur": 16.0,
                    "background_desat": 8.0,
                    "skin_smooth": 24.0,
                    "skin_whiten": 10.0,
                    "skin_tone_balance": 20.0,
                    "under_eye_soften": 8.0,
                    "eye_brighten": 5.0,
                    "teeth_whiten": 3.0,
                })
            elif profile == "Low Light":
                base.update({
                    "exposure": base["exposure"] + 3.0,
                    "brightness": base["brightness"] + 2.0,
                    "contrast": -5.0,
                    "shadows": min(base["shadows"] + 6.0, 18.0),
                    "highlights": min(base["highlights"], -10.0),
                    "denoise_luma": max(base["denoise_luma"], 14.0),
                    "denoise_color": max(base["denoise_color"], 10.0),
                    "clarity": -3.0,
                    "texture": -2.0,
                    "subject_light": 6.0 if faces_count else 0.0,
                    "skin_tone_balance": 12.0 if faces_count else 0.0,
                })
            elif profile == "Landscape":
                base.update({
                    "contrast": 1.0,
                    "clarity": 4.0,
                    "texture": 6.0,
                    "dehaze": 8.0,
                    "vibrance": 8.0,
                    "sharpness": 16.0,
                    "highlights": -8.0,
                    "shadows": 6.0,
                    "background_desat": 0.0,
                })
            elif profile == "High Contrast":
                base.update({
                    "contrast": -6.0,
                    "highlights": -18.0,
                    "shadows": 12.0,
                    "whites": -8.0,
                    "blacks": 3.0,
                    "clarity": -3.0,
                    "texture": -3.0,
                    "skin_tone_balance": 10.0 if faces_count else 0.0,
                })
            else:
                base.update({
                    "contrast": min(base["contrast"], -2.0 if faces_count else 0.0),
                    "skin_tone_balance": 10.0 if faces_count else 0.0,
                })
            return base
        if mode == "portrait":
            return {
                **base,
                "contrast": -5.0,
                "clarity": -8.0,
                "texture": -12.0,
                "subject_light": 12.0,
                "background_blur": 20.0,
                "background_desat": 10.0,
                "skin_smooth": 26.0 + portrait * 10.0,
                "skin_whiten": 10.0 + portrait * 4.0,
                "skin_tone_balance": 22.0,
                "acne_remove": 16.0,
                "under_eye_soften": 8.0,
                "eye_brighten": 6.0,
                "teeth_whiten": 4.0,
                "lip_enhance": 3.0,
                "face_slim": 8.0 + portrait * 6.0,
            }
        if mode == "group":
            return {
                **base,
                "contrast": -4.0,
                "clarity": -5.0,
                "texture": -7.0,
                "subject_light": 8.0,
                "background_blur": 8.0,
                "background_desat": 6.0,
                "skin_smooth": 18.0,
                "skin_whiten": 6.0,
                "skin_tone_balance": 20.0,
                "acne_remove": 12.0,
                "under_eye_soften": 6.0,
                "eye_brighten": 4.0,
                "face_slim": 0.0,
                "body_slim": 0.0,
            }
        return {
            **base,
            "subject_light": 14.0,
            "background_blur": 22.0,
            "background_desat": 12.0,
            "clarity": -2.0 if portrait > 0 else 2.0,
            "texture": -3.0 if portrait > 0 else 2.0,
            "vignette": 8.0,
            "skin_tone_balance": 12.0 if faces_count else 0.0,
            "eye_brighten": 4.0 if faces_count else 0.0,
        }

    def apply_ai_mode(self, mode, label):
        values = self.build_ai_mode_values(mode)
        if not values:
            return
        self.apply_preset(**values)
        scene_profile = self.classify_scene_profile(self.analyze_auto_photo(self.original_bgr)) if self.original_bgr is not None else "Unknown"
        self.status_var.set(f"{label} applied | Scene: {scene_profile}")
        self.add_history_entry(f"{label} applied")

    def ai_scene_match(self):
        self.apply_ai_mode("scene", "AI Scene")

    def ai_portrait_studio(self):
        self.apply_ai_mode("portrait", "AI Portrait")

    def ai_group_retouch(self):
        self.apply_ai_mode("group", "AI Group")

    def ai_subject_pop(self):
        self.apply_ai_mode("subject", "AI Subject")

    def apply_preset(self, **kwargs):
        if self.original_bgr is None:
            return
        self.push_undo()
        for key, value in kwargs.items():
            if hasattr(self.edit_params, key):
                setattr(self.edit_params, key, value)
        self._sync_params_to_sliders()
        self.clear_processing_cache()
        self._persist_current_record_state()
        self.refresh_filmstrip()
        self.schedule_render(immediate=True)
        self.add_history_entry("Applied preset")
        self.schedule_session_save()

    def preset_natural(self): self.apply_preset(contrast=8, highlights=-10, shadows=10, sharpness=18, vibrance=8)
    def preset_vivid(self): self.apply_preset(contrast=16, saturation=10, vibrance=28, clarity=12, sharpness=25, dehaze=10)
    def preset_portrait(self): self.apply_preset(highlights=-18, shadows=16, temperature=6, clarity=-8, texture=-5, vibrance=6, denoise_luma=12)
    def preset_landscape(self): self.apply_preset(contrast=18, dehaze=18, clarity=14, texture=12, vibrance=20, sharpness=22)
    def preset_bw(self): self.apply_preset(saturation=-100, contrast=20, clarity=12, grain=15, fade=10)
    def preset_beauty(self): self.apply_preset(highlights=-8, shadows=8, clarity=-4, texture=-8, denoise_luma=8, face_slim=14, skin_smooth=28, skin_whiten=12, acne_remove=20, under_eye_soften=6, eye_brighten=4)
    def auto_face_refine(self): self.apply_preset(**self.build_auto_retouch_values("face"))
    def auto_skin_refine(self): self.apply_preset(**self.build_auto_retouch_values("skin"))
    def auto_body_refine(self): self.apply_preset(**self.build_auto_retouch_values("body"))
    def auto_beauty_plus(self): self.apply_preset(**self.build_auto_retouch_values("beauty"))

    def apply_standard_frame(self):
        src = self.ensure_current_full_res()
        if src is None:
            return
        ratio = FRAME_RATIOS.get(self.frame_var.get())
        if ratio is None:
            self.status_var.set("Original frame kept")
            return
        self.push_undo(resolved_current=src)
        self._replace_with_new_base(self.crop_to_ratio(src, ratio[0], ratio[1]), f"Applied frame: {self.frame_var.get()}")
        self.set_zoom_fit(reset_pan=True)

    def crop_to_ratio(self, img, ratio_w, ratio_h):
        h, w = img.shape[:2]
        target_ratio, current_ratio = ratio_w / ratio_h, w / h
        if current_ratio > target_ratio:
            new_w, x1 = int(h * target_ratio), (w - int(h * target_ratio)) // 2
            return img[:, x1:x1 + new_w]
        new_h, y1 = int(w / target_ratio), (h - int(w / target_ratio)) // 2
        return img[y1:y1 + new_h, :]

    def rotate_image(self, angle):
        src = self.ensure_current_full_res()
        if src is None:
            return
        self.push_undo(resolved_current=src)
        if angle == 90:
            out = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            out = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            h, w = src.shape[:2]
            out = cv2.warpAffine(src, cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self._replace_with_new_base(out, f"Rotated {angle} degrees")

    def flip_horizontal(self):
        src = self.ensure_current_full_res()
        if src is not None:
            self.push_undo(resolved_current=src); self._replace_with_new_base(cv2.flip(src, 1), "Flipped horizontally")

    def flip_vertical(self):
        src = self.ensure_current_full_res()
        if src is not None:
            self.push_undo(resolved_current=src); self._replace_with_new_base(cv2.flip(src, 0), "Flipped vertically")

    def to_gray(self):
        src = self.ensure_current_full_res()
        if src is None:
            return
        self.push_undo(resolved_current=src)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self._replace_with_new_base(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Converted to grayscale")

    def apply_sharpen_once(self):
        src = self.ensure_current_full_res()
        if src is not None:
            self.push_undo(resolved_current=src); self._replace_with_new_base(self.apply_sharpen_advanced(src, 35, 1.6), "Sharpen applied")

    def apply_denoise_once(self):
        src = self.ensure_current_full_res()
        if src is not None:
            self.push_undo(resolved_current=src); self._replace_with_new_base(cv2.fastNlMeansDenoisingColored(src, None, 8, 8, 7, 21), "Denoise applied")

    def apply_clahe_once(self):
        src = self.ensure_current_full_res()
        if src is None:
            return
        self.push_undo(resolved_current=src)
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._replace_with_new_base(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR), "CLAHE applied")

    def auto_enhance(self):
        src = self.ensure_current_full_res()
        if src is None:
            return
        self.push_undo(resolved_current=src)
        params = self.build_soft_auto_enhance_params(src)
        auto_stamp = f"auto-soft:{self.current_source_stamp}"
        img = self.apply_pipeline(src, params, preview_mode=False, source_bgr=src, source_stamp=auto_stamp)
        blend = 0.90 if params.skin_smooth > 0 else 0.93
        softened = cv2.addWeighted(src, 1.0 - blend, img, blend, 0)
        self._replace_with_new_base(softened, "Auto enhance applied")

    def _replace_with_new_base(self, out_bgr, status_text):
        self.original_bgr = out_bgr.copy()
        self.refresh_scene_profile(out_bgr)
        self.edit_params = EditParams()
        self.local_adjustments = []
        self.local_adjustment_selection = -1
        self.frame_var.set("Original")
        self._sync_params_to_sliders()
        self.refresh_local_adjustment_stack()
        self.clear_processing_cache()
        self._set_full_resolution_state(out_bgr, source_changed=True)
        self._update_current_record_base(self.original_bgr)
        self._persist_current_record_state()
        self.update_histogram(out_bgr)
        self.update_file_info()
        self.request_preview_refresh(immediate=True)
        self.refresh_filmstrip()
        self.queue_analysis_precompute(out_bgr.copy(), self.current_source_stamp)
        self.status_var.set(status_text)
        self.add_history_entry(status_text)
        self.schedule_session_save()

    def update_file_info(self, current_only=False):
        self.file_info_text.config(state="normal")
        self.file_info_text.delete("1.0", "end")
        if self.original_bgr is None:
            self.file_info_text.insert("1.0", "No image loaded")
            self.file_info_text.config(state="disabled")
            self.info_var.set("No image")
            self.batch_title_var.set("No image loaded")
            return
        img = self.current_bgr if current_only and self.current_bgr is not None else self.original_bgr
        if current_only and self.original_bgr is not None and img.shape[:2] != self.original_bgr.shape[:2]:
            img = self.original_bgr
        h, w = img.shape[:2]
        raw_text = "RAW" if self.path and os.path.splitext(self.path)[1].lower() in {".raw", ".dng", ".nef", ".cr2", ".arw"} else "Raster"
        batch_text = f"{self.current_index + 1} / {len(self.batch_items)}" if self.batch_items and 0 <= self.current_index < len(self.batch_items) else "Single"
        info = [
            f"File: {os.path.basename(self.path) if self.path else '-'}",
            f"Batch: {batch_text}",
            f"Type: {raw_text}",
            f"Size: {w} x {h}",
            f"Channels: {img.shape[2] if img.ndim == 3 else 1}",
            f"Scene: {self.current_scene_profile}",
            f"Preview mode: {self.preview_mode_var.get()}",
            f"Zoom: {self.zoom_var.get()}",
            f"Local strokes: {len(self.local_adjustments)}",
        ]
        self.file_info_text.insert("1.0", "\n".join(info))
        self.file_info_text.config(state="disabled")
        self.info_var.set(f"{w}x{h} | {raw_text}")

    def update_histogram(self, bgr):
        self.hist_canvas.delete("all")
        if bgr is None:
            return
        w, h = max(self.hist_canvas.winfo_width(), HIST_W), max(self.hist_canvas.winfo_height(), HIST_H)
        self.hist_canvas.create_rectangle(0, 0, w, h, fill=PANEL_SECTION, outline="")
        margin, plot_w, plot_h, maxv = 8, w - 16, h - 16, 1
        colors, histograms = [(ACCENT_BLUE, 0), (ACCENT_TEAL, 1), (ACCENT_CORAL, 2)], []
        for _color, channel in colors:
            hist = cv2.calcHist([bgr], [channel], None, [256], [0, 256]).flatten()
            histograms.append(hist)
            maxv = max(maxv, hist.max())
        luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        luma_hist = cv2.calcHist([luma], [0], None, [256], [0, 256]).flatten()
        maxv = max(maxv, luma_hist.max())
        for color, channel in colors:
            hist = histograms[channel] / maxv
            pts = [(margin + int(i / 255 * plot_w), margin + plot_h - int(hist[i] * plot_h)) for i in range(256)]
            for i in range(1, len(pts)):
                self.hist_canvas.create_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], fill=color, width=1.2)
        luma_pts = [(margin + int(i / 255 * plot_w), margin + plot_h - int((luma_hist[i] / maxv) * plot_h)) for i in range(256)]
        for i in range(1, len(luma_pts)):
            self.hist_canvas.create_line(luma_pts[i - 1][0], luma_pts[i - 1][1], luma_pts[i][0], luma_pts[i][1], fill="#f4f8fb", width=1.4)
        self.hist_canvas.create_rectangle(margin, margin, margin + plot_w, margin + plot_h, outline=BORDER)

    def _set_status_ready(self):
        pose_mode = "Pose" if self.mp_pose is not None else "Heuristic"
        mesh_mode = "FaceMesh" if self.mp_face_mesh is not None else "BBox"
        subject_mode = "Segmentation" if self.mp_selfie_segmentation is not None else "Heuristic"
        self.status_var.set(f"Ready | {'RAW ready' if RAWPY_AVAILABLE else 'No rawpy'} | Face detect: {'MediaPipe' if self.mp_face_detector is not None else 'Haar'} | Mesh: {mesh_mode} | Body: {pose_mode} | Subject: {subject_mode}")


def main():
    root = tk.Tk()
    PhotoEditorProX(root)
    root.mainloop()


if __name__ == "__main__":
    main()
