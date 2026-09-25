import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk

from config.settings import SCENARIOS, GUI_TITLE, GUI_SUBTITLE, GUI_WIDTH, GUI_HEIGHT, COLORS, VIDEOS_DIR, IMAGES_DIR
from core.scenario_manager import ScenarioManager
from core.video_manager import VideoManager


class PerceptionDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(GUI_TITLE)
        self.root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.root.configure(bg=COLORS["bg_dark"])
        
        self.scenario_manager = ScenarioManager()
        self.video_manager = VideoManager()
        
        self.current_scenario_key = None
        self.is_running = False
        self.current_input_path = None
        self.current_input_mode = None
        
        self.seatbelt_status = True
        
        self._build_ui()
        self._show_welcome_screen()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=COLORS["bg_dark"])
        style.configure('Card.TFrame', background=COLORS["bg_card"])
        style.configure('Sidebar.TFrame', background=COLORS["bg_medium"])
        style.configure('TLabel', background=COLORS["bg_dark"], foreground=COLORS["text_primary"], font=("Helvetica", 11))
        style.configure('Header.TLabel', font=("Helvetica", 16, "bold"), foreground=COLORS["accent_blue"])
        style.configure('SubHeader.TLabel', font=("Helvetica", 10), foreground=COLORS["text_secondary"])
        style.configure('TButton', font=("Helvetica", 11, "bold"), background=COLORS["bg_card"], foreground=COLORS["text_primary"], borderwidth=0)
        style.map('TButton', background=[('active', COLORS["bg_card_hover"])])
        
        style.configure('Run.TButton', background=COLORS["accent_green"], foreground="white")
        style.map('Run.TButton', background=[('active', '#2ea043')])
        
        style.configure('Stop.TButton', background=COLORS["accent_red"], foreground="white")
        style.map('Stop.TButton', background=[('active', '#da3633')])
        
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.sidebar = ttk.Frame(self.main_container, style='Sidebar.TFrame', width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.sidebar.pack_propagate(False)
        
        sidebar_header = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        sidebar_header.pack(fill=tk.X, padx=20, pady=20)
        ttk.Label(sidebar_header, text=GUI_TITLE, style='Header.TLabel', background=COLORS["bg_medium"], wraplength=260).pack(anchor=tk.W)
        ttk.Label(sidebar_header, text=GUI_SUBTITLE, style='SubHeader.TLabel', background=COLORS["bg_medium"], wraplength=260).pack(anchor=tk.W, pady=(5, 0))
        
        categories = {}
        for key, data in SCENARIOS.items():
            cat = data["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((key, data))
            
        for cat, scens in categories.items():
            cat_label = ttk.Label(self.sidebar, text=cat, font=("Helvetica", 9, "bold"), foreground=COLORS["text_secondary"], background=COLORS["bg_medium"])
            cat_label.pack(anchor=tk.W, padx=20, pady=(15, 5))
            
            for key, data in scens:
                btn = ttk.Button(self.sidebar, text=data["name"], command=lambda k=key: self._select_scenario(k))
                btn.pack(fill=tk.X, padx=15, pady=2)
                
        self.content_area = ttk.Frame(self.main_container)
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.status_bar = ttk.Frame(self.root, style='Sidebar.TFrame', height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)
        self.status_label = ttk.Label(self.status_bar, text="SYSTEM READY", font=("Helvetica", 10, "bold"), foreground=COLORS["status_ready"], background=COLORS["bg_medium"])
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)

    def _show_welcome_screen(self):
        for widget in self.content_area.winfo_children():
            widget.destroy()
            
        welcome_frame = ttk.Frame(self.content_area)
        welcome_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        ttk.Label(welcome_frame, text="Welcome to " + GUI_TITLE, font=("Helvetica", 24, "bold")).pack(pady=10)
        ttk.Label(welcome_frame, text="Please select a scenario from the sidebar to begin.", font=("Helvetica", 14), foreground=COLORS["text_secondary"]).pack(pady=10)

    def _select_scenario(self, key):
        if self.is_running:
            self._stop_processing()
            
        self.current_scenario_key = key
        data = SCENARIOS[key]
        self.current_input_mode = data["default_input"]
        self.current_input_path = None
        
        demo_video = data.get("demo_video")
        if demo_video:
            demo_path = os.path.join(VIDEOS_DIR, demo_video)
            if os.path.exists(demo_path):
                self.current_input_path = demo_path
                self.current_input_mode = "video"
        
        for widget in self.content_area.winfo_children():
            widget.destroy()
            
        top_panel = ttk.Frame(self.content_area, style='Card.TFrame')
        top_panel.pack(fill=tk.X, pady=(0, 20))
        
        info_frame = ttk.Frame(top_panel, style='Card.TFrame')
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(info_frame, text=data["name"], font=("Helvetica", 18, "bold"), background=COLORS["bg_card"]).pack(anchor=tk.W)
        ttk.Label(info_frame, text=data["description"], background=COLORS["bg_card"], wraplength=600, foreground=COLORS["text_secondary"]).pack(anchor=tk.W, pady=(10, 0))
        
        controls_frame = ttk.Frame(top_panel, style='Card.TFrame')
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)
        
        inputs_frame = ttk.Frame(controls_frame, style='Card.TFrame')
        inputs_frame.pack(fill=tk.X, pady=(0, 10))
        
        if "video" in data["input_modes"]:
            ttk.Button(inputs_frame, text="Select Video", command=self._select_video).pack(side=tk.LEFT, padx=5)
        if "image" in data["input_modes"]:
            ttk.Button(inputs_frame, text="Select Image", command=self._select_image).pack(side=tk.LEFT, padx=5)
        if "live_camera" in data["input_modes"]:
            ttk.Button(inputs_frame, text="Live Camera", command=self._select_camera).pack(side=tk.LEFT, padx=5)
            
        if key == "passenger_safety":
            ttk.Button(inputs_frame, text="Toggle Seatbelt", command=self._toggle_seatbelt).pack(side=tk.LEFT, padx=5)
            
        self.btn_run = ttk.Button(controls_frame, text="Run Scenario", style='Run.TButton', command=self._run_scenario)
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(controls_frame, text="Stop", style='Stop.TButton', command=self._stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)
        
        self.lbl_input_status = ttk.Label(controls_frame, text="No input selected", background=COLORS["bg_card"], foreground=COLORS["accent_orange"])
        self.lbl_input_status.pack(pady=5)
        
        if self.current_input_path and self.current_input_mode == "video":
            self.lbl_input_status.config(
                text=f"Demo: {os.path.basename(self.current_input_path)}",
                foreground=COLORS["accent_green"]
            )
        
        display_container = ttk.Frame(self.content_area)
        display_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(display_container, bg="black", width=640, height=480, highlightthickness=0)
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.telemetry_frame = ttk.Frame(display_container, style='Card.TFrame', width=300)
        self.telemetry_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.telemetry_frame.pack_propagate(False)
        
        ttk.Label(self.telemetry_frame, text="TELEMETRY & ALERTS", font=("Helvetica", 12, "bold"), background=COLORS["bg_card"], foreground=COLORS["accent_blue"]).pack(anchor=tk.W, padx=15, pady=15)
        
        self.telemetry_text = tk.Text(self.telemetry_frame, bg=COLORS["bg_card"], fg=COLORS["text_primary"], font=("Consolas", 10), bd=0, highlightthickness=0, wrap=tk.WORD)
        self.telemetry_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.telemetry_text.config(state=tk.DISABLED)

    def _select_video(self):
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path:
            self.current_input_path = path
            self.current_input_mode = "video"
            self.lbl_input_status.config(text=f"Selected: {os.path.basename(path)}", foreground=COLORS["accent_green"])

    def _select_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.current_input_path = path
            self.current_input_mode = "image"
            self.lbl_input_status.config(text=f"Selected: {os.path.basename(path)}", foreground=COLORS["accent_green"])

    def _select_camera(self):
        self.current_input_path = 0
        self.current_input_mode = "live_camera"
        self.lbl_input_status.config(text="Selected: Live Camera", foreground=COLORS["accent_green"])

    def _toggle_seatbelt(self):
        self.seatbelt_status = not self.seatbelt_status
        status_text = "Fastened" if self.seatbelt_status else "Unfastened"
        self.lbl_input_status.config(text=f"Seatbelt: {status_text}", foreground=COLORS["accent_blue"])

    def _update_status(self, text, state="ready"):
        self.status_label.config(text=text)
        if state == "ready":
            self.status_label.config(foreground=COLORS["status_ready"])
        elif state == "running":
            self.status_label.config(foreground=COLORS["status_running"])
        elif state == "error":
            self.status_label.config(foreground=COLORS["status_error"])

    def _log_telemetry(self, text):
        self.telemetry_text.config(state=tk.NORMAL)
        self.telemetry_text.insert(tk.END, text + "\n")
        self.telemetry_text.see(tk.END)
        self.telemetry_text.config(state=tk.DISABLED)

    def _clear_telemetry(self):
        self.telemetry_text.config(state=tk.NORMAL)
        self.telemetry_text.delete(1.0, tk.END)
        self.telemetry_text.config(state=tk.DISABLED)

    def _display_telemetry(self, telemetry):
        """Display telemetry dict in the telemetry panel."""
        if not isinstance(telemetry, dict):
            return
        self.telemetry_text.config(state=tk.NORMAL)
        self.telemetry_text.delete(1.0, tk.END)
        for key, value in telemetry.items():
            if isinstance(value, float):
                self.telemetry_text.insert(tk.END, f"{key}: {value:.3f}\n")
            else:
                self.telemetry_text.insert(tk.END, f"{key}: {value}\n")
        self.telemetry_text.config(state=tk.DISABLED)

    def _run_scenario(self):
        if self.current_input_path is None and self.current_input_mode != "live_camera":
            messagebox.showwarning("Warning", "Please select an input source first.")
            return
            
        self._clear_telemetry()
        self._log_telemetry("Initializing scenario...")
        self._update_status(f"LOADING: {SCENARIOS[self.current_scenario_key]['name']}", "running")
        self.root.update()
        
        detector, error = self.scenario_manager.get_detector(self.current_scenario_key)
        
        if error:
            messagebox.showerror("Initialization Error", f"Could not load detector:\n{error}")
            self._update_status("ERROR: Detector failed to load", "error")
            return
            
        self.detector = detector
        self.scenario_manager.reset_detector(self.current_scenario_key)
        
        try:
            if self.current_input_mode == "video":
                self.video_manager.open_video(self.current_input_path)
            elif self.current_input_mode == "live_camera":
                self.video_manager.open_camera(self.current_input_path)
            elif self.current_input_mode == "image":
                img = self.video_manager.read_image(self.current_input_path)
                self._process_single_image(img)
                return
                
            self.is_running = True
            self.btn_run.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            
            fps = self.video_manager.fps
            self.delay_ms = max(15, int(1000 / fps))
            
            self._update_status(f"PROCESSING: {SCENARIOS[self.current_scenario_key]['name']}", "running")
            self._process_video_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open source:\n{str(e)}")
            self._update_status("ERROR: Source failed", "error")

    def _stop_processing(self):
        self.is_running = False
        self.video_manager.release()
        
        if hasattr(self, 'btn_run') and self.btn_run.winfo_exists():
            self.btn_run.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            
        self._update_status("SYSTEM READY", "ready")
        self._log_telemetry("Processing stopped.")

    def _process_single_image(self, img):
        self._log_telemetry("Processing image...")
        self._update_status(f"PROCESSING: {SCENARIOS[self.current_scenario_key]['name']}", "running")
        self.root.update()
        try:
            result = self.detector.process_frame(img)
            if isinstance(result, tuple):
                annotated, telemetry = result
                self._display_frame(annotated)
                self._display_telemetry(telemetry)
            else:
                self._display_frame(result)
            self._log_telemetry("Processing complete.")
            self._update_status("SYSTEM READY", "ready")
        except Exception as e:
            self._log_telemetry(f"Error processing image: {str(e)}")
            self._update_status("ERROR: Processing failed", "error")

    def _process_video_frame(self):
        if not self.is_running:
            return
            
        ret, frame = self.video_manager.read_frame()
        if not ret:
            self._stop_processing()
            self._log_telemetry("End of video stream.")
            return
            
        try:
            if self.current_scenario_key == "passenger_safety":
                if hasattr(self.detector, 'toggle_seatbelt'):
                    det_belt = getattr(self.detector, 'seatbelt_detector', None)
                    if det_belt is not None:
                        expected = "ON" if self.seatbelt_status else "OFF"
                        if det_belt.get_status() != expected:
                            self.detector.toggle_seatbelt()
                    
            result = self.detector.process_frame(frame)
            
            if isinstance(result, tuple):
                annotated, telemetry = result
                self._display_frame(annotated)
                if self.video_manager.current_frame % 15 == 0:
                    self._display_telemetry(telemetry)
            else:
                self._display_frame(result)
            
            self.root.after(self.delay_ms, self._process_video_frame)
            
        except Exception as e:
            self._log_telemetry(f"Error processing frame: {str(e)}")
            self._stop_processing()

    def _display_frame(self, frame):
        if frame is None:
            return
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
        photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        self.video_canvas.image = photo  # Keep a reference!
