import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
from pathlib import Path
import platform


def open_first_camera(max_idx=5):
    # Prefer DirectShow on Windows, then MSMF, then ANY; use V4L2 on Linux
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for api in backends:
        for i in range(max_idx):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                return cap, i, api
            cap.release()
    return None, None, None


class YOLODetectionApp:
    def __init__(
        self,
        root,
        model_path="C:/Users/Jiyaad/Documents/GitHub/Space-Station-Challenge/runs/detect/linux/weights/best.pt",
    ):
        self.root = root
        self.root.title("YOLO11n Real-Time Object Detection")
        self.root.geometry("1280x800")
        self.root.configure(bg="#fcfcf9")

        # Initialize variables
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.video_cap = None
        self.is_running = False
        self.video_running = False
        self.current_frame = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        self.load_model()
        self.create_widgets()

    def load_model(self):
        try:
            p = Path(self.model_path)
            if p.exists():
                self.model = YOLO(str(p))
                print(f"Model loaded: {p}")
            else:
                messagebox.showerror("Error", f"Model not found:\n{p}")
                self.model = None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model:\n{e}")
            self.model = None

    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg="#21808d", height=62)
        title_frame.pack(fill=tk.X, padx=11, pady=(11, 0))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="Real-Time Object Detection (YOLO11n)",
            font=("Times New Roman", 20, "bold"),
            bg="#21808d",
            fg="white",
        )
        title_label.pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#fcfcf9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=11, pady=11)

        video_frame = tk.Frame(main_frame, bg="#262828", relief=tk.SOLID, borderwidth=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.video_label = tk.Label(video_frame, bg="#262828")
        self.video_label.pack(padx=12, pady=12)

        control_frame = tk.Frame(
            main_frame, bg="#fffffe", width=325, relief=tk.SOLID, borderwidth=1
        )
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)

        control_title = tk.Label(
            control_frame,
            text="Controls",
            font=("Arial", 16, "bold"),
            bg="#fffffe",
            fg="#13343b",
        )
        control_title.pack(pady=(22, 11))

        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start Detection",
            command=self.start_detection,
            font=("Arial", 12, "bold"),
            bg="#21808d",
            fg="white",
            activebackground="#1d7480",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            width=20,
            height=2,
        )
        self.start_btn.pack(pady=11, padx=22)

        self.stop_btn = tk.Button(
            control_frame,
            text="⬛ Stop Detection",
            command=self.stop_all,
            font=("Arial", 12, "bold"),
            bg="#c0152f",
            fg="white",
            activebackground="#a0122a",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            width=20,
            height=2,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(pady=11, padx=22)

        self.screenshot_btn = tk.Button(
            control_frame,
            text="📸 Capture Screenshot",
            command=self.capture_screenshot,
            font=("Arial", 11),
            bg="#626c71",
            fg="white",
            activebackground="#505959",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            width=20,
            height=2,
            state=tk.DISABLED,
        )
        self.screenshot_btn.pack(pady=11, padx=22)

        self.video_btn = tk.Button(
            control_frame,
            text="🎞 Analyze Video",
            command=self.open_video,
            font=("Arial", 11),
            bg="#21808d",
            fg="white",
            activebackground="#1d7480",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            width=20,
            height=2,
        )
        self.video_btn.pack(pady=11, padx=22)

        separator = tk.Frame(control_frame, bg="#e0e0e0", height=2)
        separator.pack(fill=tk.X, pady=22, padx=22)

        info_title = tk.Label(
            control_frame,
            text="Information",
            font=("Arial", 14, "bold"),
            bg="#fffffe",
            fg="#13343b",
        )
        info_title.pack(pady=(11, 6))

        self.fps_label = tk.Label(
            control_frame, text="FPS: 0.0", font=("Arial", 12), bg="#fffffe", fg="#626c71"
        )
        self.fps_label.pack(pady=6)

        self.status_label = tk.Label(
            control_frame,
            text="Status: Ready",
            font=("Arial", 12),
            bg="#fffffe",
            fg="#626c71",
        )
        self.status_label.pack(pady=6)

        model_info = tk.Label(
            control_frame,
            text=f"Model: YOLO11(n)",
            font=("Arial", 10),
            bg="#fffffe",
            fg="#626c71",
            wraplength=253,
        )
        model_info.pack(pady=6)

        separator2 = tk.Frame(control_frame, bg="#e0e0e0", height=2)
        separator2.pack(fill=tk.X, pady=22, padx=22)

        instructions = tk.Label(
            control_frame,
            text=(
                "Instructions:\n\n"
                "1. Click 'Start Detection' to begin\n"
                "2. Allow camera access if prompted\n"
                "3. Objects will be detected in real-time\n"
                "4. Use 'Capture' to save frames\n"
                "5. Click 'Stop' when done"
            ),
            font=("Arial", 9),
            bg="#fffffe",
            fg="#626c71",
            justify=tk.LEFT,
            wraplength=250,
        )
        instructions.pack(pady=10, padx=20)

    def start_detection(self):
        try:
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded; check the weights path.")
                return
            cap, idx, api = open_first_camera()
            if cap is None:
                messagebox.showerror("Error", "Couldn't access webcam!")
                return

            self.cap = cap
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.screenshot_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", fg="#21808d")
            self.frame_count = 0
            self.start_time = time.time()
            self.root.after(10, self.update_frame)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")

    def stop_detection(self):
        self.is_running = False
        if getattr(self, "cap", None):
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.screenshot_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped successfully", fg="#c0152f")
        self.video_label.config(image="")

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_label.config(text="Status: Camera read failed", fg="#c0152f")
            self.root.after(300, self.update_frame)
            return

        # Inference (Ultralytics model.__call__ returns a list of Results)
        results = self.model(frame, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()
        self.current_frame = annotated.copy()

        # FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        # Show in Tkinter
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (800, 600))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture_screenshot(self):
        if self.current_frame is not None:
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/detection_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Success", f"Screenshot saved:\n{filename}")
        else:
            messagebox.showwarning("Warning", "No frame to capture!")

    def open_video(self):
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select a video", filetypes=filetypes)
        if not path:
            return
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded; check the weights path.")
            return

        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file!")
            return

        self.status_label.config(text="Status: Analyzing video", fg="#21808d")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.screenshot_btn.config(state=tk.DISABLED)
        self.video_running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.root.after(1, self._video_tick)

    def _video_tick(self):
        if not self.video_running or self.video_cap is None:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self._end_video()
            return

        results = self.model(frame, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (800, 600))
        imgtk = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(1, self._video_tick)

    def _end_video(self):
        self.video_running = False
        if getattr(self, "video_cap", None):
            self.video_cap.release()
            self.video_cap = None
        self.status_label.config(text="Status: Ready", fg="#626c71")
        self.stop_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)
        self.screenshot_btn.config(state=tk.DISABLED)

    def stop_video(self):
        self.video_running = False

    def stop_all(self):
        self.stop_video()
        self.stop_detection()

    def on_closing(self):
        self.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    # Set your weights path here (forward slashes are safe on Windows)
    model_path = "C:/Users/Jiyaad/Documents/GitHub/Space-Station-Challenge/runs/detect/linux/best.pt"
    app = YOLODetectionApp(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
