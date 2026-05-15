import os
import json
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import customtkinter as ctk
from tkinter import ttk, messagebox

# Optional preview support
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# ------------------ Optional LLM Settings ------------------
USE_LLM = False
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

# ------------------ Tracking Tuning ------------------
DEMO_MODE_DEFAULT = True
VIDEO_SOURCE_INDEX = 0
LINE_ORIENTATION = "vertical"  # "vertical" or "horizontal"
LINE_POSITION_RATIO = 0.55      # % across frame (x for vertical, y for horizontal)
MOTION_SENSITIVITY = 4500       # larger = less sensitive
CROSSING_COOLDOWN_SECONDS = 2.0

EVENT_DISTANCES = ["100m", "200m", "400m", "800m", "1600m", "3200m"]


def format_seconds(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def expected_split_seconds(event: str, pr_text: str) -> float:
    """Uses PR as rough target total time; expected split is 1/4 of PR as a simple default."""
    pr = safe_float(pr_text, 0.0)
    if pr <= 0:
        fallback = {"100m": 16, "200m": 34, "400m": 72, "800m": 150, "1600m": 340, "3200m": 760}
        pr = fallback.get(event, 120)
    return pr / 4.0


class VideoSplitTracker:
    def __init__(self, on_split, on_frame=None, demo_mode=False):
        self.on_split = on_split
        self.on_frame = on_frame
        self.demo_mode = demo_mode
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        self.last_split_time = 0.0
        self.split_count = 0
        self.last_trigger_time = 0.0

    def start(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _emit_split(self):
        now = time.time()
        elapsed = now - self.start_time
        split_time = elapsed - self.last_split_time
        self.split_count += 1
        self.last_split_time = elapsed
        self.on_split(self.split_count, elapsed, split_time)

    def _run(self):
        if self.demo_mode:
            self._run_demo()
        else:
            self._run_camera()

    def _run_demo(self):
        while not self.stop_event.is_set():
            time.sleep(3.5)
            if self.stop_event.is_set():
                break
            self._emit_split()

    def _run_camera(self):
        cap = cv2.VideoCapture(VIDEO_SOURCE_INDEX)
        if not cap.isOpened():
            return

        ret, prev = cap.read()
        if not ret:
            cap.release()
            return

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)

            h, w = frame.shape[:2]
            if LINE_ORIENTATION == "vertical":
                x = int(w * LINE_POSITION_RATIO)
                cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 2)
                region = thresh[:, max(0, x - 20):min(w, x + 20)]
            else:
                y = int(h * LINE_POSITION_RATIO)
                cv2.line(frame, (0, y), (w, y), (0, 255, 255), 2)
                region = thresh[max(0, y - 20):min(h, y + 20), :]

            crossing_score = int(cv2.countNonZero(region)) if region.size else 0
            now = time.time()
            if crossing_score > MOTION_SENSITIVITY and (now - self.last_trigger_time) > CROSSING_COOLDOWN_SECONDS:
                self.last_trigger_time = now
                self._emit_split()

            cv2.putText(frame, f"Motion:{motion_score} Line:{crossing_score}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
            if self.on_frame:
                self.on_frame(frame)

            prev_gray = gray

        cap.release()


class CoachMeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("CoachMeMac")
        self.geometry("1080x760")

        self.profile = {}
        self.splits = []
        self.analysis = {}
        self.pre_message = ""

        self.tracker = None
        self.timer_running = False
        self.workout_start = None

        self.split_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=1)

        self._build_container()
        self.show_input_page()

    def _build_container(self):
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True, padx=16, pady=16)

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def show_input_page(self):
        self.clear()
        self.splits = []
        self.analysis = {}

        title = ctk.CTkLabel(self.container, text="CoachMeMac", font=ctk.CTkFont(size=30, weight="bold"))
        title.pack(pady=(10, 20))

        body = ctk.CTkFrame(self.container)
        body.pack(fill="both", expand=True, padx=30, pady=10)
        body.grid_columnconfigure((0, 1), weight=1)

        profile_frame = ctk.CTkFrame(body)
        profile_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        workout_frame = ctk.CTkFrame(body)
        workout_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(profile_frame, text="Runner Profile", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        ctk.CTkLabel(workout_frame, text="Workout Info", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        self.event_var = ctk.StringVar(value=EVENT_DISTANCES[2])
        self.weight_entry = ctk.CTkEntry(profile_frame, placeholder_text="Weight (lbs)")
        self.sex_var = ctk.StringVar(value="Female")
        self.age_entry = ctk.CTkEntry(profile_frame, placeholder_text="Age")
        self.height_entry = ctk.CTkEntry(profile_frame, placeholder_text="Height (in)")
        self.pr_entry = ctk.CTkEntry(profile_frame, placeholder_text="Current PR (seconds)")

        ctk.CTkLabel(profile_frame, text="Event").pack(anchor="w", padx=20)
        ctk.CTkOptionMenu(profile_frame, variable=self.event_var, values=EVENT_DISTANCES).pack(fill="x", padx=20, pady=(0, 8))
        for label, widget in [
            ("Weight", self.weight_entry),
            ("Biological Sex", ctk.CTkOptionMenu(profile_frame, variable=self.sex_var, values=["Female", "Male", "Other"])),
            ("Age", self.age_entry),
            ("Height", self.height_entry),
            ("Current PR", self.pr_entry),
        ]:
            ctk.CTkLabel(profile_frame, text=label).pack(anchor="w", padx=20)
            widget.pack(fill="x", padx=20, pady=(0, 8))

        self.no_plan_var = ctk.BooleanVar(value=False)
        self.demo_var = ctk.BooleanVar(value=DEMO_MODE_DEFAULT)
        self.workout_text = ctk.CTkTextbox(workout_frame, height=260)
        self.workout_text.insert("1.0", "6x200m @ controlled pace, 90 sec rest")

        ctk.CTkLabel(workout_frame, text="Workout Description").pack(anchor="w", padx=20)
        self.workout_text.pack(fill="x", padx=20, pady=(0, 8))
        ctk.CTkCheckBox(workout_frame, text="No prior workout plan", variable=self.no_plan_var, command=self._toggle_workout_box).pack(anchor="w", padx=20, pady=5)
        ctk.CTkCheckBox(workout_frame, text="Demo mode (no camera)", variable=self.demo_var).pack(anchor="w", padx=20, pady=5)

        ctk.CTkButton(self.container, text="Continue", height=44, command=self.handle_input_submit).pack(pady=16)

    def _toggle_workout_box(self):
        if self.no_plan_var.get():
            self.workout_text.delete("1.0", "end")
            self.workout_text.configure(state="disabled")
        else:
            self.workout_text.configure(state="normal")

    def handle_input_submit(self):
        workout = "No prior plan" if self.no_plan_var.get() else self.workout_text.get("1.0", "end").strip()
        self.profile = {
            "event": self.event_var.get(),
            "weight": self.weight_entry.get().strip(),
            "sex": self.sex_var.get(),
            "age": self.age_entry.get().strip(),
            "height": self.height_entry.get().strip(),
            "pr_seconds": self.pr_entry.get().strip(),
            "workout": workout,
            "demo_mode": self.demo_var.get(),
        }
        self.show_loading_page()

    def show_loading_page(self):
        self.clear()
        ctk.CTkLabel(self.container, text="Processing runner profile...", font=ctk.CTkFont(size=28, weight="bold")).pack(expand=True)
        self.after(1200, self.show_pre_workout_page)

    def show_pre_workout_page(self):
        self.clear()
        self.pre_message = self.local_pre_workout_message()
        ctk.CTkLabel(self.container, text="Ready to Train", font=ctk.CTkFont(size=30, weight="bold")).pack(pady=16)
        summary = (
            f"Hey athlete! You are focused on the {self.profile['event']} today.\n"
            f"Goal pace is based on your PR of {self.profile.get('pr_seconds') or 'not provided'} seconds.\n\n"
            f"Plan: {self.profile.get('workout', 'No prior plan')}\n\n"
            f"{self.pre_message}"
        )
        ctk.CTkLabel(self.container, text=summary, justify="left", wraplength=850, font=ctk.CTkFont(size=18)).pack(padx=30, pady=20)
        ctk.CTkButton(self.container, text="START WORKOUT", height=54, font=ctk.CTkFont(size=20, weight="bold"), command=self.start_workout).pack(pady=10)

    def start_workout(self):
        self.clear()
        self.splits = []
        self.workout_start = time.time()
        self.timer_running = True

        top = ctk.CTkFrame(self.container)
        top.pack(fill="x", pady=10)
        self.timer_label = ctk.CTkLabel(top, text="00:00.00", font=ctk.CTkFont(size=52, weight="bold"))
        self.timer_label.pack(pady=10)

        mid = ctk.CTkFrame(self.container)
        mid.pack(fill="both", expand=True, padx=10, pady=10)
        mid.grid_columnconfigure((0, 1), weight=1)

        table_frame = ctk.CTkFrame(mid)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=8)

        columns = ("Split #", "Elapsed Time", "Split Time", "Status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        video_frame = ctk.CTkFrame(mid)
        video_frame.grid(row=0, column=1, sticky="nsew", padx=8)
        ctk.CTkLabel(video_frame, text="Live Preview", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=8)
        self.video_label = ctk.CTkLabel(video_frame, text="Demo mode active" if self.profile["demo_mode"] else "Starting camera...")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(self.container, text="END WORKOUT", fg_color="#b91c1c", hover_color="#991b1b", height=50,
                      font=ctk.CTkFont(size=18, weight="bold"), command=self.end_workout).pack(pady=12)

        self.tracker = VideoSplitTracker(on_split=self.on_split_detected, on_frame=self.on_frame, demo_mode=self.profile["demo_mode"])
        self.tracker.start()
        self.update_timer()
        self.process_queues()

    def update_timer(self):
        if self.timer_running:
            self.timer_label.configure(text=format_seconds(time.time() - self.workout_start))
            self.after(50, self.update_timer)

    def on_split_detected(self, split_number, elapsed_time, split_time):
        self.split_queue.put((split_number, elapsed_time, split_time))

    def on_frame(self, frame):
        if Image is None:
            return
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass
        self.frame_queue.put(frame)

    def process_queues(self):
        while not self.split_queue.empty():
            n, elapsed, split_t = self.split_queue.get_nowait()
            status = self.compute_status(split_t)
            self.splits.append({"split": n, "elapsed": elapsed, "split_time": split_t, "status": status})
            self.tree.insert("", "end", values=(n, format_seconds(elapsed), format_seconds(split_t), status))

        if not self.frame_queue.empty() and Image is not None and ImageTk is not None:
            frame = self.frame_queue.get_nowait()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((460, 300))
            tk_img = ImageTk.PhotoImage(img)
            self.video_label.configure(image=tk_img, text="")
            self.video_label.image = tk_img

        if self.timer_running:
            self.after(60, self.process_queues)

    def compute_status(self, split_time):
        target = expected_split_seconds(self.profile["event"], self.profile["pr_seconds"])
        if split_time > target * 1.08:
            return "🔺 Slower"
        if split_time < target * 0.92:
            return "🔻 Fast"
        return "✅ On Pace"

    def end_workout(self):
        self.timer_running = False
        if self.tracker:
            self.tracker.stop()
        self.analysis = self.generate_analysis()
        self.show_final_summary_page()

    def generate_analysis(self):
        if not self.splits:
            return {
                "final_summary": "No splits were captured. Try demo mode or adjust camera setup.",
                "future_workout_suggestion": "Start with 4 x 200m at controlled pace.",
                "next_day_suggestion": "Do easy recovery jog + mobility.",
            }
        split_times = [s["split_time"] if isinstance(s, tuple) else s["split_time"] for s in self.splits]
        total = sum(split_times)
        avg = total / len(split_times)
        fastest = min(split_times)
        slowest = max(split_times)
        consistency = (slowest - fastest)
        fatigue = "noticeable late-race slowdown" if len(split_times) >= 3 and split_times[-1] > avg * 1.1 else "good late-race control"

        summary = (
            f"You finished {len(split_times)} splits in {format_seconds(total)}. "
            f"Average split was {format_seconds(avg)}; fastest {format_seconds(fastest)}, slowest {format_seconds(slowest)}. "
            f"Pacing consistency range was {consistency:.2f}s with {fatigue}."
        )

        fut = "Next session: 5 x 300m at goal pace with 2-minute recovery."
        nxt = "Tomorrow: 20-30 min easy jog, then light core + hip mobility."
        result = {
            "final_summary": summary,
            "future_workout_suggestion": fut,
            "next_day_suggestion": nxt,
            "pre_workout_message": self.pre_message,
        }
        if USE_LLM:
            llm = self.try_llm_summary(result)
            if llm:
                result.update(llm)
        return result

    def local_pre_workout_message(self):
        return "Stay relaxed early and build rhythm one split at a time—you've got this."

    def try_llm_summary(self, fallback):
        try:
            if LLM_PROVIDER != "groq":
                return None
            import requests

            key = os.getenv("GROQ_API_KEY", "")
            if not key:
                return None

            url = "https://api.groq.com/openai/v1/chat/completions"
            payload_data = {
                "event": self.profile["event"], "age": self.profile["age"], "height": self.profile["height"],
                "weight": self.profile["weight"], "sex": self.profile["sex"], "pr": self.profile["pr_seconds"],
                "workout": self.profile["workout"], "splits": self.splits,
            }
            prompt = (
                "Return strictly JSON with keys pre_workout_message, final_summary, future_workout_suggestion, "
                "next_day_suggestion. Keep concise and encouraging. Input data: " + json.dumps(payload_data)
            )
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
            }, timeout=15)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return fallback

    def show_final_summary_page(self):
        self.clear()
        ctk.CTkLabel(self.container, text="Workout Complete", font=ctk.CTkFont(size=30, weight="bold")).pack(pady=14)
        ctk.CTkLabel(self.container, text=self.analysis.get("final_summary", ""), wraplength=900, justify="left", font=ctk.CTkFont(size=16)).pack(padx=20, pady=10)

        table = ttk.Treeview(self.container, columns=("Split", "Elapsed", "SplitTime", "Status"), show="headings", height=8)
        for c in ("Split", "Elapsed", "SplitTime", "Status"):
            table.heading(c, text=c)
            table.column(c, width=140, anchor="center")
        for s in self.splits:
            table.insert("", "end", values=(s["split"], format_seconds(s["elapsed"]), format_seconds(s["split_time"]), s["status"]))
        table.pack(pady=8)

        ctk.CTkLabel(self.container, text=f"Future workout: {self.analysis.get('future_workout_suggestion', '')}", wraplength=900).pack(pady=4)
        ctk.CTkLabel(self.container, text=f"Next-day suggestion: {self.analysis.get('next_day_suggestion', '')}", wraplength=900).pack(pady=4)

        btns = ctk.CTkFrame(self.container)
        btns.pack(pady=16)
        ctk.CTkButton(btns, text="DOWNLOAD SUMMARY", command=self.export_summary).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="RETURN TO BEGINNING", command=self.show_input_page).pack(side="left", padx=8)

    def export_summary(self):
        out_dir = Path("data/workout_summaries")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"workout_{stamp}.txt"

        lines = [
            "CoachMeMac Workout Summary",
            "=" * 40,
            f"Date: {datetime.now().isoformat()}",
            "",
            "Runner Profile:",
        ]
        for k in ["event", "weight", "sex", "age", "height", "pr_seconds"]:
            lines.append(f"- {k}: {self.profile.get(k, '')}")
        lines.extend(["", "Workout Description:", self.profile.get("workout", ""), "", "Splits:"])
        for s in self.splits:
            lines.append(f"Split {s['split']}: elapsed {format_seconds(s['elapsed'])}, split {format_seconds(s['split_time'])}, status {s['status']}")
        lines.extend([
            "",
            "Final Summary:", self.analysis.get("final_summary", ""),
            "",
            "Future Workout Suggestion:", self.analysis.get("future_workout_suggestion", ""),
            "",
            "Next Day Suggestion:", self.analysis.get("next_day_suggestion", ""),
        ])

        path.write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("Saved", f"Summary saved to:\n{path}")


if __name__ == "__main__":
    app = CoachMeApp()
    app.mainloop()
