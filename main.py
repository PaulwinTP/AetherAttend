import os
import sys
import cv2
import json
import csv
import time
import queue
import shutil
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import face_recognition

# Color Palette (Modern Slate & Light Sky Blue Theme)
BG_PRIMARY = "#F0F9FF"      # Slate-50 / Light sky blue tint
BG_CARD = "#FFFFFF"         # White containers
COLOR_BORDER = "#E0F2FE"    # Sky-100
SKY_DARK = "#0369A1"        # Sky-700
SKY_MED = "#0284C7"         # Sky-600
SKY_LIGHT = "#BAE6FD"       # Sky-200
TEXT_MAIN = "#0F172A"       # Slate-900
TEXT_MUTED = "#475569"      # Slate-600
COLOR_SUCCESS = "#10B981"   # Green-500
COLOR_SUCCESS_HOVER = "#059669"
COLOR_DANGER = "#EF4444"    # Red-500
COLOR_DANGER_HOVER = "#DC2626"
COLOR_PURPLE = "#8B5CF6"    # Purple-500
COLOR_PURPLE_HOVER = "#7C3AED"

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Attendance Register")
        self.root.geometry("1050x750")
        self.root.configure(bg=BG_PRIMARY)
        self.root.resizable(True, True)

        # File paths & directories
        self.db_dir = "database"
        self.dataset_dir = os.path.join(self.db_dir, "dataset")
        self.attendance_dir = os.path.join(self.db_dir, "attendance")
        self.students_path = os.path.join(self.db_dir, "students.json")
        
        # Init Folders
        self.init_directories()

        # Student Profile and Encoding Caches
        self.students_map = {}
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.load_students_and_encodings()

        # Threading & state variables
        self.camera_active = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.feed_thread = None
        self.rec_thread = None
        
        # Decoupled bounding box overlay caches
        self.latest_frame = None
        self.recognition_results = []
        
        # Session logging tracking (enforce log only once per active session)
        self.logged_this_session = set()
        
        # Build interface
        self.setup_ui()

        # Register cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_directories(self):
        """Creates necessary folders for database, datasets and logs."""
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)

    def load_students_and_encodings(self):
        """Loads registered student details from students.json and caches face encodings in memory."""
        self.known_face_encodings = []
        self.known_face_metadata = []
        
        if os.path.exists(self.students_path):
            try:
                with open(self.students_path, 'r') as f:
                    self.students_map = json.load(f)
                
                # Cache encodings for real-time comparison
                for reg, profile in self.students_map.items():
                    # Support both list of encodings ("encodings") and older single encoding ("encoding") formats
                    if "encodings" in profile:
                        for enc in profile["encodings"]:
                            self.known_face_encodings.append(np.array(enc))
                            self.known_face_metadata.append(profile)
                    elif "encoding" in profile:
                        self.known_face_encodings.append(np.array(profile["encoding"]))
                        self.known_face_metadata.append(profile)
                
                unique_names = list(set([s.get("name", "Unknown") for s in self.known_face_metadata]))
                print(f"Loaded student database. Cached {len(self.known_face_encodings)} face encodings. Registered profiles: {unique_names}")
            except Exception as e:
                print(f"Error loading students.json: {e}")
                self.students_map = {}
        else:
            self.students_map = {}

    def setup_ui(self):
        """Creates and layouts the main UI widgets."""
        # Top banner style
        header_frame = tk.Frame(self.root, bg=BG_PRIMARY)
        header_frame.pack(fill=tk.X, pady=(20, 10), padx=20)

        # Title Label
        title_lbl = tk.Label(
            header_frame, 
            text="Automated Attendance Register", 
            font=("Segoe UI", 26, "bold"), 
            fg=SKY_DARK, 
            bg=BG_PRIMARY
        )
        title_lbl.pack(anchor="w")

        # Subtitle and Status Indicator frame
        status_frame = tk.Frame(header_frame, bg=BG_PRIMARY)
        status_frame.pack(anchor="w", pady=(5, 0))

        # Canvas-based indicator dot
        self.status_canvas = tk.Canvas(status_frame, width=20, height=20, bg=BG_PRIMARY, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 10))
        self.status_dot = self.status_canvas.create_oval(2, 2, 18, 18, fill=COLOR_DANGER, outline="")

        # Status text label
        self.status_lbl = tk.Label(
            status_frame, 
            text="Status: Standby", 
            font=("Segoe UI", 12, "bold"), 
            fg=TEXT_MUTED, 
            bg=BG_PRIMARY
        )
        self.status_lbl.pack(side=tk.LEFT)

        # Main viewport frame (Card design)
        self.viewport_card = tk.Frame(
            self.root, 
            bg=BG_CARD, 
            highlightbackground=COLOR_BORDER, 
            highlightthickness=2,
            bd=0
        )
        self.viewport_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Video/Camera Feed Label inside the Card
        self.camera_lbl = tk.Label(
            self.viewport_card, 
            text="Camera Feed Inactive\n\nClick 'Start Detection' to begin face scanning.", 
            font=("Segoe UI", 14), 
            fg=TEXT_MUTED, 
            bg=BG_CARD,
            justify=tk.CENTER
        )
        self.camera_lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bottom Button Bar Frame
        control_frame = tk.Frame(self.root, bg=BG_PRIMARY)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=25)

        # Style configurations for custom premium buttons
        # Start/Stop Button (Left)
        self.btn_toggle = tk.Button(
            control_frame,
            text="Start Detection",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground=COLOR_SUCCESS_HOVER,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=10,
            relief="flat",
            command=self.toggle_detection
        )
        self.btn_toggle.pack(side=tk.LEFT)

        # Register Student Button (Center-Left)
        self.btn_register = tk.Button(
            control_frame,
            text="Register New Student",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_PURPLE,
            fg="white",
            activebackground=COLOR_PURPLE_HOVER,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=10,
            relief="flat",
            command=self.open_registration_dialog
        )
        self.btn_register.pack(side=tk.LEFT, padx=(15, 15))

        # Manage Directory Button (Center-Right)
        self.btn_manage = tk.Button(
            control_frame,
            text="Manage Students",
            font=("Segoe UI", 12, "bold"),
            bg="#4F46E5", # Indigo-600
            fg="white",
            activebackground="#4338CA",
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=10,
            relief="flat",
            command=self.open_manage_students
        )
        self.btn_manage.pack(side=tk.LEFT)

        # Show Today's Register Button (Right)
        self.btn_show_log = tk.Button(
            control_frame,
            text="Show Today's Register",
            font=("Segoe UI", 12, "bold"),
            bg=SKY_MED,
            fg="white",
            activebackground=SKY_DARK,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=10,
            relief="flat",
            command=self.show_today_register
        )
        self.btn_show_log.pack(side=tk.RIGHT)

        # Periodic UI queue checker
        self.check_queue_loop()

    def toggle_detection(self):
        """Starts or stops the main face detection video loop."""
        if not self.camera_active:
            # Start Camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not access system camera. Please check camera connections.")
                return

            self.camera_active = True
            
            # Clear active session logging memory and frame caches
            self.logged_this_session.clear()
            self.latest_frame = None
            self.recognition_results = []
            print("New scanning session initialized. Session logs reset.")

            self.btn_toggle.configure(text="Stop Detection", bg=COLOR_DANGER, activebackground=COLOR_DANGER_HOVER)
            self.status_canvas.itemconfig(self.status_dot, fill=COLOR_SUCCESS)
            self.status_lbl.configure(text="Status: Active Scanning")

            # Clean queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            # Start decoupled worker threads
            self.feed_thread = threading.Thread(target=self.camera_feed_worker, daemon=True)
            self.feed_thread.start()
            
            self.rec_thread = threading.Thread(target=self.recognition_worker, daemon=True)
            self.rec_thread.start()
        else:
            self.stop_camera()

    def stop_camera(self):
        """Shuts down the camera feed cleanly."""
        self.camera_active = False
        self.btn_toggle.configure(text="Start Detection", bg=COLOR_SUCCESS, activebackground=COLOR_SUCCESS_HOVER)
        self.status_canvas.itemconfig(self.status_dot, fill=COLOR_DANGER)
        self.status_lbl.configure(text="Status: Standby")

        # Wait for threads to terminate
        if self.feed_thread and self.feed_thread.is_alive():
            self.feed_thread.join(timeout=1.0)
        if self.rec_thread and self.rec_thread.is_alive():
            self.rec_thread.join(timeout=1.0)

        # Release Camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Reset UI label to default screen
        self.camera_lbl.configure(
            image="",
            text="Camera Feed Inactive\n\nClick 'Start Detection' to begin face scanning.",
            font=("Segoe UI", 14)
        )

    def camera_feed_worker(self):
        """Fast thread to read camera frames and render them immediately with current overlays (Runs at 30-60 FPS)."""
        print("Camera feed rendering thread started.")
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror frame for intuitive view
            frame = cv2.flip(frame, 1)
            
            # Cache the raw frame in thread-safe memory for the background recognition processor
            self.latest_frame = frame.copy()
            
            # Copy overlay variables locally to avoid frame-drawing race conditions
            current_results = list(self.recognition_results)
            face_in_frame = len(current_results) > 0

            # Draw coordinates calculated asynchronously by the recognizer thread
            for result in current_results:
                left, top, right, bottom = result["box"]
                color = result["color"]
                label_text = result["label"]

                # Render Bounding Box and label banner
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame, 
                    label_text, 
                    (left + 8, bottom - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.55, 
                    (255, 255, 255), 
                    2
                )

            # Resize frame to fit display viewport
            display_w, display_h = 760, 500
            frame_resized = cv2.resize(frame, (display_w, display_h))

            # Convert BGR frame to RGB for Tkinter display
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            data = {
                "frame": frame_rgb,
                "face_in_frame": face_in_frame
            }

            try:
                # Add to queue, overwrite if slow
                self.frame_queue.put(data, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(data, block=False)
                except Exception:
                    pass

            # Maximize frame-rate rendering speed (~60 FPS target display rate)
            time.sleep(0.015)

        print("Camera feed rendering thread stopped.")

    def recognition_worker(self):
        """Asynchronous worker thread to perform face recognition computations without locking video FPS."""
        print("Asynchronous recognition thread started.")
        while self.camera_active:
            if self.latest_frame is None:
                time.sleep(0.05)
                continue

            # Perform calculations on copy of latest cached camera frame
            frame = self.latest_frame.copy()
            
            # Downsize frame to 1/4 size for fast feature extraction
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Run deep-learning dlib detectors
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            new_results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale coordinates back by 4x to match source frame dimensions
                top_orig = top * 4
                right_orig = right * 4
                bottom_orig = bottom * 4
                left_orig = left * 4
                
                # Check for profile matches
                student_info = None
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if matches and len(face_distances) > 0:
                        best_match_idx = np.argmin(face_distances)
                        if matches[best_match_idx]:
                            student_info = self.known_face_metadata[best_match_idx]

                if student_info:
                    label_text = f"{student_info.get('name', 'Student')} (Roll: {student_info.get('roll_number', '-')})"
                    color = (46, 204, 113) # Green if recognized
                    self.log_attendance(student_info)
                else:
                    label_text = "Unknown Face"
                    color = (52, 152, 219) # Blue if unknown

                new_results.append({
                    "box": (left_orig, top_orig, right_orig, bottom_orig),
                    "label": label_text,
                    "color": color
                })

            # Update overlays globally
            self.recognition_results = new_results
            
            # Delay slightly to prevent 100% CPU thread starvation
            time.sleep(0.04)

        print("Asynchronous recognition thread stopped.")

    def log_attendance(self, student_info):
        """Records attendance of a recognized student to the CSV file exactly once per session."""
        reg_num = student_info.get("register_number")
        if not reg_num:
            return
        
        # Enforce "only enter a data of a student at a single time in a process" logic
        if reg_num in self.logged_this_session:
            return

        self.logged_this_session.add(reg_num)
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        csv_filename = os.path.join(self.attendance_dir, f"attendance_{date_str}.csv")
        
        file_exists = os.path.exists(csv_filename)
        
        try:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    # Extended attendance headers
                    writer.writerow(["Name", "Roll Number", "Register Number", "Class", "Year", "Date", "Timestamp"])
                writer.writerow([
                    student_info.get("name", "-"),
                    student_info.get("roll_number", "-"),
                    student_info.get("register_number", "-"),
                    student_info.get("class", "-"),
                    student_info.get("year", "-"),
                    date_str,
                    time_str
                ])
            print(f"Logged attendance: {student_info.get('name')} ({reg_num}) at {time_str}")
        except Exception as e:
            print(f"Failed to write attendance log for {student_info.get('name')}: {e}")

    def check_queue_loop(self):
        """Polls queue and updates UI with camera frames and status colors."""
        try:
            while True:
                data = self.frame_queue.get_nowait()
                frame_rgb = data["frame"]
                face_in_frame = data["face_in_frame"]

                # Convert to ImageTk format
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Render to viewport label
                self.camera_lbl.imgtk = imgtk
                self.camera_lbl.configure(image=imgtk)

                # Update Status Dot and text dynamically based on whether faces are currently being read
                if face_in_frame:
                    self.status_canvas.itemconfig(self.status_dot, fill=COLOR_SUCCESS)
                    self.status_lbl.configure(text="Status: Detecting & Reading Faces...", fg=COLOR_SUCCESS)
                else:
                    self.status_canvas.itemconfig(self.status_dot, fill=COLOR_DANGER)
                    self.status_lbl.configure(text="Status: Active Scanning (Standby)", fg=COLOR_DANGER)

        except queue.Empty:
            pass

        # Call periodically
        self.root.after(15, self.check_queue_loop)

    def show_today_register(self):
        """Displays today's attendance logs in a custom styled modern window."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_filename = os.path.join(self.attendance_dir, f"attendance_{date_str}.csv")

        # Create window
        log_win = tk.Toplevel(self.root)
        log_win.title(f"Today's Register - {date_str}")
        log_win.geometry("900x550")
        log_win.configure(bg=BG_PRIMARY)
        log_win.grab_set() # Focus lock
        log_win.transient(self.root)

        # Header Frame
        lbl_header = tk.Label(
            log_win, 
            text=f"Attendance Logs: {date_str}", 
            font=("Segoe UI", 16, "bold"), 
            fg=SKY_DARK, 
            bg=BG_PRIMARY
        )
        lbl_header.pack(pady=(20, 10), padx=20, anchor="w")

        # Main Table Container Frame
        tbl_frame = tk.Frame(log_win, bg=BG_CARD, bd=1, relief="solid")
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Scrollbars
        scrollbar = ttk.Scrollbar(tbl_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Style setup for treeview to look modern
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview", 
            background="#FFFFFF", 
            foreground=TEXT_MAIN, 
            fieldbackground="#FFFFFF",
            rowheight=30,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Treeview.Heading", 
            background=SKY_LIGHT, 
            foreground=SKY_DARK, 
            font=("Segoe UI", 11, "bold"),
            bd=0
        )
        style.map("Treeview", background=[("selected", SKY_MED)], foreground=[("selected", "white")])

        # Table/Treeview setup with extended columns
        cols = ("Name", "Roll No", "Reg No", "Class", "Year", "Timestamp")
        tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', yscrollcommand=scrollbar.set)
        
        tree.heading("Name", text="Name")
        tree.heading("Roll No", text="Roll No")
        tree.heading("Reg No", text="Reg No")
        tree.heading("Class", text="Class")
        tree.heading("Year", text="Year")
        tree.heading("Timestamp", text="Check-In Time")
        
        tree.column("Name", width=180, anchor="w")
        tree.column("Roll No", width=80, anchor="center")
        tree.column("Reg No", width=120, anchor="center")
        tree.column("Class", width=100, anchor="center")
        tree.column("Year", width=80, anchor="center")
        tree.column("Timestamp", width=120, anchor="center")
        
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=tree.yview)

        # Load logs into the table
        logs = []
        if os.path.exists(csv_filename):
            try:
                with open(csv_filename, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader, None)  # Skip header
                    for row in reader:
                        # Accommodate both old 2-column format and new 7-column formats
                        if len(row) >= 7:
                            logs.append((row[0], row[1], row[2], row[3], row[4], row[6]))
                        elif len(row) == 2:
                            logs.append((row[0], "-", "-", "-", "-", row[1]))
            except Exception as e:
                print(f"Error loading register: {e}")

        # Insert rows reversed (newest first)
        for log in reversed(logs):
            tree.insert("", tk.END, values=log)

        if not logs:
            tree.insert("", tk.END, values=("No attendees logged today", "-", "-", "-", "-", "-"))

        # Footer Actions
        bottom_frame = tk.Frame(log_win, bg=BG_PRIMARY)
        bottom_frame.pack(fill=tk.X, pady=(10, 20), padx=20)

        # Button to open file directly in default system editor
        btn_open_file = tk.Button(
            bottom_frame,
            text="Open CSV File",
            font=("Segoe UI", 10, "bold"),
            bg=SKY_MED,
            fg="white",
            activebackground=SKY_DARK,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            command=lambda: self.open_csv_file(csv_filename)
        )
        btn_open_file.pack(side=tk.LEFT)

        # Close dialog button
        btn_close = tk.Button(
            bottom_frame,
            text="Close Window",
            font=("Segoe UI", 10, "bold"),
            bg="#94A3B8",
            fg="white",
            activebackground="#64748B",
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            command=log_win.destroy
        )
        btn_close.pack(side=tk.RIGHT)

    def open_csv_file(self, file_path):
        """Triggers the operating system to open the attendance file in standard view."""
        if not os.path.exists(file_path):
            messagebox.showinfo("Register Empty", "No entries have been recorded today. File not created yet.")
            return
        
        try:
            if os.name == 'nt': # Windows
                os.startfile(file_path)
            elif os.name == 'posix': # Mac/Linux
                import subprocess
                subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', file_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

    def open_registration_dialog(self):
        """Checks for admin credentials prior to opening registration wizard."""
        self.check_admin_password(self.show_registration_wizard)

    def open_manage_students(self):
        """Checks credentials then opens the Student Management panel."""
        self.check_admin_password(self.show_management_panel)

    def check_admin_password(self, callback):
        """Asks for admin password in a styled dialog. If correct, executes callback."""
        pw_win = tk.Toplevel(self.root)
        pw_win.title("Admin Verification")
        pw_win.geometry("380x200")
        pw_win.configure(bg=BG_PRIMARY)
        pw_win.grab_set()
        pw_win.transient(self.root)
        
        lbl_title = tk.Label(
            pw_win, 
            text="Administrator Password Required", 
            font=("Segoe UI", 12, "bold"), 
            fg=SKY_DARK, 
            bg=BG_PRIMARY
        )
        lbl_title.pack(pady=(25, 5))
        
        lbl_prompt = tk.Label(
            pw_win, 
            text="Enter access password (default: admin123):", 
            font=("Segoe UI", 9), 
            fg=TEXT_MUTED, 
            bg=BG_PRIMARY
        )
        lbl_prompt.pack()
        
        pw_var = tk.StringVar()
        pw_entry = tk.Entry(
            pw_win, 
            textvariable=pw_var, 
            show="*", 
            font=("Segoe UI", 12), 
            justify="center", 
            relief="solid", 
            bd=1
        )
        pw_entry.pack(pady=12, padx=40, fill=tk.X)
        pw_entry.focus()
        
        def verify():
            if pw_var.get() == "admin123":
                pw_win.destroy()
                callback()
            else:
                messagebox.showerror("Access Denied", "Incorrect administrator password.")
                pw_entry.delete(0, tk.END)
                
        btn_ok = tk.Button(
            pw_win,
            text="Verify Access",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground=COLOR_SUCCESS_HOVER,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=20,
            pady=6,
            relief="flat",
            command=verify
        )
        btn_ok.pack(pady=5)
        
        pw_entry.bind("<Return>", lambda event: verify())

    def show_registration_wizard(self):
        """Opens a form window to register a student and capture face snapshots."""
        # Check if main camera detection is running. If so, request stop.
        was_camera_active = self.camera_active
        if self.camera_active:
            self.stop_camera()
            messagebox.showinfo("Camera Suspended", "Main scanning suspended temporarily for registration camera access.")

        # Create Dialog
        reg_win = tk.Toplevel(self.root)
        reg_win.title("Register New Student")
        reg_win.geometry("520x720")
        reg_win.configure(bg=BG_PRIMARY)
        reg_win.grab_set()
        reg_win.transient(self.root)

        lbl_header = tk.Label(
            reg_win, 
            text="Student Registration Wizard", 
            font=("Segoe UI", 18, "bold"), 
            fg=SKY_DARK, 
            bg=BG_PRIMARY
        )
        lbl_header.pack(pady=(20, 5), padx=20, anchor="w")

        lbl_instr = tk.Label(
            reg_win,
            text="Fill in the student details below, look directly at the camera,\nand press 'Begin Capture'. We will scan 10 snapshots automatically.",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_PRIMARY,
            justify=tk.LEFT
        )
        lbl_instr.pack(padx=20, pady=(0, 10), anchor="w")

        # Form Entries Frame
        form_frame = tk.Frame(reg_win, bg=BG_PRIMARY)
        form_frame.pack(fill=tk.X, padx=20, pady=5)

        fields = [
            ("Full Name:", "name"),
            ("Roll Number:", "roll"),
            ("Register Number:", "reg"),
            ("Class / Section:", "cls"),
            ("Year:", "yr")
        ]
        
        vars_dict = {
            "name": tk.StringVar(),
            "roll": tk.StringVar(),
            "reg": tk.StringVar(),
            "cls": tk.StringVar(),
            "yr": tk.StringVar()
        }

        # Render form grid
        for i, (label_text, field_key) in enumerate(fields):
            lbl = tk.Label(
                form_frame, 
                text=label_text, 
                font=("Segoe UI", 10, "bold"), 
                fg=TEXT_MAIN, 
                bg=BG_PRIMARY,
                anchor="e",
                width=18
            )
            lbl.grid(row=i, column=0, padx=(0, 10), pady=6, sticky="e")
            
            entry = tk.Entry(
                form_frame, 
                textvariable=vars_dict[field_key], 
                font=("Segoe UI", 10), 
                bg="white", 
                relief="solid", 
                bd=1,
                width=30
            )
            entry.grid(row=i, column=1, pady=6, sticky="w")
            if i == 0:
                entry.focus()

        # Visual camera preview box for snapshots
        cam_card = tk.Frame(reg_win, bg=BG_CARD, bd=1, relief="solid")
        cam_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        preview_lbl = tk.Label(
            cam_card, 
            text="Camera Ready\nEnter details above and tap 'Begin Capture'", 
            font=("Segoe UI", 11), 
            fg=TEXT_MUTED, 
            bg=BG_CARD,
            justify=tk.CENTER
        )
        preview_lbl.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        prog_var = tk.DoubleVar()
        progress = ttk.Progressbar(reg_win, variable=prog_var, maximum=10)
        progress.pack(fill=tk.X, padx=20, pady=(5, 5))

        lbl_progress_status = tk.Label(reg_win, text="", font=("Segoe UI", 10), fg=TEXT_MUTED, bg=BG_PRIMARY)
        lbl_progress_status.pack(pady=(0, 5))

        # Capture Control Functions
        def start_capture_process():
            # Get values
            name = vars_dict["name"].get().strip()
            roll = vars_dict["roll"].get().strip()
            reg = vars_dict["reg"].get().strip()
            cls = vars_dict["cls"].get().strip()
            yr = vars_dict["yr"].get().strip()
            
            if not all([name, roll, reg, cls, yr]):
                messagebox.showerror("Error", "All student fields must be filled.")
                return
            
            # Check register number uniqueness
            if reg in self.students_map:
                messagebox.showerror("Error", f"Student with Register Number '{reg}' is already registered.")
                return

            # Subfolder in dataset is named by unique Register Number
            safe_reg = reg.replace(" ", "_").replace("/", "-")
            person_dir = os.path.join(self.dataset_dir, safe_reg)
            os.makedirs(person_dir, exist_ok=True)

            # Start camera for capture
            reg_cap = cv2.VideoCapture(0)
            if not reg_cap.isOpened():
                messagebox.showerror("Camera Error", "Could not start camera feed for registration.")
                return

            btn_start_cap.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.DISABLED)
            # Disable entry inputs
            for child in form_frame.winfo_children():
                if isinstance(child, tk.Entry):
                    child.config(state=tk.DISABLED)

            # Snapshot capture loop run inside a safe local thread to avoid UI locking
            def capture_thread():
                count = 0
                student_encodings = []
                
                # Take 10 photos automatically
                while count < 10:
                    ret, img_frame = reg_cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                    img_frame = cv2.flip(img_frame, 1)
                    rgb_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces in live stream
                    face_locations = face_recognition.face_locations(rgb_frame)

                    # Create a display copy with drawings
                    display_frame = img_frame.copy()
                    
                    if len(face_locations) > 0:
                        # Process face
                        top, right, bottom, left = face_locations[0]
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (139, 92, 246), 2)
                        
                        # Extract 128-D face encoding
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        if len(face_encodings) > 0:
                            new_encoding = face_encodings[0]
                            
                            # Check similarity with already captured poses in this session
                            similar_count = 0
                            if len(student_encodings) > 0:
                                # Compare new encoding with already captured encodings
                                # face_recognition.face_distance returns distance (lower = more similar)
                                # Euclidean distance less than 0.28 means it's the exact same pose/angle
                                distances = face_recognition.face_distance([np.array(enc) for enc in student_encodings], new_encoding)
                                similar_count = sum(1 for d in distances if d < 0.28)
                            
                            # Enforce: "donot capture same faces more than two times"
                            if similar_count >= 2:
                                # Display warning message on the live stream and skip capture
                                cv2.putText(
                                    display_frame, 
                                    "Same angle! Rotate head slightly", 
                                    (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, 
                                    (0, 165, 255), # Orange warning
                                    2
                                )
                                self.root.after(0, lambda: lbl_progress_status.config(
                                    text="Angle already captured. Please rotate/tilt head slightly..."
                                ))
                            else:
                                # Angle is sufficiently unique. Save it!
                                face_crop = img_frame[top:bottom, left:right]
                                if face_crop.size > 0:
                                    face_crop_resized = cv2.resize(face_crop, (200, 200))
                                    face_file = os.path.join(person_dir, f"{count + 1}.jpg")
                                    cv2.imwrite(face_file, face_crop_resized)
                                
                                # Store encoding list
                                student_encodings.append(new_encoding.tolist())
                                
                                count += 1
                                self.root.after(0, lambda c=count: prog_var.set(c))
                                self.root.after(0, lambda c=count: lbl_progress_status.config(
                                    text=f"Photo {c}/10 saved! Adjust angle slightly..."
                                ))
                                
                                # Visual notification on frame
                                cv2.putText(
                                    display_frame, 
                                    f"Saved {count}/10", 
                                    (30, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, 
                                    (46, 204, 113), 
                                    2
                                )
                                
                                # Render visual preview of success
                                display_frame_resized = cv2.resize(display_frame, (480, 320))
                                frame_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(frame_rgb)
                                imgtk = ImageTk.PhotoImage(image=img)
                                
                                self.root.after(0, lambda p=imgtk: preview_lbl.configure(image=p, text=""))
                                preview_lbl.image = imgtk
                                
                                # Wait 600ms before taking next snapshot to allow head movements/expression changes
                                time.sleep(0.6)
                                continue
                    else:
                        cv2.putText(
                            display_frame, 
                            "Position face in center", 
                            (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (231, 76, 60), 
                            2
                        )

                    # Update preview screen for real-time tracking
                    display_frame_resized = cv2.resize(display_frame, (480, 320))
                    frame_rgb = cv2.cvtColor(display_frame_resized, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    def update_preview(photo):
                        preview_lbl.configure(image=photo, text="")
                        preview_lbl.image = photo
                      
                    self.root.after(0, update_preview, imgtk)
                    time.sleep(0.033)

                # Capture finished
                reg_cap.release()
                
                # Save student details to students_map & students.json
                self.students_map[reg] = {
                    "name": name,
                    "roll_number": roll,
                    "register_number": reg,
                    "class": cls,
                    "year": yr,
                    "encodings": student_encodings
                }
                
                try:
                    with open(self.students_path, 'w') as f:
                        json.dump(self.students_map, f, indent=4)
                except Exception as e:
                    print(f"Error saving students.json: {e}")
                
                # Reload metadata encodings cache
                self.root.after(0, self.load_students_and_encodings)
                
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Student '{name}' registered successfully with 10 face angles!"))
                self.root.after(0, reg_win.destroy)
                if was_camera_active:
                    self.root.after(100, self.toggle_detection)  # Automatically restart main camera scanning

            # Start capture thread
            threading.Thread(target=capture_thread, daemon=True).start()

        # Action Buttons frame for registration wizard
        reg_btn_frame = tk.Frame(reg_win, bg=BG_PRIMARY)
        reg_btn_frame.pack(fill=tk.X, pady=15, padx=20)

        btn_start_cap = tk.Button(
            reg_btn_frame,
            text="Begin Capture",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground=COLOR_SUCCESS_HOVER,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            command=start_capture_process
        )
        btn_start_cap.pack(side=tk.LEFT)

        def close_wizard():
            reg_win.destroy()
            if was_camera_active:
                self.toggle_detection()  # Restart camera feed

        btn_cancel = tk.Button(
            reg_btn_frame,
            text="Cancel Wizard",
            font=("Segoe UI", 11, "bold"),
            bg="#94A3B8",
            fg="white",
            activebackground="#64748B",
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            command=close_wizard
        )
        btn_cancel.pack(side=tk.RIGHT)

        # Handle window close behavior manually to restore camera
        reg_win.protocol("WM_DELETE_WINDOW", close_wizard)

    def show_management_panel(self):
        """Opens a split window showing the student directory list and details panel."""
        # Suspend camera if running
        was_camera_active = self.camera_active
        if self.camera_active:
            self.stop_camera()
            messagebox.showinfo("Camera Suspended", "Main scanning suspended temporarily for database management access.")

        m_win = tk.Toplevel(self.root)
        m_win.title("Student Database Directory")
        m_win.geometry("920x580")
        m_win.configure(bg=BG_PRIMARY)
        m_win.grab_set()
        m_win.transient(self.root)

        # Main Split Pane Layout
        # Left Pane (List View)
        left_frame = tk.Frame(m_win, bg=BG_PRIMARY)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 10), pady=20)

        lbl_left_hdr = tk.Label(
            left_frame, 
            text="Registered Students", 
            font=("Segoe UI", 14, "bold"), 
            fg=SKY_DARK, 
            bg=BG_PRIMARY
        )
        lbl_left_hdr.pack(anchor="w", pady=(0, 10))

        tbl_frame = tk.Frame(left_frame, bg=BG_CARD, bd=1, relief="solid")
        tbl_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tbl_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Table Styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview", 
            background="#FFFFFF", 
            foreground=TEXT_MAIN, 
            fieldbackground="#FFFFFF",
            rowheight=30,
            font=("Segoe UI", 9)
        )
        style.configure(
            "Treeview.Heading", 
            background=SKY_LIGHT, 
            foreground=SKY_DARK, 
            font=("Segoe UI", 10, "bold"),
            bd=0
        )
        style.map("Treeview", background=[("selected", SKY_MED)], foreground=[("selected", "white")])

        cols = ("Name", "Roll No", "Reg No")
        tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', yscrollcommand=scrollbar.set)
        tree.heading("Name", text="Name")
        tree.heading("Roll No", text="Roll No")
        tree.heading("Reg No", text="Reg No")
        tree.column("Name", width=180, anchor="w")
        tree.column("Roll No", width=80, anchor="center")
        tree.column("Reg No", width=120, anchor="center")
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=tree.yview)

        # Action Buttons frame below Treeview in the left pane
        action_frame = tk.Frame(left_frame, bg=BG_PRIMARY)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        # Delete Button (placed prominently below the student table list)
        btn_delete = tk.Button(
            action_frame,
            text="Delete Selected Student",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_DANGER,
            fg="white",
            activebackground=COLOR_DANGER_HOVER,
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            state=tk.DISABLED
        )
        btn_delete.pack(side=tk.LEFT)

        # Right Pane (Details View)
        right_frame = tk.Frame(m_win, bg=BG_CARD, highlightbackground=COLOR_BORDER, highlightthickness=2, bd=0, width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 20), pady=20)
        right_frame.pack_propagate(False)

        lbl_right_hdr = tk.Label(
            right_frame, 
            text="Student Profile Detail", 
            font=("Segoe UI", 14, "bold"), 
            fg=SKY_DARK, 
            bg=BG_CARD
        )
        lbl_right_hdr.pack(pady=15, padx=20, anchor="w")

        # Photo Preview Canvas
        photo_lbl = tk.Label(
            right_frame,
            text="Select a student\nto view profile photo",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_PRIMARY,
            width=20,
            height=8,
            relief="solid",
            bd=1
        )
        photo_lbl.pack(pady=(0, 15))

        # Metadata Display Area
        details_frame = tk.Frame(right_frame, bg=BG_CARD)
        details_frame.pack(fill=tk.X, padx=20, pady=10)

        detail_vars = {
            "name": tk.StringVar(value="-"),
            "roll": tk.StringVar(value="-"),
            "reg": tk.StringVar(value="-"),
            "cls": tk.StringVar(value="-"),
            "yr": tk.StringVar(value="-")
        }

        labels = [
            ("Name:", "name"),
            ("Roll Number:", "roll"),
            ("Register Number:", "reg"),
            ("Class / Section:", "cls"),
            ("Year:", "yr")
        ]

        for i, (lbl_txt, var_key) in enumerate(labels):
            lbl = tk.Label(details_frame, text=lbl_txt, font=("Segoe UI", 10, "bold"), fg=TEXT_MUTED, bg=BG_CARD, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=4)
            
            val_lbl = tk.Label(details_frame, textvariable=detail_vars[var_key], font=("Segoe UI", 10), fg=TEXT_MAIN, bg=BG_CARD, anchor="w")
            val_lbl.grid(row=i, column=1, sticky="w", padx=10, pady=4)

        # Bottom Frame Close Button Setup
        def close_management():
            m_win.destroy()
            if was_camera_active:
                self.toggle_detection()

        m_win.protocol("WM_DELETE_WINDOW", close_management)

        # Close Button
        btn_close_m = tk.Button(
            right_frame,
            text="Close Directory",
            font=("Segoe UI", 11, "bold"),
            bg="#94A3B8",
            fg="white",
            activebackground="#64748B",
            activeforeground="white",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            relief="flat",
            command=close_management
        )
        btn_close_m.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 20))

        # Refresh directory list helper
        def refresh_list():
            # Force load latest details from JSON file to keep directory synchronized
            self.load_students_and_encodings()
            
            # Clear list
            for item in tree.get_children():
                tree.delete(item)
            
            # Load database profiles safely using .get() to prevent crashes on legacy profiles
            for reg_no, s in self.students_map.items():
                tree.insert("", tk.END, values=(
                    s.get("name", "-"), 
                    s.get("roll_number", "-"), 
                    s.get("register_number", "-")
                ))
            
            # Clear detail cards
            for k in detail_vars:
                detail_vars[k].set("-")
            photo_lbl.configure(image="", text="Select a student\nto view profile photo")
            btn_delete.config(state=tk.DISABLED)

        # Handle Treeview row selection change
        def on_select(event):
            selected = tree.selection()
            if not selected:
                return
            
            item_id = selected[0]
            values = tree.item(item_id)['values']
            if not values or len(values) < 3:
                return
            
            reg_no = str(values[2])
            s = self.students_map.get(reg_no)
            if not s:
                return
            
            # Update labels safely using .get()
            detail_vars["name"].set(s.get("name", "-"))
            detail_vars["roll"].set(s.get("roll_number", "-"))
            detail_vars["reg"].set(s.get("register_number", "-"))
            detail_vars["cls"].set(s.get("class", "-"))
            detail_vars["yr"].set(s.get("year", "-"))
            
            # Update photo preview
            self.update_student_photo_preview(reg_no, photo_lbl)
            
            # Enable Delete button
            btn_delete.config(state=tk.NORMAL, command=lambda: confirm_delete(reg_no))

        tree.bind("<<TreeviewSelect>>", on_select)

        # Delete student actions
        def confirm_delete(reg_no):
            s = self.students_map.get(reg_no)
            if not s:
                return
            
            if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the student profile for '{s.get('name', 'selected student')}'?\nThis will remove their face models and references forever."):
                # Delete dataset directory
                safe_reg = reg_no.replace(" ", "_").replace("/", "-")
                folder_path = os.path.join(self.dataset_dir, safe_reg)
                if os.path.exists(folder_path):
                    try:
                        shutil.rmtree(folder_path)
                    except Exception as e:
                        print(f"Error removing dataset folder: {e}")

                # Remove from map & save database
                if reg_no in self.students_map:
                    del self.students_map[reg_no]
                
                try:
                    with open(self.students_path, 'w') as f:
                        json.dump(self.students_map, f, indent=4)
                except Exception as e:
                    print(f"Error saving students.json: {e}")
                
                # Reload in-memory encodings cache
                self.load_students_and_encodings()
                
                # Refresh panel
                refresh_list()
                messagebox.showinfo("Success", f"Student profile deleted successfully.")

        # Initial directory population
        refresh_list()

    def update_student_photo_preview(self, reg_no, photo_lbl):
        """Loads and resizes student reference photo 1.jpg for preview."""
        # Clean folder naming
        safe_reg = reg_no.replace(" ", "_").replace("/", "-")
        photo_path = os.path.join(self.dataset_dir, safe_reg, "1.jpg")
        
        if os.path.exists(photo_path):
            try:
                img = Image.open(photo_path)
                img_resized = img.resize((150, 150), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(img_resized)
                photo_lbl.configure(image=imgtk, text="")
                photo_lbl.image = imgtk
            except Exception as e:
                print(f"Error loading preview: {e}")
                photo_lbl.configure(image="", text="Photo Load Error\n(Corrupted)")
        else:
            photo_lbl.configure(image="", text="No Profile Photo\nAvailable")

    def on_close(self):
        """Cleans resources and closes Tkinter GUI loop."""
        if messagebox.askyesno("Quit Application", "Are you sure you want to exit?"):
            self.camera_active = False
            if self.feed_thread and self.feed_thread.is_alive():
                self.feed_thread.join(timeout=1.0)
            if self.rec_thread and self.rec_thread.is_alive():
                self.rec_thread.join(timeout=1.0)
            if self.cap:
                self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
