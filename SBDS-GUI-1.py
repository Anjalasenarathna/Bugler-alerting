import tkinter as tk
from tkinter import filedialog, ttk, Scale, HORIZONTAL
import cv2
import os
import numpy as np
from pathlib import Path
import pytesseract
from ultralytics import YOLO
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import defaultdict
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import subprocess
import json

class SecurityDetectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Security Detection System")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f0f0f0")
        
        self.preview_width = 640
        self.preview_height = 480
    
        # Variables
        self.video_path = None
        self.frame_skip = 60
        self.roi_points = []
        self.drawing = False
        self.cap = None
        self.processing_thread = None
        self.stop_processing = False
        
        # Feature flags
        self.enable_face_detection = tk.BooleanVar(value=True)
        self.enable_face_recognition = tk.BooleanVar(value=True)
        self.enable_vehicle_detection = tk.BooleanVar(value=True)
        self.enable_plate_detection = tk.BooleanVar(value=True)
        self.enable_unauthorized_access = tk.BooleanVar(value=True)
        self.enable_weapon_detection = tk.BooleanVar(value=True)
        self.enable_save_logs = tk.BooleanVar(value=True)
        self.enable_save_captures = tk.BooleanVar(value=True)
        self.enable_email_alerts = tk.BooleanVar(value=True)
        self.enable_frame_skipping = tk.BooleanVar(value=False)
        
        self.car_count = 0
        self.is_vehicle_sus = False
        self.reported_suspicious_vehicles = set()
        self.vehicle_tracker = defaultdict(list)
        self.last_email_times = {}
        self.email_cooldown = 60  # seconds between emails of the same category

        # Add to initialize_components method:
        self.last_email_times = {
            "Face": 0,
            "Vehicle": 0,
            "Weapon": 0,
            "Plate": 0,
            "Unauthorized": 0
        }
        self.email_cooldown = 10  # 1 minute cooldown in seconds
        self.car_count=0
        
        self.is_human_sus = False  # Or whatever default value you want
        self.is_vehicle_sus = False  # Or whatever default value you want
         
        # Set up the main layout
        self.setup_ui()
        
        # Initialize models and utilities
        self.initialize_models()
        
        # Set up directories
        self.setup_directories()
        
        # Initialize other components
        self.initialize_components()
        
        # Load settings from file
        self.load_settings_from_file()    
        
    def setup_ui(self):
        # Left side - Preview and buttons
        left_frame = tk.Frame(self.root, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Top buttons
        btn_frame = tk.Frame(left_frame, bg="#f0f0f0")
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.select_btn = tk.Button(btn_frame, text="Select Video File", command=self.select_video, bg="#c0c0c0", width=15, height=2)
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        self.start_btn = tk.Button(btn_frame, text="Start Detection", command=self.start_detection, bg="#c0c0c0", width=15, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        # Add Stop Detection button
        self.stop_btn = tk.Button(btn_frame, text="Stop Detection", command=self.stop_detection, bg="#c0c0c0", width=15, height=2)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.settings_btn = tk.Button(btn_frame, text="Settings", command=self.open_settings, bg="#c0c0c0", width=15, height=2)
        self.settings_btn.pack(side=tk.RIGHT, padx=10)
        
         # Video preview with fixed dimensions
        self.preview_frame = tk.Frame(left_frame, bg="black", width=self.preview_width, height=self.preview_height)
        self.preview_frame.pack(side=tk.TOP, pady=10, fill=tk.BOTH, expand=True)
        self.preview_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its children
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls and checkboxes
        right_frame = tk.Frame(self.root, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        # Detection options
        options_frame = tk.Frame(right_frame, bg="#f0f0f0")
        options_frame.pack(side=tk.TOP, pady=20, fill=tk.X)
        
        # First row of checkboxes
        row1 = tk.Frame(options_frame, bg="#f0f0f0")
        row1.pack(fill=tk.X, pady=5)
        
        face_detection_cb = tk.Checkbutton(row1, text="Face Detection", variable=self.enable_face_detection, bg="#f0f0f0")
        face_detection_cb.pack(side=tk.LEFT, padx=(0, 50))
        
        face_recognition_cb = tk.Checkbutton(row1, text="Face Recognition", variable=self.enable_face_recognition, bg="#f0f0f0")
        face_recognition_cb.pack(side=tk.LEFT)
        
        # Second row of checkboxes
        row2 = tk.Frame(options_frame, bg="#f0f0f0")
        row2.pack(fill=tk.X, pady=5)
        
        vehicle_detection_cb = tk.Checkbutton(row2, text="Vehicle Detection", variable=self.enable_vehicle_detection, bg="#f0f0f0")
        vehicle_detection_cb.pack(side=tk.LEFT, padx=(0, 50))
        
        plate_detection_cb = tk.Checkbutton(row2, text="Number Plate Detection", variable=self.enable_plate_detection, bg="#f0f0f0")
        plate_detection_cb.pack(side=tk.LEFT)
        
        # Third row - Unauthorized access and region selection
        row3 = tk.Frame(options_frame, bg="#f0f0f0")
        row3.pack(fill=tk.X, pady=5)
        
        unauthorized_cb = tk.Checkbutton(row3, text="Unauthorized Access", variable=self.enable_unauthorized_access, bg="#f0f0f0")
        unauthorized_cb.pack(side=tk.LEFT)
        
        region_frame = tk.Frame(row3, bg="#f0f0f0")
        region_frame.pack(side=tk.RIGHT)
        
        select_region_btn = tk.Button(region_frame, text="Select Region", command=self.select_region, bg="#c0c0c0")
        select_region_btn.pack(side=tk.LEFT, padx=5)
        
        clear_region_btn = tk.Button(region_frame, text="Clear Region", command=self.clear_region, bg="#c0c0c0")
        clear_region_btn.pack(side=tk.LEFT, padx=5)
        
        # Fourth row - Weapon detection
        row4 = tk.Frame(options_frame, bg="#f0f0f0")
        row4.pack(fill=tk.X, pady=5)
        
        weapon_detection_cb = tk.Checkbutton(row4, text="Weapon Detection", variable=self.enable_weapon_detection, bg="#f0f0f0")
        weapon_detection_cb.pack(side=tk.LEFT)
        
        # Fifth row - Saving options
        row5 = tk.Frame(options_frame, bg="#f0f0f0")
        row5.pack(fill=tk.X, pady=5)
        
        save_logs_cb = tk.Checkbutton(row5, text="Save Logs", variable=self.enable_save_logs, bg="#f0f0f0")
        save_logs_cb.pack(side=tk.LEFT, padx=(0, 10))
        
        save_captures_cb = tk.Checkbutton(row5, text="Save Captures", variable=self.enable_save_captures, bg="#f0f0f0")
        save_captures_cb.pack(side=tk.LEFT, padx=(0, 10))
        
        email_alerts_cb = tk.Checkbutton(row5, text="Send Email Alerts", variable=self.enable_email_alerts, bg="#f0f0f0")
        email_alerts_cb.pack(side=tk.LEFT)
        
        # Frame skipping option
        row6 = tk.Frame(options_frame, bg="#f0f0f0")
        row6.pack(fill=tk.X, pady=5)
        
        frame_skipping_cb = tk.Checkbutton(row6, text="Frame Skipping", variable=self.enable_frame_skipping, bg="#f0f0f0")
        frame_skipping_cb.pack(side=tk.LEFT, anchor=tk.W)
        
        # Frame skipping slider
        self.skip_slider = Scale(options_frame, from_=1, to=120, orient=HORIZONTAL, length=400)
        self.skip_slider.set(self.frame_skip)
        self.skip_slider.pack(pady=5)
        
        # Bottom buttons for logs and captures
        bottom_frame = tk.Frame(right_frame, bg="#f0f0f0")
        bottom_frame.pack(side=tk.BOTTOM, pady=20, fill=tk.X)
        
        view_log_btn = tk.Button(bottom_frame, text="View Log File", command=self.view_logs, bg="#c0c0c0", width=15, height=2)
        view_log_btn.pack(side=tk.LEFT, padx=10)
        
        view_captures_btn = tk.Button(bottom_frame, text="View Captured Images", command=self.view_captures, bg="#c0c0c0", width=20, height=2)
        view_captures_btn.pack(side=tk.RIGHT, padx=10)
    
    def initialize_models(self):
        # Define model paths with instance variables so they can be changed
        self.weapon_model_path = Path("weapon.pt")
        self.vehicle_model_path = Path("vehicle.pt")
        self.plate_model_path = Path("plate.pt")
        self.tesseract_path = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        
        # Load YOLO models
        self.weapon_model = None
        self.vehicle_model = None
        self.plate_model = None
        
        try:
            if self.weapon_model_path.exists():
                self.weapon_model = YOLO(str(self.weapon_model_path))
                print("Weapon Model loaded")
        except Exception as e:
            print(f"Error loading weapon model: {e}")
            
        try:
            if self.vehicle_model_path.exists():
                self.vehicle_model = YOLO(str(self.vehicle_model_path))
                print("Vehicle Model loaded")
        except Exception as e:
            print(f"Error loading vehicle model: {e}")
            
        try:
            if self.plate_model_path.exists():
                self.plate_model = YOLO(str(self.plate_model_path))
                print("Plate Model loaded")
        except Exception as e:
            print(f"Error loading plate model: {e}")
        
        # Set Tesseract path
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            print("Tesseract loaded")
        except Exception as e:
            print(f"Error loading Tesseract: {e}")
        
        # Initialize MTCNN for face detection
        try:
            self.mtcnn = MTCNN()
            print("MTCNN loaded")
        except Exception as e:
            print(f"Error loading MTCNN: {e}")
        
        # Load pre-trained FaceNet model for face recognition
        try:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            print("FaceNet loaded")
        except Exception as e:
            print(f"Error loading FaceNet: {e}")
    
    def setup_directories(self):
        # Directories for saving detections
        self.OUTPUT_DIR = Path("output_images")
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        
        self.SAVE_DIRS = {
            "weapons": self.OUTPUT_DIR / "weapons",
            "plates": self.OUTPUT_DIR / "number_plates",
            "vehicles": self.OUTPUT_DIR / "vehicles",
            "faces": self.OUTPUT_DIR / "faces",
            "suspicious": self.OUTPUT_DIR / "suspicious"  # Added for suspicious detections
        }
        
        for dir_path in self.SAVE_DIRS.values():
            dir_path.mkdir(exist_ok=True)
        
        # Log file paths
        self.LOG_TEXT_FILE = self.OUTPUT_DIR / "detections_log.txt"
        self.LOG_EXCEL_FILE = self.OUTPUT_DIR / "detections_log.xlsx"
    
    def save_settings_to_file(self):
        """Save current settings to a JSON file"""
        settings = {
            "tesseract_path": self.tesseract_path,
            "weapon_model_path": str(self.weapon_model_path),
            "vehicle_model_path": str(self.vehicle_model_path),
            "plate_model_path": str(self.plate_model_path),
            "similarity_threshold": self.SIMILARITY_THRESHOLD,
            "face_count": self.face_count,
            "plate_count": self.plate_count,
            "suspicious_vehicle_threshold": self.suspicious_vehicle_threshold,
            "suspicious_face_threshold": self.suspicious_face_threshold,
            "frame_skip": self.frame_skip,
            "email_address": "detectionthreat932@gmail.com",  # Add more email settings if needed
            "email_cooldown": self.email_cooldown,
            # Feature flags
            "enable_face_detection": self.enable_face_detection.get(),
            "enable_face_recognition": self.enable_face_recognition.get(),
            "enable_vehicle_detection": self.enable_vehicle_detection.get(),
            "enable_plate_detection": self.enable_plate_detection.get(),
            "enable_unauthorized_access": self.enable_unauthorized_access.get(),
            "enable_weapon_detection": self.enable_weapon_detection.get(),
            "enable_save_logs": self.enable_save_logs.get(),
            "enable_save_captures": self.enable_save_captures.get(),
            "enable_email_alerts": self.enable_email_alerts.get(),
            "enable_frame_skipping": self.enable_frame_skipping.get()
        }
        
        try:
            with open('security_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
            print("[INFO] Settings saved to security_settings.json")
        except Exception as e:
            print(f"[ERROR] Failed to save settings: {e}")

    def load_settings_from_file(self):
        """Load settings from a JSON file if it exists"""
        try:
            if not os.path.exists('security_settings.json'):
                print("[INFO] Settings file not found, using defaults")
                return
                
            with open('security_settings.json', 'r') as f:
                settings = json.load(f)
            
            # Load paths
            self.tesseract_path = settings.get('tesseract_path', self.tesseract_path)
            self.weapon_model_path = Path(settings.get('weapon_model_path', str(self.weapon_model_path)))
            self.vehicle_model_path = Path(settings.get('vehicle_model_path', str(self.vehicle_model_path)))
            self.plate_model_path = Path(settings.get('plate_model_path', str(self.plate_model_path)))
            
            # Load threshold values
            self.SIMILARITY_THRESHOLD = settings.get('similarity_threshold', self.SIMILARITY_THRESHOLD)
            self.face_count = settings.get('face_count', self.face_count)
            self.plate_count = settings.get('plate_count', self.plate_count)
            self.suspicious_vehicle_threshold = settings.get('suspicious_vehicle_threshold', self.suspicious_vehicle_threshold)
            self.suspicious_face_threshold = settings.get('suspicious_face_threshold', self.suspicious_face_threshold)
            self.frame_skip = settings.get('frame_skip', self.frame_skip)
            self.email_cooldown = settings.get('email_cooldown', self.email_cooldown)
            
            # Load feature flags
            self.enable_face_detection.set(settings.get('enable_face_detection', True))
            self.enable_face_recognition.set(settings.get('enable_face_recognition', True))
            self.enable_vehicle_detection.set(settings.get('enable_vehicle_detection', True))
            self.enable_plate_detection.set(settings.get('enable_plate_detection', True)) 
            self.enable_unauthorized_access.set(settings.get('enable_unauthorized_access', True))
            self.enable_weapon_detection.set(settings.get('enable_weapon_detection', True))
            self.enable_save_logs.set(settings.get('enable_save_logs', True))
            self.enable_save_captures.set(settings.get('enable_save_captures', True))
            self.enable_email_alerts.set(settings.get('enable_email_alerts', True))
            self.enable_frame_skipping.set(settings.get('enable_frame_skipping', False))
            
            # Update tesseract path
            try:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                print("Tesseract path loaded from settings")
            except Exception as e:
                print(f"Error updating Tesseract path: {e}")
            
            # Load models if paths exist
            try:
                if self.weapon_model_path.exists():
                    self.weapon_model = YOLO(str(self.weapon_model_path))
                    print("Weapon model loaded from settings")
            except Exception as e:
                print(f"Error loading weapon model: {e}")
            
            try:
                if self.vehicle_model_path.exists():
                    self.vehicle_model = YOLO(str(self.vehicle_model_path))
                    print("Vehicle model loaded from settings")
            except Exception as e:
                print(f"Error loading vehicle model: {e}")
            
            try:
                if self.plate_model_path.exists():
                    self.plate_model = YOLO(str(self.plate_model_path))
                    print("Plate model loaded from settings")
            except Exception as e:
                print(f"Error loading plate model: {e}")
            
            print("[INFO] Settings loaded from security_settings.json")
        except Exception as e:
            print(f"[ERROR] Failed to load settings: {e}")
        
    def initialize_components(self):
        # Dictionary to store face embeddings and their counts
        self.face_embeddings = {}
        self.face_counter = defaultdict(int)
        self.face_count = 60
        
        # Dictionary to track detected plates and their counts
        self.plate_tracker = defaultdict(int)
        self.plate_count = 60
        
        # Add to initialize_components method:
        # Dictionary to track detected vehicles and their frames
        self.vehicle_tracker = defaultdict(list)
        self.suspicious_vehicle_threshold = 5  # Number of detections to consider suspicious
        self.reported_suspicious_vehicles = set()  # Track already reported suspicious vehicles

        # Dictionary to track detected faces and their frames
        self.face_tracker = defaultdict(list)
        self.suspicious_face_threshold = 5  # Number of detections to consider suspicious
        self.reported_suspicious_faces = set()  # Track already reported suspicious faces
        
        
        # Threshold for face similarity
        self.SIMILARITY_THRESHOLD = 0.7
        
        # Initialize log data
        self.log_data = []
        
        # Global variables
        self.last_detected_object = "No detections yet"
        self.last_detected_frame = 0
        
        # Queue for frames
        self.frame_queue = queue.Queue(maxsize=20)
        self.stop_threads = False
        self.num_threads = 4
        
        self.car_counter = 0
    
    def browse_file(self, entry_widget, title, filetypes):
        """Open a file browser dialog and update the entry widget with selected path"""
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)
    
    def _cleanup_processing_thread(self):
        """Clean up the processing thread safely without blocking the GUI"""
        if self.processing_thread:
            self.processing_thread.join(timeout=10.0)  # Wait up to 10 seconds
            
            # Update the GUI from the main thread
            self.root.after(0, lambda: print("Detection stopped"))
            
            # Close any open windows
            cv2.destroyAllWindows()
            
    def select_video(self):
        video_file = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if video_file and os.path.exists(video_file):
            self.video_path = video_file
            self.show_preview_frame()

    def update_preview(self, frame):
        """Update the preview label with the processed frame"""
        try:
            # Use fixed dimensions instead of getting from the widget
            frame_resized = cv2.resize(frame, (self.preview_width, self.preview_height))
            
            # Convert the OpenCV frame to a format tkinter can display
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
            
            # Update the preview label in the main thread
            self.root.after(1, lambda: self._update_label(img))
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def _update_label(self, img):
        """Update the label in the main thread"""
        self.preview_label.configure(image=img)
        self.preview_label.image = img  # Keep a reference to prevent garbage collection
    
    def show_preview_frame(self):
        if not self.video_path:
            return
                
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
                
        # Get the middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            # Use fixed dimensions for preview
            frame = cv2.resize(frame, (self.preview_width, self.preview_height))
            
            # Convert to PIL format for tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
            self.preview_label.configure(image=img)
            self.preview_label.image = img  # Keep a reference
        
        cap.release()
    
    def start_detection(self):
        if not self.video_path:
            tk.messagebox.showerror("Error", "Please select a video file first.")
            return
        
        if not self.roi_points and self.enable_unauthorized_access.get():
            tk.messagebox.showwarning("Warning", "Unauthorized access detection is enabled but no region is selected. Please select a region first.")
            return
        
        # Update frame skip value from slider
        self.frame_skip = self.skip_slider.get() if self.enable_frame_skipping.get() else 1
        
        # Create a new thread for processing to avoid freezing the UI
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing = True
            self.processing_thread.join()
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_detection(self):
        """Safely stop the detection process without freezing the GUI"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing = True
            
            # Don't join the thread here as it will block the GUI
            # Instead use a separate thread to join the processing thread
            threading.Thread(target=self._cleanup_processing_thread, daemon=True).start()
            
            # Let the user know what's happening
            tk.messagebox.showinfo("Stopping Detection", "Detection stopping. Please wait...")
    
    def open_settings(self):
        # Create a settings window
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x400")  # Made wider for the file paths
        
        # Face detection settings
        face_frame = tk.LabelFrame(settings_window, text="Face Detection Settings")
        face_frame.pack(fill=tk.X, padx=10, pady=5)
        
        face_threshold_label = tk.Label(face_frame, text="Face Similarity Threshold:")
        face_threshold_label.pack(side=tk.LEFT, padx=5)
        
        face_threshold_slider = Scale(face_frame, from_=0.1, to=1.0, resolution=0.1, orient=HORIZONTAL)
        face_threshold_slider.set(self.SIMILARITY_THRESHOLD)
        face_threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Counter settings
        counter_frame = tk.LabelFrame(settings_window, text="Counter Settings")
        counter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        face_count_label = tk.Label(counter_frame, text="Face Detection Count:")
        face_count_label.pack(side=tk.LEFT, padx=5)
        
        face_count_entry = tk.Entry(counter_frame, width=5)
        face_count_entry.insert(0, str(self.face_count))
        face_count_entry.pack(side=tk.LEFT, padx=5)
        
        plate_count_label = tk.Label(counter_frame, text="Plate Detection Count:")
        plate_count_label.pack(side=tk.LEFT, padx=5)
        
        plate_count_entry = tk.Entry(counter_frame, width=5)
        plate_count_entry.insert(0, str(self.plate_count))
        plate_count_entry.pack(side=tk.LEFT, padx=5)
        
        vehicle_threshold_label = tk.Label(counter_frame, text="Suspicious Vehicle Threshold:")
        vehicle_threshold_label.pack(side=tk.LEFT, padx=5)

        vehicle_threshold_entry = tk.Entry(counter_frame, width=5)
        vehicle_threshold_entry.insert(0, str(self.suspicious_vehicle_threshold))
        vehicle_threshold_entry.pack(side=tk.LEFT, padx=5)

        face_threshold_label = tk.Label(counter_frame, text="Suspicious Face Threshold:")
        face_threshold_label.pack(side=tk.LEFT, padx=5)

        face_threshold_entry = tk.Entry(counter_frame, width=5)
        face_threshold_entry.insert(0, str(self.suspicious_face_threshold))
        face_threshold_entry.pack(side=tk.LEFT, padx=5)

        # And in the save_settings function:
        self.suspicious_vehicle_threshold = int(vehicle_threshold_entry.get())
        self.suspicious_face_threshold = int(face_threshold_entry.get())

        
        # File paths frame
        file_paths_frame = tk.LabelFrame(settings_window, text="File Paths")
        file_paths_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Tesseract path
        tesseract_frame = tk.Frame(file_paths_frame)
        tesseract_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tesseract_label = tk.Label(tesseract_frame, text="Tesseract Path:", width=15, anchor="w")
        tesseract_label.pack(side=tk.LEFT, padx=5)
        
        tesseract_entry = tk.Entry(tesseract_frame, width=50)
        tesseract_entry.insert(0, self.tesseract_path)
        tesseract_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tesseract_btn = tk.Button(tesseract_frame, text="Browse", 
                                 command=lambda: self.browse_file(tesseract_entry, "Select Tesseract Executable", 
                                                                [("Executable", "*.exe")]))
        tesseract_btn.pack(side=tk.RIGHT, padx=5)
        
        
        # Weapon model path
        weapon_frame = tk.Frame(file_paths_frame)
        weapon_frame.pack(fill=tk.X, padx=5, pady=2)
        
        weapon_label = tk.Label(weapon_frame, text="Weapon Model:", width=15, anchor="w")
        weapon_label.pack(side=tk.LEFT, padx=5)
        
        weapon_entry = tk.Entry(weapon_frame, width=50)
        weapon_entry.insert(0, str(self.weapon_model_path))
        weapon_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        weapon_btn = tk.Button(weapon_frame, text="Browse", 
                              command=lambda: self.browse_file(weapon_entry, "Select Weapon Model", 
                                                             [("PyTorch Model", "*.pt")]))
        weapon_btn.pack(side=tk.RIGHT, padx=5)
        
        # Vehicle model path
        vehicle_frame = tk.Frame(file_paths_frame)
        vehicle_frame.pack(fill=tk.X, padx=5, pady=2)
        
        vehicle_label = tk.Label(vehicle_frame, text="Vehicle Model:", width=15, anchor="w")
        vehicle_label.pack(side=tk.LEFT, padx=5)
        
        vehicle_entry = tk.Entry(vehicle_frame, width=50)
        vehicle_entry.insert(0, str(self.vehicle_model_path))
        vehicle_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        vehicle_btn = tk.Button(vehicle_frame, text="Browse", 
                               command=lambda: self.browse_file(vehicle_entry, "Select Vehicle Model", 
                                                              [("PyTorch Model", "*.pt")]))
        vehicle_btn.pack(side=tk.RIGHT, padx=5)
        
        # Plate model path
        plate_frame = tk.Frame(file_paths_frame)
        plate_frame.pack(fill=tk.X, padx=5, pady=2)
        
        plate_label = tk.Label(plate_frame, text="Plate Model:", width=15, anchor="w")
        plate_label.pack(side=tk.LEFT, padx=5)
        
        plate_entry = tk.Entry(plate_frame, width=50)
        plate_entry.insert(0, str(self.plate_model_path))
        plate_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        plate_btn = tk.Button(plate_frame, text="Browse", 
                             command=lambda: self.browse_file(plate_entry, "Select Plate Model", 
                                                            [("PyTorch Model", "*.pt")]))
        plate_btn.pack(side=tk.RIGHT, padx=5)
        
        # Email settings
        email_frame = tk.LabelFrame(settings_window, text="Email Settings")
        email_frame.pack(fill=tk.X, padx=10, pady=5)
        
        email_label = tk.Label(email_frame, text="Email:")
        email_label.pack(side=tk.LEFT, padx=5)
        
        email_entry = tk.Entry(email_frame, width=30)
        email_entry.insert(0, "detectionthreat932@gmail.com")
        email_entry.pack(side=tk.LEFT, padx=5)
        
        # Save button
        def save_settings():
            try:
                if face_threshold_slider.winfo_exists():
                    self.SIMILARITY_THRESHOLD = float(face_threshold_slider.get())
                else:
                    self.SIMILARITY_THRESHOLD=0.7
                self.face_count = int(face_count_entry.get())
                self.plate_count = int(plate_count_entry.get())
                self.suspicious_vehicle_threshold = int(vehicle_threshold_entry.get())
                self.suspicious_face_threshold = int(face_threshold_entry.get())
                
                # Save file paths
                old_tesseract_path = self.tesseract_path
                old_weapon_path = self.weapon_model_path
                old_vehicle_path = self.vehicle_model_path
                old_plate_path = self.plate_model_path
                
                self.tesseract_path = tesseract_entry.get()
                self.weapon_model_path = Path(weapon_entry.get())
                self.vehicle_model_path = Path(vehicle_entry.get())
                self.plate_model_path = Path(plate_entry.get())
                
                # Check if paths have changed and reload models if needed
                path_changed = False
                
                if old_tesseract_path != self.tesseract_path:
                    try:
                        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                        print("Tesseract path updated")
                        path_changed = True
                    except Exception as e:
                        print(f"Error updating Tesseract path: {e}")
                
                if old_weapon_path != self.weapon_model_path and self.weapon_model_path.exists():
                    try:
                        self.weapon_model = YOLO(str(self.weapon_model_path))
                        print("Weapon model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading weapon model: {e}")
                
                if old_vehicle_path != self.vehicle_model_path and self.vehicle_model_path.exists():
                    try:
                        self.vehicle_model = YOLO(str(self.vehicle_model_path))
                        print("Vehicle model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading vehicle model: {e}")
                
                if old_plate_path != self.plate_model_path and self.plate_model_path.exists():
                    try:
                        self.plate_model = YOLO(str(self.plate_model_path))
                        print("Plate model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading plate model: {e}")
                
                # Save settings to file
                self.save_settings_to_file()
                
                if path_changed:
                    tk.messagebox.showinfo("Info", "Model paths updated and models reloaded. Settings saved to file.")
                else:
                    tk.messagebox.showinfo("Info", "Settings saved to file.")
                
                settings_window.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter valid numbers for count values.")
            try:
                self.SIMILARITY_THRESHOLD = float(face_threshold_slider.get())
                self.face_count = int(face_count_entry.get())
                self.plate_count = int(plate_count_entry.get())
                
                # Save file paths
                old_tesseract_path = self.tesseract_path
                old_weapon_path = self.weapon_model_path
                old_vehicle_path = self.vehicle_model_path
                old_plate_path = self.plate_model_path
                
                self.tesseract_path = tesseract_entry.get()
                self.weapon_model_path = Path(weapon_entry.get())
                self.vehicle_model_path = Path(vehicle_entry.get())
                self.plate_model_path = Path(plate_entry.get())
                
                # Check if paths have changed and reload models if needed
                path_changed = False
                
                if old_tesseract_path != self.tesseract_path:
                    try:
                        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                        print("Tesseract path updated")
                        path_changed = True
                    except Exception as e:
                        print(f"Error updating Tesseract path: {e}")
                
                if old_weapon_path != self.weapon_model_path and self.weapon_model_path.exists():
                    try:
                        self.weapon_model = YOLO(str(self.weapon_model_path))
                        print("Weapon model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading weapon model: {e}")
                
                if old_vehicle_path != self.vehicle_model_path and self.vehicle_model_path.exists():
                    try:
                        self.vehicle_model = YOLO(str(self.vehicle_model_path))
                        print("Vehicle model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading vehicle model: {e}")
                
                if old_plate_path != self.plate_model_path and self.plate_model_path.exists():
                    try:
                        self.plate_model = YOLO(str(self.plate_model_path))
                        print("Plate model reloaded")
                        path_changed = True
                    except Exception as e:
                        print(f"Error loading plate model: {e}")
                
                if path_changed:
                    tk.messagebox.showinfo("Info", "Model paths updated and models reloaded.")
                
                settings_window.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter valid numbers for count values.")
        
        save_btn = tk.Button(settings_window, text="Save Settings", command=save_settings)
        save_btn.pack(pady=10)
    
    def select_region(self):
        if not self.video_path:
            tk.messagebox.showerror("Error", "Please select a video file first.")
            return
        
        # Open a new window for region selection
        self.roi_points = []
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            tk.messagebox.showerror("Error", "Could not open video file.")
            return
        
        # Get the middle frame for selection
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if not ret:
            tk.messagebox.showerror("Error", "Could not read frame from video.")
            cap.release()
            return
        
        # Create a window for region selection
        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", self.draw_roi_points, frame.copy())
        
        # Display the frame and wait for user input
        temp_frame = frame.copy()
        while True:
            display_frame = temp_frame.copy()
            
            # Draw the existing points and connections
            for i in range(1, len(self.roi_points)):
                cv2.line(display_frame, self.roi_points[i-1], self.roi_points[i], (0, 0, 255), 2)
            
            # If there are points, connect the last point to the first
            if len(self.roi_points) > 1:
                cv2.line(display_frame, self.roi_points[-1], self.roi_points[0], (0, 0, 255), 2)
            
            cv2.imshow("Select Region", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter key to save
                break
            elif key == 27:  # Escape key to cancel
                self.roi_points = []
                break
        
        cv2.destroyAllWindows()
        cap.release()
        
        print("ROI Points:", self.roi_points)
    
    def draw_roi_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))
            print(f"Added point at {x}, {y}")
    
    def clear_region(self):
        self.roi_points = []
        print("ROI points cleared")
    
    def view_logs(self):
        if os.path.exists(self.LOG_EXCEL_FILE):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.LOG_EXCEL_FILE)
                else:  # macOS or Linux
                    subprocess.run(['xdg-open', self.LOG_EXCEL_FILE])
            except Exception as e:
                print(f"Error opening log file: {e}")
        else:
            tk.messagebox.showinfo("Info", "No log file found.")
    
    def view_captures(self):
        if os.path.exists(self.OUTPUT_DIR):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.OUTPUT_DIR)
                else:  # macOS or Linux
                    subprocess.run(['xdg-open', str(self.OUTPUT_DIR)])
            except Exception as e:
                print(f"Error opening captures directory: {e}")
        else:
            tk.messagebox.showinfo("Info", "No captures directory found.")
    
    def process_video(self):
        if not self.video_path:
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        frame_index = 0
        ret = True
        
        # Start worker threads
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.process_frame_worker, daemon=True)
            t.start()
            threads.append(t)
        
        # Process frames
        while ret and not self.stop_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if self.enable_frame_skipping.get() and frame_index % self.frame_skip != 0:
                frame_index += 1
                continue
            
            frame_index += 1
            self.frame_queue.put((frame, frame_index))
        
        # Clean up
        for _ in range(self.num_threads):
            self.frame_queue.put(None)  # Send sentinel to stop threads
        
        for t in threads:
            t.join()
        
        cap.release()
    
    def process_frame_worker(self):
        while not self.stop_processing:
            try:
                item = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if item is None:  # Sentinel value to stop thread
                break
            
            frame, frame_index = item
            processed_frame = self.process_frame(frame)
            
            # Update the preview label
            self.update_preview(processed_frame)
            
            self.frame_queue.task_done()
    
    def process_frame(self, frame):
        # Skip processing if feature is disabled
        original_frame = frame.copy()
        frame_copy = frame.copy()
        
        # Face detection and recognition
        if self.enable_face_detection.get():
            faces, _ = self.mtcnn.detect(frame)
            if faces is not None and self.enable_face_recognition.get():
                for bbox in faces:
                    x1, y1, x2, y2 = map(int, bbox)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    # If recognition is enabled, process the face
                    if self.enable_face_recognition.get():
                        embedding = self.get_face_embedding(face_img)
                        embedding /= np.linalg.norm(embedding)  # Normalize embedding
                        
                        match_id = None
                        for known_id, known_embedding in self.face_embeddings.items():
                            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
                            if similarity > self.SIMILARITY_THRESHOLD:
                                match_id = known_id
                                break
                        
                        if match_id is None:
                            match_id = f"face_{len(self.face_embeddings) + 1}"
                            self.face_embeddings[match_id] = embedding
                            self.face_counter[match_id] = 0
                        
                        self.face_counter[match_id] += 1
                        
                        # Track this face's appearances for suspicious activity detection
                        self.face_tracker[match_id].append(time.time())
                        
                        # Remove timestamps older than 5 minutes
                        current_time = time.time()
                        self.face_tracker[match_id] = [t for t in self.face_tracker[match_id] 
                                                     if current_time - t < 300]  # 5 minutes
                        
                        # Check if face has been seen multiple times in a short period
                        is_suspicious = len(self.face_tracker[match_id]) >= self.suspicious_face_threshold
                        self.is_human_sus=is_suspicious
                        # If the face is detected multiple times, save a snapshot
                        if self.enable_save_captures.get() and self.face_counter[match_id] == self.face_count:
                            save_path = self.save_detection(frame_copy, (x1, y1, x2, y2), "faces")
                            if self.enable_save_logs.get():
                                self.log_detection("Face", match_id)
                            print(f"[INFO] Saved snapshot for face ID: {match_id}")
                            
                            # Reset the counter for this face
                            self.face_counter[match_id] = 0
                        
                        # Draw rectangle - red if suspicious, green if normal
                        color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add face ID and suspicious indicator if necessary
                        label_text = f"Face {match_id}"
                        if is_suspicious:
                            label_text += " SUSPICIOUS"
                            # Add text on screen for suspicious face
                            cv2.putText(frame, "SUSPICIOUS HUMAN MOVEMENT DETECTED", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.putText(frame, label_text, 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Handle suspicious face that hasn't been reported yet
                        if is_suspicious and match_id not in self.reported_suspicious_faces:
                            self.reported_suspicious_faces.add(match_id)
                            
                            # Save the suspicious face image
                            suspicious_path = None
                            if self.enable_save_captures.get():
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                filename = f"suspicious_face_{timestamp}.jpg"
                                suspicious_path = self.OUTPUT_DIR / "suspicious" / filename
                                suspicious_path.parent.mkdir(exist_ok=True)
                                cv2.imwrite(str(suspicious_path), frame_copy[y1:y2, x1:x2])
                            
                            # Send email with image if email alerts are enabled
                            if self.enable_email_alerts.get():
                                self.send_email_alert_sus(
                                    "Suspicious Human", 
                                    f"Repetitive movement detected ({len(self.face_tracker[match_id])} times)",
                                    None,
                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                    str(suspicious_path) if suspicious_path else None
                                )
                    else:
                        # If recognition is disabled, just draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Face", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Weapon detection
        if self.enable_weapon_detection.get() and self.weapon_model:
            weapon_results = self.weapon_model(frame, conf=0.5)
            for result in weapon_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    label = self.weapon_model.names[cls].lower()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if self.enable_save_captures.get():
                        self.save_detection(frame, (x1, y1, x2, y2), "weapons")
                    if self.enable_save_logs.get():
                        self.log_detection("Weapon", label, conf)
        
        # Vehicle detection
        if self.enable_vehicle_detection.get() and self.vehicle_model:
            vehicle_results = self.vehicle_model(frame, conf=0.5)
            for result in vehicle_results:
                for box in result.boxes:
                    if not hasattr(self, 'frame_vehicle_count'):
                        self.frame_vehicle_count = 0
                        
                    self.frame_vehicle_count += 1
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    label = self.vehicle_model.names[cls].lower()
                    
                    # Generate a vehicle ID based on position and size
                    vehicle_id = f"{label}_{x1//50}_{y1//50}_{(x2-x1)//50}_{(y2-y1)//50}"
                    
                    # Track this vehicle's appearances
                    self.vehicle_tracker[vehicle_id].append(time.time())
                    
                    # Remove timestamps older than 5 minutes
                    current_time = time.time()
                    self.vehicle_tracker[vehicle_id] = [t for t in self.vehicle_tracker[vehicle_id] 
                                                      if current_time - t < 300]  # 5 minutes
                    
                    # Check if vehicle has been seen multiple times in a short period
                    is_suspicious = len(self.vehicle_tracker[vehicle_id]) >= self.suspicious_vehicle_threshold
                    
                    self.is_vehicle_sus = len(self.vehicle_tracker[vehicle_id]) >= self.suspicious_vehicle_threshold
                    # Draw rectangle - red if suspicious, blue if normal
                    color = (0, 0, 255) if is_suspicious else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with vehicle info
                    label_text = f"{label} {conf:.2f}"
                    if is_suspicious:
                        label_text += " SUSPICIOUS"
                        # Add text on screen for suspicious vehicle
                        cv2.putText(frame, "SUSPICIOUS VEHICLE MOVEMENT DETECTED", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Save detection image
                    if self.enable_save_captures.get():
                        save_path = self.save_detection(frame_copy, (x1, y1, x2, y2), "vehicles", margin=50)
                    
                    # Log detection
                    if self.enable_save_logs.get():
                        self.log_detection("Vehicle", label, conf)
                    
                    # Handle suspicious vehicle that hasn't been reported yet
                    if is_suspicious and vehicle_id not in self.reported_suspicious_vehicles:
                        self.reported_suspicious_vehicles.add(vehicle_id)
                        
                        # Save the suspicious vehicle image
                        suspicious_path = None
                        if self.enable_save_captures.get():
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"suspicious_vehicle_{timestamp}.jpg"
                            suspicious_path = self.OUTPUT_DIR / "suspicious" / filename
                            suspicious_path.parent.mkdir(exist_ok=True)
                            cv2.imwrite(str(suspicious_path), frame_copy[y1:y2, x1:x2])
                        
                        # Send email with image if email alerts are enabled
                        if self.enable_email_alerts.get():
                            self.send_email_alert_sus(
                                "Suspicious Vehicle", 
                                f"Repetitive movement detected ({len(self.vehicle_tracker[vehicle_id])} times)",
                                conf,
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                str(suspicious_path) if suspicious_path else None
                            )
                            
        # License plate detection
        if self.enable_plate_detection.get() and self.plate_model:
            plate_results = self.plate_model(frame, conf=0.5)
            for result in plate_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_img = frame[y1:y2, x1:x2]
                    plate_text = self.extract_plate_text(plate_img)
                    
                    if plate_text:
                        self.plate_tracker[plate_text] += 1
                        print(f"[INFO] Plate '{plate_text}' detected {self.plate_tracker[plate_text]} times.")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"Plate: {plate_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        if self.enable_save_captures.get() and self.plate_tracker[plate_text] == self.plate_count:
                            self.save_detection(frame, (x1, y1, x2, y2), "plates", margin=20)
                            print(f"[INFO] Saved snapshot for plate '{plate_text}'.")
                            if self.enable_save_logs.get():
                                self.log_detection("Plate", plate_text)
                            # Reset counter
                            self.plate_tracker[plate_text] = 0
        
        # Unauthorized access detection
        if self.enable_unauthorized_access.get() and len(self.roi_points) >= 3:
            # Convert frame to gray for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create ROI mask
            roi_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(self.roi_points, dtype=np.int32)], 255)
            
            # Blur for noise reduction
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # If we have a previous frame, check for motion
            if hasattr(self, 'prev_gray'):
                # Calculate difference between current and previous frame
                frame_delta = cv2.absdiff(self.prev_gray, gray_blurred)
                _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
                
                # Apply ROI mask to focus only on the region of interest
                masked_thresh = cv2.bitwise_and(thresh, roi_mask)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(masked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check if any contour is large enough to be considered motion
                motion_detected = any(cv2.contourArea(cnt) > 500 for cnt in contours)
                
                if motion_detected:
                    # Highlight the ROI in red
                    cv2.polylines(frame, [np.array(self.roi_points, dtype=np.int32)], True, (0, 0, 255), 2)
                    cv2.putText(frame, "UNAUTHORIZED ACCESS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if self.enable_save_captures.get():
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"unauthorized_{timestamp}.jpg"
                        save_path = self.OUTPUT_DIR / "unauthorized" / filename
                        save_path.parent.mkdir(exist_ok=True)
                        cv2.imwrite(str(save_path), frame)
                    
                    if self.enable_save_logs.get():
                        self.log_detection("Unauthorized", "Motion detected")
                else:
                    # Draw ROI in green if no motion
                    cv2.polylines(frame, [np.array(self.roi_points, dtype=np.int32)], True, (0, 255, 0), 2)
            
            # Update previous frame
            self.prev_gray = gray_blurred
        
        # Display detection status on frame
        status_text = "Detection active" if not self.stop_processing else "Detection stopped"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.is_human_sus:
            label_text = " SUSPICIOUS HUMAN ACTIVITY DETECTED"
            # Add text on screen for suspicious face
            cv2.putText(frame, "SUSPICIOUS HUMAN MOVEMENT DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        if self.is_vehicle_sus:
            label_text = " SUSPICIOUS VEHICLE ACTIVITY DETECTED"
            # Add text on screen for suspicious vehicle
            cv2.putText(frame, "SUSPICIOUS VEHICLE MOVEMENT DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    # Send email with image if email alerts are enabled
            
        return frame
    
    def save_detection(self, frame, bbox, category, margin=20):
        """Save detected objects to appropriate folders"""
        if not self.enable_save_captures.get():
            return
            
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        obj_img = frame[y1:y2, x1:x2].copy()
        if obj_img.size == 0:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_{timestamp}.jpg"
        save_path = self.SAVE_DIRS[category] / filename
        
        cv2.imwrite(str(save_path), obj_img)
        print(f"[INFO] Saved {category} snapshot to {save_path}")
        
        return save_path
    
    def log_detection(self, category, name, confidence=None):
        """Log detected objects to text and Excel files"""
        if not self.enable_save_logs.get():
            return
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "Timestamp": timestamp,
            "Category": category,
            "Name": name,
            "Confidence": confidence
        }
        self.log_data.append(log_entry)
        
        # Append to text file
        try:
            with open(self.LOG_TEXT_FILE, "a") as f:
                f.write(f"{timestamp}, {category}, {name}, {confidence}\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to text file: {e}")
        
        # Save to Excel file
        try:
            df = pd.DataFrame(self.log_data)
            df.to_excel(self.LOG_EXCEL_FILE, index=False)
        except Exception as e:
            print(f"[ERROR] Failed to save Excel file: {e}")
        
        # Send email alert if enabled
        if self.enable_email_alerts.get():
            if self.enable_email_alerts.get():
                self.send_email_alert(category, name, confidence, timestamp)
    
    # Enhance the send_email_alert method with better logging
    def send_email_alert(self, category, name, confidence, timestamp, image_path=None):
        """Send email alert for detected objects with cooldown"""
        current_time = time.time()
        
        # Check if enough time has passed since the last email for this category
        if current_time - self.last_email_times.get(category, 0) < self.email_cooldown:
            print(f"[INFO] Email cooldown active for {category}, skipping alert")
            return
        
        subject = f"Security Alert: {category} Detected"
        body = f"The following object was detected by the Security Detection System:\n\n" \
               f"Category: {category}\n" \
               f"Name: {name}\n" \
               f"Confidence: {confidence}\n" \
               f"Timestamp: {timestamp}"
        
        sender_email = "detectionthreat932@gmail.com"
        receiver_email = "detectionthreat932@gmail.com"
        password = "oasa wpww jwoi kacd"
        
        print(f"[INFO] Attempting to send email alert for {category}...")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                print(f"[INFO] Attaching image: {image_path}")
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            elif image_path:
                print(f"[WARNING] Image path does not exist: {image_path}")
            
            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                print("[INFO] Connecting to SMTP server...")
                server.starttls()
                print("[INFO] Logging into email account...")
                server.login(sender_email, password)
                print("[INFO] Sending email...")
                server.sendmail(sender_email, receiver_email, msg.as_string())
            
            # Update last email time for this category
            self.last_email_times[category] = current_time
            print(f"[INFO] Email alert successfully sent for {category} detection")
        except Exception as e:
            print(f"[ERROR] Failed to send email alert: {e}")
            import traceback
            traceback.print_exc()
    
    def send_email_alert_sus(self, category, name, confidence, timestamp, image_path=None):
        """Send email alert for detected objects with cooldown"""
        current_time = time.time()
        
        subject = f"Security Alert: {category} Detected"
        body = f"The following object was detected by the Security Detection System:\n\n" \
               f"Category: {category}\n" \
               f"Name: {name}\n" \
               f"Confidence: {confidence}\n" \
               f"Timestamp: {timestamp}"
        
        sender_email = "detectionthreat932@gmail.com"
        receiver_email = "detectionthreat932@gmail.com"
        password = "oasa wpww jwoi kacd"
        
        print(f"[INFO] Attempting to send email alert for {category}...")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                print(f"[INFO] Attaching image: {image_path}")
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            elif image_path:
                print(f"[WARNING] Image path does not exist: {image_path}")
            
            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                print("[INFO] Connecting to SMTP server...")
                server.starttls()
                print("[INFO] Logging into email account...")
                server.login(sender_email, password)
                print("[INFO] Sending email...")
                server.sendmail(sender_email, receiver_email, msg.as_string())
            
            # Update last email time for this category
            self.last_email_times[category] = current_time
            print(f"[INFO] Email alert successfully sent for {category} detection")
        except Exception as e:
            print(f"[ERROR] Failed to send email alert: {e}")
            import traceback
            traceback.print_exc()
    
    
    def get_face_embedding(self, face_image):
        """Convert face image to a FaceNet embedding"""
        face_image = cv2.resize(face_image, (160, 160))
        face_image = torch.tensor(face_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embedding = self.resnet(face_image)
        return embedding.numpy().flatten()
    
    def preprocess_plate_image(self, plate_img):
        """Preprocess the number plate image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def extract_plate_text(self, plate_img):
        """Extract text from the number plate image using Tesseract OCR"""
        processed_img = self.preprocess_plate_image(plate_img)
        custom_config = r'--oem 3 --psm 7'  
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        text = "".join([char for char in text if char.isalnum()])  
        return text.strip()

# Main function to run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityDetectionApp(root)
    root.mainloop()