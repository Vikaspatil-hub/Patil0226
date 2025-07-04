from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_file
import cv2
import os
import pickle
import pandas as pd
from datetime import datetime, date
import threading
import time
from camera import VideoCamera
from face_recognition_module import FaceRecognizer
from train_model import FaceTrainer
from utils.helpers import format_time, get_attendance_stats, ensure_directories

app = Flask(__name__)
app.secret_key = 'facetrack_pro_secret_key_2024'

# Global variables
video_camera = None
face_recognizer = None
attendance_logged_today = set()  # Track who's already logged today
is_camera_active = False

def initialize_system():
    """Initialize the FaceTrack Pro system"""
    global face_recognizer
    ensure_directories()
    face_recognizer = FaceRecognizer()
    face_recognizer.load_model()

def gen(camera):
    """Video streaming generator function."""
    global attendance_logged_today
    
    while True:
        if not is_camera_active:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is not None:
            # Process frame for face recognition
            processed_frame, detected_names = face_recognizer.process_frame(frame)
            
            # Log attendance for new faces detected
            current_date = date.today().strftime("%Y-%m-%d")
            for name in detected_names:
                if name != "Unknown" and f"{name}_{current_date}" not in attendance_logged_today:
                    log_attendance(name)
                    attendance_logged_today.add(f"{name}_{current_date}")
            
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def log_attendance(name):
    """Log attendance for a student"""
    try:
        current_time = datetime.now()
        attendance_data = {
            'Name': name,
            'Date': current_time.strftime("%Y-%m-%d"),
            'Time': current_time.strftime("%H:%M:%S"),
            'Status': 'Present'
        }
        
        attendance_file = 'face-track-pro/attendance/attendance.csv'
        
        # Check if file exists and has data
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            # Check if already logged today
            today_records = df[(df['Name'] == name) & (df['Date'] == attendance_data['Date'])]
            if len(today_records) == 0:
                df = pd.concat([df, pd.DataFrame([attendance_data])], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                print(f"Attendance logged for {name} at {attendance_data['Time']}")
        else:
            # Create new file with headers
            df = pd.DataFrame([attendance_data])
            df.to_csv(attendance_file, index=False)
            print(f"Attendance logged for {name} at {attendance_data['Time']}")
            
    except Exception as e:
        print(f"Error logging attendance: {e}")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Admin panel page"""
    students = get_registered_students()
    return render_template('admin.html', students=students)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start the camera feed"""
    global is_camera_active
    is_camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera feed"""
    global is_camera_active
    is_camera_active = False
    return jsonify({'status': 'Camera stopped'})

@app.route('/attendance_stats')
def attendance_stats():
    """Get real-time attendance statistics"""
    stats = get_attendance_stats()
    return jsonify(stats)

@app.route('/register_student', methods=['POST'])
def register_student():
    """Register a new student"""
    try:
        name = request.form.get('student_name')
        usn = request.form.get('student_usn')
        
        if not name or not usn:
            flash('Name and USN are required', 'error')
            return redirect(url_for('admin'))
        
        # Handle file uploads
        if 'images' not in request.files:
            flash('No images uploaded', 'error')
            return redirect(url_for('admin'))
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            flash('No images selected', 'error')
            return redirect(url_for('admin'))
        
        # Create student directory
        student_dir = f'face-track-pro/dataset/{name.lower().replace(" ", "_")}'
        os.makedirs(student_dir, exist_ok=True)
        
        # Save uploaded images
        saved_files = []
        for i, file in enumerate(files):
            if file and file.filename != '':
                filename = f"{name.lower().replace(' ', '_')}_{i+1}.jpg"
                filepath = os.path.join(student_dir, filename)
                file.save(filepath)
                saved_files.append(filepath)
        
        if saved_files:
            # Train the model with new student
            trainer = FaceTrainer()
            trainer.train_model()
            
            # Reload the face recognizer
            face_recognizer.load_model()
            
            flash(f'Student {name} registered successfully with {len(saved_files)} images', 'success')
        else:
            flash('No valid images were saved', 'error')
            
    except Exception as e:
        flash(f'Error registering student: {str(e)}', 'error')
    
    return redirect(url_for('admin'))

@app.route('/retrain_model')
def retrain_model():
    """Retrain the face recognition model"""
    try:
        trainer = FaceTrainer()
        trainer.train_model()
        face_recognizer.load_model()
        flash('Model retrained successfully', 'success')
    except Exception as e:
        flash(f'Error retraining model: {str(e)}', 'error')
    
    return redirect(url_for('admin'))

@app.route('/download_attendance')
def download_attendance():
    """Download attendance CSV file"""
    try:
        attendance_file = 'face-track-pro/attendance/attendance.csv'
        if os.path.exists(attendance_file):
            return send_file(attendance_file, as_attachment=True)
        else:
            flash('No attendance data available', 'error')
            return redirect(url_for('admin'))
    except Exception as e:
        flash(f'Error downloading attendance: {str(e)}', 'error')
        return redirect(url_for('admin'))

def get_registered_students():
    """Get list of registered students"""
    students = []
    dataset_dir = 'face-track-pro/dataset'
    if os.path.exists(dataset_dir):
        for student_folder in os.listdir(dataset_dir):
            student_path = os.path.join(dataset_dir, student_folder)
            if os.path.isdir(student_path):
                image_count = len([f for f in os.listdir(student_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                students.append({
                    'name': student_folder.replace('_', ' ').title(),
                    'folder': student_folder,
                    'images': image_count
                })
    return students

if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)