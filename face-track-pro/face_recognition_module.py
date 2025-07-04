import cv2
import face_recognition
import pickle
import numpy as np
import os
from datetime import datetime

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = 'face-track-pro/local.pkl'
        
        # Face detection optimization
        self.face_detection_confidence = 0.6
        self.face_recognition_tolerance = 0.5
        self.process_every_n_frames = 3  # Process every 3rd frame for speed
        self.frame_count = 0
        
        # Last known face locations for interpolation
        self.last_face_locations = []
        self.last_face_names = []
    
    def load_model(self):
        """Load the trained face recognition model from pickle file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces from model")
            else:
                print("No existing model found. Please train the model first.")
                self.known_face_encodings = []
                self.known_face_names = []
        except Exception as e:
            print(f"Error loading model: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def process_frame(self, frame):
        """Process a frame for face recognition"""
        self.frame_count += 1
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
        
        detected_names = []
        
        # Only process every N frames for speed optimization
        if self.frame_count % self.process_every_n_frames == 0:
            # Find faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # Check if face matches any known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=self.face_recognition_tolerance
                )
                name = "Unknown"
                
                # Use the known face with the smallest distance
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < self.face_recognition_tolerance:
                        name = self.known_face_names[best_match_index]
                        detected_names.append(name)
                
                face_names.append(name)
            
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            self.last_face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
            self.last_face_names = face_names
        
        # Draw the results on the frame
        processed_frame = self.draw_results(frame, self.last_face_locations, self.last_face_names)
        
        return processed_frame, detected_names
    
    def draw_results(self, frame, face_locations, face_names):
        """Draw bounding boxes and names on the frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Choose color based on recognition status
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                text_color = (255, 255, 255)  # White text
            else:
                color = (0, 255, 0)  # Green for known faces
                text_color = (255, 255, 255)  # White text
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, text_color, 1)
            
            # Add timestamp for known faces
            if name != "Unknown":
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (left + 6, top - 10), font, 0.4, color, 1)
        
        # Add system info overlay
        self.add_info_overlay(frame)
        
        return frame
    
    def add_info_overlay(self, frame):
        """Add system information overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Add background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "FaceTrack Pro - Live Detection", (20, 35), font, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Known Faces: {len(self.known_face_names)}", (20, 55), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 75), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, 95), font, 0.4, (255, 255, 255), 1)
    
    def recognize_face(self, face_encoding):
        """Recognize a single face encoding"""
        if len(self.known_face_encodings) == 0:
            return "Unknown", 1.0
        
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < self.face_recognition_tolerance:
            return self.known_face_names[best_match_index], face_distances[best_match_index]
        else:
            return "Unknown", face_distances[best_match_index]
    
    def get_face_encoding(self, image_path):
        """Get face encoding from an image file"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None