import os
import cv2
import face_recognition
import pickle
import numpy as np
from pathlib import Path

class FaceTrainer:
    def __init__(self):
        self.dataset_path = 'face-track-pro/dataset'
        self.model_path = 'face-track-pro/local.pkl'
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    def train_model(self):
        """Train the face recognition model with all images in the dataset"""
        print("Starting face recognition model training...")
        
        known_encodings = []
        known_names = []
        
        # Create dataset directory if it doesn't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Process each person's folder in the dataset
        for person_folder in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_folder)
            
            if not os.path.isdir(person_path):
                continue
            
            print(f"Processing images for: {person_folder}")
            person_encodings = []
            
            # Process all images for this person
            for image_file in os.listdir(person_path):
                if not any(image_file.lower().endswith(fmt) for fmt in self.supported_formats):
                    continue
                
                image_path = os.path.join(person_path, image_file)
                print(f"  Processing: {image_file}")
                
                try:
                    # Load and process the image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings (there should be exactly one face per image)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) == 0:
                        print(f"    Warning: No face found in {image_file}")
                        continue
                    elif len(face_encodings) > 1:
                        print(f"    Warning: Multiple faces found in {image_file}, using the first one")
                    
                    # Add the face encoding
                    face_encoding = face_encodings[0]
                    person_encodings.append(face_encoding)
                    
                except Exception as e:
                    print(f"    Error processing {image_file}: {e}")
                    continue
            
            # Add all encodings for this person
            if person_encodings:
                # Convert folder name to display name
                display_name = person_folder.replace('_', ' ').title()
                
                for encoding in person_encodings:
                    known_encodings.append(encoding)
                    known_names.append(display_name)
                
                print(f"  Added {len(person_encodings)} encodings for {display_name}")
            else:
                print(f"  No valid encodings found for {person_folder}")
        
        # Save the model
        if known_encodings:
            model_data = {
                'encodings': known_encodings,
                'names': known_names
            }
            
            # Create directory for model if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\nModel training completed!")
            print(f"Total faces trained: {len(known_encodings)}")
            print(f"Unique persons: {len(set(known_names))}")
            print(f"Model saved to: {self.model_path}")
            
            # Print summary by person
            from collections import Counter
            person_counts = Counter(known_names)
            print("\nTraining summary:")
            for person, count in person_counts.items():
                print(f"  {person}: {count} images")
                
        else:
            print("No valid face encodings found. Please check your dataset.")
            return False
        
        return True
    
    def add_person(self, person_name, image_paths):
        """Add a new person to the dataset and retrain the model"""
        person_folder = person_name.lower().replace(' ', '_')
        person_path = os.path.join(self.dataset_path, person_folder)
        
        # Create person directory
        os.makedirs(person_path, exist_ok=True)
        
        valid_images = 0
        for i, image_path in enumerate(image_paths):
            try:
                # Load and validate the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image: {image_path}")
                    continue
                
                # Check if face is detected
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if len(face_locations) == 0:
                    print(f"No face detected in: {image_path}")
                    continue
                elif len(face_locations) > 1:
                    print(f"Multiple faces detected in: {image_path}, using first face")
                
                # Save the image to the person's folder
                filename = f"{person_folder}_{i+1}.jpg"
                save_path = os.path.join(person_path, filename)
                cv2.imwrite(save_path, image)
                valid_images += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        if valid_images > 0:
            print(f"Added {valid_images} images for {person_name}")
            # Retrain the model
            return self.train_model()
        else:
            print(f"No valid images added for {person_name}")
            return False
    
    def remove_person(self, person_name):
        """Remove a person from the dataset and retrain the model"""
        person_folder = person_name.lower().replace(' ', '_')
        person_path = os.path.join(self.dataset_path, person_folder)
        
        if os.path.exists(person_path):
            import shutil
            shutil.rmtree(person_path)
            print(f"Removed {person_name} from dataset")
            return self.train_model()
        else:
            print(f"Person {person_name} not found in dataset")
            return False
    
    def validate_dataset(self):
        """Validate the dataset and report any issues"""
        print("Validating dataset...")
        
        if not os.path.exists(self.dataset_path):
            print("Dataset directory does not exist")
            return False
        
        total_images = 0
        valid_images = 0
        issues = []
        
        for person_folder in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_folder)
            
            if not os.path.isdir(person_path):
                continue
            
            person_images = 0
            person_valid = 0
            
            for image_file in os.listdir(person_path):
                if not any(image_file.lower().endswith(fmt) for fmt in self.supported_formats):
                    continue
                
                total_images += 1
                person_images += 1
                image_path = os.path.join(person_path, image_file)
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) == 0:
                        issues.append(f"No face found: {person_folder}/{image_file}")
                    elif len(face_encodings) > 1:
                        issues.append(f"Multiple faces: {person_folder}/{image_file}")
                        valid_images += 1
                        person_valid += 1
                    else:
                        valid_images += 1
                        person_valid += 1
                        
                except Exception as e:
                    issues.append(f"Error processing {person_folder}/{image_file}: {e}")
            
            print(f"{person_folder}: {person_valid}/{person_images} valid images")
        
        print(f"\nDataset validation completed:")
        print(f"Total images: {total_images}")
        print(f"Valid images: {valid_images}")
        print(f"Issues found: {len(issues)}")
        
        if issues:
            print("\nIssues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
        
        return len(issues) == 0

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FaceTrack Pro Model Training')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--validate', action='store_true', help='Validate the dataset')
    parser.add_argument('--add-person', type=str, help='Add a new person to the dataset')
    parser.add_argument('--images', nargs='+', help='Image paths for adding a person')
    
    args = parser.parse_args()
    
    trainer = FaceTrainer()
    
    if args.validate:
        trainer.validate_dataset()
    elif args.train:
        trainer.train_model()
    elif args.add_person and args.images:
        trainer.add_person(args.add_person, args.images)
    else:
        print("Please specify an action: --train, --validate, or --add-person")

if __name__ == '__main__':
    main()