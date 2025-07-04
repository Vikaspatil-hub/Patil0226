#!/usr/bin/env python3
"""
FaceTrack Pro - Demo and Testing Script
This script helps test the system components and validate the installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2', 'face_recognition', 'flask', 'pandas', 'numpy', 'pickle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV {cv2.__version__}")
            elif package == 'face_recognition':
                import face_recognition
                print(f"✅ face_recognition")
            elif package == 'flask':
                import flask
                print(f"✅ Flask {flask.__version__}")
            elif package == 'pandas':
                import pandas as pd
                print(f"✅ Pandas {pd.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✅ NumPy {np.__version__}")
            elif package == 'pickle':
                import pickle
                print(f"✅ Pickle (built-in)")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - NOT FOUND")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_camera():
    """Test camera functionality"""
    print("\n📷 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"✅ Camera working - Resolution: {width}x{height}")
            cap.release()
            return True
        else:
            print("❌ Cannot read from camera")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def check_face_recognition():
    """Test face recognition functionality"""
    print("\n🧠 Testing face recognition...")
    
    try:
        import face_recognition
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Try to find faces (should find none)
        face_locations = face_recognition.face_locations(test_image)
        print(f"✅ Face recognition library working")
        print(f"   Found {len(face_locations)} faces in test image (expected: 0)")
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

def check_directory_structure():
    """Verify directory structure"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        'face-track-pro',
        'face-track-pro/templates',
        'face-track-pro/static',
        'face-track-pro/dataset',
        'face-track-pro/attendance',
        'face-track-pro/utils'
    ]
    
    required_files = [
        'face-track-pro/app.py',
        'face-track-pro/camera.py',
        'face-track-pro/face_recognition_module.py',
        'face-track-pro/train_model.py',
        'face-track-pro/requirements.txt',
        'face-track-pro/templates/index.html',
        'face-track-pro/templates/admin.html',
        'face-track-pro/static/style.css',
        'face-track-pro/utils/helpers.py'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def create_sample_data():
    """Create sample attendance data for testing"""
    print("\n📊 Creating sample data...")
    
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create sample attendance data
        sample_data = [
            {'Name': 'John Doe', 'Date': '2024-12-01', 'Time': '09:15:30', 'Status': 'Present'},
            {'Name': 'Jane Smith', 'Date': '2024-12-01', 'Time': '09:22:45', 'Status': 'Present'},
            {'Name': 'Bob Johnson', 'Date': '2024-12-01', 'Time': '09:35:12', 'Status': 'Present'},
        ]
        
        df = pd.DataFrame(sample_data)
        attendance_file = 'face-track-pro/attendance/attendance.csv'
        
        # Only create if file doesn't exist
        if not os.path.exists(attendance_file):
            df.to_csv(attendance_file, index=False)
            print(f"✅ Created sample attendance data: {attendance_file}")
        else:
            print(f"✅ Attendance file already exists: {attendance_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def run_tests():
    """Run all system tests"""
    print("🚀 FaceTrack Pro - System Validation\n")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Camera", check_camera),
        ("Face Recognition", check_face_recognition),
        ("Sample Data", create_sample_data)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your FaceTrack Pro system is ready!")
        print("\nTo start the system:")
        print("1. cd face-track-pro")
        print("2. python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-deps":
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "face-track-pro/requirements.txt"])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    
    success = run_tests()
    sys.exit(0 if success else 1)