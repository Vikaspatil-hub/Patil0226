# FaceTrack Pro - Real-Time Smart Attendance System

A sophisticated face recognition-based attendance system built with Python, OpenCV, and Flask. FaceTrack Pro provides real-time face detection, recognition, and attendance logging with a modern web interface.

## 🌟 Features

- **Real-time Face Recognition**: Live camera feed with instant face detection and recognition
- **Automated Attendance Logging**: Automatic attendance marking with timestamp
- **Web-based Dashboard**: Modern, responsive UI for monitoring and management
- **Student Registration**: Easy student enrollment with multiple face images
- **Admin Panel**: Complete system management and controls
- **Data Export**: CSV export of attendance records
- **Performance Optimized**: Efficient processing for real-time operation
- **Scalable Architecture**: Modular design for easy extension

## 🏗️ System Architecture

```
face-track-pro/
│
├── app.py                    # Main Flask application
├── camera.py                 # Webcam/video stream logic
├── face_recognition_module.py # Face recognition functions
├── train_model.py           # Model training script
├── local.pkl                # Face encodings database
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
│
├── templates/              # HTML templates
│   ├── index.html         # Dashboard
│   └── admin.html         # Admin panel
│
├── static/                # Static assets
│   └── style.css         # CSS styling
│
├── dataset/               # Face images dataset
│   └── [student_name]/   # Student image folders
│
├── attendance/            # Attendance logs
│   └── attendance.csv    # CSV attendance records
│
└── utils/                 # Utility functions
    └── helpers.py        # Helper functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV compatible camera (webcam/USB camera)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-track-pro
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you encounter issues with `dlib` installation:
   - On Windows: Install Visual Studio Build Tools
   - On Ubuntu/Debian: `sudo apt-get install build-essential cmake`
   - On macOS: `brew install cmake`

3. **Initialize the system**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Dashboard: `http://localhost:5000/`
   - Admin Panel: `http://localhost:5000/admin`

## 📖 Usage Guide

### 1. Register Students

1. Navigate to the Admin Panel (`/admin`)
2. Fill in student details:
   - **Name**: Full name of the student
   - **USN/ID**: Unique student identifier
   - **Images**: Upload 3-5 clear face images
3. Click "Register Student" to add to the system
4. The system will automatically train the model with new data

### 2. Start Attendance Monitoring

1. Go to the Dashboard (`/`)
2. Click "Start Camera" to begin live monitoring
3. Students will be automatically recognized and attendance logged
4. View real-time statistics and attendance list

### 3. Manage System

**Retrain Model**: Update the recognition model with new data
**Download Data**: Export attendance records as CSV
**View Statistics**: Monitor attendance rates and trends
**System Status**: Check model status and dataset information

## 🔧 Configuration

### Camera Settings
Modify camera parameters in `camera.py`:
```python
self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
self.video.set(cv2.CAP_PROP_FPS, 30)
```

### Recognition Parameters
Adjust recognition sensitivity in `face_recognition_module.py`:
```python
self.face_recognition_tolerance = 0.5  # Lower = more strict
self.process_every_n_frames = 3        # Process every nth frame
```

### Late Arrival Time
Modify late arrival threshold in `utils/helpers.py`:
```python
late_time = pd.to_datetime('09:00:00', format='%H:%M:%S').time()
```

## 📊 Data Management

### Database Structure
The system uses a pickle file (`local.pkl`) to store face encodings:
```python
{
    'encodings': [face_encoding_arrays],
    'names': [corresponding_names]
}
```

### Attendance Records
CSV format with columns: Name, Date, Time, Status
```csv
Name,Date,Time,Status
John Doe,2024-12-01,09:15:30,Present
Jane Smith,2024-12-01,09:22:45,Present
```

## 🛠️ Advanced Features

### Custom Training
Train the model manually:
```bash
python train_model.py --train
```

### Dataset Validation
Check dataset quality:
```bash
python train_model.py --validate
```

### Add Person via CLI
```bash
python train_model.py --add-person "John Doe" --images path/to/image1.jpg path/to/image2.jpg
```

## 🔍 Troubleshooting

### Common Issues

**Camera not detected**
- Check camera permissions
- Ensure camera is not used by another application
- Try changing camera index in `camera.py`

**Poor recognition accuracy**
- Add more training images per person
- Ensure good lighting conditions
- Use high-quality, clear face images
- Adjust recognition tolerance

**Model training fails**
- Verify all images contain exactly one face
- Check image file formats (JPG, PNG, BMP)
- Ensure sufficient disk space

**Performance issues**
- Reduce camera resolution
- Increase `process_every_n_frames` value
- Close unnecessary applications

### System Requirements

**Minimum Requirements**:
- 2GB RAM
- 1GB disk space
- USB 2.0 camera
- Python 3.8+

**Recommended Requirements**:
- 4GB+ RAM
- 2GB+ disk space
- USB 3.0 camera with 720p resolution
- Multi-core processor

## 📈 Performance Optimization

1. **Frame Processing**: Adjust `process_every_n_frames` based on system performance
2. **Image Scaling**: Reduce frame size for faster processing
3. **Model Optimization**: Retrain periodically with quality images
4. **Memory Management**: Monitor memory usage during long sessions

## 🔒 Security & Privacy

- Face encodings are stored locally (no cloud storage)
- Images are processed on-device only
- HTTPS recommended for production deployment
- Regular backup of attendance data recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔄 Updates & Changelog

### Version 1.0.0
- Initial release with core functionality
- Real-time face recognition
- Web-based dashboard
- Student registration system
- Attendance logging and export

---

**FaceTrack Pro** - Revolutionizing attendance management with AI-powered face recognition technology.