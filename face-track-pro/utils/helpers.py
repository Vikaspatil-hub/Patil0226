import os
import pandas as pd
from datetime import datetime, date, timedelta
import csv

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'face-track-pro/dataset',
        'face-track-pro/attendance',
        'face-track-pro/templates',
        'face-track-pro/static',
        'face-track-pro/utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize attendance.csv if it doesn't exist
    attendance_file = 'face-track-pro/attendance/attendance.csv'
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time', 'Status'])

def format_time(timestamp=None):
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%H:%M:%S")

def format_date(date_obj=None):
    """Format date for display"""
    if date_obj is None:
        date_obj = date.today()
    return date_obj.strftime("%Y-%m-%d")

def format_datetime(datetime_obj=None):
    """Format datetime for display"""
    if datetime_obj is None:
        datetime_obj = datetime.now()
    return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

def get_attendance_stats():
    """Get comprehensive attendance statistics"""
    attendance_file = 'face-track-pro/attendance/attendance.csv'
    
    stats = {
        'total_students': 0,
        'present_today': 0,
        'absent_today': 0,
        'late_today': 0,
        'attendance_rate': 0.0,
        'recent_attendance': [],
        'today_attendance': []
    }
    
    try:
        # Get registered students count
        dataset_dir = 'face-track-pro/dataset'
        if os.path.exists(dataset_dir):
            stats['total_students'] = len([d for d in os.listdir(dataset_dir) 
                                         if os.path.isdir(os.path.join(dataset_dir, d))])
        
        # Read attendance data
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            
            if not df.empty:
                today = date.today().strftime("%Y-%m-%d")
                
                # Today's attendance
                today_df = df[df['Date'] == today]
                stats['present_today'] = len(today_df)
                stats['absent_today'] = max(0, stats['total_students'] - stats['present_today'])
                
                # Calculate attendance rate
                if stats['total_students'] > 0:
                    stats['attendance_rate'] = (stats['present_today'] / stats['total_students']) * 100
                
                # Today's attendance list
                stats['today_attendance'] = today_df.to_dict('records')
                
                # Recent attendance (last 7 days)
                recent_dates = [(date.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                recent_df = df[df['Date'].isin(recent_dates)]
                stats['recent_attendance'] = recent_df.tail(10).to_dict('records')
                
                # Calculate late arrivals (after 9:00 AM)
                today_df['Time'] = pd.to_datetime(today_df['Time'], format='%H:%M:%S').dt.time
                late_time = pd.to_datetime('09:00:00', format='%H:%M:%S').time()
                stats['late_today'] = len(today_df[today_df['Time'] > late_time])
        
    except Exception as e:
        print(f"Error calculating attendance stats: {e}")
    
    return stats

def get_weekly_attendance():
    """Get weekly attendance statistics"""
    attendance_file = 'face-track-pro/attendance/attendance.csv'
    weekly_stats = {}
    
    try:
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            
            # Get last 7 days
            for i in range(7):
                check_date = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
                day_attendance = len(df[df['Date'] == check_date])
                weekly_stats[check_date] = day_attendance
    
    except Exception as e:
        print(f"Error getting weekly attendance: {e}")
    
    return weekly_stats

def export_attendance_report(start_date=None, end_date=None):
    """Export attendance report for a date range"""
    attendance_file = 'face-track-pro/attendance/attendance.csv'
    
    if start_date is None:
        start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    
    try:
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            
            # Filter by date range
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            filtered_df = df[mask]
            
            # Generate report
            report = {
                'total_records': len(filtered_df),
                'unique_students': filtered_df['Name'].nunique(),
                'date_range': f"{start_date} to {end_date}",
                'daily_summary': filtered_df.groupby('Date').size().to_dict(),
                'student_summary': filtered_df.groupby('Name').size().to_dict()
            }
            
            return report, filtered_df
    
    except Exception as e:
        print(f"Error generating attendance report: {e}")
        return None, None

def validate_image_for_face(image_path):
    """Validate if an image contains a detectable face"""
    try:
        import face_recognition
        
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        return len(face_locations) > 0, len(face_locations)
    
    except Exception as e:
        print(f"Error validating image {image_path}: {e}")
        return False, 0

def cleanup_old_logs(days_to_keep=90):
    """Clean up old attendance logs"""
    attendance_file = 'face-track-pro/attendance/attendance.csv'
    
    try:
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            
            cutoff_date = (date.today() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            
            # Keep only recent records
            recent_df = df[df['Date'] >= cutoff_date]
            
            if len(recent_df) < len(df):
                # Backup old file
                backup_file = f"face-track-pro/attendance/attendance_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(backup_file, index=False)
                
                # Save cleaned data
                recent_df.to_csv(attendance_file, index=False)
                
                print(f"Cleaned up {len(df) - len(recent_df)} old records")
                print(f"Backup saved to {backup_file}")
    
    except Exception as e:
        print(f"Error cleaning up logs: {e}")

def get_system_status():
    """Get system status information"""
    status = {
        'model_exists': os.path.exists('face-track-pro/local.pkl'),
        'dataset_size': 0,
        'attendance_records': 0,
        'last_training': None
    }
    
    try:
        # Check dataset size
        dataset_dir = 'face-track-pro/dataset'
        if os.path.exists(dataset_dir):
            total_images = 0
            for root, dirs, files in os.walk(dataset_dir):
                total_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            status['dataset_size'] = total_images
        
        # Check attendance records
        attendance_file = 'face-track-pro/attendance/attendance.csv'
        if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
            df = pd.read_csv(attendance_file)
            status['attendance_records'] = len(df)
        
        # Check last training time
        model_file = 'face-track-pro/local.pkl'
        if os.path.exists(model_file):
            import time
            last_modified = os.path.getmtime(model_file)
            status['last_training'] = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
    
    except Exception as e:
        print(f"Error getting system status: {e}")
    
    return status