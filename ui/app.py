"""
Main Streamlit application for Vehicle Counting System
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import tempfile
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import VideoProcessor, get_config
from detection import VehicleDetector
from tracking import VehicleTracker
from counting import Checkpoint, VehicleCounter
from analysis import CrossingTimeAnalyzer


# Page configuration
st.set_page_config(
    page_title="AI Vehicle Counting System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'checkpoint_configured' not in st.session_state:
        st.session_state.checkpoint_configured = False


def process_video(video_path: str, checkpoint: Checkpoint, 
                 progress_bar, status_text) -> dict:
    """
    Process video and return results
    
    Args:
        video_path: Path to video file
        checkpoint: Configured checkpoint
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        
    Returns:
        Dictionary with processing results
    """
    # Initialize components
    config = get_config()
    
    status_text.text("Loading video...")
    video_processor = VideoProcessor(video_path)
    video_info = video_processor.get_video_info()
    
    status_text.text("Loading AI model...")
    detector = VehicleDetector()
    
    tracker_config = config.get_tracking_config()
    tracker = VehicleTracker(
        max_age=tracker_config.get('max_age', 30),
        min_hits=tracker_config.get('min_hits', 3),
        iou_threshold=tracker_config.get('iou_threshold', 0.3)
    )
    
    counter = VehicleCounter(checkpoint)
    analyzer = CrossingTimeAnalyzer(fps=video_info['fps'])
    
    # Process video
    total_frames = video_info['total_frames']
    processed_frames = []
    frame_count = 0
    
    status_text.text("Processing video...")
    
    for frame_num, frame in video_processor.extract_frames():
        # Update progress
        progress = (frame_num + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_num + 1}/{total_frames}")
        
        # Detect vehicles
        detections = detector.detect_vehicles(frame)
        
        # Track vehicles
        tracks = tracker.update(detections)
        
        # Count crossings
        new_crossings = counter.update(tracks)
        
        # Record crossing times
        for crossing in new_crossings:
            analyzer.record_crossing(
                crossing['track_id'],
                frame_num,
                crossing['class_name'],
                crossing['direction']
            )
        
        # Draw visualizations (sample every 10th frame to save memory)
        if frame_num % 10 == 0:
            vis_frame = frame.copy()
            
            # Draw detections with vehicle type
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                class_name = track.get('class_name', 'vehicle')
                track_id = track['id']
                
                # Color coding by vehicle type
                color_map = {
                    'car': (0, 255, 0),       # Green
                    'motorcycle': (255, 0, 0),  # Blue
                    'bus': (0, 165, 255),     # Orange
                    'truck': (0, 255, 255)    # Yellow
                }
                color = color_map.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with ID and vehicle type
                label = f"ID:{track_id} {class_name.upper()}"
                cv2.putText(vis_frame, label, 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
            
            # Draw checkpoint
            vis_frame = checkpoint.draw(vis_frame)
            
            # Add stats
            cv2.putText(vis_frame, f"Count: {counter.get_total_count()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 255), 2)
            
            processed_frames.append(vis_frame)
        
        frame_count += 1
    
    video_processor.release()
    
    # Compile results
    results = {
        'video_info': video_info,
        'total_count': counter.get_total_count(),
        'direction_counts': counter.get_direction_counts(),
        'statistics': analyzer.get_statistics(),
        'crossing_data': analyzer.get_all_crossing_data(),
        'processed_frames': processed_frames,
        'analyzer': analyzer
    }
    
    return results


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚗 AI-Based Vehicle Counting System</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Video upload
        st.subheader("1. Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a demo video of vehicles crossing a checkpoint"
        )
        
        # Checkpoint configuration
        st.subheader("2. Configure Checkpoint")
        
        checkpoint_type = st.selectbox(
            "Checkpoint Type",
            ["line", "polygon"],
            help="Select checkpoint type"
        )
        
        if checkpoint_type == "line":
            st.write("Define checkpoint line coordinates:")
            st.info("💡 Tip: Set the line to span across ALL lanes for accurate counting")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("Start X", value=50, min_value=0, 
                                    help="Left edge of checkpoint line")
                y1 = st.number_input("Start Y", value=360, min_value=0,
                                    help="Vertical position (same for both points for horizontal line)")
            with col2:
                x2 = st.number_input("End X", value=1230, min_value=0,
                                    help="Right edge of checkpoint line (should cover all lanes)")
                y2 = st.number_input("End Y", value=360, min_value=0,
                                    help="Vertical position (same as Start Y for horizontal line)")
            
            checkpoint = Checkpoint("line")
            checkpoint.set_line((x1, y1), (x2, y2))
            st.session_state.checkpoint_configured = True
        
        # Process button
        st.subheader("3. Process Video")
        process_button = st.button("🚀 Start Processing", 
                                   disabled=uploaded_file is None)
    
    # Main content
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video info
        st.subheader("📹 Video Information")
        try:
            video_processor = VideoProcessor(video_path)
            video_info = video_processor.get_video_info()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", video_info['duration_formatted'])
            col2.metric("FPS", video_info['fps'])
            col3.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            col4.metric("Total Frames", video_info['total_frames'])
            
            video_processor.release()
        except Exception as e:
            st.error(f"Error loading video: {e}")
        
        # Process video
        if process_button and st.session_state.checkpoint_configured:
            st.subheader("🔄 Processing...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                results = process_video(video_path, checkpoint, 
                                      progress_bar, status_text)
                st.session_state.results = results
                st.session_state.processed = True
                
                status_text.text("✅ Processing complete!")
                st.success("Video processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing video: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.processed and st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Results")
        
        results = st.session_state.results
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Vehicles", results['total_count'])
        col2.metric("Avg Crossing Time", 
                   f"{results['statistics']['average_crossing_time']:.2f}s")
        col3.metric("Min Time", 
                   f"{results['statistics']['min_crossing_time']:.2f}s")
        col4.metric("Max Time", 
                   f"{results['statistics']['max_crossing_time']:.2f}s")
        
        # Direction counts
        st.subheader("📈 Direction Analysis")
        direction_counts = results['direction_counts']
        st.bar_chart(direction_counts)
        
        # Vehicle type breakdown
        st.subheader("🚗 Vehicle Type Breakdown")
        if results['crossing_data']:
            import pandas as pd
            df_temp = pd.DataFrame(results['crossing_data'])
            vehicle_counts = df_temp['class_name'].value_counts().to_dict()
            
            # Display vehicle type counts
            cols = st.columns(len(vehicle_counts) if vehicle_counts else 1)
            
            # Color mapping for display
            color_emoji = {
                'car': '🚗',
                'motorcycle': '🏍️',
                'bus': '🚌',
                'truck': '🚚'
            }
            
            for idx, (vehicle_type, count) in enumerate(vehicle_counts.items()):
                emoji = color_emoji.get(vehicle_type, '🚙')
                cols[idx].metric(f"{emoji} {vehicle_type.capitalize()}", count)
        else:
            st.info("No vehicle data available")
        
        # Crossing data table
        st.subheader("📋 Crossing Details")
        if results['crossing_data']:
            import pandas as pd
            df = pd.DataFrame(results['crossing_data'])
            st.dataframe(df, width='stretch')
            
            # Export options
            st.subheader("💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"vehicle_counting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                import json
                json_data = json.dumps({
                    'crossing_records': results['crossing_data'],
                    'statistics': results['statistics']
                }, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    f"vehicle_counting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        else:
            st.info("No vehicles crossed the checkpoint")
        
        # Sample frames
        if results['processed_frames']:
            st.subheader("🎬 Sample Processed Frames")
            num_samples = min(5, len(results['processed_frames']))
            cols = st.columns(num_samples)
            
            for i, col in enumerate(cols):
                if i < len(results['processed_frames']):
                    frame = results['processed_frames'][i]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    col.image(frame_rgb, width='stretch')
    
    else:
        # Welcome message
        st.info("👆 Upload a video and configure the checkpoint to get started!")
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload Video**: Choose a video file showing vehicles crossing a checkpoint
        2. **Configure Checkpoint**: Define the checkpoint line coordinates
        3. **Process**: Click 'Start Processing' to analyze the video
        4. **View Results**: See vehicle counts, crossing times, and statistics
        5. **Export**: Download results as CSV or JSON
        """)


if __name__ == "__main__":
    main()
