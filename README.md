# AI-Based Vehicle Counting and Crossing Time Analysis System

An intelligent computer vision system that analyzes vehicle traffic by detecting, tracking, and counting vehicles crossing a checkpoint, while calculating their crossing times.

## Features

- 🎥 Video upload and processing
- 🚗 AI-powered vehicle detection (YOLOv8)
- 🎯 Multi-object tracking with unique IDs
- 📍 Customizable checkpoint definition
- 📊 Automated vehicle counting
- ⏱️ Crossing time analysis
- 📈 Results visualization and export

## Project Structure

```
AI-Based Vehicle Counting and Crossing Time Analysis System/
├── data/
│   ├── videos/           # Input demo videos
│   ├── models/           # Pre-trained models
│   └── results/          # Output data (CSV, JSON)
├── src/
│   ├── detection/        # Vehicle detection module
│   ├── tracking/         # Vehicle tracking module
│   ├── counting/         # Counting logic
│   ├── analysis/         # Crossing time analysis
│   └── utils/            # Helper functions
├── ui/
│   ├── app.py           # Main Streamlit application
│   └── components/      # UI components
├── tests/               # Unit tests
├── config/
│   └── config.yaml      # Configuration file
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run ui/app.py
```

## Technology Stack

- **Python 3.8+**
- **OpenCV** - Video processing
- **YOLOv8** - Object detection
- **DeepSORT** - Object tracking
- **Streamlit** - Web interface
- **Pandas** - Data management
- **Matplotlib/Plotly** - Visualization

## License

MIT License
