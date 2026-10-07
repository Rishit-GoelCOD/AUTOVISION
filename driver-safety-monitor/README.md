# Driver Safety Monitor

An AI-powered computer vision system to monitor driver attentiveness, drowsiness, and distraction in real time using MediaPipe and OpenCV.

## Directory Structure

```
driver-safety-monitor/
├── data/
│   ├── raw/          # Raw dataset files
│   └── own/          # Custom/recorded dataset files
├── models/           # Trained models and weights
├── notebooks/        # Jupyter notebooks for experiments and prototyping
├── results/          # Output plots, logs, and evaluation metrics
├── src/              # Source code modules
│   └── check_setup.py # Environment and webcam verification script
├── requirements.txt  # Project dependencies
└── README.md         # Documentation
```

## Setup & Installation (macOS)

### 1. Navigate to Project Directory
```bash
cd driver-safety-monitor
```

### 2. Create Virtual Environment
Using Python 3.10+ (recommended for MediaPipe compatibility):
```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Setup
Run the environment verification script to test webcam access and verify package versions:
```bash
python src/check_setup.py
```
*(Press `q` anytime during webcam preview to close it early.)*
