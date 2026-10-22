# RF-DETR Small Mobile Phone Detection & ByteTrack Studio 📱🚀

End-to-end, high-performance Computer Vision system for real-time **Mobile Phone Detection** and **ByteTrack Multi-Object Tracking** running fully on **NVIDIA RTX GPU**.

---

## ⚡ Key Highlights

1. **Model: RF-DETR Small (Roboflow Detection Transformer)**
   - End-to-end transformer detection architecture combining DINOv2 vision backbone with Neural Architecture Search (NAS) and deformable attention.
   - Operates without Non-Maximum Suppression (NMS) bottlenecks.
   - Filtered specifically for **Mobile / Cell Phone** targets.
   - Accelerated via **PyTorch FP16 Tensor Cores** on `cuda:0` (NVIDIA GeForce RTX 2060 SUPER).

2. **GPU Video Decoding (NVDEC)**
   - Hardware-accelerated frame decoding via PyAV using `h264_cuvid` / `hevc_cuvid` / `hwaccel=cuda`.
   - Asynchronous threaded reader ensures frame decoding never stalls GPU inference.

3. **ByteTrack Sticky Tracking**
   - High-persistence multi-object tracker (`trackers.ByteTrackTracker`).
   - Assigns permanent 1-based IDs (`Phone #1`, `Phone #2`...).
   - Kalman filter motion prediction maintains tracking across hand occlusions, motion blur, and tilt.
   - Visual motion trajectory curves (trails) drawn on video.

4. **Pristine Video Quality Preservation (NVENC CQ 18)**
   - Unlike standard OpenCV writers that introduce blocky pixelation artifacts, output video is encoded using **NVIDIA NVENC (`h264_nvenc`)** with Constant Quality `CQ=18` and `preset=p4`.
   - Full input resolution (1080p / 720p / 4K) is preserved with zero downsampling.
   - Exported video can be directly downloaded in full high-definition from the UI.

5. **Modern Glassmorphic Web UI**
   - Cyberpunk dark theme with glowing cyan and emerald HUD accents.
   - Drag-and-drop video upload zone.
   - Real-time video player with low-latency MJPEG stream.
   - Dynamic Confidence Threshold slider (5% – 95%) with instant live response.
   - Live telemetry cards: FPS, Latency (ms), Active Phones count, Unique Tracks count.

---

## 📁 Project Architecture

```
d:/New mobile model det/
├── backend/
│   ├── app.py                 # Flask server & REST API
│   ├── config.py              # Centralized GPU & pipeline settings
│   ├── detector_rfdetr.py     # RF-DETR Small GPU detector
│   ├── tracker_bytetrack.py   # ByteTrack tracker with Kalman persistence
│   ├── video_gpu_io.py        # NVDEC GPU reader & NVENC GPU writer
│   ├── models/
│   │   └── rf-detr-small.pth  # RF-DETR Small pre-trained model weights (386 MB)
│   ├── data/
│   │   ├── sample.mp4         # High-resolution sample video
│   │   ├── uploads/           # Uploaded user videos
│   │   └── outputs/           # High-quality NVENC annotated output videos
│   ├── static/
│   │   ├── css/style.css      # Cyber HUD glassmorphism UI styles
│   │   └── js/app.js          # Video streaming & telemetry client
│   └── templates/
│       └── index.html         # Web studio dashboard
├── requirements.txt
├── run.bat                    # 1-Click launcher for Windows
└── README.md
```

---

## 🚀 Quick Start

### 1. Launch Server
Double-click `run.bat` or run in terminal:
```powershell
& "C:\Users\raham\.gemini\antigravity-ide\scratch\mobile_detection\.venv\Scripts\python.exe" backend/app.py
```

### 2. Open Web Browser
Navigate to:
```
http://127.0.0.1:5001
```

### 3. Process Video
- Click **"Load Sample Video"** or drag-and-drop any `.mp4`, `.mov`, `.avi`, `.mkv` video.
- Click **"START"** to launch real-time GPU detection and tracking.
- Adjust the **Confidence Slider** in real time as the video plays.
- Click **"Download Video"** to save the crystal-clear annotated MP4.
