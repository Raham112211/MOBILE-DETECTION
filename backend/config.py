"""
Configuration module for RF-DETR Small Mobile Phone Detection & ByteTrack Studio.
Optimized for NVIDIA GPU acceleration (NVDEC decoding, Tensor Core inference, NVENC encoding).
"""
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
SAMPLE_VIDEO_PATH = DATA_DIR / "sample.mp4"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Model Settings
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"

# RF-DETR Model
MODEL_LOCAL_PATH = MODELS_DIR / "rf-detr-small.pth"
ROBOFLOW_GLOBAL_MODEL = Path(os.path.expanduser("~")) / ".roboflow" / "models" / "rf-detr-small.pth"
MODEL_PATH = str(MODEL_LOCAL_PATH if MODEL_LOCAL_PATH.exists() else ROBOFLOW_GLOBAL_MODEL)

TARGET_CLASS_NAME = "cell phone"
DEFAULT_CONF_THRESHOLD = 0.22
INFERENCE_SHAPE = (1024, 1024)   # Ultra-HD 1024x1024 resolution: deep per-frame inspection with 4x pixel density
ENABLE_FP16 = True

# Phone Physical Geometry & False Positive Suppression
# Strictly eliminates 'bari cheezen' (laptops, TVs, tables, persons, background furniture)
MAX_PHONE_AREA_FRACTION = 0.14     # Phone cannot exceed 14% of the video frame area
MAX_PHONE_WIDTH_FRACTION = 0.42    # Phone cannot exceed 42% of the frame width
MAX_PHONE_HEIGHT_FRACTION = 0.52   # Phone cannot exceed 52% of the frame height
MIN_PHONE_AREA_PIXELS = 36         # Minimum valid bounding box area in pixels (allows distant/small phones)
MAX_PHONE_ASPECT_RATIO = 3.8       # Maximum aspect ratio max(w/h, h/w) (phones are handheld rectangles)

# Tracking Settings (ByteTrack + Kalman Sticky Persistence)
TRACK_ACTIVATION_THRESHOLD = 0.20
LOST_TRACK_BUFFER = 60             # Keep tracking active for up to 2 seconds of occlusion
MINIMUM_MATCHING_THRESHOLD = 0.80
PERSISTENCE_FRAMES = 15            # Keep tracking phone across occlusions/drops for up to ~0.50s
BOX_SMOOTHING_ALPHA = 0.80         # Smooth moving average for zero box jitter

# Video I/O & Encoding Quality
USE_GPU_DECODING = True
GPU_DECODER_CODEC = "h264_cuvid"

# High-Quality Output Encoding: NVENC (h264_nvenc) with visually lossless CQ 16 + Spatial AQ
USE_GPU_ENCODING = True
NVENC_PRESET = "p6"              # Turing NVENC high quality preset
NVENC_CQ = 16                    # Reference-grade visual quality (zero pixelation/blocking)
NVENC_SPATIAL_AQ = "1"           # Spatial Adaptive Quantization for fine detail preservation
NVENC_TEMPORAL_AQ = "1"          # Temporal Adaptive Quantization for inter-frame stability
LIBX264_CRF = 17                 # CPU fallback visually lossless CRF

# Live Stream Quality
STREAM_JPEG_QUALITY = 95         # High fidelity 95% JPEG quality for zero pixel distortion

# Web Server Settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5001
