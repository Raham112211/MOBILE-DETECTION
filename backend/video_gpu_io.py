"""
Hardware-Accelerated Video I/O Module.
Uses NVIDIA NVDEC for GPU video decoding (h264_cuvid/hevc_cuvid) and
NVIDIA NVENC for crystal-clear, studio-grade GPU video encoding (h264_nvenc, CQ 16 + Spatial AQ).
"""
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import av
import cv2
import numpy as np

import config


class GPUVideoReader:
    """
    Decodes video frames using NVIDIA NVDEC hardware acceleration.
    Operates in a dedicated background thread to ensure zero inference blocking.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        queue_size: int = 45,
        use_gpu: bool = config.USE_GPU_DECODING,
    ):
        self.video_path = Path(video_path)
        self.queue_size = queue_size
        self.use_gpu = use_gpu

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path.resolve()}")

        # Probe container and extract video stream info
        self.container = None
        self.stream = None
        self.decoder_type = "CPU"
        self._open_container()

        self.width = self.stream.width
        self.height = self.stream.height
        self.fps = float(self.stream.average_rate or self.stream.base_rate or 30.0)
        if self.fps <= 0 or self.fps is None:
            self.fps = 30.0

        # Calculate total frames
        if self.stream.frames > 0:
            self.total_frames = int(self.stream.frames)
        else:
            # Estimate using duration
            duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration else 0.0
            self.total_frames = int(round(duration * self.fps)) if duration > 0 else 0

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.stopped = threading.Event()
        self.read_complete = threading.Event()

        # Start background decoder thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _open_container(self):
        """Open video container with GPU NVDEC options or fallback to CPU."""
        if self.use_gpu:
            for opt in [{"c:v": config.GPU_DECODER_CODEC}, {"c:v": "hevc_cuvid"}, {"hwaccel": "cuda"}]:
                try:
                    c = av.open(str(self.video_path), options=opt)
                    if len(c.streams.video) > 0:
                        self.container = c
                        self.stream = c.streams.video[0]
                        self.decoder_type = f"GPU NVDEC ({opt.get('c:v', 'cuda')})"
                        print(f"[VideoReader] Video opened successfully with {self.decoder_type}")
                        return
                    c.close()
                except Exception:
                    continue

        # Fallback to standard CPU PyAV decoding
        print("[VideoReader] Using standard CPU PyAV decoding...")
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        self.decoder_type = "CPU PyAV"

    def _worker(self):
        """Background thread decodes frames and enqueues them."""
        frame_idx = 0
        try:
            for frame in self.container.decode(self.stream):
                if self.stopped.is_set():
                    break

                # Convert decoded frame to RGB numpy array
                rgb_array = frame.to_ndarray(format="rgb24")
                frame_idx += 1
                item = (frame_idx, rgb_array)

                # Push to queue (wait if full)
                while not self.stopped.is_set():
                    try:
                        self.frame_queue.put(item, timeout=0.08)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            print(f"[VideoReader] Decoding ended or interrupted: {e}")
        finally:
            self.read_complete.set()
            try:
                self.container.close()
            except Exception:
                pass

    def read(self, timeout: float = 0.3) -> Tuple[bool, Optional[int], Optional[np.ndarray]]:
        """
        Get next (frame_number, rgb_array) tuple.
        Returns: (has_frame, frame_idx, rgb_frame)
        """
        while not self.stopped.is_set():
            try:
                frame_idx, frame = self.frame_queue.get(timeout=timeout)
                return True, frame_idx, frame
            except queue.Empty:
                if self.read_complete.is_set() and self.frame_queue.empty():
                    return False, None, None
                continue
        return False, None, None

    def stop(self):
        """Stop reader worker thread."""
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class GPUVideoWriter:
    """
    High-fidelity video writer powered by NVIDIA NVENC (h264_nvenc) hardware acceleration.
    Preserves pristine, studio-grade video quality with CQ 16 and Spatial Adaptive Quantization.
    Runs asynchronously in a background thread to prevent I/O blocking.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float = 30.0,
        use_gpu: bool = config.USE_GPU_ENCODING,
        queue_size: int = 90,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = max(1.0, float(fps))
        self.use_gpu = use_gpu
        self.queue_size = queue_size

        self.container = None
        self.stream = None
        self.encoder_name = "CPU"
        self._init_encoder()

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _init_encoder(self):
        """Initialize NVENC high-quality stream or fallback to libx264."""
        self.container = av.open(str(self.output_path), mode="w")

        if self.use_gpu:
            try:
                # Reference-grade NVENC configuration:
                # CQ=16 + Spatial AQ + Temporal AQ + Preset p6 ensures zero pixelation/blocking
                self.stream = self.container.add_stream("h264_nvenc", rate=int(round(self.fps)))
                self.stream.width = self.width
                self.stream.height = self.height
                self.stream.pix_fmt = "yuv420p"
                self.stream.options = {
                    "preset": config.NVENC_PRESET,
                    "cq": str(config.NVENC_CQ),
                    "spatial-aq": config.NVENC_SPATIAL_AQ,
                    "temporal-aq": config.NVENC_TEMPORAL_AQ,
                }
                self.encoder_name = f"NVIDIA NVENC (CQ {config.NVENC_CQ} Studio Quality)"
                print(f"[VideoWriter] Initialized {self.encoder_name} at {self.width}x{self.height} @ {self.fps:.1f} FPS")
                return
            except Exception as e:
                print(f"[VideoWriter] NVENC notice ({e}). Falling back to libx264...")

        # Fallback to libx264 with visually lossless CRF=17
        self.stream = self.container.add_stream("libx264", rate=int(round(self.fps)))
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "crf": str(config.LIBX264_CRF),
            "preset": "fast",
        }
        self.encoder_name = f"libx264 (CRF {config.LIBX264_CRF} Studio Quality)"
        print(f"[VideoWriter] Initialized {self.encoder_name} at {self.width}x{self.height} @ {self.fps:.1f} FPS")

    def write(self, rgb_frame: np.ndarray):
        """Enqueue an RGB frame for encoding (strictly zero frame loss)."""
        if self.stopped.is_set():
            return
        while not self.stopped.is_set():
            try:
                self.frame_queue.put(rgb_frame, timeout=0.2)
                break
            except queue.Full:
                continue

    def _worker(self):
        """Worker thread to encode and mux video packets."""
        while not self.stopped.is_set() or not self.frame_queue.empty():
            try:
                rgb_frame = self.frame_queue.get(timeout=0.08)
                av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                for packet in self.stream.encode(av_frame):
                    self.container.mux(packet)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VideoWriter] Error during frame encoding: {e}")
                break

        # Flush remaining frames from encoder buffer
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        except Exception:
            pass

        try:
            self.container.close()
            print(f"[VideoWriter] Finished writing: {self.output_path.name}")
        except Exception as e:
            print(f"[VideoWriter] Error closing container: {e}")

    def close(self):
        """Close writer and wait for pending frames to be muxed."""
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10.0)


def draw_hud_annotations(
    rgb_frame: np.ndarray,
    tracked_detections: Any,
    tracker_trails: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> np.ndarray:
    """
    Renders crystal-clear HUD visual annotations on an RGB video frame:
    - High-contrast dual-stroke bounding boxes (visible on both dark and light phone cases)
    - Glowing corner brackets
    - Sleek pill badge placed neatly outside the box (never obscuring the phone screen)
    - Smooth motion trail history curves
    """
    canvas = rgb_frame.copy()
    h, w = canvas.shape[:2]

    # Palettes for track IDs: Distinct high-contrast modern colors
    color_palette = [
        (0, 242, 254),   # Cyber Cyan
        (16, 185, 129),  # Emerald Green
        (245, 158, 11),  # Amber Gold
        (168, 85, 247),  # Electric Purple
        (236, 72, 153),  # Neon Pink
        (59, 130, 246),  # Deep Blue
        (20, 184, 166),  # Teal
        (249, 115, 22),  # Bright Orange
    ]

    if tracked_detections is None or len(tracked_detections) == 0:
        return canvas

    boxes = tracked_detections.xyxy
    confs = tracked_detections.confidence if tracked_detections.confidence is not None else [0.9] * len(boxes)
    tids = tracked_detections.tracker_id if tracked_detections.tracker_id is not None else list(range(len(boxes)))

    # 1. Draw Motion Trails (fading glow curve)
    if tracker_trails:
        for tid in tids:
            if tid in tracker_trails and len(tracker_trails[tid]) > 1:
                trail_pts = tracker_trails[tid]
                col = color_palette[tid % len(color_palette)]
                num_pts = len(trail_pts)
                for k in range(1, num_pts):
                    pt1 = trail_pts[k - 1]
                    pt2 = trail_pts[k]
                    thickness = max(1, int(round((k / num_pts) * 3)))
                    cv2.line(canvas, pt1, pt2, col, thickness, cv2.LINE_AA)

    # 2. Draw Bounding Boxes and Badges
    for box, conf, tid in zip(boxes, confs, tids):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        bw = x2 - x1
        bh = y2 - y1
        b_area = bw * bh

        # Reject oversized boxes (bari cheezen) from HUD rendering
        frame_area = float(w * h)
        if b_area > (frame_area * config.MAX_PHONE_AREA_FRACTION):
            continue
        if bw > (w * config.MAX_PHONE_WIDTH_FRACTION) or bh > (h * config.MAX_PHONE_HEIGHT_FRACTION):
            continue
        if bw <= 6 or bh <= 6:
            continue

        col = color_palette[tid % len(color_palette)]

        # Dual-stroke box: 1px dark border behind + 2px bright neon border
        cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (5, 8, 15), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        # High-tech HUD Corner Brackets
        corner_len = min(18, max(5, int(min(bw, bh) * 0.22)))
        t_corner = 3
        # Top-Left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), col, t_corner, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), col, t_corner, cv2.LINE_AA)
        # Top-Right
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), col, t_corner, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), col, t_corner, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), col, t_corner, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), col, t_corner, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), col, t_corner, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), col, t_corner, cv2.LINE_AA)

        # Badge: "Phone #X • 87%"
        conf_pct = int(round(float(conf) * 100))
        label_text = f"Phone #{tid}  {conf_pct}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Position badge ABOVE box; if too close to top border, place it below box
        if y1 - th - 10 >= 0:
            badge_y1 = y1 - th - 10
            badge_y2 = y1
        else:
            badge_y1 = y2
            badge_y2 = y2 + th + 10

        badge_x1 = x1
        badge_x2 = min(w - 1, x1 + tw + 12)

        # High-contrast pill background with neon border
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (8, 12, 22), -1)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), col, 1, cv2.LINE_AA)

        # White crisp text inside badge
        text_y = badge_y2 - 6 if badge_y1 < y1 else badge_y2 - 5
        cv2.putText(canvas, label_text, (badge_x1 + 6, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas
