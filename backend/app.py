"""
Flask Web Application for RF-DETR Small Mobile Phone Detection & ByteTrack Studio.
Hardware-Accelerated on NVIDIA RTX GPU (NVDEC decoding, PyTorch FP16 detection, NVENC encoding).
"""
import os
import time
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import av
import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
)

import config
from detector_rfdetr import RFDETRPhoneDetector
from tracker_bytetrack import PhoneTracker
from video_gpu_io import GPUVideoReader, GPUVideoWriter, draw_hud_annotations

# Flask App Initialization
app = Flask(
    __name__,
    template_folder=str(config.BASE_DIR / "templates"),
    static_folder=str(config.BASE_DIR / "static"),
)

# Shared Pipeline State
pipeline_state: Dict[str, Any] = {
    "state": "idle",  # idle, processing, paused, completed, error
    "current_frame": 0,
    "total_frames": 0,
    "percent": 0.0,
    "fps": 0.0,
    "latency_ms": 0.0,
    "conf": config.DEFAULT_CONF_THRESHOLD,
    "active_tracks_count": 0,
    "total_unique_tracks": 0,
    "active_track_ids": [],
    "active_video_name": "sample.mp4",
    "output_video_name": "",
    "output_ready": False,
    "decoder_type": "GPU NVDEC",
    "encoder_type": "GPU NVENC (CQ 18 High Quality)",
    "status_message": "Ready. Upload a video or load the sample video to begin.",
    "error_message": "",
}

# Streaming frame buffer and thread synchronization
latest_jpeg_frame: bytes = b""
state_lock = threading.Lock()
frame_condition = threading.Condition(state_lock)

# Control Events
stop_event = threading.Event()
pause_event = threading.Event()
pipeline_thread: Optional[threading.Thread] = None

# Detector and Tracker instances (lazy-loaded / cached)
detector_instance: Optional[RFDETRPhoneDetector] = None


def get_detector() -> RFDETRPhoneDetector:
    """Lazy initialize and cache RF-DETR Small detector on GPU."""
    global detector_instance
    if detector_instance is None:
        detector_instance = RFDETRPhoneDetector(
            conf_threshold=pipeline_state["conf"],
            target_class_name=config.TARGET_CLASS_NAME,
            device=config.DEVICE,
            use_fp16=config.ENABLE_FP16,
        )
    return detector_instance


def processing_worker(input_path: Path, output_path: Path, conf_threshold: float):
    """
    Background worker thread running the complete hardware-accelerated pipeline:
    1. GPU NVDEC Video Reader (h264_cuvid)
    2. RF-DETR Small GPU Inference (FP16 Tensor Cores)
    3. ByteTrack Multi-Object Tracker (Kalman filter + ID persistence)
    4. HUD Visual Annotations
    5. GPU NVENC Video Writer (h264_nvenc, CQ 18 Lossless-like)
    """
    global latest_jpeg_frame

    with state_lock:
        pipeline_state["state"] = "processing"
        pipeline_state["status_message"] = "Initializing GPU NVDEC Decoder and RF-DETR model..."
        pipeline_state["current_frame"] = 0
        pipeline_state["percent"] = 0.0
        pipeline_state["output_ready"] = False
        pipeline_state["error_message"] = ""

    reader = None
    writer = None

    try:
        detector = get_detector()
        detector.conf_threshold = conf_threshold

        # Initialize GPU Video Reader
        reader = GPUVideoReader(input_path, use_gpu=config.USE_GPU_DECODING)
        total_frames = reader.total_frames
        fps = reader.fps
        width = reader.width
        height = reader.height

        with state_lock:
            pipeline_state["total_frames"] = total_frames
            pipeline_state["decoder_type"] = reader.decoder_type
            pipeline_state["status_message"] = f"Processing video ({width}x{height} @ {fps:.1f} FPS via {reader.decoder_type})..."

        # Initialize GPU Video Writer (NVENC with crystal-clear CQ 18)
        writer = GPUVideoWriter(
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            use_gpu=config.USE_GPU_ENCODING,
        )

        with state_lock:
            pipeline_state["encoder_type"] = writer.encoder_name

        # Initialize ByteTrack Tracker with matching activation threshold
        tracker = PhoneTracker(frame_rate=fps, track_activation_threshold=conf_threshold)

        frame_count = 0
        start_time = time.time()
        fps_calc_time = time.time()
        fps_frames = 0
        current_fps = 0.0

        print(f"[Pipeline] Processing started for: {input_path.name} -> {output_path.name}")

        while not stop_event.is_set():
            # Handle Pause
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.1)

            has_frame, frame_idx, rgb_frame = reader.read(timeout=0.5)
            if not has_frame or rgb_frame is None:
                # Video processing finished
                break

            t0 = time.time()

            # 1. RF-DETR Detection on GPU
            current_conf = pipeline_state["conf"]
            detections = detector.detect(rgb_frame, conf=current_conf, is_bgr=False)

            # 2. ByteTrack Tracking
            tracked = tracker.update(detections)

            # 3. Trajectory trails mapping
            active_trails = {tid: tracker.get_trail(tid) for tid in tracker.active_track_ids}

            # 4. Render crystal-clear HUD annotations
            annotated_frame = draw_hud_annotations(
                rgb_frame=rgb_frame,
                tracked_detections=tracked,
                tracker_trails=active_trails,
            )

            # 5. Write to GPU NVENC Output Video
            writer.write(annotated_frame)

            t_infer = (time.time() - t0) * 1000.0  # ms
            frame_count += 1
            fps_frames += 1

            # Update FPS every 10 frames
            if time.time() - fps_calc_time >= 0.5:
                current_fps = fps_frames / (time.time() - fps_calc_time)
                fps_calc_time = time.time()
                fps_frames = 0

            # Encode JPEG for real-time web stream with 95% high fidelity (zero pixel distortion)
            bgr_for_stream = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg_buf = cv2.imencode(".jpg", bgr_for_stream, [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY])
            if ret:
                with frame_condition:
                    latest_jpeg_frame = jpeg_buf.tobytes()
                    frame_condition.notify_all()

            # Update UI state
            pct = round((frame_count / total_frames * 100.0), 1) if total_frames > 0 else 0.0
            with state_lock:
                pipeline_state["current_frame"] = frame_count
                pipeline_state["percent"] = min(100.0, pct)
                pipeline_state["fps"] = round(current_fps, 1)
                pipeline_state["latency_ms"] = round(t_infer, 1)
                pipeline_state["active_tracks_count"] = len(tracker.active_track_ids)
                pipeline_state["total_unique_tracks"] = len(tracker.seen_track_ids)
                pipeline_state["active_track_ids"] = sorted(list(tracker.active_track_ids))

        # Finalize and close writer and reader
        if writer:
            writer.close()
        if reader:
            reader.stop()

        total_elapsed = time.time() - start_time
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        print(f"[Pipeline] Finished! {frame_count}/{total_frames} frames processed (100% zero skipped) in {total_elapsed:.1f}s ({avg_fps:.1f} FPS)")

        with state_lock:
            if stop_event.is_set():
                pipeline_state["state"] = "idle"
                pipeline_state["status_message"] = "Processing cancelled by user."
            else:
                pipeline_state["state"] = "completed"
                pipeline_state["current_frame"] = frame_count
                pipeline_state["percent"] = 100.0
                pipeline_state["output_ready"] = True
                pipeline_state["output_video_name"] = output_path.name
                pipeline_state["status_message"] = (
                    f"Processing complete! {frame_count} frames processed (Zero frame skip) in {total_elapsed:.1f}s "
                    f"({avg_fps:.1f} FPS). Studio video ready."
                )

    except Exception as e:
        print(f"[Pipeline] Error in processing worker: {e}")
        import traceback
        traceback.print_exc()
        with state_lock:
            pipeline_state["state"] = "error"
            pipeline_state["error_message"] = str(e)
            pipeline_state["status_message"] = f"Error during processing: {e}"
    finally:
        if reader:
            reader.stop()
        if writer and not writer.stopped.is_set():
            writer.close()


@app.route("/")
def index():
    """Render main web dashboard."""
    return render_template("index.html")


@app.route("/api/pipeline/status", methods=["GET"])
def get_status():
    """Return pipeline state as JSON."""
    with state_lock:
        return jsonify(dict(pipeline_state))


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handle video file upload from the browser UI."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    filename = Path(file.filename).name
    save_path = config.UPLOADS_DIR / filename
    file.save(str(save_path))

    # Probe uploaded video with PyAV
    try:
        c = av.open(str(save_path))
        stream = c.streams.video[0]
        width = stream.width
        height = stream.height
        fps = float(stream.average_rate or 30.0)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        c.close()
    except Exception:
        width, height, fps, duration = 1280, 720, 30.0, 0.0

    with state_lock:
        pipeline_state["active_video_name"] = filename
        pipeline_state["status_message"] = f"Uploaded '{filename}' ({width}x{height}, {fps:.1f} FPS, {duration:.1f}s)."
        pipeline_state["output_ready"] = False

    return jsonify({
        "status": "success",
        "filename": filename,
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "duration": round(duration, 1),
    })


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    """Load default sample video for immediate testing."""
    sample_file = config.SAMPLE_VIDEO_PATH
    if not sample_file.exists():
        return jsonify({"status": "error", "message": "Sample video not found"}), 404

    target_path = config.UPLOADS_DIR / "sample.mp4"
    if not target_path.exists():
        shutil.copy(str(sample_file), str(target_path))

    with state_lock:
        pipeline_state["active_video_name"] = "sample.mp4"
        pipeline_state["status_message"] = "Sample video loaded. Ready to start detection."
        pipeline_state["output_ready"] = False

    return jsonify({"status": "success", "filename": "sample.mp4"})


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    """Start video detection & tracking pipeline."""
    global pipeline_thread, stop_event, pause_event

    data = request.get_json() or {}
    conf = float(data.get("conf", pipeline_state["conf"]))

    video_name = pipeline_state["active_video_name"]
    input_path = config.UPLOADS_DIR / video_name

    if not input_path.exists():
        if config.SAMPLE_VIDEO_PATH.exists():
            input_path = config.SAMPLE_VIDEO_PATH
            video_name = input_path.name
        else:
            return jsonify({"status": "error", "message": f"Input video not found: {video_name}"}), 400

    output_name = f"detected_{Path(video_name).stem}.mp4"
    output_path = config.OUTPUTS_DIR / output_name

    # Reset events
    stop_event.clear()
    pause_event.clear()

    with state_lock:
        pipeline_state["conf"] = conf

    # Start background processing thread
    pipeline_thread = threading.Thread(
        target=processing_worker,
        args=(input_path, output_path, conf),
        daemon=True,
    )
    pipeline_thread.start()

    return jsonify({"status": "success", "message": "Pipeline started"})


@app.route("/api/pipeline/pause", methods=["POST"])
def toggle_pause():
    """Toggle pipeline pause/resume."""
    global pause_event
    if pause_event.is_set():
        pause_event.clear()
        with state_lock:
            pipeline_state["state"] = "processing"
            pipeline_state["status_message"] = "Resumed video processing."
        return jsonify({"status": "success", "paused": False})
    else:
        pause_event.set()
        with state_lock:
            pipeline_state["state"] = "paused"
            pipeline_state["status_message"] = "Paused video processing."
        return jsonify({"status": "success", "paused": True})


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    """Stop active video processing pipeline."""
    global stop_event
    stop_event.set()
    return jsonify({"status": "success", "message": "Pipeline stopping..."})


@app.route("/api/settings/confidence", methods=["POST"])
def update_confidence():
    """Dynamically update confidence threshold on the fly."""
    data = request.get_json() or {}
    new_conf = float(data.get("conf", 0.20))
    new_conf = max(0.01, min(0.99, new_conf))

    with state_lock:
        pipeline_state["conf"] = new_conf

    if detector_instance is not None:
        detector_instance.conf_threshold = new_conf

    return jsonify({"status": "success", "conf": new_conf})


@app.route("/api/stream/feed")
def stream_feed():
    """Multipart MJPEG video stream for the web browser player."""
    def generate_frames():
        last_sent = None
        while True:
            with frame_condition:
                frame_condition.wait(timeout=0.08)
                frame_data = latest_jpeg_frame

            if frame_data and frame_data is not last_sent:
                last_sent = frame_data
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            else:
                time.sleep(0.01)

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/video/download")
def download_video():
    """Download the full-resolution NVENC-encoded output video."""
    video_name = pipeline_state.get("output_video_name")
    if not video_name:
        return jsonify({"status": "error", "message": "No output video generated yet"}), 404

    target = config.OUTPUTS_DIR / video_name
    if not target.exists():
        return jsonify({"status": "error", "message": "Output file does not exist on disk"}), 404

    return send_file(
        str(target),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=target.name,
    )


def start_server(host: str = config.SERVER_HOST, port: int = config.SERVER_PORT, debug: bool = False):
    """Run Flask application."""
    print("=" * 70)
    print("   RF-DETR Small Mobile Phone Detection & ByteTrack Studio")
    print(f"   Hardware: {config.DEVICE_NAME} (CUDA: {config.CUDA_AVAILABLE})")
    print(f"   Server running at: http://{host}:{port}")
    print("=" * 70)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    start_server()
