"""
RF-DETR Small Object Detector for Mobile Phone Detection.
Runs with PyTorch Tensor Core acceleration (FP16) on NVIDIA RTX GPU.
Optimized for 512x512 native resolution for sub-25ms inference latency.
"""
import time
import logging
from typing import Optional, Union, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import supervision as sv

from rfdetr import RFDETRSmall
import config

logger = logging.getLogger("rfdetr.detector")


class RFDETRPhoneDetector:
    """
    High-accuracy, transformer-based mobile phone detector powered by RF-DETR Small.
    Operates on GPU with FP16 half precision for real-time video processing.
    """

    def __init__(
        self,
        conf_threshold: float = config.DEFAULT_CONF_THRESHOLD,
        target_class_name: str = config.TARGET_CLASS_NAME,
        device: str = config.DEVICE,
        use_fp16: bool = config.ENABLE_FP16,
    ):
        self.conf_threshold = conf_threshold
        self.target_class_name = target_class_name.lower().strip()
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        print(f"[RF-DETR] Initializing RF-DETR Small on device: {self.device} (FP16: {self.use_fp16})...")

        # Enable CuDNN benchmark for maximum GPU throughput
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Instantiate RFDETRSmall
        self.model = RFDETRSmall()

        # Configure model resolution for fine-detail detection
        target_res = config.INFERENCE_SHAPE[0]
        self.model.resolution = target_res
        if hasattr(self.model, "model") and hasattr(self.model.model, "resolution"):
            self.model.model.resolution = target_res

        # Optimize for GPU inference
        if self.use_fp16 and torch.cuda.is_available():
            try:
                print(f"[RF-DETR] Optimizing inference pipeline (torch.float16 at {target_res}x{target_res})...")
                self.model.inference(dtype=torch.float16)
                print(f"[RF-DETR] FP16 optimization ({target_res}x{target_res}) successfully enabled!")
            except Exception as e:
                print(f"[RF-DETR] FP16 optimization notice: {e}")

        # Run warmup pass to ensure zero lag on the very first video frame
        self._warmup()

    def _warmup(self):
        """Warm up GPU CUDA kernels and memory allocators."""
        if not torch.cuda.is_available():
            return
        try:
            print("[RF-DETR] Warming up GPU CUDA kernels...")
            h, w = config.INFERENCE_SHAPE
            dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.detect(dummy_frame, conf=0.5)
            torch.cuda.synchronize()
            print("[RF-DETR] GPU Warmup complete! Detector is ready for real-time video stream.")
        except Exception as e:
            print(f"[RF-DETR] Warmup note: {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        is_bgr: bool = True,
        min_tracking_conf: Optional[float] = None,
    ) -> sv.Detections:
        """
        Detect mobile phones in a video frame with sub-25ms latency.
        Strictly filters out non-phone objects and 'bari cheezen' (large furniture, laptops, TVs).

        Args:
            frame: Numpy array image (H, W, 3).
            conf: Target confidence threshold.
            is_bgr: True if frame is in OpenCV BGR format.
            min_tracking_conf: Optional lower threshold to pass weak proposals for tracking.

        Returns:
            sv.Detections: Supervision detections object filtered strictly for mobile phones.
        """
        target_conf = float(conf) if conf is not None else float(self.conf_threshold)
        
        # Dual-threshold: extract even faint, tilted, or partially occluded proposals
        # so ByteTrack's 2nd stage can associate subtle phone details across frames
        cutoff_conf = min_tracking_conf if min_tracking_conf is not None else max(0.08, target_conf * 0.50)

        # Ensure RGB format for RF-DETR transformer backbone
        if is_bgr:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = frame

        frame_h, frame_w = img_rgb.shape[:2]
        frame_area = float(frame_h * frame_w)
        max_box_area = frame_area * config.MAX_PHONE_AREA_FRACTION
        max_box_w = frame_w * config.MAX_PHONE_WIDTH_FRACTION
        max_box_h = frame_h * config.MAX_PHONE_HEIGHT_FRACTION
        min_box_area = float(config.MIN_PHONE_AREA_PIXELS)
        max_aspect_ratio = float(config.MAX_PHONE_ASPECT_RATIO)

        # Run transformer inference at ultra-high 1024x1024 resolution for deep fine-detail detection
        with torch.inference_mode():
            detections: sv.Detections = self.model.predict(
                images=img_rgb,
                threshold=cutoff_conf,
                shape=config.INFERENCE_SHAPE,
                include_source_image=False,
            )

        if len(detections) == 0:
            return sv.Detections.empty()

        class_names = detections.data.get("class_name", [])
        class_ids = detections.class_id if detections.class_id is not None else []
        boxes = detections.xyxy

        mask = []
        for idx, box in enumerate(boxes):
            c_name = str(class_names[idx]).lower().strip() if idx < len(class_names) else ""
            c_id = int(class_ids[idx]) if idx < len(class_ids) else -1

            # 1. Strict Class Verification:
            # Must strictly be 'cell phone'. Exclude all background categories
            is_cell_phone = (c_name == "cell phone") or ("cell phone" in c_name)
            if not is_cell_phone:
                mask.append(False)
                continue

            # 2. Geometric Dimensions & Size Validation (Eliminate 'bari cheezen'):
            bw = float(box[2] - box[0])
            bh = float(box[3] - box[1])
            b_area = bw * bh

            # Reject if box exceeds maximum mobile phone size (e.g. laptop, table, TV, wall)
            if b_area > max_box_area or bw > max_box_w or bh > max_box_h:
                mask.append(False)
                continue

            # Reject if tiny noise artifact (allows distant handheld phones down to 5px)
            if b_area < min_box_area or bw < 5.0 or bh < 5.0:
                mask.append(False)
                continue

            # Reject abnormal aspect ratio (phones have aspect ratio between 1.0 and 3.8)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect > max_aspect_ratio:
                mask.append(False)
                continue

            mask.append(True)

        mask = np.array(mask, dtype=bool)
        detections = detections[mask]

        # Deduplicate overlapping box proposals without suppressing adjacent phones (IoU 0.45)
        if len(detections) > 1:
            try:
                detections = detections.with_nms(threshold=0.45)
            except Exception:
                pass

        return detections
