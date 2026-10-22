"""
Robust, Sticky ByteTrack Engine for Mobile Phone Detection & Tracking.
Includes Kalman state prediction persistence (prevents phone dropouts during hand occlusions),
Exponential Moving Average (EMA) box smoothing, and continuous trajectory recording.
"""
import collections
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

import config


class PhoneTracker:
    """
    ByteTrack wrapper optimized for mobile phones with Sticky Tracking:
    1. Kalman Persistence: Retains phone tracks across momentary dropouts/occlusions (up to 10 frames).
    2. EMA Box Smoothing: Eliminates bounding box jittering.
    3. Permanent Stable IDs: Assigns clean 1-based IDs (Phone #1, Phone #2...) with zero jumping.
    4. Motion Trails: Records trajectory curves for visual tracking.
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_activation_threshold: float = config.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = config.LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = config.MINIMUM_MATCHING_THRESHOLD,
        persistence_frames: int = config.PERSISTENCE_FRAMES,
        smoothing_alpha: float = config.BOX_SMOOTHING_ALPHA,
        trail_max_len: int = 40,
    ):
        self.frame_rate = max(1.0, float(frame_rate))
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.persistence_frames = persistence_frames
        self.smoothing_alpha = smoothing_alpha
        self.trail_max_len = trail_max_len

        # Initialize internal ByteTrack tracker
        # high_conf_det_threshold activates new tracks; lower proposals associate existing tracks
        self.byte_tracker = ByteTrackTracker(
            lost_track_buffer=self.lost_track_buffer,
            frame_rate=self.frame_rate,
            track_activation_threshold=self.track_activation_threshold,
            high_conf_det_threshold=self.track_activation_threshold,
            minimum_iou_threshold=0.10,
            minimum_consecutive_frames=1,
        )

        # Patch _spawn_new_tracks to allocate IDs immediately on creation
        self._patch_tracker_spawning()

        # Mapping internal IDs to clean user-facing IDs (1, 2, 3...)
        self._internal_to_clean_id: Dict[int, int] = {}
        self._next_clean_id: int = 1

        # Smooth box history: clean_id -> np.ndarray [x1, y1, x2, y2]
        self._smoothed_boxes: Dict[int, np.ndarray] = {}

        # Unique tracks tracking
        self.seen_track_ids: Set[int] = set()
        self.active_track_ids: Set[int] = set()

        # Motion trail history: track_id -> deque of (x, y) center points
        self.trails: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.trail_max_len)
        )

    def _patch_tracker_spawning(self):
        """Patch ByteTrackTracker._spawn_new_tracks to allocate IDs on creation."""
        original_spawn = self.byte_tracker._spawn_new_tracks

        def immediate_spawn(detection_boxes, confidences, unmatched_high_local, high_indices, out_det_indices, out_tracker_ids):
            original_spawn(detection_boxes, confidences, unmatched_high_local, high_indices, out_det_indices, out_tracker_ids)
            for i, tid in enumerate(out_tracker_ids):
                if tid == -1:
                    for trk in reversed(self.byte_tracker.tracks):
                        if getattr(trk, "tracker_id", -1) == -1:
                            trk.tracker_id = self.byte_tracker._allocate_tracker_id()
                            out_tracker_ids[i] = trk.tracker_id
                            break

        self.byte_tracker._spawn_new_tracks = immediate_spawn

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with current detections and smoothly persist active tracks.

        Args:
            detections: sv.Detections from detector.

        Returns:
            sv.Detections: Tracked detections including Kalman-persisted occluded phones.
        """
        # Run ByteTrack association
        tracked: sv.Detections = self.byte_tracker.update(detections)

        active_boxes: List[np.ndarray] = []
        active_confs: List[float] = []
        active_clean_ids: List[int] = []
        matched_internal_ids: Set[int] = set()

        # 1. Process actively matched tracks
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for box, conf, raw_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                rid = int(raw_id)
                if rid < 0:
                    continue

                matched_internal_ids.add(rid)

                # Map to clean 1-based ID
                if rid not in self._internal_to_clean_id:
                    self._internal_to_clean_id[rid] = self._next_clean_id
                    self._next_clean_id += 1

                cid = self._internal_to_clean_id[rid]
                self.seen_track_ids.add(cid)

                # Apply Exponential Moving Average (EMA) box smoothing
                if cid in self._smoothed_boxes:
                    prev_box = self._smoothed_boxes[cid]
                    smooth_box = self.smoothing_alpha * box + (1.0 - self.smoothing_alpha) * prev_box
                else:
                    smooth_box = box.copy()
                self._smoothed_boxes[cid] = smooth_box

                active_boxes.append(smooth_box)
                active_confs.append(float(conf) if conf is not None else 0.85)
                active_clean_ids.append(cid)

                # Record center point for motion trail
                cx = int((smooth_box[0] + smooth_box[2]) / 2.0)
                cy = int((smooth_box[1] + smooth_box[3]) / 2.0)
                self.trails[cid].append((cx, cy))

        # 2. Sticky Kalman Persistence: Keep phones visible during momentary occlusions (up to persistence_frames)
        # Prevents "kuch detect karke chor deta hai" (phones vanishing when partially occluded or blurred)
        if self.persistence_frames > 0 and len(self.byte_tracker.tracks) > 0:
            for trk in self.byte_tracker.tracks:
                rid = getattr(trk, "tracker_id", -1)
                if rid < 0 or rid in matched_internal_ids:
                    continue

                time_lost = getattr(trk, "time_since_update", 0)
                if 1 <= time_lost <= self.persistence_frames:
                    # Get Kalman predicted box
                    pred_box = trk.get_state_bbox()
                    pw = float(pred_box[2] - pred_box[0])
                    ph = float(pred_box[3] - pred_box[1])
                    if pw < 8 or ph < 8:
                        continue
                    aspect = max(pw, ph) / max(1.0, min(pw, ph))
                    if aspect > 4.0:
                        continue

                    if rid not in self._internal_to_clean_id:
                        continue

                    cid = self._internal_to_clean_id[rid]

                    # Smooth predicted box
                    if cid in self._smoothed_boxes:
                        prev_box = self._smoothed_boxes[cid]
                        smooth_box = self.smoothing_alpha * pred_box + (1.0 - self.smoothing_alpha) * prev_box
                    else:
                        smooth_box = pred_box.copy()
                    self._smoothed_boxes[cid] = smooth_box

                    # Slightly decay confidence during prediction
                    decayed_conf = max(0.40, 0.80 - (time_lost * 0.05))
                    active_boxes.append(smooth_box)
                    active_confs.append(decayed_conf)
                    active_clean_ids.append(cid)

                    cx = int((smooth_box[0] + smooth_box[2]) / 2.0)
                    cy = int((smooth_box[1] + smooth_box[3]) / 2.0)
                    self.trails[cid].append((cx, cy))

        self.active_track_ids = set(active_clean_ids)

        if not active_boxes:
            return sv.Detections.empty()

        # Build output Detections with active and persisted tracks
        result = sv.Detections(
            xyxy=np.array(active_boxes, dtype=np.float32),
            confidence=np.array(active_confs, dtype=np.float32),
            class_id=np.array([67] * len(active_boxes), dtype=int),
            data={"class_name": np.array(["cell phone"] * len(active_boxes))},
        )
        result.tracker_id = np.array(active_clean_ids, dtype=int)
        return result

    def get_trail(self, track_id: int) -> List[Tuple[int, int]]:
        """Return motion trail coordinates for a given track ID."""
        return list(self.trails.get(track_id, []))

    def reset(self):
        """Reset tracking state for new video."""
        self.byte_tracker = ByteTrackTracker(
            lost_track_buffer=self.lost_track_buffer,
            frame_rate=self.frame_rate,
            track_activation_threshold=self.track_activation_threshold,
            high_conf_det_threshold=self.track_activation_threshold,
            minimum_iou_threshold=0.10,
            minimum_consecutive_frames=1,
        )
        self._patch_tracker_spawning()
        self._internal_to_clean_id.clear()
        self._next_clean_id = 1
        self._smoothed_boxes.clear()
        self.seen_track_ids.clear()
        self.active_track_ids.clear()
        self.trails.clear()
