"""
dual_loop_tracker.py
====================
Dual-loop target tracker for the interceptor drone.

Architecture:
  - Slow loop: YOLO11n detector runs every N frames (default 5) or when the
    track is lost, providing drift-free ground-truth bounding boxes.
  - Fast loop: OpenCV CSRT tracker runs on every intermediate frame (~2-3 ms,
    60+ FPS on a laptop CPU) to propagate the target between YOLO updates.
  - Re-sync: On every YOLO frame, the CSRT box is snapped to the matched YOLO
    detection to eliminate long-term drift.

Usage:
  tracker = DualLoopTracker(yolo_period=5)
  while True:
      frame = grab_frame()
      track = tracker.update(frame, yolo_detections)   # yolo_detections None off-frame
      if track:
          cx, cy = track["cx"], track["cy"]
          cmd = aim.compute(cx, cy, W, H)
"""
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# A detection tuple: (x1, y1, x2, y2, confidence, class_id)
Detection = Tuple[float, float, float, float, float, int]
Track = Dict[str, object]


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class DualLoopTracker:
    """
    Dual-loop object tracker: slow YOLO loop + fast CSRT loop.

    Parameters
    ----------
    yolo_period : int
        Run YOLO every N frames (default 5). Intermediate frames use CSRT.
    iou_threshold : float
        Minimum IoU to match a YOLO detection to the CSRT track for re-sync.
    lost_thresh : int
        Consecutive frames without a YOLO match before the track is declared lost.
    max_age : int
        Frames to keep a lost track alive before fully dropping it.
    """

    def __init__(self, yolo_period: int = 5,
                 iou_threshold: float = 0.30,
                 lost_thresh: int = 10,
                 max_age: int = 30):
        self.yolo_period = max(1, int(yolo_period))
        self.iou_threshold = float(iou_threshold)
        self.lost_thresh = int(lost_thresh)
        self.max_age = int(max_age)

        # CSRT tracker (accurate, ~2-3 ms/frame on CPU)
        self._tracker: Optional[cv2.Tracker] = None
        self._tracker_box: Optional[Tuple[float, float, float, float]] = None
        self._tracker_ok = False

        # Loop counters / state
        self._frame_idx = 0
        self._lost_counter = 0
        self._tracker_age = 0

        self._last_box: Optional[Tuple[float, float, float, float]] = None
        self._last_conf = 0.0
        self._last_cls = -1
        self._locked = False

        # FPS bookkeeping
        self._fps_times: Deque[float] = deque(maxlen=60)
        self._yolo_ms: Deque[float] = deque(maxlen=60)
        self._csrt_ms: Deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray,
               yolo_detections: Optional[List[Detection]] = None,
               run_yolo: bool = False) -> Optional[Track]:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        yolo_detections : list of (x1,y1,x2,y2,conf,cls) or None
            YOLO detections from the slow loop (only used when run_yolo=True).
        run_yolo : bool
            True when YOLO was executed this frame (every N frames or on loss).

        Returns
        -------
        track : dict or None
            {"cx","cy","box","conf","cls","locked","source":"yolo"|"csrt"}
            or None if no target is being tracked.
        """
        t0 = time.perf_counter()
        self._frame_idx += 1
        H, W = frame.shape[:2]

        track: Optional[Track] = None

        # ---------------------------------------------------------- #
        # Slow loop: YOLO re-detection / re-sync
        # ---------------------------------------------------------- #
        run_slow = run_yolo or (self._frame_idx % self.yolo_period == 0)
        if run_slow and yolo_detections:
            ts = time.perf_counter()
            best = self._match_best_yolo(yolo_detections)
            te = time.perf_counter()
            self._yolo_ms.append((te - ts) * 1000.0)

            if best is not None:
                box, conf, cls = best
                cx, cy = _center(box)
                self._init_csrt(frame, box)
                self._last_box = box
                self._last_conf = conf
                self._last_cls = cls
                self._locked = True
                self._lost_counter = 0
                track = {
                    "cx": cx, "cy": cy, "box": box,
                    "conf": conf, "cls": int(cls),
                    "locked": True, "source": "yolo",
                }
            else:
                # YOLO ran but nothing matched: track is lost
                self._locked = False
                self._lost_counter += 1
        elif run_slow and not yolo_detections:
            self._locked = False
            self._lost_counter += 1

        # ---------------------------------------------------------- #
        # Fast loop: CSRT propagation on intermediate frames
        # ---------------------------------------------------------- #
        if track is None and self._tracker_ok and self._tracker_box is not None:
            ts = time.perf_counter()
            ok, new_box = self._tracker.update(frame)
            te = time.perf_counter()
            self._csrt_ms.append((te - ts) * 1000.0)

            if ok:
                x1, y1, x2, y2 = new_box
                # Clamp to frame bounds
                x1 = max(0.0, min(x1, W))
                y1 = max(0.0, min(y1, H))
                x2 = max(0.0, min(x2, W))
                y2 = max(0.0, min(y2, H))
                box = (x1, y1, x2, y2)
                cx, cy = _center(box)

                # Validate box size (reject degenerate)
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    self._tracker_box = box
                    self._last_box = box
                    self._lost_counter = 0
                    self._tracker_age = 0
                    track = {
                        "cx": cx, "cy": cy, "box": box,
                        "conf": self._last_conf, "cls": self._last_cls,
                        "locked": self._locked, "source": "csrt",
                    }
                else:
                    self._tracker_ok = False
            else:
                self._tracker_ok = False

        # ---------------------------------------------------------- #
        # Track-loss housekeeping
        # ---------------------------------------------------------- #
        if track is None:
            self._lost_counter += 1
            self._tracker_age += 1
            if self._lost_counter >= self.lost_thresh:
                self._locked = False
                self._tracker_ok = False
                self._tracker_box = None
            elif self._tracker_age > self.max_age:
                self._locked = False
                self._tracker_ok = False
                self._tracker_box = None

        # FPS bookkeeping
        t1 = time.perf_counter()
        self._fps_times.append(t1 - t0)
        return track

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _match_best_yolo(self,
                         dets: List[Detection]
                         ) -> Optional[Tuple[Tuple[float, float, float, float],
                                             float, int]]:
        """Pick the YOLO detection closest to the current track (IoU match)."""
        if self._last_box is not None and self._locked:
            best_iou = self.iou_threshold
            best = None
            for d in dets:
                box = d[:4]
                iou = _iou(self._last_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best = box
            if best is not None:
                # find conf/cls for chosen box
                for d in dets:
                    if d[:4] == best:
                        return best, d[4], d[5]
                return best, best[4] if len(best) > 4 else 0.0, int(best[5]) if len(best) > 5 else -1
            return None

        # No active track: pick the highest-confidence detection
        best = None
        best_conf = -1.0
        for d in dets:
            if d[4] > best_conf:
                best_conf = d[4]
                best = d
        if best is not None:
            return best[:4], best[4], int(best[5]) if len(best) > 5 else -1
        return None

    def _init_csrt(self, frame: np.ndarray,
                   box: Tuple[float, float, float, float]) -> None:
        (major, minor) = cv2.__version__.split(".")[:2]
        try:
            if int(major) >= 4 and int(minor) >= 5:
                self._tracker = cv2.TrackerCSRT_create()
            else:
                self._tracker = cv2.TrackerKCF_create()
        except Exception:
            self._tracker = cv2.TrackerKCF_create()

        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        opencv_box = (float(x1), float(y1), float(w), float(h))
        self._tracker_ok = self._tracker.init(frame, opencv_box)
        self._tracker_box = box

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        if not self._fps_times:
            return 0.0
        return 1.0 / (sum(self._fps_times) / len(self._fps_times))

    @property
    def yolo_ms(self) -> float:
        if not self._yolo_ms:
            return 0.0
        return sum(self._yolo_ms) / len(self._yolo_ms)

    @property
    def csrt_ms(self) -> float:
        if not self._csrt_ms:
            return 0.0
        return sum(self._csrt_ms) / len(self._csrt_ms)

    @property
    def locked(self) -> bool:
        return self._locked

    def reset(self) -> None:
        """Reset all tracker state (call on target re-acquisition)."""
        self._tracker = None
        self._tracker_ok = False
        self._tracker_box = None
        self._frame_idx = 0
        self._lost_counter = 0
        self._tracker_age = 0
        self._last_box = None
        self._last_conf = 0.0
        self._last_cls = -1
        self._locked = False
        self._fps_times.clear()
        self._yolo_ms.clear()
        self._csrt_ms.clear()
