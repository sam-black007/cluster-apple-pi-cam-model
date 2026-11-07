"""
web_dashboard.py
================
Localhost Tactical Web Dashboard for the DroneInterceptor.

FastAPI + Uvicorn server serving an HTML5/Tailwind CSS tactical HUD on
http://127.0.0.1:8000.

Features:
  - MJPEG video stream at /video_feed with HUD overlay (lock box, crosshair,
    coordinates, FPS, telemetry).
  - Source toggle: live webcam (source 0 / picamera) vs uploaded MP4 video.
  - GET  /api/telemetry   -> live target lock, coords, yaw/pitch, commands.
  - POST /api/settings    -> dynamic confidence / Kp / Kd / deadband sliders.
  - POST /api/upload_video -> upload & switch to an MP4 clip.
  - Integrates AimController (filtered PD + deadband) for actuator commands.

Usage:
    python web_dashboard.py
    # open http://127.0.0.1:8000 in a browser
"""
import asyncio
import io
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(ROOT, "deploy")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from aim_controller import AimController  # noqa: E402
except Exception:  # pragma: no cover - fallback for non-RPi environments
    AimController = None  # type: ignore

try:
    from dual_loop_tracker import DualLoopTracker  # noqa: E402
except Exception:  # pragma: no cover
    DualLoopTracker = None  # type: ignore

UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="DroneInterceptor Tactical Dashboard")

# ---------------------------------------------------------------------------
# Shared runtime state (thread-safe via locks)
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
app_state = {
    "source": "webcam",        # "webcam" | "video"
    "video_path": None,        # path to uploaded MP4
    "conf": 0.25,
    "kp": 0.04,
    "kd": 0.012,
    "deadband": 1.5,
    "max_cmd": 1.0,
    "running": False,
    "cap": None,               # cv2.VideoCapture
    "tracker": None,           # DualLoopTracker
    "aim": None,               # AimController
    "telemetry": {
        "locked": False,
        "cx": None,
        "cy": None,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "cmd_yaw": 0.0,
        "cmd_pitch": 0.0,
        "fps": 0.0,
        "latency_ms": 0.0,
        "source": "idle",
    },
    "last_frame_jpeg": None,   # latest annotated MJPEG frame
    "clients": set(),          # active WebSocket telemetry clients
}


def _new_aim() -> Optional[AimController]:
    if AimController is None:
        return None
    with state_lock:
        kp = app_state["kp"]
        kd = app_state["kd"]
        db = app_state["deadband"]
        mc = app_state["max_cmd"]
    return AimController(fov_h=62.0, fov_v=48.0,
                         kp=kp, kd=kd,
                         deadband_deg=db, lpf_alpha=0.7,
                         max_cmd=mc)


def _new_tracker() -> Optional[DualLoopTracker]:
    if DualLoopTracker is None:
        return None
    return DualLoopTracker(yolo_period=5)


def _open_capture(source: str, video_path: Optional[str] = None):
    """Open cv2 VideoCapture for webcam (0) or video file."""
    if source == "video" and video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # webcam index 0 (or picamera via V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def _release_cap():
    cap = app_state["cap"]
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    app_state["cap"] = None


# ---------------------------------------------------------------------------
# Video capture / processing loop (background thread)
# ---------------------------------------------------------------------------
_stop_event = threading.Event()
_process_thread: Optional[threading.Thread] = None


def _annotate_frame(frame, track, aim_ctrl, conf):
    """Draw HUD overlay: boxes, crosshair, telemetry text."""
    H, W = frame.shape[:2]
    cx0, cy0 = W // 2, H // 2

    # center crosshair
    cv2.line(frame, (cx0, 0), (cx0, H), (40, 40, 40), 1)
    cv2.line(frame, (0, cy0), (W, cy0), (40, 40, 40), 1)

    # deadband ring (approx. 1.5 deg of 31-deg half-FOV -> px)
    r = int((1.5 / 31.0) * (W / 2))
    cv2.circle(frame, (cx0, cy0), r, (60, 60, 60), 1, cv2.LINE_AA)

    locked = False
    t_cx = t_cy = None
    yaw_deg = pitch_deg = 0.0
    cmd_yaw = cmd_pitch = 0.0

    if track is not None:
        x1, y1, x2, y2 = track["box"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_cx, t_cy = int(track["cx"]), int(track["cy"])
        cls_id = int(track.get("cls", 0))
        color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)  # green drone, orange bird
        label = "DRONE" if cls_id == 0 else "BIRD"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {track.get('conf',0):.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (t_cx, t_cy), 5, (0, 255, 255), -1)
        # vector center -> target
        cv2.line(frame, (cx0, cy0), (t_cx, t_cy), (0, 255, 255), 1, cv2.LINE_AA)
        locked = track.get("locked", False)

        if aim_ctrl is not None:
            yaw_a, pitch_a, cmd_yaw, cmd_pitch, _ = aim_ctrl.compute(
                t_cx, t_cy, W, H)
            yaw_deg = float(np.degrees(yaw_a))
            pitch_deg = float(np.degrees(pitch_a))

    # HUD text panel
    lock_txt = "LOCKED" if locked else "SEARCHING"
    lock_color = (0, 255, 0) if locked else (0, 0, 255)
    fps = app_state["telemetry"]["fps"]
    lat = app_state["telemetry"]["latency_ms"]

    hud_lines = [
        (f"TARGET LOCK: {lock_txt}", lock_color),
        (f"CX,CY: {t_cx if t_cx is not None else '-'}, "
         f"{t_cy if t_cy is not None else '-'}",
         (255, 255, 255)),
        (f"YAW: {yaw_deg:+.1f} deg   PITCH: {pitch_deg:+.1f} deg",
         (255, 255, 255)),
        (f"CMD_YAW: {cmd_yaw:+.3f}   CMD_PITCH: {cmd_pitch:+.3f}",
         (255, 255, 255)),
        (f"FPS: {fps:4.1f}   LAT: {lat:5.1f} ms",
         (255, 255, 255)),
        (f"SRC: {app_state['source'].upper()}  CONF: {conf:.2f}",
         (200, 200, 200)),
    ]
    for i, (txt, col) in enumerate(hud_lines):
        y = 24 + i * 22
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    return {
        "locked": locked,
        "cx": t_cx,
        "cy": t_cy,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "cmd_yaw": cmd_yaw,
        "cmd_pitch": cmd_pitch,
    }


def _process_loop():
    """Background thread: capture -> detect/track -> annotate -> MJPEG buffer."""
    _stop_event.clear()
    while not _stop_event.is_set():
        with state_lock:
            source = app_state["source"]
            video_path = app_state["video_path"]
            conf = app_state["conf"]
            running = app_state["running"]

        if not running:
            time.sleep(0.05)
            continue

        # (re)open capture if needed
        cap = app_state["cap"]
        need_open = cap is None or not cap.isOpened()
        if need_open:
            _release_cap()
            cap = _open_capture(source, video_path)
            with state_lock:
                app_state["cap"] = cap
            if cap is None:
                time.sleep(0.1)
                continue

        # (re)create aim/tracker if settings changed
        aim = app_state["aim"]
        tracker = app_state["tracker"]
        if aim is None:
            aim = _new_aim()
            with state_lock:
                app_state["aim"] = aim
        if tracker is None:
            tracker = _new_tracker()
            with state_lock:
                app_state["tracker"] = tracker

        ok, frame = cap.read()
        if not ok:
            # end of video file -> loop or stop
            if source == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.01)
            continue

        t0 = time.perf_counter()

        # Simulate YOLO detections on every 5th frame via tracker's slow loop.
        # In production this calls the YOLO model; here we drive the tracker
        # with a synthetic detection derived from motion/color to keep the demo
        # self-contained without a trained model dependency.
        run_yolo = (tracker is not None and
                    tracker._frame_idx % tracker.yolo_period == 0)

        # Build a lightweight pseudo-detection from a red target if present
        # (so the demo works without a YOLO model file on this machine).
        yolo_dets = None
        if run_yolo:
            yolo_dets = _pseudo_red_detections(frame, conf)

        track = None
        if tracker is not None:
            track = tracker.update(frame, yolo_dets, run_yolo=run_yolo)

        # Update aim controller settings dynamically
        if aim is not None:
            with state_lock:
                aim.kp = app_state["kp"]
                aim.kd = app_state["kd"]
                aim.deadband_deg = app_state["deadband"]

        hud = _annotate_frame(frame, track, aim, conf)

        t1 = time.perf_counter()
        with state_lock:
            app_state["telemetry"].update({
                "locked": hud["locked"],
                "cx": hud["cx"],
                "cy": hud["cy"],
                "yaw_deg": hud["yaw_deg"],
                "pitch_deg": hud["pitch_deg"],
                "cmd_yaw": hud["cmd_yaw"],
                "cmd_pitch": hud["cmd_pitch"],
                "fps": round(tracker.fps, 1) if tracker else 0.0,
                "latency_ms": round((t1 - t0) * 1000.0, 1),
                "source": source,
            })
            # encode JPEG for MJPEG
            ok2, buf = cv2.imencode(".jpg", frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok2:
                app_state["last_frame_jpeg"] = buf.tobytes()

    _release_cap()


def _pseudo_red_detections(frame, conf):
    """Detect red blobs as pseudo-YOLO drone detections (demo fallback)."""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            if cv2.contourArea(c) < 40:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append((float(x), float(y), float(x + w),
                         float(y + h), 0.9, 0))  # class 0 = drone
        return dets
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(ROOT, "dashboard.html")
    if os.path.exists(html_path):
        return HTMLResponse(content=open(html_path, "r", encoding="utf-8").read())
    # Fallback minimal HTML inline
    return HTMLResponse(content="""
    <html><body><h2>DroneInterceptor Tactical Dashboard</h2>
    <p>Open <code>/video_feed</code> for the MJPEG stream.</p>
    <p>API: <code>GET /api/telemetry</code>, <code>POST /api/settings</code>,
    <code>POST /api/upload_video</code></p>
    </body></html>""")


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the annotated video."""
    async def frame_generator():
        while True:
            with state_lock:
                jpeg = app_state["last_frame_jpeg"]
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.033)  # ~30 FPS cap for the web socket
    return StreamingResponse(frame_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/telemetry")
async def telemetry():
    with state_lock:
        tel = dict(app_state["telemetry"])
    return JSONResponse(content=tel)


class Settings(BaseModel):
    conf: Optional[float] = None
    kp: Optional[float] = None
    kd: Optional[float] = None
    deadband: Optional[float] = None


@app.post("/api/settings")
async def settings(s: Settings):
    with state_lock:
        if s.conf is not None:
            app_state["conf"] = float(s.conf)
        if s.kp is not None:
            app_state["kp"] = float(s.kp)
        if s.kd is not None:
            app_state["kd"] = float(s.kd)
        if s.deadband is not None:
            app_state["deadband"] = float(s.deadband)
        snap = {
            "conf": app_state["conf"],
            "kp": app_state["kp"],
            "kd": app_state["kd"],
            "deadband": app_state["deadband"],
        }
    return JSONResponse(content={"ok": True, "settings": snap})


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload an MP4 and switch the source to video playback."""
    ext = os.path.splitext(file.filename or "upload")[1].lower() or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"upload_{int(time.time())}{ext}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    with state_lock:
        # stop current capture so thread re-opens with new source
        app_state["running"] = False
        _release_cap()
        app_state["source"] = "video"
        app_state["video_path"] = save_path
        app_state["aim"] = None
        app_state["tracker"] = None
        app_state["running"] = True
    return JSONResponse(content={"ok": True, "path": save_path,
                                 "source": "video"})


@app.post("/api/source")
async def set_source(source: str):
    """Switch between 'webcam' and 'video'."""
    with state_lock:
        app_state["running"] = False
        _release_cap()
        app_state["source"] = source
        app_state["aim"] = None
        app_state["tracker"] = None
        app_state["running"] = True
    return JSONResponse(content={"ok": True, "source": source})


@app.post("/api/reset")
async def reset_tracker():
    """Reset the dual-loop tracker state (prevents derivative kick on re-acquire)."""
    with state_lock:
        app_state["tracker"] = None
        app_state["aim"] = None
        app_state["telemetry"]["locked"] = False
        app_state["telemetry"]["cx"] = None
        app_state["telemetry"]["cy"] = None
    return JSONResponse(content={"ok": True})


@app.websocket("/api/telemetry/ws")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    with state_lock:
        app_state["clients"].add(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
            with state_lock:
                tel = dict(app_state["telemetry"])
            await websocket.send_json(tel)
    except WebSocketDisconnect:
        pass
    finally:
        with state_lock:
            app_state["clients"].discard(websocket)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    global _process_thread
    with state_lock:
        app_state["running"] = True
    _stop_event.clear()
    _process_thread = threading.Thread(target=_process_loop, daemon=True)
    _process_thread.start()


@app.on_event("shutdown")
async def _shutdown():
    _stop_event.set()
    if _process_thread is not None:
        _process_thread.join(timeout=2)
    _release_cap()


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
