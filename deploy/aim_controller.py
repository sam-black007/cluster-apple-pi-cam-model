"""
aim_controller.py  (Raspberry Pi 5)
=================================
Target-tracking FLIGHT CONTROL loop for the interceptor drone.

Pipeline:
  Pi camera -> INT8 YOLO11n (drone detection + tracking)
            -> target angle from camera center (yaw/pitch)
            -> P-controller -> pitch/yaw/roll commands
            -> link to flight controller

The camera + Pi are assumed ONBOARD the interceptor drone.

Links (--link):
  print    - just show computed angles/commands (default, no hardware)
  serial   - write "yaw,pitch,roll" lines over UART (pyserial)
  mavlink  - SET_ATTITUDE_TARGET over UART (pymavlink; Pixhawk/ArduPilot/PX4)
  pwm      - pulse widths on GPIO pins (pigpio; direct ESC/receiver)

Roll is kept at ~0 (the FC auto-levels). Yaw aims left/right, pitch aims up/down.

Usage:
  python aim_controller.py --link print          # verify logic, no hardware
  python aim_controller.py --link serial --port COM3 --baud 115200
  python aim_controller.py --link mavlink --port /dev/ttyAMA0 --baud 57600
  python aim_controller.py --link pwm --pins 17,18,19
"""

import argparse
import time
import math
import json

import numpy as np
import cv2

try:
    from rpi_infer import (
        load_interpreter, open_source, preprocess, quantize_input,
        decode, nms, DroneTracker, CLASS_NAMES, DRONE_ID, IMG_SIZE,
    )
except ImportError:
    from deploy.rpi_infer import (
        load_interpreter, open_source, preprocess, quantize_input,
        decode, nms, DroneTracker, CLASS_NAMES, DRONE_ID, IMG_SIZE,
    )

# ---------------------------------------------------------------------------
# Angle / command math
# ---------------------------------------------------------------------------
class AimController:
    def __init__(self, fov_h=62.0, fov_v=48.0, kp=0.04, kd=0.012,
                 deadband_deg=1.5, lpf_alpha=0.7, max_cmd=1.0):
        self.fov_h = math.radians(fov_h)
        self.fov_v = math.radians(fov_v)
        self.kp = kp
        self.kd = kd
        self.deadband_deg = deadband_deg
        self.lpf_alpha = lpf_alpha
        self.max_cmd = max_cmd
        self.reset()

    def reset(self):
        """Reset timing/error/derivative state (avoids derivative kicks)."""
        self.last_time = None
        self.last_err_yaw = 0.0
        self.last_err_pitch = 0.0
        self.prev_d_yaw = 0.0
        self.prev_d_pitch = 0.0

    @staticmethod
    def _apply_deadband(err, deadband_deg):
        if abs(err) <= deadband_deg:
            return 0.0
        return math.copysign(abs(err) - deadband_deg, err)

    def compute(self, cx, cy, W, H):
        """Return (yaw_angle, pitch_angle, cmd_yaw, cmd_pitch, cmd_roll)."""
        now = time.perf_counter()
        dt = now - self.last_time if self.last_time is not None else 0.0
        self.last_time = now

        # normalized offset from center, [-1, 1]
        nx = (cx - W / 2) / (W / 2)
        ny = (cy - H / 2) / (H / 2)

        yaw_angle = nx * (self.fov_h / 2.0)      # +right (radians)
        pitch_angle = -ny * (self.fov_v / 2.0)   # +up (radians)

        # work in degrees; apply deadband per axis
        err_yaw = self._apply_deadband(math.degrees(yaw_angle), self.deadband_deg)
        err_pitch = self._apply_deadband(math.degrees(pitch_angle), self.deadband_deg)

        # filtered derivative of error
        if dt > 0.0:
            raw_d_yaw = (err_yaw - self.last_err_yaw) / dt
            raw_d_pitch = (err_pitch - self.last_err_pitch) / dt
        else:
            raw_d_yaw = 0.0
            raw_d_pitch = 0.0

        d_yaw_filt = self.lpf_alpha * raw_d_yaw + (1.0 - self.lpf_alpha) * self.prev_d_yaw
        d_pitch_filt = self.lpf_alpha * raw_d_pitch + (1.0 - self.lpf_alpha) * self.prev_d_pitch

        self.prev_d_yaw = d_yaw_filt
        self.prev_d_pitch = d_pitch_filt
        self.last_err_yaw = err_yaw
        self.last_err_pitch = err_pitch

        # PD command clamped to [-max_cmd, max_cmd]
        cmd_yaw = float(np.clip(self.kp * err_yaw + self.kd * d_yaw_filt,
                                -self.max_cmd, self.max_cmd))
        cmd_pitch = float(np.clip(self.kp * err_pitch + self.kd * d_pitch_filt,
                                  -self.max_cmd, self.max_cmd))
        cmd_roll = 0.0
        return yaw_angle, pitch_angle, cmd_yaw, cmd_pitch, cmd_roll


# ---------------------------------------------------------------------------
# Link backends
# ---------------------------------------------------------------------------
class PrintLink:
    def send(self, yaw, pitch, roll, extra=""):
        print(f"[CMD] yaw={yaw:+.3f} pitch={pitch:+.3f} roll={roll:+.3f} {extra}")
    def close(self):
        pass


class SerialLink:
    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
    def send(self, yaw, pitch, roll, extra=""):
        self.ser.write((json.dumps({"yaw": yaw, "pitch": pitch,
                                    "roll": roll}) + "\n").encode())
    def close(self):
        self.ser.close()


class MavlinkLink:
    def __init__(self, port, baud):
        from pymavlink import mavutil
        self.master = mavutil.mavlink_connection(port, baud=baud)
        self.master.wait_heartbeat()
        print("[MAVLINK] heartbeat from FC")
    def send(self, yaw, pitch, roll, extra=""):
        # Rate control: turn toward target using body yaw/pitch rates.
        # yaw/pitch here are normalized commands in [-1, 1].
        MAX_YAW_RATE = 30.0    # deg/s at full command
        MAX_PITCH_RATE = 20.0  # deg/s at full command
        self.master.mav.set_attitude_target_send(
            int(time.time() * 1e3) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            0b01110000,  # ignore attitude, use body rates + thrust
            [1.0, 0.0, 0.0, 0.0],  # quaternion (unused)
            math.radians(roll * 0.0),       # body roll rate (level)
            math.radians(pitch * MAX_PITCH_RATE),
            math.radians(yaw * MAX_YAW_RATE),
            0.5,  # thrust (hover)
        )
    def close(self):
        self.master.close()


class PwmLink:
    def __init__(self, pins):
        import pigpio
        self.pi = pigpio.pi()
        self.pins = [int(p) for p in pins.split(",")]
        self.min_us, self.max_us = 1000, 2000
    def send(self, yaw, pitch, roll, extra=""):
        for val, pin in zip((yaw, pitch, roll), self.pins):
            us = int(1500 + val * 500)  # 1500 = neutral
            us = max(self.min_us, min(self.max_us, us))
            self.pi.set_servo_pulsewidth(pin, us)
    def close(self):
        for pin in self.pins:
            self.pi.set_servo_pulsewidth(pin, 0)
        self.pi.stop()


def make_link(args):
    if args.link == "print":
        return PrintLink()
    if args.link == "serial":
        return SerialLink(args.port, args.baud)
    if args.link == "mavlink":
        return MavlinkLink(args.port, args.baud)
    if args.link == "pwm":
        return PwmLink(args.pins)
    raise SystemExit(f"Unknown link: {args.link}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="picamera")
    ap.add_argument("--model", default="best_new.tflite")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--link", default="print",
                    choices=["print", "serial", "mavlink", "pwm"])
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--pins", default="17,18,19")
    ap.add_argument("--fov-h", type=float, default=62.0)
    ap.add_argument("--fov-v", type=float, default=48.0)
    ap.add_argument("--kp", type=float, default=0.04)
    args = ap.parse_args()

    interp = load_interpreter(args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]

    getter, stop = open_source(args.source)
    tracker = DroneTracker()
    aim = AimController(fov_h=args.fov_h, fov_v=args.fov_v, kp=args.kp)
    link = make_link(args)

    print(f"[INFO] aim controller | link={args.link} | FOV={args.fov_h}x{args.fov_v} | kp={args.kp}")

    try:
        while True:
            frame = getter()
            if frame is None:
                break
            H, W = frame.shape[:2]
            x = quantize_input(preprocess(frame), inp)
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            oscale, ozp = outp["quantization"]
            pred = (interp.get_tensor(outp["index"]).astype(np.float32) - ozp) * oscale
            pred = np.reshape(pred, (-1, pred.shape[-1]))
            if pred.shape[0] != 8400 and pred.shape[1] == 8400:
                pred = pred.T

            boxes, cls, scores = decode(pred, args.conf)
            result = []
            for c in range(len(CLASS_NAMES)):
                ci = np.where(cls == c)[0]
                if not len(ci):
                    continue
                keep = nms(boxes[ci], scores[ci], args.iou)
                for k in keep:
                    i = ci[k]
                    x1, y1, x2, y2 = boxes[i]
                    sx, sy = W / IMG_SIZE, H / IMG_SIZE
                    result.append((int(x1 * sx), int(y1 * sy),
                                   int(x2 * sx), int(y2 * sy),
                                   float(scores[i]), int(c)))

            drone_dets = [( (x1+x2)//2, (y1+y2)//2 )
                         for (x1,y1,x2,y2,_,c) in result if c == DRONE_ID]
            tracked = tracker.update(drone_dets) if drone_dets else []

            for (x1,y1,x2,y2,sc,c) in result:
                col = (0,255,0) if c == DRONE_ID else (0,165,255)
                label = CLASS_NAMES[c] + (" (ignored)" if c != DRONE_ID else "")
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"{label} {sc:.2f}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            # ---- aim at the best (highest-confidence) drone ----
            best = None
            for (x1,y1,x2,y2,sc,c) in result:
                if c == DRONE_ID and (best is None or sc > best[4]):
                    best = (x1,y1,x2,y2,sc)
            if best:
                x1,y1,x2,y2,sc = best
                cx, cy = (x1+x2)//2, (y1+y2)//2
                yaw_a, pitch_a, cmd_yaw, cmd_pitch, cmd_roll = aim.compute(cx, cy, W, H)
                link.send(cmd_yaw, cmd_pitch, cmd_roll,
                          extra=f"| target=({cx},{cy}) yaw_deg={math.degrees(yaw_a):+.1f} pitch_deg={math.degrees(pitch_a):+.1f}")
                cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
                cv2.putText(frame, f"AIM ({cx},{cy})", (cx+8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            else:
                # No drone detected this frame: reset controller state so a
                # later re-acquisition does not produce a derivative kick.
                aim.reset()

            cv2.imshow("Aim Controller (Pi 5)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop()
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
