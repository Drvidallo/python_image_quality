"""Live webcam ID quality checker.

Usage:
    uv run python webcam/app.py
    uv run python webcam/app.py --camera 0 --width 1280 --height 720
    uv run python webcam/app.py --source path/to/frame.jpg --alpha 1.0 --port 8010
    uv run python webcam/app.py --source path/to/clip.mp4

Serves a browser page with the live feed and per-dimension blur / lighting /
glare / resolution scores. Thresholds start from inference/inference_config.json;
every committed change is saved to webcam/inference_config_1.json with an
incremented revision counter.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tornado.iostream
import tornado.web
from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = Path(__file__).resolve().parent
EDA_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EDA_ROOT / "inference"))
sys.path.insert(0, str(EDA_ROOT / "training"))

from data_common import detect_document_quad  # noqa: E402
from score_image import (  # noqa: E402
    DIMENSIONS,
    extract_features,
    load_inference_config,
    load_models,
    score_features,
)

DEFAULT_HOLD_FRAMES = 15
DEFAULT_SCORE_EVERY = 2
DEFAULT_ALPHA = 0.3
DEFAULT_WINDOW = 3
DEFAULT_HYSTERESIS = 2.0
RESET_AFTER_SECONDS = 1.0
STREAM_FPS = 30
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def friendly(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def guide_quad(frame_shape):
    height, width = frame_shape[:2]
    aspect = 85.6 / 53.98
    guide_w = 0.8 * width
    guide_h = guide_w / aspect
    if guide_h > 0.8 * height:
        guide_h = 0.8 * height
        guide_w = guide_h * aspect
    x0 = (width - guide_w) / 2.0
    y0 = (height - guide_h) / 2.0
    return np.array(
        [[x0, y0], [x0 + guide_w, y0], [x0 + guide_w, y0 + guide_h], [x0, y0 + guide_h]],
        dtype=np.float64,
    )


def draw_dashed_rect(img, quad, color, thickness=2, dash=14):
    points = quad.astype(int).tolist()
    for index in range(4):
        p1 = points[index]
        p2 = points[(index + 1) % 4]
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if length <= 0:
            continue
        steps = max(1, int(length // dash))
        for step in range(steps):
            start = step / steps
            end = min(1.0, start + 0.55 / steps)
            a = (int(p1[0] + (p2[0] - p1[0]) * start), int(p1[1] + (p2[1] - p1[1]) * start))
            b = (int(p1[0] + (p2[0] - p1[0]) * end), int(p1[1] + (p2[1] - p1[1]) * end))
            cv2.line(img, a, b, color, thickness)


class Smoother:
    def __init__(self, alpha, window, hysteresis):
        self.alpha = alpha
        self.window = window
        self.hysteresis = hysteresis
        self.buffer = deque(maxlen=window)
        self.ema = None
        self.pass_state = None

    def reset(self):
        self.buffer.clear()
        self.ema = None
        self.pass_state = None

    @property
    def ready(self):
        return len(self.buffer) >= self.window

    def update(self, raw):
        self.buffer.append(float(raw))
        median = float(np.median(self.buffer))
        self.ema = median if self.ema is None else self.alpha * median + (1 - self.alpha) * self.ema
        return self.ema

    def verdict(self, value, threshold):
        if value >= threshold + self.hysteresis:
            self.pass_state = True
        elif value <= threshold - self.hysteresis:
            self.pass_state = False
        elif self.pass_state is None:
            self.pass_state = bool(value >= threshold)
        return self.pass_state


class Engine:
    def __init__(self, args, models, spec, thresholds, reference_px, revision):
        self.args = args
        self.models = models
        self.spec = spec
        self.thresholds = dict(thresholds)
        self.reference_px = float(reference_px)
        self.revision = int(revision)
        self.config_path = Path(args.config_1)
        self.snapshots_dir = Path(args.snapshots_dir)
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.last_status = {
            "fps": 0.0,
            "detection": "starting",
            "stabilizing": True,
            "revision": self.revision,
            "thresholds": {key: friendly(value) for key, value in self.thresholds.items()},
            "resolution_reference_px": self.reference_px,
            "dimensions": {name: None for name in DIMENSIONS},
            "overall_pass": None,
            "warnings": [],
            "raw": {},
        }
        self.smoothers = {
            name: Smoother(args.alpha, args.window, args.hysteresis) for name in DIMENSIONS
        }
        self.last_quad = None
        self.frames_since_detect = 0
        self.last_valid_time = 0.0
        self.frame_times = deque(maxlen=60)
        self.capture = None
        self.image_frame = None
        self.source_is_video = False

    def open_source(self):
        source = self.args.source
        if source.isdigit():
            index = int(source)
            backend = None
            if self.args.api == "msmf":
                backend = cv2.CAP_MSMF
            elif self.args.api == "dshow":
                backend = cv2.CAP_DSHOW
            self.capture = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            if not self.capture.isOpened():
                raise RuntimeError(f"could not open camera {index}")
            return
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise RuntimeError(f"could not open video {path}")
            self.source_is_video = True
            return
        self.image_frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.image_frame is None:
            raise RuntimeError(f"could not read image {path}")

    def read_frame(self):
        if self.image_frame is not None:
            time.sleep(1.0 / STREAM_FPS)
            return self.image_frame.copy()
        ok, frame = self.capture.read()
        if not ok:
            if self.source_is_video:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                return None
        return frame

    def _choose_quad(self, frame):
        quad = detect_document_quad(frame)
        if quad is not None:
            self.last_quad = quad
            self.frames_since_detect = 0
            return quad, "detected"
        if self.last_quad is not None and self.frames_since_detect < self.args.hold_frames:
            self.frames_since_detect += 1
            return self.last_quad, "held"
        self.last_quad = None
        return guide_quad(frame.shape), "guide"

    def _score_cycle(self, frame):
        now = time.time()
        if now - self.last_valid_time > RESET_AFTER_SECONDS:
            for smoother in self.smoothers.values():
                smoother.reset()
        self.last_valid_time = now

        quad, detection = self._choose_quad(frame)
        warnings = []
        if detection == "guide":
            warnings.append("document_not_detected: align the ID inside the guide area")
        elif detection == "held":
            warnings.append("detection_held: reusing the last detected boundary")

        features = extract_features(
            frame, quad, self.args.document_type, self.args.capture_condition_code
        )
        scores, feature_warnings = score_features(
            features, self.spec, self.models, self.thresholds, self.reference_px
        )
        warnings.extend(feature_warnings)

        dimensions = {}
        for name in DIMENSIONS:
            smoother = self.smoothers[name]
            smoothed = smoother.update(scores[name]["score"])
            verdict = smoother.verdict(smoothed, self.thresholds[name])
            entry = {
                "score": round(float(smoothed), 1),
                "threshold": friendly(self.thresholds[name]),
                "is_pass": bool(verdict),
            }
            if name == "resolution":
                entry["short_side_px"] = scores[name].get("short_side_px")
                entry["est_dpi"] = scores[name].get("est_dpi")
            dimensions[name] = entry

        status = {
            "fps": self._fps(),
            "detection": detection,
            "stabilizing": not all(smoother.ready for smoother in self.smoothers.values()),
            "revision": self.revision,
            "thresholds": {key: friendly(value) for key, value in self.thresholds.items()},
            "resolution_reference_px": self.reference_px,
            "dimensions": dimensions,
            "overall_pass": all(dimensions[name]["is_pass"] for name in DIMENSIONS),
            "warnings": warnings,
            "raw": {
                "blur_var": round(float(features["blur_var"]), 2),
                "brightness_L": round(float(features["brightness_L"]), 2),
                "overexposed_pct": round(float(features["overexposed_pct"]), 3),
                "underexposed_pct": round(float(features["underexposed_pct"]), 3),
                "glare_pct": round(float(features["glare_pct"]), 3),
                "glare_blob_pct": round(float(features["glare_blob_pct"]), 3),
                "doc_short_side_px": round(float(features["doc_short_side_px"]), 1),
            },
            "quad": quad.astype(int).tolist(),
        }
        with self.lock:
            self.last_status = status

    def _fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return round((len(self.frame_times) - 1) / span, 1) if span > 0 else 0.0

    def _annotate(self, frame):
        with self.lock:
            status = dict(self.last_status)
        detection = status.get("detection", "starting")
        quad = status.get("quad")
        if quad is not None:
            quad_array = np.array(quad, dtype=np.int32)
            if detection == "detected":
                cv2.polylines(frame, [quad_array], True, (0, 200, 0), 3)
            elif detection == "held":
                cv2.polylines(frame, [quad_array], True, (0, 180, 255), 2)
            else:
                draw_dashed_rect(frame, np.array(quad, dtype=np.float64), (0, 140, 255), 2)
        label = {
            "detected": "DETECTED",
            "held": "HELD",
            "guide": "GUIDE - ALIGN ID",
            "starting": "STARTING",
        }.get(detection, detection.upper())
        cv2.putText(frame, f"{label}  {status.get('fps', 0.0):.0f} FPS", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def run(self):
        frame_index = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                continue
            self.frame_times.append(time.time())
            frame_index += 1
            if frame_index % self.args.score_every == 0:
                try:
                    self._score_cycle(frame)
                except Exception as exc:
                    with self.lock:
                        self.last_status["warnings"] = [f"scoring_error: {exc}"]
            annotated = self._annotate(frame)
            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()

    def status_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.last_status))

    def update_thresholds(self, payload):
        changed = False
        with self.lock:
            for name in DIMENSIONS:
                if name in payload:
                    value = float(payload[name])
                    if value != self.thresholds[name]:
                        self.thresholds[name] = value
                        changed = True
            if changed:
                self.revision += 1
                self._write_config()
                self._refresh_status_thresholds()
            return {
                "changed": changed,
                "revision": self.revision,
                "thresholds": {key: friendly(value) for key, value in self.thresholds.items()},
            }

    def _refresh_status_thresholds(self):
        status = self.last_status
        status["revision"] = self.revision
        status["thresholds"] = {key: friendly(value) for key, value in self.thresholds.items()}
        dimensions = status.get("dimensions") or {}
        for name in DIMENSIONS:
            entry = dimensions.get(name)
            if not entry:
                continue
            smoothed = self.smoothers[name].ema
            if smoothed is None:
                continue
            entry["threshold"] = friendly(self.thresholds[name])
            entry["is_pass"] = bool(self.smoothers[name].verdict(smoothed, self.thresholds[name]))
        if all(dimensions.get(name) for name in DIMENSIONS):
            status["overall_pass"] = all(dimensions[name]["is_pass"] for name in DIMENSIONS)

    def _write_config(self):
        payload = {
            "thresholds": {key: float(value) for key, value in self.thresholds.items()},
            "resolution_reference_px": self.reference_px,
            "revision": self.revision,
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def save_snapshot(self):
        with self.lock:
            jpeg = self.latest_jpeg
            status = json.loads(json.dumps(self.last_status))
        if jpeg is None:
            return {"error": "no frame available yet"}
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_path = self.snapshots_dir / f"{stamp}_capture.jpg"
        json_path = self.snapshots_dir / f"{stamp}_capture.json"
        with open(image_path, "wb") as handle:
            handle.write(jpeg)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "revision": status.get("revision"),
            "thresholds": status.get("thresholds"),
            "detection": status.get("detection"),
            "dimensions": status.get("dimensions"),
            "overall_pass": status.get("overall_pass"),
            "warnings": status.get("warnings"),
            "raw": status.get("raw"),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return {"image": image_path.name, "json": json_path.name}


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        engine = self.application.settings["engine"]
        template = self.application.settings["template"]
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(template.render())


class StatusHandler(tornado.web.RequestHandler):
    def get(self):
        engine = self.application.settings["engine"]
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(engine.status_snapshot()))


class ThresholdsHandler(tornado.web.RequestHandler):
    def post(self):
        engine = self.application.settings["engine"]
        try:
            payload = json.loads(self.request.body or b"{}")
            result = engine.update_thresholds(payload)
        except Exception as exc:
            self.set_status(400)
            result = {"error": str(exc)}
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(result))


class SnapshotHandler(tornado.web.RequestHandler):
    def post(self):
        engine = self.application.settings["engine"]
        result = engine.save_snapshot()
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(result))


class VideoHandler(tornado.web.RequestHandler):
    async def get(self):
        engine = self.application.settings["engine"]
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.set_header("Cache-Control", "no-store")
        try:
            while True:
                with engine.lock:
                    frame = engine.latest_jpeg
                if frame:
                    self.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.write(str(len(frame)).encode())
                    self.write(b"\r\n\r\n")
                    self.write(frame)
                    self.write(b"\r\n")
                    await self.flush()
                await asyncio.sleep(1.0 / STREAM_FPS)
        except (tornado.iostream.StreamClosedError, asyncio.CancelledError):
            return


def load_initial_thresholds(args):
    config_path = Path(args.config_1)
    base_thresholds, base_reference = load_inference_config(Path(args.base_config))
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        thresholds = {key: float(value) for key, value in base_thresholds.items()}
        thresholds.update({key: float(value) for key, value in saved.get("thresholds", {}).items()})
        reference = float(saved.get("resolution_reference_px", base_reference))
        revision = int(saved.get("revision", 0))
        return thresholds, reference, revision
    return base_thresholds, base_reference, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Live webcam ID quality checker.")
    parser.add_argument("--source", default="0", help="Camera index, video file or image file.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index shortcut.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--api", choices=["auto", "msmf", "dshow"], default="auto")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--document-type", choices=["id_card", "passport"], default="id_card")
    parser.add_argument("--capture-condition-code", type=int, default=0)
    parser.add_argument("--score-every", type=int, default=DEFAULT_SCORE_EVERY)
    parser.add_argument("--hold-frames", type=int, default=DEFAULT_HOLD_FRAMES)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--hysteresis", type=float, default=DEFAULT_HYSTERESIS)
    parser.add_argument("--models-dir", default=str(EDA_ROOT / "models"))
    parser.add_argument(
        "--spec", default=str(EDA_ROOT / "data" / "midv500" / "processed" / "feature_spec.json")
    )
    parser.add_argument(
        "--base-config", default=str(EDA_ROOT / "inference" / "inference_config.json")
    )
    parser.add_argument("--config-1", default=str(SCRIPT_DIR / "inference_config_1.json"))
    parser.add_argument("--snapshots-dir", default=str(SCRIPT_DIR / "snapshots"))
    args = parser.parse_args()
    if args.camera is not None:
        args.source = str(args.camera)
    return args


def main():
    args = parse_args()
    models = load_models(Path(args.models_dir))
    with open(args.spec, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    thresholds, reference_px, revision = load_initial_thresholds(args)
    engine = Engine(args, models, spec, thresholds, reference_px, revision)
    engine.open_source()
    environment = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR / "templates")), autoescape=True
    )
    template = environment.get_template("index.html")
    application = tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/status", StatusHandler),
            (r"/thresholds", ThresholdsHandler),
            (r"/snapshot", SnapshotHandler),
            (r"/video", VideoHandler),
        ],
        engine=engine,
        template=template,
    )
    application.listen(args.port, address="127.0.0.1")
    threading.Thread(target=engine.run, daemon=True).start()
    print(f"Webcam quality checker running at http://localhost:{args.port}")
    print(f"Thresholds revision: {revision} | config: {args.config_1}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()