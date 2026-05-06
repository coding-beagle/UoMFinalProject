"""
camera_thread.py
================
Background camera capture with MediaPipe Pose + Hands inference.

Each CameraThread runs in its own OS thread; the main loop reads
results via read_camera() which briefly acquires the thread lock.
"""

import threading

import cv2
import mediapipe as mp
import numpy as np

from .config import FINGER_CURL_THRESHOLD, FINGER_CLOSED_COUNT, GRIPPER_DEBOUNCE_FRAMES
from .hand_gesture import compute_finger_curls


class CameraThread(threading.Thread):
    """
    Public attributes (protected by self.lock):
        frame           : latest BGR frame (or None)
        world_landmarks : latest pose_world_landmarks (or None)
        pose_landmarks  : latest pose_landmarks for drawing (or None)
        tracking        : True when a pose is detected
        hand_landmarks  : list of detected hand landmark sets (may be empty)
        hand_open       : bool — True = open hand, False = closed fist, None = not detected
        hand_curl_ratios: list of 4 floats (Index→Pinky) or None
    """

    def __init__(self, cam_index: int, cam_id: int):
        super().__init__(daemon=True, name=f"CamThread-{cam_id}")
        self.cam_index = cam_index
        self.cam_id = cam_id
        self.lock = threading.Lock()
        self.frame = None
        self.world_landmarks = None
        self.pose_landmarks = None
        self.tracking = False
        self.hand_landmarks = []
        self.hand_open = None
        self.hand_curl_ratios = None
        self._stop_event = threading.Event()
        self.flipped = False

    def flip(self, val):
        self.flipped = val

    def stop(self):
        self._stop_event.set()

    def run(self):
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        cap = cv2.VideoCapture(self.cam_index + cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(
                f"[CamThread-{self.cam_id}] WARNING: could not open camera {self.cam_index}"
            )
            return
        print(f"[CamThread-{self.cam_id}] Camera {self.cam_index} opened.")

        _pending_open = None
        _pending_frames = 0

        try:
            while not self._stop_event.is_set():
                ret, bgr = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.flip(rgb, 1) if self.flipped else rgb

                pose_results = pose.process(rgb)
                hand_results = hands.process(rgb)

                # ── raw hand state ────────────────────────────────────────────
                raw_hand_lms = []
                raw_open = None
                raw_curls = None
                if hand_results.multi_hand_landmarks:
                    raw_hand_lms = hand_results.multi_hand_landmarks
                    raw_curls = compute_finger_curls(raw_hand_lms[0])
                    raw_open = (
                        sum(1 for r in raw_curls if r < FINGER_CURL_THRESHOLD)
                        < FINGER_CLOSED_COUNT
                    )

                # ── debounce ──────────────────────────────────────────────────
                if raw_open is None:
                    _pending_open = None
                    _pending_frames = 0
                    debounced_open = None
                else:
                    if raw_open == _pending_open:
                        _pending_frames += 1
                    else:
                        _pending_open = raw_open
                        _pending_frames = 1
                    debounced_open = (
                        raw_open if _pending_frames >= GRIPPER_DEBOUNCE_FRAMES else None
                    )

                with self.lock:
                    self.frame = cv2.flip(bgr, 1) if self.flipped else bgr
                    self.world_landmarks = pose_results.pose_world_landmarks
                    self.pose_landmarks = pose_results.pose_landmarks
                    self.tracking = pose_results.pose_world_landmarks is not None
                    self.hand_landmarks = raw_hand_lms
                    self.hand_curl_ratios = raw_curls
                    if debounced_open is not None:
                        self.hand_open = debounced_open
                    elif raw_open is None:
                        self.hand_open = None
        finally:
            cap.release()
            pose.close()
            hands.close()
            print(f"[CamThread-{self.cam_id}] Camera {self.cam_index} released.")


# ── snapshot helpers ──────────────────────────────────────────────────────────


def read_camera(cam_thread: CameraThread, flipped: bool = False):
    """Return a consistent snapshot tuple from a CameraThread."""
    cam_thread.flip(flipped)
    frame = cam_thread.frame.copy() if cam_thread.frame is not None else None

    with cam_thread.lock:
        return (
            frame,
            cam_thread.world_landmarks,
            cam_thread.pose_landmarks,
            cam_thread.tracking,
            list(cam_thread.hand_landmarks),
            cam_thread.hand_open,
            list(cam_thread.hand_curl_ratios) if cam_thread.hand_curl_ratios else None,
        )


def tile_frames(frames: list, widths: list) -> np.ndarray:
    """Resize each frame to its target width then hstack them."""
    resized = []
    for f, tw in zip(frames, widths):
        if f is None:
            f = np.zeros((360, tw, 3), dtype=np.uint8)
        h, w = f.shape[:2]
        th = max(1, int(h * tw / w))
        resized.append(cv2.resize(f, (tw, th)))
    max_h = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        dh = max_h - r.shape[0]
        if dh:
            r = np.vstack([r, np.zeros((dh, r.shape[1], 3), dtype=np.uint8)])
        padded.append(r)
    return np.hstack(padded)
