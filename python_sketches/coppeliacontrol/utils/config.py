"""
config.py
=========
All tunable constants and configuration for YuMi Pose Control.
Edit values here; nothing else needs to change.
"""

from math import pi

# ── camera configuration ──────────────────────────────────────────────────────
CAMERA_INDICES = [0]  # OpenCV device indices; edit to match your setup
PRIMARY_CAMERA = 0  # which index in CAMERA_INDICES is the preferred source
TILE_WIDTH = 640  # display width of the primary (HUD) camera tile
SECONDARY_TILE_WIDTH = 240
HUD_CAMERA = 0  # index within CAMERA_INDICES that gets the experiment HUD

# ── constants ─────────────────────────────────────────────────────────────────
DEG_TO_RAD = pi / 180

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

ROBOT_ARM_LENGTH = 0.21492 + 0.24129

# ── gripper signal ────────────────────────────────────────────────────────────
GRIPPER_SIGNAL = "gripper_close"  # 1 = close, 0 = open

# ── hand gesture configuration ────────────────────────────────────────────────
# Finger curl threshold: fraction of max curl distance at which a finger counts
# as "closed".  Lower = more sensitive (easier to trigger closed).
FINGER_CURL_THRESHOLD = 1.2
# How many of the 4 non-thumb fingers must be curled to call the hand "closed"
FINGER_CLOSED_COUNT = 3
# Hysteresis: once open/closed, require this many consecutive frames of the
# opposite state before flipping (prevents jitter).
GRIPPER_DEBOUNCE_FRAMES = 4

# ── pose smoothing ────────────────────────────────────────────────────────────
# EMA alpha: 0 = frozen, 1 = no filtering.  Typical range: 0.15–0.25
POSE_SMOOTH_ALPHA = 0.2

# ── reach experiment configuration ────────────────────────────────────────────
EXP_N_TRIALS = 10
EXP_RADIUS = 0.05
EXP_DWELL_TIME = 0.5
EXP_TIMEOUT = 20.0
EXP_MIN_REACH = 0.7
EXP_MAX_REACH = 0.9
EXP_MIN_ELEVATION = -35.0
EXP_MAX_ELEVATION = 50.0
EXP_AZ_MIN = 20.0
EXP_AZ_MAX = 110.0
EXP_SEED = None

# ── transport experiment configuration ────────────────────────────────────────
TRANSPORT_N_TRIALS = 10
TRANSPORT_PICK_RADIUS = 0.06
TRANSPORT_DROP_RADIUS = 0.06
TRANSPORT_TIMEOUT = 30.0
TRANSPORT_MIN_REACH = 0.85
TRANSPORT_MAX_REACH = 0.9
TRANSPORT_MIN_ELEV = -25.0
TRANSPORT_MAX_ELEV = 40.0
TRANSPORT_AZ_MIN = 50.0
TRANSPORT_AZ_MAX = 90.0
TRANSPORT_SEED = None

# ── obstacle transport experiment configuration ───────────────────────────────
OBS_N_TRIALS = 10
OBS_PICK_RADIUS = 0.06
OBS_DROP_RADIUS = 0.06
OBS_TIMEOUT = 100.0
OBS_MIN_REACH = 0.93
OBS_MAX_REACH = 0.93
OBS_MIN_ELEV = -25.0
OBS_MAX_ELEV = 40.0
OBS_AZ_MIN = 50.0
OBS_AZ_MAX = 90.0
OBS_SEED = None

# ObstacleConfig fields – edit these to change the obstacle cloud:
OBS_N_OBSTACLES = 10  # number of spherical obstacles
OBS_RADIUS_MIN = 0.01  # smallest sphere radius (m)
OBS_RADIUS_MAX = 0.03  # largest sphere radius (m)
OBS_MARGIN = 0.12  # keep-clear radius around cube & drop zone (m)
OBS_SHOULDER_MARGIN = 0.3  # keep-clear radius around shoulder origin (m)
OBS_PENALTY_ON_HIT = False  # add time penalty to reported duration on each hit
OBS_PENALTY_SECONDS = 2.0  # seconds added per obstacle contact event
# Obstacle cloud hemisphere bounds – same meaning as the OBS_* reach params above
OBS_CLOUD_MIN_REACH = 0.3  # nearest obstacle (fraction of arm length)
OBS_CLOUD_MAX_REACH = 0.90  # furthest obstacle (fraction of arm length)
OBS_CLOUD_MIN_ELEVATION = -20.0  # degrees below horizontal
OBS_CLOUD_MAX_ELEVATION = 70.0  # degrees above horizontal
OBS_CLOUD_AZ_MIN = 20.0  # azimuth spread around OBS_CLOUD_AZ_CENTRE
OBS_CLOUD_AZ_MAX = 90.0
OBS_CLOUD_AZ_CENTRE = -90.0  # should match TRN_AZ_* centre for consistency

# ── experiment mode tokens ────────────────────────────────────────────────────
MODE_REACH = "reach"
MODE_TRANSPORT = "transport"
MODE_OBSTACLE = "obstacle"
MODE_SELECT = "select"  # lobby / menu screen
