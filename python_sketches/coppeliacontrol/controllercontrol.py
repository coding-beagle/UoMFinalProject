from math import pi, sqrt
import pygame
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import csv
import datetime
from reach_experiment import (
    Experiment,
    TransportExperiment,
    ObstacleTransportExperiment,
    ObstacleConfig,
)

# ── constants ────────────────────────────────────────────────────────────────
ROBOT_ARM_LENGTH = 0.21492 + 0.24129

MOVE_SPEED = 0.005  # metres per sim step
ROT_SPEED = 0.02  # radians per sim step

GRIPPER_SIGNAL = "gripper_close"
TRIGGER_THRESHOLD = 0.1
DEADZONE = 0.1
MODE_BUTTON = 5  # RB
RESET_BUTTON = 1  # B
RESYNC_BUTTON = 3  # Y – snap controller target to current wrist position

COL_POSITION = (100, 220, 100)
COL_ROTATION = (220, 160, 60)

# ── display layout ────────────────────────────────────────────────────────────
W, H = 700, 420
STICK_R = 50
STICK_DOT_R = 8
TRIG_W = 30
TRIG_H = 80

LS_CX, LS_CY = 140, 260
RS_CX, RS_CY = 370, 260

LT_X, LT_Y = 55, 60
RT_X, RT_Y = 610, 60

BUTTON_LAYOUT = {
    0: (580, 200, "A"),
    1: (605, 175, "B"),
    2: (555, 175, "X"),
    3: (580, 150, "Y"),
    4: (120, 110, "LB"),
    5: (580, 110, "RB"),
}

# ── experiment type ───────────────────────────────────────────────────────────
# Set to "reach"     – standard reach-to-target experiment
#        "transport" – pick-and-place cube transport
#        "obstacle"  – pick-and-place with spherical obstacle cloud
EXP_TYPE = "transport"  # "reach" | "transport" | "obstacle"

# ── reach experiment configuration ───────────────────────────────────────────
EXP_N_TRIALS = 10
EXP_RADIUS = 0.05  # success zone radius (m)
EXP_DWELL_TIME = 0.5  # seconds to hold inside zone
EXP_TIMEOUT = 20.0  # seconds per trial before fail
EXP_MIN_REACH = 0.7  # nearest target (fraction of arm length)
EXP_MAX_REACH = 0.9  # furthest target (fraction of arm length)
EXP_MIN_ELEVATION = -35.0  # degrees – allow slightly below horizontal
EXP_MAX_ELEVATION = 50.0  # degrees – cap well before overhead singularity
EXP_AZ_MIN = 20.0  # degrees – quarter-sphere spread
EXP_AZ_MAX = 110.0
EXP_SEED = None  # set an int for reproducible target placement

# ── transport experiment configuration ───────────────────────────────────────
TRN_N_TRIALS = 10
TRN_PICK_RADIUS = 0.06  # must approach within this distance to pick (m)
TRN_DROP_RADIUS = 0.06  # must be within this distance to place (m)
TRN_TIMEOUT = 100.0  # seconds per trial before fail
TRN_MIN_REACH = 0.8
TRN_MAX_REACH = 0.9
TRN_MIN_ELEVATION = -35.0
TRN_MAX_ELEVATION = 50.0
TRN_AZ_MIN = 20.0
TRN_AZ_MAX = 90.0
TRN_SEED = None  # set an int for reproducible positions

# ── obstacle transport experiment configuration ───────────────────────────────
OBS_N_TRIALS = 10
OBS_PICK_RADIUS = 0.06
OBS_DROP_RADIUS = 0.06
OBS_TIMEOUT = 60.0
OBS_MIN_REACH = 0.8
OBS_MAX_REACH = 0.9
OBS_MIN_ELEVATION = -35.0
OBS_MAX_ELEVATION = 50.0
OBS_AZ_MIN = 20.0
OBS_AZ_MAX = 90.0
OBS_SEED = None  # set an int for reproducible positions + obstacle layout

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


# ── helpers ───────────────────────────────────────────────────────────────────
def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_clamp_to_sphere(origin, point, radius):
    diff = [point[i] - origin[i] for i in range(3)]
    length = sum(x**2 for x in diff) ** 0.5
    if length > radius:
        scale = radius / length
        diff = [d * scale for d in diff]
    return [origin[i] + diff[i] for i in range(3)]


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def normalise_quaternion(q):
    n = sum(v**2 for v in q) ** 0.5
    return [v / n for v in q] if n > 1e-9 else q


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_stick(surf, cx, cy, ax, ay, label, active_col):
    pygame.draw.circle(surf, (70, 70, 70), (cx, cy), STICK_R, 2)
    pygame.draw.line(surf, (50, 50, 50), (cx - STICK_R, cy), (cx + STICK_R, cy), 1)
    pygame.draw.line(surf, (50, 50, 50), (cx, cy - STICK_R), (cx, cy + STICK_R), 1)
    dot_x = int(cx + ax * STICK_R)
    dot_y = int(cy + ay * STICK_R)
    pygame.draw.circle(surf, active_col, (dot_x, dot_y), STICK_DOT_R)
    pygame.draw.circle(surf, (255, 255, 255), (dot_x, dot_y), STICK_DOT_R, 1)
    lbl = pygame.font.SysFont("monospace", 13).render(label, True, (160, 160, 160))
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy + STICK_R + 6))


def draw_trigger(surf, tx, ty, value, label, active_col):
    pygame.draw.rect(surf, (60, 60, 60), (tx, ty, TRIG_W, TRIG_H), 2)
    fill_h = int(TRIG_H * value)
    if fill_h > 0:
        pygame.draw.rect(
            surf, active_col, (tx + 2, ty + TRIG_H - fill_h, TRIG_W - 4, fill_h)
        )
    lbl = pygame.font.SysFont("monospace", 13).render(label, True, (160, 160, 160))
    surf.blit(lbl, (tx + TRIG_W // 2 - lbl.get_width() // 2, ty + TRIG_H + 4))


def draw_button(surf, cx, cy, label, pressed):
    col_fill = (220, 180, 50) if pressed else (55, 55, 55)
    col_border = (255, 220, 100) if pressed else (110, 110, 110)
    col_text = (20, 20, 20) if pressed else (160, 160, 160)
    pygame.draw.circle(surf, col_fill, (cx, cy), 16)
    pygame.draw.circle(surf, col_border, (cx, cy), 16, 2)
    lbl = pygame.font.SysFont("monospace", 11, bold=pressed).render(
        label, True, col_text
    )
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


# ── CoppeliaSim setup ─────────────────────────────────────────────────────────
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require("sim")
simIK = client.require("simIK")

rightShoulderAbduct = sim.getObject("/rightJoint1")
rightWristLink = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7"
)
rightGripperObject = sim.getObject(
    "/rightJoint1/rightLink1/rightJoint2/rightLink2"
    "/rightJoint3/rightLink3/rightJoint4/rightLink4"
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7/rightConnector/YuMiGripper/centerJoint/leftFinger"
)

# ── joint handles for kinematics logging ──────────────────────────────────────
# All 7 joints of the right arm, in proximal→distal order.
joint_handles = [
    sim.getObject("/rightJoint1"),
    sim.getObject("/rightJoint1/rightLink1/rightJoint2"),
    sim.getObject("/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3"),
    sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4"
    ),
    sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5"
    ),
    sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5/rightLink5/rightJoint6"
    ),
    sim.getObject(
        "/rightJoint1/rightLink1/rightJoint2/rightLink2/rightJoint3/rightLink3/rightJoint4/rightLink4/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7"
    ),
]
N_JOINTS = len(joint_handles)
print(f"Tracking {N_JOINTS} joints for kinematics logging.")

# ── arm collision collection ───────────────────────────────────────────────────
# A CoppeliaSim collection covering the entire right arm, used by
# sim.checkCollision in ObstacleTransportTrial.  Adding the shoulder root with
# the "include all descendants" option captures every link automatically.
arm_collection = sim.createCollection(0)
sim.addItemToCollection(arm_collection, sim.handle_tree, rightShoulderAbduct, 0)
print(f"Arm collision collection created (handle={arm_collection}).")

simIndex = 0
simObject = 0

# wristTarget cleanup
while simObject != -1:
    simObject = sim.getObjects(simIndex, sim.handle_all)
    if simObject != -1:
        alias = sim.getObjectAlias(simObject)
        if alias == "WristTarget":
            print("Removing Object!")
            sim.removeObjects([simObject])
    simIndex += 1

target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")
robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

target_pos = [
    robot_shoulder_world[0] + 0.3,
    robot_shoulder_world[1],
    robot_shoulder_world[2],
]
target_quat = [0.0, 0.0, 0.0, 1.0]

# ── IK setup ──────────────────────────────────────────────────────────────────
ikEnv = simIK.createEnvironment()

ikGroupUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(ikEnv, ikGroupUndamped, simIK.method_pseudo_inverse, 0, 6)
simIK.addElementFromScene(
    ikEnv,
    ikGroupUndamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_pose,
)

ikGroupDamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(
    ikEnv, ikGroupDamped, simIK.method_damped_least_squares, 1, 99
)
simIK.addElementFromScene(
    ikEnv,
    ikGroupDamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_position,
)

# ── Pygame setup ──────────────────────────────────────────────────────────────
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected. Please connect a joystick and retry.")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("YuMi 7-DOF Joystick Control")
font_sm = pygame.font.SysFont("monospace", 13)
font_md = pygame.font.SysFont("monospace", 15)
font_lg = pygame.font.SysFont("monospace", 17, bold=True)
clock = pygame.time.Clock()

fonts = {"sm": font_sm, "md": font_md, "lg": font_lg}

# ── kinematics buffer ─────────────────────────────────────────────────────────
# One row appended per sim.step() call; written to CSV in the finally block.
kinematics_buffer = []

try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    sim.setInt32Signal(GRIPPER_SIGNAL, 1)
    print("Simulation started OK")
    print(f"Experiment type: {EXP_TYPE}")

    # ── create experiment ─────────────────────────────────────────────────────
    if EXP_TYPE == "obstacle":
        obs_cfg = ObstacleConfig(
            n_obstacles=OBS_N_OBSTACLES,
            radius_min=OBS_RADIUS_MIN,
            radius_max=OBS_RADIUS_MAX,
            margin=OBS_MARGIN,
            shoulder_margin=OBS_SHOULDER_MARGIN,
            seed=OBS_SEED,
            penalty_on_hit=OBS_PENALTY_ON_HIT,
            penalty_seconds=OBS_PENALTY_SECONDS,
            min_reach=OBS_CLOUD_MIN_REACH,
            max_reach=OBS_CLOUD_MAX_REACH,
            min_elevation=OBS_CLOUD_MIN_ELEVATION,
            max_elevation=OBS_CLOUD_MAX_ELEVATION,
            az_min=OBS_CLOUD_AZ_MIN,
            az_max=OBS_CLOUD_AZ_MAX,
            az_centre=OBS_CLOUD_AZ_CENTRE,
        )
        experiment = ObstacleTransportExperiment.from_random(
            sim,
            shoulder_pos=robot_shoulder_world,
            arm_length=ROBOT_ARM_LENGTH,
            arm_collection=arm_collection,
            n_trials=OBS_N_TRIALS,
            obstacle_cfg=obs_cfg,
            pick_radius=OBS_PICK_RADIUS,
            drop_radius=OBS_DROP_RADIUS,
            timeout=OBS_TIMEOUT,
            min_reach=OBS_MIN_REACH,
            max_reach=OBS_MAX_REACH,
            min_elevation=OBS_MIN_ELEVATION,
            max_elevation=OBS_MAX_ELEVATION,
            az_min=OBS_AZ_MIN,
            az_max=OBS_AZ_MAX,
            seed=OBS_SEED,
            start_pos=target_pos,
        )
        print(
            f"Obstacle transport experiment created — {OBS_N_TRIALS} trials, {OBS_N_OBSTACLES} obstacles each."
        )
        for i, t in enumerate(experiment._trial_defs):
            print(f"  {i+1}. cube={t['cube_pos']}  drop={t['drop_pos']}")

    elif EXP_TYPE == "transport":
        experiment = TransportExperiment.from_random(
            sim,
            shoulder_pos=robot_shoulder_world,
            arm_length=ROBOT_ARM_LENGTH,
            n_trials=TRN_N_TRIALS,
            pick_radius=TRN_PICK_RADIUS,
            drop_radius=TRN_DROP_RADIUS,
            timeout=TRN_TIMEOUT,
            min_reach=TRN_MIN_REACH,
            max_reach=TRN_MAX_REACH,
            min_elevation=TRN_MIN_ELEVATION,
            max_elevation=TRN_MAX_ELEVATION,
            az_min=TRN_AZ_MIN,
            az_max=TRN_AZ_MAX,
            seed=TRN_SEED,
            start_pos=target_pos,
        )
        print(f"Transport experiment created — {TRN_N_TRIALS} pick-and-place trials.")
        for i, t in enumerate(experiment._trial_defs):
            print(f"  {i+1}. cube={t['cube_pos']}  drop={t['drop_pos']}")

    else:
        experiment = Experiment.from_hemisphere(
            sim,
            shoulder_pos=robot_shoulder_world,
            arm_length=ROBOT_ARM_LENGTH,
            n_trials=EXP_N_TRIALS,
            radius=EXP_RADIUS,
            dwell_time=EXP_DWELL_TIME,
            timeout=EXP_TIMEOUT,
            min_reach=EXP_MIN_REACH,
            max_reach=EXP_MAX_REACH,
            min_elevation=EXP_MIN_ELEVATION,
            max_elevation=EXP_MAX_ELEVATION,
            az_min=EXP_AZ_MIN,
            az_max=EXP_AZ_MAX,
            seed=EXP_SEED,
        )
        print(
            f"Reach experiment created — {EXP_N_TRIALS} targets placed on reachable hemisphere."
        )
        for i, t in enumerate(experiment._trial_defs):
            print(f"  {i+1}. {t['pos']}")

    running = True
    gripper_open = False  # signal=1 (close) is set at startup; open=False matches that
    wrist_pos = list(target_pos)

    while running:
        dt = clock.get_time() / 1000.0

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN and event.button == 7:
                running = False

        # ── read inputs ───────────────────────────────────────────────────────
        ls_x_raw = joystick.get_axis(0)
        ls_y_raw = joystick.get_axis(1)
        rs_x_raw = joystick.get_axis(2)
        rs_y_raw = joystick.get_axis(3)
        lt_raw = joystick.get_axis(4)
        rt_raw = joystick.get_axis(5)

        ls_x = apply_deadzone(ls_x_raw, DEADZONE)
        ls_y = apply_deadzone(ls_y_raw, DEADZONE)
        rs_y = apply_deadzone(rs_y_raw, DEADZONE)
        lt = (lt_raw + 1.0) / 2.0
        rt = (rt_raw + 1.0) / 2.0

        rotation_mode = joystick.get_button(MODE_BUTTON)
        reset_rot = joystick.get_button(RESET_BUTTON)
        resync = joystick.get_button(RESYNC_BUTTON)

        num_buttons = joystick.get_numbuttons()
        button_states = [joystick.get_button(i) for i in range(num_buttons)]

        # ── gripper ───────────────────────────────────────────────────────────
        # LT (left trigger)  → close gripper  (gripper_open = False)
        # RT (right trigger) → open gripper   (gripper_open = True)
        # Neither held       → stay open by default
        if lt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 1)  # 1 = close
            gripper_open = False
        elif rt > TRIGGER_THRESHOLD:
            sim.setInt32Signal(GRIPPER_SIGNAL, 0)  # 0 = open
            gripper_open = True
        else:
            # Neither trigger held — gripper stays in last commanded state
            # but we always reflect the hardware truth for the experiment logic
            gripper_open = sim.getInt32Signal(GRIPPER_SIGNAL) == 0

        # ── resync ────────────────────────────────────────────────────────────
        if resync and not rotation_mode:
            target_pos = list(wrist_pos)

        # ── position mode ─────────────────────────────────────────────────────
        if not rotation_mode:
            target_pos = vec_add(
                target_pos,
                [ls_y * MOVE_SPEED, ls_x * MOVE_SPEED, -rs_y * MOVE_SPEED],
            )
            target_pos = vec_clamp_to_sphere(
                robot_shoulder_world, target_pos, ROBOT_ARM_LENGTH
            )

        # ── rotation mode ─────────────────────────────────────────────────────
        else:
            d_pitch = ls_y * ROT_SPEED
            d_yaw = ls_x * ROT_SPEED
            d_roll = rs_y * ROT_SPEED
            if d_pitch or d_yaw or d_roll:
                delta = euler_to_quaternion(d_roll, d_pitch, d_yaw)
                target_quat = normalise_quaternion(
                    quaternion_multiply(target_quat, delta)
                )
            if reset_rot:
                target_quat = [0.0, 0.0, 0.0, 1.0]

        # ── push to sim ───────────────────────────────────────────────────────
        sim.setObjectPosition(target, target_pos)
        sim.setObjectQuaternion(target, target_quat)

        res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
        if res != simIK.result_success:
            res, *_ = simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})
            ik_status = "damped" if res == simIK.result_success else "failed"
        else:
            ik_status = "ok"

        sim.step()

        # ── read actual wrist position ────────────────────────────────────────
        wrist_pos = sim.getObjectPosition(rightGripperObject, -1)

        # ── update experiment ─────────────────────────────────────────────────
        if EXP_TYPE in ("transport", "obstacle"):
            experiment.update(wrist_pos, gripper_open, dt)
        else:
            experiment.update(wrist_pos, dt)

        # print summary once when experiment finishes
        if experiment.finished and not getattr(experiment, "_summary_printed", False):
            print(experiment.summary())
            experiment._summary_printed = True

        # ── sample joint angles + kinematics row ──────────────────────────────
        # Logged once per sim.step() so the cadence matches the physics.
        joint_angles_rad = [sim.getJointPosition(h) for h in joint_handles]

        # Determine current trial index and phase for context columns.
        trial_idx = getattr(experiment, "_index", 0)
        active = getattr(experiment, "_active", None)
        phase = getattr(active, "_phase", "") if active is not None else "done"

        # Obstacle hits so far this trial (0 for non-obstacle experiments).
        hits_so_far = getattr(active, "_total_hits", 0) if active is not None else 0

        kinematics_buffer.append(
            {
                "sim_time": round(sim.getSimulationTime(), 4),
                "trial": trial_idx,
                "phase": phase,
                "gripper_open": int(gripper_open),
                "ik_status": ik_status,
                "wrist_x": round(wrist_pos[0], 5),
                "wrist_y": round(wrist_pos[1], 5),
                "wrist_z": round(wrist_pos[2], 5),
                **{
                    f"j{i+1}_rad": round(joint_angles_rad[i], 6)
                    for i in range(N_JOINTS)
                },
                **{
                    f"j{i+1}_deg": round(joint_angles_rad[i] * 180.0 / pi, 4)
                    for i in range(N_JOINTS)
                },
                "obstacle_hits": hits_so_far,
            }
        )

        # ── draw ──────────────────────────────────────────────────────────────
        screen.fill((22, 22, 28))

        # Mode banner
        mode_label = "[ ROTATION MODE ]" if rotation_mode else "[ POSITION MODE ]"
        mode_col = COL_ROTATION if rotation_mode else COL_POSITION
        hint_label = (
            "holding RB"
            if rotation_mode
            else "hold RB to rotate  |  B to reset orientation"
        )
        screen.blit(font_lg.render(mode_label, True, mode_col), (10, 8))
        screen.blit(font_sm.render(hint_label, True, (130, 130, 130)), (10, 32))

        # Experiment type badge
        badge_map = {
            "transport": ("TRANSPORT", (230, 130, 40)),
            "obstacle": ("OBS-TRANSPORT", (200, 80, 80)),
            "reach": ("REACH", (100, 160, 220)),
        }
        badge_txt, badge_col = badge_map.get(EXP_TYPE, ("REACH", (100, 160, 220)))
        screen.blit(font_sm.render(f"[ {badge_txt} EXP ]", True, badge_col), (10, 52))

        ls_col = COL_ROTATION if rotation_mode else COL_POSITION
        rs_col = COL_ROTATION if rotation_mode else COL_POSITION
        ls_label = "pitch / yaw" if rotation_mode else "X / Y pos"
        rs_label = "roll" if rotation_mode else "Z pos"

        draw_stick(screen, LS_CX, LS_CY, ls_x_raw, ls_y_raw, ls_label, ls_col)
        draw_stick(screen, RS_CX, RS_CY, rs_x_raw, rs_y_raw, rs_label, rs_col)

        lt_col = (220, 100, 100) if lt > TRIGGER_THRESHOLD else (100, 160, 220)
        rt_col = (100, 220, 130) if rt > TRIGGER_THRESHOLD else (100, 160, 220)
        draw_trigger(screen, LT_X, LT_Y, lt, "LT close", lt_col)
        draw_trigger(screen, RT_X, RT_Y, rt, "RT open", rt_col)

        for idx, (bx, by, blabel) in BUTTON_LAYOUT.items():
            pressed = button_states[idx] if idx < num_buttons else False
            draw_button(screen, bx, by, blabel, pressed)

        experiment.draw(screen, wrist_pos, fonts, dt)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    pygame.quit()
    print("Simulation stopped.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_tag = EXP_TYPE  # "reach" | "transport" | "obstacle"
    results_dir = f"controller{exp_tag.capitalize()}Results"

    # ── Save trial results CSV ────────────────────────────────────────────────
    results = experiment.results if "experiment" in dir() else []
    if results:
        results_filename = f"{results_dir}/{exp_tag}_results_{ts}.csv"

        if EXP_TYPE in ("transport", "obstacle"):
            # Base fieldnames shared by transport + obstacle
            fieldnames = [
                "trial",
                "label",
                "result",
                "duration_s",
                "cube_x",
                "cube_y",
                "cube_z",
                "drop_x",
                "drop_y",
                "drop_z",
                "start_x",
                "start_y",
                "start_z",
                "dist_start_to_cube",
                "dist_start_to_drop",
                "phase_approach_s",
                "phase_grip_s",
                "phase_carry_s",
                "phase_place_s",
            ]
            # Obstacle-only extra columns
            if EXP_TYPE == "obstacle":
                fieldnames += [
                    "n_obstacles",
                    "total_hits",
                    "penalty_accumulated_s",
                    "adjusted_duration_s",
                ]

            trial_defs = {i + 1: t for i, t in enumerate(experiment._trial_defs)}
            with open(results_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    cp = trial_defs.get(r["trial"], {}).get("cube_pos", [None] * 3)
                    dp = trial_defs.get(r["trial"], {}).get("drop_pos", [None] * 3)
                    sp = r.get("phase_splits", {})
                    spos = r.get("start_pos") or [None, None, None]
                    row = {
                        "trial": r["trial"],
                        "label": r["label"],
                        "result": r["result"],
                        "duration_s": round(r["duration"], 3),
                        "cube_x": round(cp[0], 4) if cp[0] is not None else "",
                        "cube_y": round(cp[1], 4) if cp[1] is not None else "",
                        "cube_z": round(cp[2], 4) if cp[2] is not None else "",
                        "drop_x": round(dp[0], 4) if dp[0] is not None else "",
                        "drop_y": round(dp[1], 4) if dp[1] is not None else "",
                        "drop_z": round(dp[2], 4) if dp[2] is not None else "",
                        "start_x": round(spos[0], 4) if spos[0] is not None else "",
                        "start_y": round(spos[1], 4) if spos[1] is not None else "",
                        "start_z": round(spos[2], 4) if spos[2] is not None else "",
                        "dist_start_to_cube": (
                            round(r["dist_start_to_cube"], 4)
                            if r.get("dist_start_to_cube") is not None
                            else ""
                        ),
                        "dist_start_to_drop": (
                            round(r["dist_start_to_drop"], 4)
                            if r.get("dist_start_to_drop") is not None
                            else ""
                        ),
                        "phase_approach_s": round(sp.get("approach", 0.0), 3),
                        "phase_grip_s": round(sp.get("grip", 0.0), 3),
                        "phase_carry_s": round(sp.get("carry", 0.0), 3),
                        "phase_place_s": round(sp.get("place", 0.0), 3),
                    }
                    if EXP_TYPE == "obstacle":
                        row["n_obstacles"] = r.get("n_obstacles", "")
                        row["total_hits"] = r.get("total_hits", 0)
                        row["penalty_accumulated_s"] = round(
                            r.get("penalty_accumulated", 0.0), 3
                        )
                        row["adjusted_duration_s"] = round(
                            r.get("adjusted_duration", r["duration"]), 3
                        )
                    writer.writerow(row)

        else:  # reach
            fieldnames = [
                "trial",
                "label",
                "result",
                "duration_s",
                "target_x",
                "target_y",
                "target_z",
            ]
            trial_defs = {i + 1: t for i, t in enumerate(experiment._trial_defs)}
            with open(results_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    pos = trial_defs.get(r["trial"], {}).get("pos", [None] * 3)
                    writer.writerow(
                        {
                            "trial": r["trial"],
                            "label": r["label"],
                            "result": r["result"],
                            "duration_s": round(r["duration"], 3),
                            "target_x": round(pos[0], 4) if pos[0] is not None else "",
                            "target_y": round(pos[1], 4) if pos[1] is not None else "",
                            "target_z": round(pos[2], 4) if pos[2] is not None else "",
                        }
                    )

        print(f"Trial results saved → {results_filename}")
        print(experiment.summary())
    else:
        print("No trial results to save.")

    # ── Save kinematics CSV ───────────────────────────────────────────────────
    if kinematics_buffer and results:
        kin_filename = f"{results_dir}/{exp_tag}_kinematics_{ts}.csv"

        kin_fieldnames = [
            "sim_time",
            "trial",
            "phase",
            "gripper_open",
            "ik_status",
            "wrist_x",
            "wrist_y",
            "wrist_z",
            *[f"j{i+1}_rad" for i in range(N_JOINTS)],
            *[f"j{i+1}_deg" for i in range(N_JOINTS)],
            "obstacle_hits",
        ]
        with open(kin_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=kin_fieldnames)
            writer.writeheader()
            writer.writerows(kinematics_buffer)

        print(
            f"Kinematics saved      → {kin_filename}  ({len(kinematics_buffer)} frames)"
        )
    else:
        print("No kinematics to save.")
