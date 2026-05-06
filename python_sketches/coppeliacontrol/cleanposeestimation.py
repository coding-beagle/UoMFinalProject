"""
main.py
=======
YuMi Pose Control — entry point.

Starts camera threads, connects to CoppeliaSim, then runs the main
control loop: pose fusion → IK → experiment update → render.

Keyboard controls
-----------------
R  start / restart Reach experiment
T  start / restart Transport experiment
Q  quit
"""

import sys
import traceback


def _crash(type, value, tb):
    with open("crash.txt", "w", encoding="utf-8") as f:
        traceback.print_exception(type, value, tb, file=f)
    traceback.print_exception(type, value, tb)
    input("Press enter to close...")


sys.excepthook = _crash

import cv2
import mediapipe as mp
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from math import pi

from utils import (
    # config
    CAMERA_INDICES,
    PRIMARY_CAMERA,
    TILE_WIDTH,
    SECONDARY_TILE_WIDTH,
    HUD_CAMERA,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    ROBOT_ARM_LENGTH,
    GRIPPER_SIGNAL,
    POSE_SMOOTH_ALPHA,
    MODE_REACH,
    MODE_TRANSPORT,
    MODE_OBSTACLE,
    MODE_SELECT,
    # math / pose
    vec3,
    PoseFilter,
    ArmCalibrator,
    retarget,
    compute_wrist_quaternion,
    # camera
    CameraThread,
    read_camera,
    tile_frames,
    # hand
    draw_curl_meter,
    # hud
    draw_mode_select_hud,
    draw_experiment_hud,
    _cv_col,
    # experiments
    make_reach_experiment,
    make_transport_experiment,
    make_obstacle_experiment,
    save_results,
    save_kinematics,
)

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
    "/rightJoint5/rightLink5/rightJoint6/rightLink6/rightJoint7/rightLink7"
    "/rightConnector/YuMiGripper/centerJoint/leftFinger"
)

target = sim.createDummy(0.02)
sim.setObjectAlias(target, "WristTarget")

robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
print(f"Robot shoulder origin: {robot_shoulder_world}")

# ── joint handles for kinematics logging ──────────────────────────────────────
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

# ── arm collision collection (used by obstacle experiment) ────────────────────
arm_collection = sim.createCollection(0)
sim.addItemToCollection(arm_collection, sim.handle_tree, rightShoulderAbduct, 0)
print(f"Arm collision collection created (handle={arm_collection}).")

# ── kinematics buffer (one row per sim.step) ──────────────────────────────────
kinematics_buffer = []

# IK groups
ikEnv = simIK.createEnvironment()
ikGroupUndamped = simIK.createGroup(ikEnv)
simIK.setGroupCalculation(ikEnv, ikGroupUndamped, simIK.method_pseudo_inverse, 0, 6)
simIK.addElementFromScene(
    ikEnv,
    ikGroupUndamped,
    rightShoulderAbduct,
    rightWristLink,
    target,
    simIK.constraint_position,
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

# ── camera threads ────────────────────────────────────────────────────────────
cam_threads = [CameraThread(idx, i) for i, idx in enumerate(CAMERA_INDICES)]
for ct in cam_threads:
    ct.start()

mp_draw = mp.solutions.drawing_utils
mp_pose_mod = mp.solutions.pose
mp_hands_mod = mp.solutions.hands
hand_draw_spec = mp_draw.DrawingSpec(color=(255, 200, 0), thickness=1, circle_radius=2)
hand_conn_spec = mp_draw.DrawingSpec(color=(200, 150, 0), thickness=1)

flipped = False

# ── main loop ─────────────────────────────────────────────────────────────────
try:
    print("Starting simulation...")
    sim.setStepping(True)
    sim.startSimulation()
    sim.setInt32Signal(GRIPPER_SIGNAL, 0)
    print("Simulation started OK")

    robot_shoulder_world = sim.getObjectPosition(rightShoulderAbduct, -1)
    calibrator = ArmCalibrator(ROBOT_ARM_LENGTH)
    pose_filter = PoseFilter(alpha=POSE_SMOOTH_ALPHA)
    wrist_pos = list(robot_shoulder_world)
    prev_time = cv2.getTickCount()
    current_mode = MODE_SELECT
    experiment = None
    summary_printed = False
    gripper_open = True

    print("Press [R] for Reach experiment, [T] for Transport, [Q] to quit.")
    print("Stretch your arm fully to calibrate reach mapping.")
    print("Open hand = gripper open | Closed fist = gripper closed.")

    while True:
        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now

        snapshots = [read_camera(ct, flipped) for ct in cam_threads]

        # ── gripper state: OR gate across all cameras ─────────────────────────
        hand_states = [snap[5] for snap in snapshots if snap[5] is not None]
        if hand_states:
            new_gripper_open = all(hand_states)
            if new_gripper_open != gripper_open:
                gripper_open = new_gripper_open
                sim.setInt32Signal(GRIPPER_SIGNAL, 0 if gripper_open else 1)

        # ── pose fusion: primary camera first, fallback to others ─────────────
        ordered = [PRIMARY_CAMERA] + [
            i for i in range(len(cam_threads)) if i != PRIMARY_CAMERA
        ]
        target_pos = None
        target_quat = None
        source_idx = None

        for ci in ordered:
            frame, wl_world, wl_img, tracking, hand_lms, _, _curl = snapshots[ci]
            if not tracking:
                continue
            wl = wl_world.landmark
            hs = vec3(wl[RIGHT_SHOULDER])
            he = vec3(wl[RIGHT_ELBOW])
            hw = vec3(wl[RIGHT_WRIST])
            if source_idx is None:
                calibrator.update(hs, he, hw, dt)
            target_pos = retarget(
                hs,
                hw,
                robot_shoulder_world,
                ROBOT_ARM_LENGTH,
                human_scale=calibrator.scale,
            )
            target_quat = compute_wrist_quaternion(hs, he, hw)
            source_idx = ci
            break

        # ── smooth → IK ───────────────────────────────────────────────────────
        target_pos = pose_filter.update_pos(target_pos)
        target_quat = pose_filter.update_quat(target_quat)

        if target_pos:
            sim.setObjectPosition(target, target_pos)
        if target_quat:
            sim.setObjectQuaternion(target, target_quat)
        if target_pos or target_quat:
            res, *_ = simIK.handleGroup(ikEnv, ikGroupUndamped, {"syncWorlds": True})
            if res != simIK.result_success:
                simIK.handleGroup(ikEnv, ikGroupDamped, {"syncWorlds": True})

        sim.step()
        wrist_pos = sim.getObjectPosition(rightGripperObject, -1)

        # ── kinematics sample ─────────────────────────────────────────────────
        joint_angles_rad = [sim.getJointPosition(h) for h in joint_handles]
        trial_idx = getattr(experiment, "_index", 0) if experiment else 0
        active = getattr(experiment, "_active", None) if experiment else None
        phase = getattr(active, "_phase", "") if active is not None else ""
        hits = getattr(active, "_total_hits", 0) if active is not None else 0
        ik_ok = "ok" if (target_pos or target_quat) else "no_pose"
        kinematics_buffer.append(
            {
                "sim_time": round(sim.getSimulationTime(), 4),
                "trial": trial_idx,
                "phase": phase,
                "gripper_open": int(gripper_open),
                "ik_status": ik_ok,
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
                "obstacle_hits": hits,
            }
        )

        # ── experiment update ─────────────────────────────────────────────────
        if experiment is not None and current_mode != MODE_SELECT:
            if current_mode == MODE_REACH:
                experiment.update(wrist_pos, dt)
            else:
                experiment.update(wrist_pos, gripper_open, dt)
            if experiment.finished and not summary_printed:
                print(experiment.summary())
                save_results(experiment, current_mode)
                summary_printed = True

        # ── render ────────────────────────────────────────────────────────────
        display_frames = []
        tile_widths = []

        for ci, (
            frame,
            wl_world,
            wl_img,
            tracking,
            hand_lms,
            hand_open_ci,
            curl_ratios,
        ) in enumerate(snapshots):

            is_hud_cam = ci == HUD_CAMERA
            tw = TILE_WIDTH if is_hud_cam else SECONDARY_TILE_WIDTH

            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)

            # Pose skeleton
            if wl_img is not None:
                mp_draw.draw_landmarks(
                    frame,
                    wl_img,
                    mp_pose_mod.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            # Hand skeleton
            for hl in hand_lms:
                mp_draw.draw_landmarks(
                    frame,
                    hl,
                    mp_hands_mod.HAND_CONNECTIONS,
                    hand_draw_spec,
                    hand_conn_spec,
                )

            # Finger curl meter
            if curl_ratios is not None:
                H_f = frame.shape[0]
                origin = (10, H_f - 120) if is_hud_cam else (4, 30)
                prefix = "finger curl" if is_hud_cam else "curl"
                draw_curl_meter(frame, curl_ratios, origin, label_prefix=prefix)

            is_active = ci == source_idx
            label_txt = f"Cam {ci}" + (" [ACTIVE]" if is_active else "")
            status_col = (0, 255, 0) if tracking else (0, 0, 255)

            if is_hud_cam:
                cv2.putText(
                    frame,
                    label_txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_col,
                    2,
                )
                cv2.putText(
                    frame,
                    "IK: tracking" if tracking else "IK: no pose",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_col,
                    2,
                )

                if target_pos and is_active:
                    cv2.putText(
                        frame,
                        f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

                # Calibration status
                if calibrator.calibrated:
                    cal_txt = (
                        f"Reach cal: {calibrator.human_max_reach*100:.1f} cm"
                        f"  scale={calibrator.scale:.2f}"
                    )
                    cal_col = (0, 220, 100)
                else:
                    cal_txt = (
                        f"Reach cal: stretch arm to calibrate"
                        f" ({calibrator._samples}/{calibrator.MIN_SAMPLES})"
                    )
                    cal_col = (0, 180, 220)
                cv2.putText(
                    frame,
                    cal_txt,
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    cal_col,
                    1,
                )

                # Gripper indicator
                seeing = [i for i, snap in enumerate(snapshots) if snap[5] is not None]
                cam_note = (
                    f"  (cam {','.join(str(i) for i in seeing)})" if seeing else ""
                )
                if not hand_states:
                    g_txt, g_col = "HAND: not detected", (80, 80, 80)
                elif gripper_open:
                    g_txt = f"HAND: OPEN  [gripper open]{cam_note}"
                    g_col = _cv_col(100, 220, 130)
                else:
                    g_txt = f"HAND: CLOSED  [gripper closed]{cam_note}"
                    g_col = _cv_col(100, 180, 255)
                cv2.putText(
                    frame, g_txt, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g_col, 1
                )

                # Experiment overlay
                if current_mode == MODE_SELECT:
                    draw_mode_select_hud(frame)
                elif experiment is not None:
                    draw_experiment_hud(frame, experiment, wrist_pos, dt, current_mode)
            else:
                cv2.putText(
                    frame,
                    label_txt,
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_col,
                    1,
                )

            display_frames.append(frame)
            tile_widths.append(tw)

        combined = tile_frames(display_frames, tile_widths)
        cv2.imshow("YuMi Pose Control — Multi-Camera", combined)

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            flipped = not (flipped)
        elif key == ord("r"):
            if experiment is not None and experiment.results:
                save_results(experiment, current_mode)
            print("Starting Reach experiment...")
            experiment = make_reach_experiment(sim, robot_shoulder_world)
            current_mode = MODE_REACH
            summary_printed = False
        elif key == ord("t"):
            if experiment is not None and experiment.results:
                save_results(experiment, current_mode)
            print("Starting Transport experiment...")
            experiment = make_transport_experiment(
                sim, robot_shoulder_world, start_pos=wrist_pos
            )
            current_mode = MODE_TRANSPORT
            summary_printed = False
        elif key == ord("o"):
            if experiment is not None and experiment.results:
                save_results(experiment, current_mode)
            print("Starting Obstacle Transport experiment...")
            experiment = make_obstacle_experiment(
                sim, robot_shoulder_world, arm_collection, start_pos=wrist_pos
            )
            current_mode = MODE_OBSTACLE
            summary_printed = False

except KeyboardInterrupt:
    print("Interrupted.")

except Exception as e:
    import traceback

    print("\n── FATAL ERROR ──────────────────────────────")
    traceback.print_exc()
    print("─────────────────────────────────────────────\n")

finally:
    for ct in cam_threads:
        ct.stop()
    for ct in cam_threads:
        ct.join(timeout=2.0)
    cv2.destroyAllWindows()
    sim.clearInt32Signal(GRIPPER_SIGNAL)
    sim.stopSimulation()
    print("Simulation stopped.")

    if "experiment" in dir() and experiment is not None and experiment.results:
        if not summary_printed:
            save_results(experiment, current_mode)
        save_kinematics(kinematics_buffer, current_mode, N_JOINTS)
    else:
        print("No results to save.")
