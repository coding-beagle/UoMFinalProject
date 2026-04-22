import click
import os
import cv2
import numpy as np
from math import floor
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime
import sys

from typing import Tuple, List


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
import csv

from .plotter import CSVPlotter
from .utils.utils import *
from .mediapipe_wrapper.mediapipe_utils import *


@click.group()
def cli():
    """GData - Bulk Video Editor + Pose Analysis CLI tool"""
    pass


OUT_FORMAT = ".mp4"


def resize_and_write(
    video_path: str,
    outfile: str,
    width: int,
    height: int,
    fps: int,
    rotate: bool = False,
) -> None:
    cap = cv2.VideoCapture(video_path)

    if rotate:
        output_width, output_height = height, width  # swapped after 90 deg rotation
    else:
        output_width, output_height = width, height

    output = cv2.VideoWriter(
        outfile, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.resize(image, (width, height))
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        output.write(image)

    cap.release()
    output.release()


def trim_video_file_and_write(video_path, out_path, start, end):
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = floor(float(start) * fps)
    end_frame = min(floor(float(end) * fps), frames)

    output_width, output_height = width, height

    output = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps),
        (output_width, output_height),
    )

    # Seek to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame = start_frame
    while video.isOpened() and frame <= end_frame:
        success, image = video.read()
        if not success:
            break

        output.write(image)
        frame += 1

    video.release()
    output.release()


def process_one_video(video_path, out_csv_path, draw=False):
    click.echo(f"Processing {video_path}, will write to {out_csv_path}")

    # Check if video file exists
    import os

    if not os.path.exists(video_path):
        click.echo(f"ERROR: Video file not found: {video_path}")
        return

    click.echo("Video file exists, attempting to open...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            click.echo("ERROR: Failed to open video capture")
            return
        click.echo("Video capture opened successfully")
    except Exception as e:
        click.echo(f"ERROR opening video: {e}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width, output_height = width, height

    fps = cap.get(cv2.CAP_PROP_FPS)
    output = None

    out_csv_name = out_csv_path
    frame = 0

    mp_drawing = None
    if draw:
        mp_drawing = mp.solutions.drawing_utils
        click.echo(
            f"Creating drawing output to {video_path.split('.MP4')[0] + '_drawn.mp4'}"
        )
        output = cv2.VideoWriter(
            video_path.split(".MP4")[0] + "_drawn.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps),
            (output_height, output_width),
        )

    mp_holistic = mp.solutions.holistic

    try:
        click.echo("About to create Holistic context...")
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            click.echo("Successfully started holistic")
            with open(out_csv_name, "w") as f:
                click.echo("Successfully started csv writing")
                csv_writer = csv.writer(f, delimiter=";", lineterminator=";\n")
                csv_writer.writerow(["Frame", "Elbow Angle"])
                while cap.isOpened():
                    success, image = cap.read()

                    if not success:
                        click.echo("REACHED END OF VIDEO")
                        break

                    if draw:
                        image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if draw:
                        pose_connections = [
                            conn
                            for conn in mp_holistic.POSE_CONNECTIONS
                            if conn[0] > 10 and conn[1] > 10
                        ]

                        body_landmarks = None

                        # Only draw body landmarks (indices 11 and above)
                        if results.pose_landmarks:
                            # Create a copy of pose_landmarks with only body landmarks
                            from mediapipe.framework.formats import landmark_pb2

                            body_landmarks = landmark_pb2.NormalizedLandmarkList()
                            for i, landmark in enumerate(
                                results.pose_landmarks.landmark
                            ):
                                if i >= 11:  # Skip face landmarks (0-10)
                                    body_landmarks.landmark.add().CopyFrom(landmark)
                                else:
                                    # Add invisible dummy landmarks to maintain indexing
                                    dummy = body_landmarks.landmark.add()
                                    dummy.x = 0
                                    dummy.y = 0
                                    dummy.z = 0
                                    dummy.visibility = 0

                        mp_drawing.draw_landmarks(
                            image, body_landmarks, pose_connections
                        )

                        # cv2.imshow("Image drawn", image)

                    # Extract positions and angles
                    if results.pose_landmarks:
                        landmarks = results.pose_world_landmarks.landmark

                        # # Calculate body reference frame
                        # hip_center, forward_vec, up_vec, right_vec = (
                        #     calculate_body_reference_frame(landmarks, mp_holistic)
                        # )

                        # # Get specific joint positions
                        # left_shoulder = landmarks[
                        #     mp_holistic.PoseLandmark.RIGHT_SHOULDER
                        # ]
                        left_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                        left_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
                        # left_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]

                        # # Calculate simple 3D angles
                        # left_elbow_angle = calculate_angle_3d(
                        #     left_shoulder, left_elbow, left_wrist
                        # )

                        # # Calculate upper arm orientation relative to body
                        # upper_arm_orientation = (
                        #     calculate_limb_orientation_relative_to_body(
                        #         left_shoulder,
                        #         left_elbow,
                        #         forward_vec,
                        #         up_vec,
                        #         right_vec,
                        #     )
                        # )

                        # Calculate forearm orientation relative to body
                        # forearm_orientation = (
                        #     calculate_limb_orientation_relative_to_body(
                        #         left_elbow, left_wrist, forward_vec, up_vec, right_vec
                        #     )
                        # )

                        # Print results
                        # print(f"Left Elbow Flexion: {left_elbow_angle:.2f}°")
                        # print(
                        #     f"Upper Arm - Flex/Ext: {upper_arm_orientation['flexion_extension']:.1f}°, "
                        #     f"Abd/Add: {upper_arm_orientation['abduction_adduction']:.1f}°, "
                        #     f"Rotation: {upper_arm_orientation['rotation']:.1f}°"
                        # )
                        if draw:
                            landmarks_2d = results.pose_landmarks.landmark

                            left_elbow_2d = landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_ELBOW
                            ]
                            left_shoulder_2d = landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_SHOULDER
                            ]
                            # Display on image
                            # h, w, _ = image.shape
                            # cv2.putText(
                            #     image,
                            #     f"Flex: {forearm_orientation['flexion_extension']:.1f}",
                            #     (
                            #         int(left_elbow_2d.x * w),
                            #         int(left_elbow_2d.y * h) - 20,
                            #     ),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     2,
                            #     (255, 255, 0),
                            #     2,
                            # )
                            output.write(image)

                        # csv_writer.writerow(
                        #     [frame, forearm_orientation["flexion_extension"]]
                        # )

                        # cv2.putText(
                        #     image,
                        #     f"Flex: {forearm_orientation['abduction_adduction']:.1f}",
                        #     (int(left_elbow_2d.x * w), int(left_elbow_2d.y * h)),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )

                        # cv2.putText(
                        #     image,
                        #     f"Flex: {upper_arm_orientation['flexion_extension']:.1f}",
                        #     (
                        #         int(left_shoulder_2d.x * w),
                        #         int(left_shoulder_2d.y * h) - 20,
                        #     ),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )

                        # cv2.putText(
                        #     image,
                        #     f"Abd: {upper_arm_orientation['abduction_adduction']:.1f}",
                        #     (int(left_shoulder_2d.x * w), int(left_shoulder_2d.y * h)),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (255, 255, 0),
                        #     1,
                        # )
                    else:
                        click.echo(f"No pose analysed in frame {frame}")
                    frame += 1
    except Exception as e:
        click.echo(f"ERROR creating Holistic: {e}")
        import traceback

        click.echo(traceback.format_exc())

    cap.release()
    output.release()


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("outdirectory", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--width", default=1920)
@click.option("--height", default=1080)
@click.option("--fps", default=60)
@click.option("--rotate", default=False)
def batch_resize(directory, outdirectory, width, height, fps, rotate):
    try:
        os.makedirs(outdirectory)
    except OSError:
        click.echo("Output directory exists, try another path!")
        exit(-1)

    click.echo(f"Resizing everything in {directory} to {width}, {height}")
    videos = []
    for video in os.listdir(directory):
        videos.append({"path": directory + video, "file_name": video})

    LEN_VIDS = len(videos)

    click.echo(f"Found {LEN_VIDS} video(s), including: {videos[0]['file_name']})")

    for index, video_to_resize in enumerate(videos):
        click.echo(f"Resizing {index} / {LEN_VIDS} files...")
        outfile_location: str = (
            outdirectory
            + "/"
            + video_to_resize["file_name"].split(".")[0]
            + "_resized"
            + OUT_FORMAT
        )
        resize_and_write(
            video_to_resize["path"], outfile_location, width, height, fps, rotate
        )

    click.echo("Files resized and/or rotated!")


@cli.command()
@click.argument("video", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    "outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False)
)
@click.argument("start")
@click.argument("end")
def trim_video(video, outfile, start, end):
    # start and end are in seconds
    click.echo(f"Trimming video: {video} into {outfile}...")
    trim_video_file_and_write(video, outfile, start, end)
    click.echo(f"Done!")


@cli.command()
@click.argument(
    "video_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--draw", default=False, is_flag=True)
def process_one(video_path, draw):
    file_name = video_path.split("/")[-1]
    file_name_no_extension = file_name.split(".")[0]
    folder_path = video_path[: -len(file_name)]
    csv_path = f"{folder_path}{file_name_no_extension}.csv"
    click.echo(f"Processing {file_name_no_extension}")

    process_one_video(video_path, csv_path, draw)


@cli.command()
@click.argument(
    "file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--save", "-s", is_flag=True, help="Save plot to file instead of displaying"
)
@click.option("--start", type=int, default=0, help="Start frame (inclusive)")
@click.option(
    "--end", type=int, default=None, help="End frame (inclusive, None for last frame)"
)
def plot_csv(file_path, save, start, end):
    import csv
    import os

    frames = []
    elbow_angles = []

    click.echo(f"Reading file: {file_path}")

    with open(file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=";", lineterminator=";\n")
        header = next(csv_reader, None)
        click.echo(f"Header: {header}")

        for row in csv_reader:
            if row and len(row) >= 2:
                try:
                    frame = float(row[0])
                    angle = float(row[1])

                    # Apply frame filtering
                    if frame < start:
                        continue
                    if end is not None and frame > end:
                        continue

                    frames.append(frame)
                    elbow_angles.append(angle)
                except (ValueError, IndexError) as e:
                    click.echo(f"Skipping row {row}: {e}")
                    continue

    click.echo(f"Loaded {len(frames)} data points (frames {start} to {end or 'end'})")

    if len(frames) == 0:
        click.echo("ERROR: No data loaded!")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(frames, elbow_angles, marker="o", linestyle="-", linewidth=2, markersize=4)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Elbow Angle", fontsize=12)

    # Update title to show trim range
    if start > 0 or end is not None:
        title = f"Elbow Angle vs Frame (frames {start}-{end or 'end'})"
    else:
        title = "Elbow Angle vs Frame"
    plt.title(title, fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        # Get absolute path and change extension
        output_path = os.path.splitext(os.path.abspath(file_path))[0] + "_plot.png"
        click.echo(f"Attempting to save to: {output_path}")
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            click.echo(f"✓ Plot saved successfully to: {output_path}")
        except Exception as e:
            click.echo(f"ERROR saving plot: {e}")
    else:
        click.echo("Displaying plot...")
        plt.show(block=True)

    plt.close()


# def forward_kinematics_arm(
#     shoulder_pos,
#     upper_arm_length,
#     forearm_length,
#     shoulder_flexion,
#     shoulder_abduction,
#     shoulder_rotation,
#     elbow_flexion,
# ):
#     """
#     Calculate elbow and wrist positions using forward kinematics

#     Args:
#         shoulder_pos: 3D position of shoulder (x, y, z)
#         upper_arm_length: length of upper arm segment
#         forearm_length: length of forearm segment
#         shoulder_flexion: shoulder flexion angle (0-180°, 0=down, 90=forward, 180=up)
#         shoulder_abduction: shoulder abduction angle (0-180°, 0=down, 90=lateral)
#         shoulder_rotation: shoulder rotation angle (0-180°)
#         elbow_flexion: elbow flexion angle (0-180°, 0=straight, 180=fully bent)

#     Returns:
#         elbow_pos, wrist_pos: 3D coordinates
#     """
#     # Convert angles to radians and adjust coordinate system
#     sf = np.radians(shoulder_flexion - 90)
#     sa = np.radians(shoulder_abduction)
#     sr = np.radians(shoulder_rotation)
#     ef = np.radians(elbow_flexion - 180)

#     # Calculate elbow position relative to shoulder
#     elbow_dir = np.array([0, -1, 0])

#     # Apply flexion rotation (around x-axis)
#     flex_rot = np.array(
#         [[1, 0, 0], [0, np.cos(sf), -np.sin(sf)], [0, np.sin(sf), np.cos(sf)]]
#     )

#     # Apply abduction rotation (around z-axis)
#     abd_rot = np.array(
#         [[np.cos(sa), -np.sin(sa), 0], [np.sin(sa), np.cos(sa), 0], [0, 0, 1]]
#     )

#     # Combine rotations for upper arm
#     upper_arm_dir = abd_rot @ flex_rot @ elbow_dir
#     elbow_pos = shoulder_pos + upper_arm_length * upper_arm_dir

#     # Calculate forearm direction with elbow flexion
#     forearm_dir = flex_rot @ np.array([0, np.cos(ef), np.sin(ef)])
#     forearm_dir = abd_rot @ forearm_dir

#     wrist_pos = elbow_pos + forearm_length * forearm_dir

#     return elbow_pos, wrist_pos


# @cli.command()
# @click.option(
#     "--upper-arm-length", "-a", default=0.3, help="Upper arm length in meters"
# )
# @click.option("--forearm-length", "-b", default=0.3, help="Forearm length in meters")
# def pose_estimation_3d_with_plotting(upper_arm_length, forearm_length):
#     """
#     Process video with pose estimation and show dual 3D plots:
#     1. Actual joint positions from MediaPipe
#     2. Forward kinematics reconstruction from angles
#     """

#     # Initialize MediaPipe
#     mp_holistic = mp.solutions.holistic
#     mp_drawing = mp.solutions.drawing_utils

#     # Open video
#     cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         click.echo("ERROR: Cannot open video")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Setup matplotlib with 3 subplots
#     plt.ion()
#     fig = plt.figure(figsize=(18, 6))

#     # Left: video feed
#     ax_video = fig.add_subplot(131)
#     ax_video.axis("off")
#     ax_video.set_title("Video Feed", fontsize=14)

#     # Middle: actual joint positions
#     ax_actual = fig.add_subplot(132, projection="3d")
#     ax_actual.set_xlabel("X (m)")
#     ax_actual.set_ylabel("Y (m)")
#     ax_actual.set_zlabel("Z (m)")
#     ax_actual.set_title("Actual Joint Positions (MediaPipe)", fontsize=14)

#     # Right: forward kinematics
#     ax_fk = fig.add_subplot(133, projection="3d")
#     ax_fk.set_xlabel("X (m)")
#     ax_fk.set_ylabel("Y (m)")
#     ax_fk.set_zlabel("Z (m)")
#     ax_fk.set_title("Forward Kinematics Reconstruction", fontsize=14)

#     # Set consistent limits for both 3D plots
#     for ax in [ax_actual, ax_fk]:
#         ax.set_xlim([-0.6, 0.6])
#         ax.set_ylim([-0.6, 0.6])
#         ax.set_zlim([-0.8, 0.4])

#     # Initialize plot elements for actual joints
#     (actual_arm_line,) = ax_actual.plot(
#         [], [], [], "b-", linewidth=3, marker="o", markersize=8, label="Arm"
#     )
#     (actual_wrist_trail,) = ax_actual.plot(
#         [], [], [], "r-", linewidth=1, alpha=0.5, label="Wrist Trail"
#     )

#     # Initialize plot elements for FK
#     (fk_arm_line,) = ax_fk.plot(
#         [], [], [], "g-", linewidth=3, marker="o", markersize=8, label="FK Arm"
#     )
#     (fk_wrist_trail,) = ax_fk.plot(
#         [], [], [], "m-", linewidth=1, alpha=0.5, label="FK Wrist Trail"
#     )

#     # Add legends
#     ax_actual.legend(loc="upper right")
#     ax_fk.legend(loc="upper right")

#     # Trail storage
#     actual_wrist_positions = []
#     fk_wrist_positions = []
#     max_trail_length = 50

#     video_img = None

#     # Create checkbox for toggling FK plot
#     checkbox_ax = plt.axes([0.01, 0.5, 0.15, 0.15])
#     checkbox = CheckButtons(checkbox_ax, ["Show FK Plot", "Show Trails"], [True, True])

#     show_fk = [True]  # Use list to allow modification in nested function
#     show_trails = [True]

#     def toggle_visibility(label):
#         if label == "Show FK Plot":
#             show_fk[0] = not show_fk[0]
#             ax_fk.set_visible(show_fk[0])
#         elif label == "Show Trails":
#             show_trails[0] = not show_trails[0]
#             actual_wrist_trail.set_visible(show_trails[0])
#             fk_wrist_trail.set_visible(show_trails[0])
#         fig.canvas.draw_idle()

#     checkbox.on_clicked(toggle_visibility)

#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5, min_tracking_confidence=0.5
#     ) as holistic:

#         frame_count = 0

#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 click.echo("End of video")
#                 break

#             # Process image
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image_rgb)

#             # Draw pose landmarks on video
#             if results.pose_landmarks:
#                 pose_connections = [
#                     conn
#                     for conn in mp_holistic.POSE_CONNECTIONS
#                     if conn[0] > 10 and conn[1] > 10
#                 ]
#                 mp_drawing.draw_landmarks(
#                     image_rgb, results.pose_landmarks, pose_connections
#                 )

#                 # Get 3D world landmarks
#                 landmarks = results.pose_world_landmarks.landmark

#                 # Get arm landmarks (RIGHT arm)
#                 shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
#                 elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
#                 wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]

#                 shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
#                 elbow_pos_actual = np.array([elbow.x, elbow.y, elbow.z])
#                 wrist_pos_actual = np.array([wrist.x, wrist.y, wrist.z])

#                 # Calculate body reference frame
#                 from .utils.utils import (
#                     calculate_body_reference_frame,
#                     calculate_limb_orientation_relative_to_body,
#                 )

#                 hip_center, forward_vec, up_vec, right_vec = (
#                     calculate_body_reference_frame(landmarks, mp_holistic)
#                 )

#                 # Get upper arm and forearm orientations
#                 upper_arm_orientation = calculate_limb_orientation_relative_to_body(
#                     shoulder, elbow, forward_vec, up_vec, right_vec
#                 )

#                 forearm_orientation = calculate_limb_orientation_relative_to_body(
#                     elbow, wrist, forward_vec, up_vec, right_vec
#                 )

#                 # Extract angles
#                 shoulder_flexion = upper_arm_orientation["flexion_extension"]
#                 shoulder_abduction = upper_arm_orientation["abduction_adduction"]
#                 shoulder_rotation = upper_arm_orientation["rotation"]
#                 elbow_flexion = forearm_orientation["flexion_extension"]

#                 # Forward kinematics
#                 elbow_fk, wrist_fk = forward_kinematics_arm(
#                     shoulder_pos,
#                     upper_arm_length,
#                     forearm_length,
#                     shoulder_flexion,
#                     shoulder_abduction,
#                     shoulder_rotation,
#                     elbow_flexion,
#                 )

#                 # Update actual joint positions plot
#                 actual_x = [shoulder_pos[0], elbow_pos_actual[0], wrist_pos_actual[0]]
#                 actual_y = [shoulder_pos[1], elbow_pos_actual[1], wrist_pos_actual[1]]
#                 actual_z = [shoulder_pos[2], elbow_pos_actual[2], wrist_pos_actual[2]]

#                 actual_arm_line.set_data(actual_x, actual_y)
#                 actual_arm_line.set_3d_properties(actual_z)

#                 # Update FK plot
#                 fk_x = [shoulder_pos[0], elbow_fk[0], wrist_fk[0]]
#                 fk_y = [shoulder_pos[1], elbow_fk[1], wrist_fk[1]]
#                 fk_z = [shoulder_pos[2], elbow_fk[2], wrist_fk[2]]

#                 fk_arm_line.set_data(fk_x, fk_y)
#                 fk_arm_line.set_3d_properties(fk_z)

#                 # Update trails if enabled
#                 if show_trails[0]:
#                     # Actual wrist trail
#                     actual_wrist_positions.append(wrist_pos_actual)
#                     if len(actual_wrist_positions) > max_trail_length:
#                         actual_wrist_positions.pop(0)

#                     if len(actual_wrist_positions) > 1:
#                         trail = np.array(actual_wrist_positions)
#                         actual_wrist_trail.set_data(trail[:, 0], trail[:, 1])
#                         actual_wrist_trail.set_3d_properties(trail[:, 2])

#                     # FK wrist trail
#                     fk_wrist_positions.append(wrist_fk)
#                     if len(fk_wrist_positions) > max_trail_length:
#                         fk_wrist_positions.pop(0)

#                     if len(fk_wrist_positions) > 1:
#                         trail = np.array(fk_wrist_positions)
#                         fk_wrist_trail.set_data(trail[:, 0], trail[:, 1])
#                         fk_wrist_trail.set_3d_properties(trail[:, 2])

#                 # Add angle text to video
#                 cv2.putText(
#                     image_rgb,
#                     f"Shoulder Flex: {shoulder_flexion:.1f}deg",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     image_rgb,
#                     f"Shoulder Abd: {shoulder_abduction:.1f}deg",
#                     (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )
#                 cv2.putText(
#                     image_rgb,
#                     f"Elbow Flex: {elbow_flexion:.1f}deg",
#                     (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (255, 255, 0),
#                     2,
#                 )

#                 # Calculate and display error between actual and FK
#                 wrist_error = np.linalg.norm(wrist_pos_actual - wrist_fk)
#                 cv2.putText(
#                     image_rgb,
#                     f"FK Error: {wrist_error*100:.1f}cm",
#                     (10, 120),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 255),
#                     2,
#                 )

#             # Update video display
#             if video_img is None:
#                 video_img = ax_video.imshow(image_rgb)
#             else:
#                 video_img.set_data(image_rgb)

#             Check for window close
#                         if not plt.fignum_exists(fig.number):
#                             break

#             # Update display
#             plt.pause(0.001)

#             frame_count += 1
#             if frame_count % 30 == 0:
#                 click.echo(f"Processed frame {frame_count}")

#             # Check for window close
#             if not plt.fignum_exists(fig.number):
#                 break

#     cap.release()
#     plt.ioff()
#     plt.show()
#     click.echo("Processing complete!")


@cli.command()
@click.argument(
    "file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("-s", "--show", is_flag=True, default=False)
def process_img(file_path, show):
    click.echo(f"Searching for file {file_path}")

    img = cv2.imread(file_path)
    img_copy = img.copy()

    try:
        positions = return_all_relevant_joint_positions(img, show)
    except Exception as ex:
        click.echo(ex)

    if show:
        try:
            img_resized = cv2.resize(img_copy, [500, 580])
            cv2.imshow("Found image", img_resized)

            resized_landmarks = cv2.resize(positions.image, [500, 580])

            # get midpoint of hip
            left_hip = positions.joint_pos["LEFT_HIP"]
            right_hip = positions.joint_pos["RIGHT_HIP"]
            midpoint_hip = midpoint(left_hip, right_hip)

            left_shoulder = positions.joint_pos["LEFT_SHOULDER"]
            right_shoulder = positions.joint_pos["RIGHT_SHOULDER"]

            left_elbow = positions.joint_pos["LEFT_ELBOW"]
            right_elbow = positions.joint_pos["RIGHT_ELBOW"]

            # click.echo(f"Left Hip {left_hip}")
            # click.echo(f"Right Hip {right_hip}")
            # click.echo(f"Midpoint Hip {midpoint_hip}")

            # create normal vector from both shoulders + middle hip
            body_plane_normal = normal_vector_of_plane_on_three_points(
                left_shoulder, right_shoulder, midpoint_hip
            )

            right_upper_arm_vector = vector_between_two_points(
                right_shoulder, right_elbow
            )
            right_vector = vector_between_two_points(left_shoulder, right_shoulder)

            down_vector = np.cross(body_plane_normal, right_vector)

            A_dot_n = np.dot(right_upper_arm_vector, body_plane_normal)
            A_dot_down = np.dot(right_upper_arm_vector, down_vector)
            A_dot_right = np.dot(right_upper_arm_vector, right_vector)

            shoulder_flexion = math.atan2(
                A_dot_n, math.sqrt(A_dot_right**2 + A_dot_down**2)
            )

            shoulder_abduction = math.atan2(A_dot_right, A_dot_down)

            right_shoulder_2d_pos = convert_landmark_2d_to_pixel_coordinates(
                700, 500, positions.joint_pos2d["RIGHT_SHOULDER"]
            )

            # cv2.putText(
            #     resized_landmarks,
            #     f"Shoulder flexion = {shoulder_flexion * RADIAN_TO_DEGREES}",
            #     [right_shoulder_2d_pos[0] + 20, right_shoulder_2d_pos[1] + 20],
            #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #     1,
            #     [255, 255, 0],
            #     1,
            # )

            # cv2.putText(
            #     resized_landmarks,
            #     f"Shoulder abduction = {shoulder_abduction * RADIAN_TO_DEGREES}",
            #     [right_shoulder_2d_pos[0] + 20, right_shoulder_2d_pos[1] + 40],
            #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #     1,
            #     [255, 255, 0],
            #     1,
            # )

            # click.echo(f"Shoulder flexion = {shoulder_flexion * RADIAN_TO_DEGREES}")
            # click.echo(f"Shoulder abduction = {shoulder_abduction * RADIAN_TO_DEGREES}")

            cv2.imshow("Landmark Positions", resized_landmarks)

            cv2.waitKey(0)
        except Exception as Ex:
            click.echo(Ex)


def calculate_forward_kinematics(
    shoulder_flexion,
    shoulder_abduction,
    elbow_flexion,
    wrist_flexion,
    upper_arm_length=0.3,
    forearm_length=0.25,
    hand_length=0.1,
):
    """
    Calculate FK from shoulder, elbow, and wrist angles
    Returns: shoulder_pos, elbow_pos, wrist_pos, hand_pos as arrays
    """
    # Shoulder is at origin
    shoulder = np.array([0.0, 0.0, 0.0])

    # Calculate elbow position from shoulder angles
    # Using spherical coordinates
    elbow_x = upper_arm_length * np.sin(shoulder_abduction) * np.cos(shoulder_flexion)
    elbow_y = upper_arm_length * np.cos(shoulder_abduction) * np.cos(shoulder_flexion)
    elbow_z = upper_arm_length * np.sin(shoulder_flexion)
    elbow = shoulder + np.array([elbow_x, elbow_y, elbow_z])

    # Upper arm direction vector
    upper_arm_dir = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)

    # For elbow flexion, we need to bend the forearm
    # The forearm rotates around an axis perpendicular to the upper arm
    # For simplicity, assume it bends in a plane

    # Calculate forearm direction after elbow flexion
    # elbow_flexion of 0 = straight arm, π = fully bent
    forearm_extension = forearm_length * np.cos(np.pi - elbow_flexion)
    forearm_lateral = forearm_length * np.sin(np.pi - elbow_flexion)

    # Create perpendicular vector for the bend direction
    if abs(upper_arm_dir[2]) < 0.9:
        perp = np.cross(upper_arm_dir, np.array([0, 0, 1]))
    else:
        perp = np.cross(upper_arm_dir, np.array([1, 0, 0]))
    perp = perp / np.linalg.norm(perp)

    wrist = elbow + upper_arm_dir * forearm_extension + perp * forearm_lateral

    # Calculate hand position from wrist flexion
    forearm_dir = (wrist - elbow) / np.linalg.norm(wrist - elbow)

    hand_extension = hand_length * np.cos(np.pi - wrist_flexion)
    hand_lateral = hand_length * np.sin(np.pi - wrist_flexion)

    # Use same perpendicular for wrist bend
    hand = wrist + forearm_dir * hand_extension + perp * hand_lateral

    return shoulder, elbow, wrist, hand


@cli.command()
@click.option("-v", "video", help="Use video file instead of webcam")
@click.option(
    "-s", "save_path", help="Save recorded joint angles to this path as a csv file"
)
@click.option(
    "-j",
    "joint_positions",
    help="Save recorded joint positions to a csv with this path",
)
@click.option("-r", "rotate", is_flag=True, help="Rotate video counter clockwise")
@click.option("-m", "mirror", is_flag=True, help="Mirror video")
def process_four_joints(video, save_path, joint_positions, rotate, mirror):
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2

    cap = None
    if not (video):
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        click.echo(f"Video provided! Analysing {video}")
        cap = cv2.VideoCapture(video)
    video_img = None

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    plt.ion()
    fig = plt.figure(figsize=(18, 6))

    # Left: video feed
    ax_video = fig.add_subplot(131)
    ax_video.axis("off")
    ax_video.set_title("Video Feed", fontsize=14)

    # Middle: actual joint positions
    ax_actual = fig.add_subplot(132, projection="3d")
    ax_actual.set_xlabel("X (m)")
    ax_actual.set_ylabel("Y (m)")
    ax_actual.set_zlabel("Z (m)")
    ax_actual.set_title("Actual Joint Positions (MediaPipe)", fontsize=14)

    # Right: forward kinematics
    ax_fk = fig.add_subplot(133, projection="3d")
    ax_fk.set_xlabel("X (m)")
    ax_fk.set_ylabel("Y (m)")
    ax_fk.set_zlabel("Z (m)")
    ax_fk.set_title("Forward Kinematics", fontsize=14)

    angle_text = fig.text(0.5, 0.95, "", fontsize=12, ha="center", color="black")

    for ax in [ax_actual, ax_fk]:
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0.0, 0.8])

    (actual_arm_line,) = ax_actual.plot(
        [], [], [], "b-", linewidth=3, marker="o", markersize=8, label="Arm"
    )

    (fk_arm_line,) = ax_fk.plot(
        [], [], [], "r-", linewidth=3, marker="o", markersize=8, label="FK"
    )

    joint_positions_dict = {
        "frame": [],
        "shoulder_flexion": [],
        "shoulder_abduction": [],
        "elbow_flexion": [],
        "wrist_flexion": [],
    }

    joint_coordinates_dict = {
        "frame": [],
        "Shoulder": [],
        "Elbow": [],
        "Wrist": [],
        "Index": [],
        "Thumb": [],
        "Pinky": [],
        "HipL": [],
        "HipR": [],
        "ShoulderL": [],
    }

    frame = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while True:
            try:
                success, image = cap.read()
                if not (success):
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if rotate:
                    image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
                if mirror:
                    image_rgb = cv2.flip(image_rgb, 1)

                results = holistic.process(image_rgb)
                height, width = image_rgb.shape[:2]
                positions = JointPositions(
                    image=None, joint_pos=None, joint_pos2d=None, successful=False
                )

                if results:
                    if results.pose_world_landmarks.landmark:
                        positions.successful = True
                        landmarks = results.pose_world_landmarks.landmark
                        landmarks_2d = results.pose_landmarks.landmark
                        positions.joint_pos = {
                            "RIGHT_SHOULDER": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                            ),
                            "RIGHT_WRIST": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
                            ),
                            "RIGHT_ELBOW": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                            ),
                            "LEFT_SHOULDER": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                            ),
                            "LEFT_WRIST": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
                            ),
                            "LEFT_ELBOW": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
                            ),
                            "LEFT_HIP": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
                            ),
                            "RIGHT_HIP": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
                            ),
                            "LEFT_THUMB": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_THUMB]
                            ),
                            "LEFT_PINKY": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_PINKY]
                            ),
                            "RIGHT_THUMB": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_THUMB]
                            ),
                            "RIGHT_PINKY": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_PINKY]
                            ),
                            "LEFT_INDEX": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.LEFT_INDEX]
                            ),
                            "RIGHT_INDEX": landmark_to_dict(
                                landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX]
                            ),
                        }
                        positions.joint_pos2d = {
                            "RIGHT_SHOULDER": landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_SHOULDER
                            ],
                            "RIGHT_WRIST": landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_WRIST
                            ],
                            "RIGHT_ELBOW": landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_ELBOW
                            ],
                            "LEFT_SHOULDER": landmarks_2d[
                                mp_holistic.PoseLandmark.LEFT_SHOULDER
                            ],
                            "LEFT_WRIST": landmarks_2d[
                                mp_holistic.PoseLandmark.LEFT_WRIST
                            ],
                            "LEFT_ELBOW": landmarks_2d[
                                mp_holistic.PoseLandmark.LEFT_ELBOW
                            ],
                            "LEFT_HIP": landmarks_2d[mp_holistic.PoseLandmark.LEFT_HIP],
                            "RIGHT_HIP": landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_HIP
                            ],
                            "RIGHT_WRIST": landmarks_2d[
                                mp_holistic.PoseLandmark.RIGHT_WRIST
                            ],
                        }

                        pose_connections = [
                            conn
                            for conn in mp_holistic.POSE_CONNECTIONS
                            if conn[0] > 10 and conn[1] > 10
                        ]

                        body_landmarks = None

                        # Only draw body landmarks (indices 11 and above)
                        if results.pose_landmarks:
                            # Create a copy of pose_landmarks with only body landmarks

                            body_landmarks = landmark_pb2.NormalizedLandmarkList()
                            for i, landmark in enumerate(
                                results.pose_landmarks.landmark
                            ):
                                if i >= 11:  # Skip face landmarks (0-10)
                                    body_landmarks.landmark.add().CopyFrom(landmark)
                                else:
                                    # Add invisible dummy landmarks to maintain indexing
                                    dummy = body_landmarks.landmark.add()
                                    dummy.x = 0
                                    dummy.y = 0
                                    dummy.z = 0
                                    dummy.visibility = 0

                    mp_drawing.draw_landmarks(
                        image_rgb, body_landmarks, pose_connections
                    )
                    positions.image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    ## START OF SHOULDER CALCULATIONS
                    # get midpoint of hip
                    left_hip = positions.joint_pos["LEFT_HIP"]
                    right_hip = positions.joint_pos["RIGHT_HIP"]
                    midpoint_hip = midpoint(left_hip, right_hip)

                    left_shoulder = positions.joint_pos["LEFT_SHOULDER"]
                    right_shoulder = positions.joint_pos["RIGHT_SHOULDER"]

                    left_elbow = positions.joint_pos["LEFT_ELBOW"]
                    right_elbow = positions.joint_pos["RIGHT_ELBOW"]

                    right_wrist = positions.joint_pos["RIGHT_WRIST"]
                    hand = positions.joint_pos["RIGHT_INDEX"]

                    joint_coordinates_dict["HipL"].append(left_hip)
                    joint_coordinates_dict["HipR"].append(right_hip)
                    joint_coordinates_dict["ShoulderL"].append(left_shoulder)
                    joint_coordinates_dict["Shoulder"].append(right_shoulder)
                    joint_coordinates_dict["Elbow"].append(right_elbow)
                    joint_coordinates_dict["Wrist"].append(right_wrist)
                    joint_coordinates_dict["Index"].append(hand)
                    joint_coordinates_dict["Thumb"].append(
                        positions.joint_pos["RIGHT_THUMB"]
                    )
                    joint_coordinates_dict["Pinky"].append(
                        positions.joint_pos["RIGHT_PINKY"]
                    )

                    # click.echo(f"Left Hip {left_hip}")
                    # click.echo(f"Right Hip {right_hip}")
                    # click.echo(f"Midpoint Hip {midpoint_hip}")

                    # create normal vector from both shoulders + middle hip
                    body_plane_normal = normal_vector_of_plane_on_three_points(
                        left_shoulder, right_shoulder, midpoint_hip
                    )

                    right_upper_arm_vector = vector_between_two_points(
                        right_shoulder, right_elbow
                    )
                    right_vector = vector_between_two_points(
                        left_shoulder, right_shoulder
                    )

                    down_vector = np.cross(body_plane_normal, right_vector)

                    A_dot_n = np.dot(right_upper_arm_vector, body_plane_normal)
                    A_dot_down = np.dot(right_upper_arm_vector, down_vector)
                    A_dot_right = np.dot(right_upper_arm_vector, right_vector)

                    shoulder_flexion = math.atan2(
                        A_dot_n, math.sqrt(A_dot_right**2 + A_dot_down**2)
                    )

                    shoulder_abduction = math.atan2(A_dot_right, A_dot_down)

                    elbow_flexion = angle_between_three_points(
                        right_shoulder, right_elbow, right_wrist
                    )
                    wrist_flexion = angle_between_three_points(
                        right_elbow, right_wrist, hand
                    )

                    joint_coordinates_dict["frame"].append(frame)
                    joint_positions_dict["frame"].append(frame)
                    frame += 1
                    joint_positions_dict["shoulder_flexion"].append(
                        shoulder_flexion * RADIAN_TO_DEGREES
                    )
                    joint_positions_dict["shoulder_abduction"].append(
                        shoulder_abduction * RADIAN_TO_DEGREES
                    )
                    joint_positions_dict["elbow_flexion"].append(
                        elbow_flexion * RADIAN_TO_DEGREES
                    )
                    joint_positions_dict["wrist_flexion"].append(
                        wrist_flexion * RADIAN_TO_DEGREES
                    )

                    angle_text.set_text(
                        f"Shoulder Flexion: {shoulder_flexion * RADIAN_TO_DEGREES:.1f}° | "
                        f"Shoulder Abduction: {shoulder_abduction * RADIAN_TO_DEGREES:.1f}°"
                        f"Elbow Flexion: {elbow_flexion * RADIAN_TO_DEGREES:.1f}° | "
                        f"Wrist Flexion: {wrist_flexion * RADIAN_TO_DEGREES:.1f}°"
                    )

                    # Calculate forward kinematics
                    fk_shoulder, fk_elbow, fk_wrist, fk_hand = (
                        calculate_forward_kinematics(
                            shoulder_flexion,
                            shoulder_abduction,
                            elbow_flexion,
                            wrist_flexion,
                        )
                    )

                    # Prepare FK data for plotting (now includes hand)
                    fk_x = [fk_shoulder[0], fk_elbow[0], fk_wrist[0], fk_hand[0]]
                    fk_z = [-fk_shoulder[1], -fk_elbow[1], -fk_wrist[1], -fk_hand[1]]
                    fk_y = [fk_shoulder[2], fk_elbow[2], fk_wrist[2], fk_hand[2]]

                    # Update FK plot
                    fk_arm_line.set_data(fk_x, fk_y)
                    fk_arm_line.set_3d_properties(fk_z)

                    # UNCOMMENT TO GET TEXT ON IMG

                    # right_shoulder_2d_pos = convert_landmark_2d_to_pixel_coordinates(
                    # height, width, positions.joint_pos2d["RIGHT_SHOULDER"]
                    # )

                    # right_elbow_2d_pos = convert_landmark_2d_to_pixel_coordinates(
                    # height, width, positions.joint_pos2d["RIGHT_ELBOW"]
                    # )
                    # right_wrist_2d_pos = convert_landmark_2d_to_pixel_coordinates(
                    # height, width, positions.joint_pos2d["RIGHT_WRIST"]
                    # )
                    # cv2.putText(
                    #     positions.image,
                    #     f"Shoulder flexion = {shoulder_flexion * RADIAN_TO_DEGREES:.2f}",
                    #     [right_shoulder_2d_pos[0] + 5, right_shoulder_2d_pos[1] + 5],
                    #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #     0.5,
                    #     [255, 0, 255],
                    #     1,
                    # )

                    # cv2.putText(
                    #     positions.image,
                    #     f"Shoulder abduction = {shoulder_abduction * RADIAN_TO_DEGREES:.2f}",
                    #     [right_shoulder_2d_pos[0] + 5, right_shoulder_2d_pos[1] + 20],
                    #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #     0.5,
                    #     [255, 0, 255],
                    #     1,
                    # )

                    # cv2.putText(
                    #     positions.image,
                    #     f"Elbow flexion = {elbow_flexion * RADIAN_TO_DEGREES:.2f}",
                    #     [right_elbow_2d_pos[0], right_elbow_2d_pos[1]],
                    #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #     0.5,
                    #     [255, 0, 255],
                    #     1,
                    # )

                    # cv2.putText(
                    #     positions.image,
                    #     f"Wrist Flexion = {wrist_flexion * RADIAN_TO_DEGREES:.2f}",
                    #     [right_wrist_2d_pos[0], right_wrist_2d_pos[1]],
                    #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #     0.5,
                    #     [255, 0, 255],
                    #     1,
                    # )

                    actual_x = [
                        right_shoulder[0],
                        right_elbow[0],
                        right_wrist[0],
                        hand[0],
                    ]
                    actual_z = [
                        -right_shoulder[1],
                        -right_elbow[1],
                        -right_wrist[1],
                        -hand[1],
                    ]
                    actual_y = [
                        right_shoulder[2],
                        right_elbow[2],
                        right_wrist[2],
                        hand[2],
                    ]
                    ## END SHOULDER ROTATION STUFF

                # cv2.imshow("Webcam Feed", image

                ## MATPLOTLIB PLOTTING STUFF

                # actual_x = [shoulder_pos[0], elbow_pos_actual[0], wrist_pos_actual[0]]
                # actual_y = [shoulder_pos[1], elbow_pos_actual[1], wrist_pos_actual[1]]
                # actual_z = [shoulder_pos[2], elbow_pos_actual[2], wrist_pos_actual[2]]

                actual_arm_line.set_data(actual_x, actual_y)
                actual_arm_line.set_3d_properties(actual_z)

                # Update video display
                if video_img is None:
                    video_img = ax_video.imshow(
                        cv2.cvtColor(positions.image, cv2.COLOR_BGR2RGB)
                    )
                else:
                    video_img.set_data(cv2.cvtColor(positions.image, cv2.COLOR_BGR2RGB))

                plt.pause(0.001)

                # Check for window close
                if not plt.fignum_exists(fig.number):
                    break

                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            except Exception as ex:
                click.echo("Error!")
                click.echo(ex)

    if save_path:
        with open(save_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame",
                "shoulder_flexion",
                "shoulder_abduction",
                "elbow_flexion",
                "wrist_flexion",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for frame in joint_positions_dict["frame"]:
                row = {
                    "frame": frame,
                    "shoulder_flexion": joint_positions_dict["shoulder_flexion"][frame],
                    "shoulder_abduction": joint_positions_dict["shoulder_abduction"][
                        frame
                    ],
                    "elbow_flexion": joint_positions_dict["elbow_flexion"][frame],
                    "wrist_flexion": joint_positions_dict["wrist_flexion"][frame],
                }
                writer.writerow(row)

    if joint_positions:
        with open(joint_positions, "w", newline="") as csvfile:
            fieldnames = ["frame"]
            joint_names = [
                "Shoulder",
                "Elbow",
                "Wrist",
                "Index",
                "Thumb",
                "Pinky",
                "HipL",
                "HipR",
                "ShoulderL",
            ]

            for joint in joint_names:
                for coord in ["x", "y", "z"]:
                    fieldnames.append(f"{joint} {coord}")

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for frame in joint_coordinates_dict["frame"]:
                # click.echo(f"Processing frame: {frame}")
                # click.echo(joint_coordinates_dict)
                row = {
                    "frame": frame,
                    "Shoulder x": joint_coordinates_dict["Shoulder"][frame][0],
                    "Shoulder y": joint_coordinates_dict["Shoulder"][frame][1],
                    "Shoulder z": joint_coordinates_dict["Shoulder"][frame][2],
                    "Elbow x": joint_coordinates_dict["Elbow"][frame][0],
                    "Elbow y": joint_coordinates_dict["Elbow"][frame][1],
                    "Elbow z": joint_coordinates_dict["Elbow"][frame][2],
                    "Wrist x": joint_coordinates_dict["Wrist"][frame][0],
                    "Wrist y": joint_coordinates_dict["Wrist"][frame][1],
                    "Wrist z": joint_coordinates_dict["Wrist"][frame][2],
                    "Index x": joint_coordinates_dict["Index"][frame][0],
                    "Index y": joint_coordinates_dict["Index"][frame][1],
                    "Index z": joint_coordinates_dict["Index"][frame][2],
                    "Pinky x": joint_coordinates_dict["Pinky"][frame][0],
                    "Pinky y": joint_coordinates_dict["Pinky"][frame][1],
                    "Pinky z": joint_coordinates_dict["Pinky"][frame][2],
                    "Thumb x": joint_coordinates_dict["Thumb"][frame][0],
                    "Thumb y": joint_coordinates_dict["Thumb"][frame][1],
                    "Thumb z": joint_coordinates_dict["Thumb"][frame][2],
                    "ShoulderL x": joint_coordinates_dict["ShoulderL"][frame][0],
                    "ShoulderL y": joint_coordinates_dict["ShoulderL"][frame][1],
                    "ShoulderL z": joint_coordinates_dict["ShoulderL"][frame][2],
                    "HipL x": joint_coordinates_dict["HipL"][frame][0],
                    "HipL y": joint_coordinates_dict["HipL"][frame][1],
                    "HipL z": joint_coordinates_dict["HipL"][frame][2],
                    "HipR x": joint_coordinates_dict["HipR"][frame][0],
                    "HipR y": joint_coordinates_dict["HipR"][frame][1],
                    "HipR z": joint_coordinates_dict["HipR"][frame][2],
                }
                # click.echo(row)
                writer.writerow(row)

    cv2.destroyAllWindows()
    cap.release()


@cli.command()
@click.option("-u", "upper_arm_length", default=0.28, help="Upper arm length (m)")
@click.option("-f", "lower_arm_length", default=0.25, help="Forearm length (m)")
@click.option("-t", "time_step", default=0.1, help="Time step (s)")
@click.option("-s", "time_start", default=0.0, help="Time start (s)")
@click.option("-e", "time_end", default=1.0, help="Time end (s)")
@click.option(
    "-i", "interp_type", default=0, help="Interpolation type, 0=linear, 1=cubic"
)
@click.option("-q1s", "q1s", default=0.0, help="Q1 Start Angle (degrees)")
@click.option("-q1e", "q1e", default=0.0, help="Q1 End Angle (degrees)")
@click.option("-q2s", "q2s", default=0.0, help="Q2 Start Angle (degrees)")
@click.option("-q2e", "q2e", default=0.0, help="Q2 End Angle (degrees)")
@click.option("-q3s", "q3s", default=0.0, help="Q3 Start Angle (degrees)")
@click.option("-q3e", "q3e", default=0.0, help="Q3 End Angle (degrees)")
@click.option("-q4s", "q4s", default=0.0, help="Q4 Start Angle (degrees)")
@click.option("-q4e", "q4e", default=0.0, help="Q4 End Angle (degrees)")
@click.option("-q5s", "q5s", default=0.0, help="Q5 Start Angle (degrees)")
@click.option("-q5e", "q5e", default=0.0, help="Q5 End Angle (degrees)")
@click.option("-q6s", "q6s", default=0.0, help="Q6 Start Angle (degrees)")
@click.option("-q6e", "q6e", default=0.0, help="Q6 End Angle (degrees)")
@click.option("-q7s", "q7s", default=0.0, help="Q7 Start Angle (degrees)")
@click.option("-q7e", "q7e", default=0.0, help="Q7 End Angle (degrees)")
@click.option("-n", "file_name", help="Output file name")
def create_path(
    upper_arm_length,
    lower_arm_length,
    time_step,
    time_start,
    time_end,
    interp_type,
    q1s,
    q1e,
    q2s,
    q2e,
    q3s,
    q3e,
    q4s,
    q4e,
    q5s,
    q5e,
    q6s,
    q6e,
    q7s,
    q7e,
    file_name,
):

    num_iterations = (time_end - time_start) / time_step

    delta_q1 = ((q1e - q1s) * DEGREES_TO_RADIANS) / num_iterations
    q1s_rad, q1e_rad = (q1s * DEGREES_TO_RADIANS, q1e * DEGREES_TO_RADIANS)

    delta_q2 = ((q2e - q2s) * DEGREES_TO_RADIANS) / num_iterations
    q2s_rad, q2e_rad = (q2s * DEGREES_TO_RADIANS, q2e * DEGREES_TO_RADIANS)

    delta_q3 = ((q3e - q3s) * DEGREES_TO_RADIANS) / num_iterations
    q3s_rad, q3e_rad = (q3s * DEGREES_TO_RADIANS, q3e * DEGREES_TO_RADIANS)

    delta_q4 = ((q4e - q4s) * DEGREES_TO_RADIANS) / num_iterations
    q4s_rad, q4e_rad = (q4s * DEGREES_TO_RADIANS, q4e * DEGREES_TO_RADIANS)

    delta_q5 = ((q5e - q5s) * DEGREES_TO_RADIANS) / num_iterations
    q5s_rad, q5e_rad = (q5s * DEGREES_TO_RADIANS, q5e * DEGREES_TO_RADIANS)

    delta_q6 = ((q6e - q6s) * DEGREES_TO_RADIANS) / num_iterations
    q6s_rad, q6e_rad = (q6s * DEGREES_TO_RADIANS, q6e * DEGREES_TO_RADIANS)

    delta_q7 = ((q7e - q7s) * DEGREES_TO_RADIANS) / num_iterations
    q7s_rad, q7e_rad = (q7s * DEGREES_TO_RADIANS, q7e * DEGREES_TO_RADIANS)

    def create_dh_matrix(theta_n, alpha_n, d_n, r_n) -> np.ndarray:
        sin_thetha_n = math.sin(theta_n)
        cos_thetha_n = math.cos(theta_n)
        sin_alpha_n = math.sin(alpha_n)
        cos_alpha_n = math.cos(alpha_n)
        row_1 = np.array(
            [
                cos_thetha_n,
                -sin_thetha_n * cos_alpha_n,
                sin_thetha_n * sin_alpha_n,
                r_n * cos_thetha_n,
            ]
        )
        row_2 = np.array(
            [
                sin_thetha_n,
                cos_thetha_n * cos_alpha_n,
                -cos_thetha_n * sin_alpha_n,
                r_n * sin_thetha_n,
            ]
        )
        row_3 = np.array([0, sin_alpha_n, cos_alpha_n, d_n])
        row_4 = np.array([0, 0, 0, 1])

        dh_matrix = np.array([row_1, row_2, row_3, row_4])

        return dh_matrix

    pi_over_2 = math.pi / 2

    with open(file_name, "w", newline="") as csvfile:
        field_names = [
            "Time (s)",
            "Shoulder x",
            "Shoulder y",
            "Shoulder z",
            "Elbow x",
            "Elbow y",
            "Elbow z",
            "Wrist x",
            "Wrist y",
            "Wrist z",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        for index, i in enumerate(
            np.arange(time_start, time_end + time_step, time_step)
        ):
            row = {}
            q1 = q1s_rad + delta_q1 * index
            q2 = q2s_rad + delta_q2 * index
            q3 = q3s_rad + delta_q3 * index
            q4 = q4s_rad + delta_q4 * index
            q5 = q5s_rad + delta_q5 * index
            q6 = q6s_rad + delta_q6 * index
            q7 = q7s_rad + delta_q7 * index

            dh_1 = create_dh_matrix(q1, pi_over_2, 0, 0)  # Shoulder abduction/adduction
            dh_2 = create_dh_matrix(q2, pi_over_2, 0, 0)  # Shoulder flexion/extension
            dh_3 = create_dh_matrix(
                q3, -pi_over_2, 0, upper_arm_length
            )  # Shoulder rotation + upper arm
            dh_4 = create_dh_matrix(q4, pi_over_2, 0, 0)  # Elbow flexion/extension
            dh_5 = create_dh_matrix(
                q5, -pi_over_2, 0, lower_arm_length
            )  # Forearm rotation + lower arm
            dh_6 = create_dh_matrix(q6, pi_over_2, 0, 0)  # Wrist flexion/extension
            dh_7 = create_dh_matrix(q7, 0, 0, 0)  # Wrist radial/ulnar deviation

            shoulder_pos = (0, 0, 0)

            dh_to_elbow = dh_1 @ dh_2 @ dh_3
            elbow_pos = [dh_to_elbow[0][3], dh_to_elbow[1][3], dh_to_elbow[2][3]]

            dh_to_wrist = dh_to_elbow @ dh_4 @ dh_5
            wrist_pos = [dh_to_wrist[0][3], dh_to_wrist[1][3], dh_to_wrist[2][3]]

            upper_segment_length = np.linalg.norm(
                np.array(elbow_pos) - np.array(shoulder_pos)
            )
            lower_segment_length = np.linalg.norm(
                np.array(wrist_pos) - np.array(elbow_pos)
            )
            print(
                f"Upper segment: {upper_segment_length}, Lower segment: {lower_segment_length}"
            )

            row["Time (s)"] = i
            row["Shoulder x"] = shoulder_pos[0]
            row["Shoulder y"] = shoulder_pos[1]
            row["Shoulder z"] = shoulder_pos[2]
            row["Elbow x"] = elbow_pos[0]
            row["Elbow y"] = elbow_pos[1]
            row["Elbow z"] = elbow_pos[2]
            row["Wrist x"] = wrist_pos[0]
            row["Wrist y"] = wrist_pos[1]
            row["Wrist z"] = wrist_pos[2]

            writer.writerow(row)

    click.echo(f"Successfully written csv to {file_name}")


def plot_arm_joints(csv_file, x_axis_name="Time (s)", start_frame=0, end_frame=-1):
    """
    Plot arm joint positions from CSV file with interactive time slider.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing joint positions
    """
    # Read the CSV file
    df_raw = pd.read_csv(csv_file)

    df = df_raw.iloc[start_frame:end_frame]

    # Get the time steps
    times = df[x_axis_name].values
    num_frames = len(times)

    # Extract joint positions
    shoulder_x = df["Shoulder x"].values
    shoulder_y = df["Shoulder y"].values
    shoulder_z = df["Shoulder z"].values

    elbow_x = df["Elbow x"].values
    elbow_y = df["Elbow y"].values
    elbow_z = df["Elbow z"].values

    wrist_x = df["Wrist x"].values
    wrist_y = df["Wrist y"].values
    wrist_z = df["Wrist z"].values

    # Calculate bounds for consistent scaling
    all_x = np.concatenate([shoulder_x, elbow_x, wrist_x])
    all_y = np.concatenate([shoulder_y, elbow_y, wrist_z])
    all_z = np.concatenate([shoulder_z, elbow_z, wrist_z])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()

    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Adjust position to make room for slider
    plt.subplots_adjust(bottom=0.15)

    # Initialize the plot with first frame
    def plot_frame(frame_idx):
        # Store current view angles before clearing
        if hasattr(ax, "elev"):  # Check if view has been set
            current_elev = ax.elev
            current_azim = ax.azim
        else:
            current_elev = 20
            current_azim = 45

        ax.clear()

        # Get positions for this frame
        shoulder_pos = [
            shoulder_x[frame_idx],
            shoulder_y[frame_idx],
            shoulder_z[frame_idx],
        ]
        elbow_pos = [elbow_x[frame_idx], elbow_y[frame_idx], elbow_z[frame_idx]]
        wrist_pos = [wrist_x[frame_idx], wrist_y[frame_idx], wrist_z[frame_idx]]

        # Plot the arm segments
        # Upper arm (shoulder to elbow)
        ax.plot(
            [shoulder_pos[0], elbow_pos[0]],
            [shoulder_pos[1], elbow_pos[1]],
            [shoulder_pos[2], elbow_pos[2]],
            "b-",
            linewidth=3,
            label="Upper Arm",
        )

        # Forearm (elbow to wrist)
        ax.plot(
            [elbow_pos[0], wrist_pos[0]],
            [elbow_pos[1], wrist_pos[1]],
            [elbow_pos[2], wrist_pos[2]],
            "g-",
            linewidth=3,
            label="Forearm",
        )

        # Plot the joints
        ax.scatter(
            *shoulder_pos, c="red", s=100, marker="o", label="Shoulder", depthshade=True
        )
        ax.scatter(
            *elbow_pos, c="orange", s=100, marker="o", label="Elbow", depthshade=True
        )
        ax.scatter(
            *wrist_pos, c="purple", s=100, marker="o", label="Wrist", depthshade=True
        )

        # Plot trajectory traces (optional - shows path over time)
        ax.plot(
            shoulder_x[: frame_idx + 1],
            shoulder_y[: frame_idx + 1],
            shoulder_z[: frame_idx + 1],
            "r--",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            elbow_x[: frame_idx + 1],
            elbow_y[: frame_idx + 1],
            elbow_z[: frame_idx + 1],
            "orange",
            alpha=0.3,
            linewidth=1,
            linestyle="--",
        )
        ax.plot(
            wrist_x[: frame_idx + 1],
            wrist_y[: frame_idx + 1],
            wrist_z[: frame_idx + 1],
            "m--",
            alpha=0.3,
            linewidth=1,
        )

        # Set consistent axis limits
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])

        # Labels and title
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_zlabel("Z (m)", fontsize=12)
        ax.set_title(
            f"Arm Joint Positions - Time: {times[frame_idx]:.3f}s (Frame {frame_idx+1}/{num_frames})",
            fontsize=20,
            fontweight="bold",
        )

        # Add legend
        ax.legend(loc="upper right", fontsize=15)

        # Set viewing angle
        ax.view_init(elev=current_elev, azim=current_azim)

        fig.canvas.draw_idle()

    # Create slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Time Step",
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1,
    )

    # Update function for slider
    def update(val):
        frame_idx = int(slider.val)
        plot_frame(frame_idx)

    slider.on_changed(update)

    # Plot initial frame
    plot_frame(0)

    plt.show()


@cli.command()
@click.option("-f", "file_path", help="Path to csv file to plot")
@click.option("-i", "index_name", help="Column name for the x axis variable in the csv")
@click.option("-s", "start_frame", help="Start frame in csv")
@click.option("-e", "end_frame", help="End frame in csv")
def plot_arm_csv(file_path, index_name, start_frame, end_frame):
    click.echo(f"Plotting {file_path}!")
    plot_arm_joints(file_path, index_name, int(start_frame), int(end_frame))


@cli.command()
def plot():
    plotter = CSVPlotter()
    plotter.run()


def add_csv_files(file1, file2):
    """Element-wise addition of CSV files (excluding Time column)."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Create result dataframe starting with Time column
    result = pd.DataFrame()
    result["Time (s)"] = df1["Time (s)"]

    # Add all other columns element-wise
    for col in df1.columns:
        if col != "Time (s)":
            result[col] = df1[col] + df2[col]

    return result


@cli.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def combine_csvs(file1, file2, output):
    """
    Element-wise add two CSV files (excluding Time column).

    Usage: python script.py FILE1 FILE2 OUTPUT
    """
    result = add_csv_files(file1, file2)
    result.to_csv(output, index=False)
    click.echo(f"✓ Saved result to {output}")


def append_csv_files(file1, file2):
    """
    Append movements from file2 to file1.
    Takes the relative movement (deltas) from file2 and applies them
    starting from the final position in file1.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Get the final positions from file1
    final_positions = df1.iloc[-1].copy()
    final_time = final_positions["Time (s)"]

    # Calculate deltas in file2 (movement relative to its start)
    df2_start = df2.iloc[0].copy()

    # Create result by concatenating
    result = df1.copy()

    # For each row in file2, apply the delta to file1's final position
    for idx in range(1, len(df2)):  # Skip first row since it's the starting position
        new_row = {}
        new_row["Time (s)"] = final_time + df2.iloc[idx]["Time (s)"]

        for col in df2.columns:
            if col != "Time (s)":
                # Delta from file2's start position
                delta = df2.iloc[idx][col] - df2_start[col]
                # Apply delta to file1's final position
                new_row[col] = final_positions[col] + delta

        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)

    return result


@cli.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def append(file1, file2, output):
    """
    Append movements from FILE2 to FILE1.
    Takes the relative motion from FILE2 and continues from FILE1's end position.

    Usage: python script.py append FILE1 FILE2 OUTPUT
    """
    result = append_csv_files(file1, file2)
    result.to_csv(output, index=False)
    click.echo(f"✓ Appended motion and saved to {output}")


def calc_wrist_flex(elbow_pos, wrist_pos, hand_pos):
    return angle_between_three_points(elbow_pos, wrist_pos, hand_pos)


def calc_elbow_flex(shoulder_pos, elbow_pos, wrist_pos):
    return angle_between_three_points(shoulder_pos, elbow_pos, wrist_pos)


def calc_shoulder(left_shoulder, right_shoulder, hip_left, hip_right, elbow):
    midpoint_hip = midpoint(hip_left, hip_right)
    body_plane_normal = normal_vector_of_plane_on_three_points(
        left_shoulder, right_shoulder, midpoint_hip
    )

    right_upper_arm_vector = vector_between_two_points(right_shoulder, elbow)
    right_vector = vector_between_two_points(left_shoulder, right_shoulder)

    down_vector = np.cross(body_plane_normal, right_vector)

    A_dot_n = np.dot(right_upper_arm_vector, body_plane_normal)
    A_dot_down = np.dot(right_upper_arm_vector, down_vector)
    A_dot_right = np.dot(right_upper_arm_vector, right_vector)

    shoulder_flexion = math.atan2(A_dot_n, math.sqrt(A_dot_right**2 + A_dot_down**2))

    shoulder_abduction = math.atan2(A_dot_right, A_dot_down)

    return [shoulder_flexion, shoulder_abduction]


def calc_joint_angles_from_data_dict(in_data):
    output = {
        "frame": [],
        "shoulder_flexion": [],
        "shoulder_abduction": [],
        "elbow_flexion": [],
        "wrist_flexion": [],
    }

    for frame in in_data:
        output["frame"].append(frame["frame"])

        hip_left = frame["HipL"]
        hip_right = frame["HipR"]
        hand = frame["Index"]
        elbow = frame["Elbow"]
        wrist = frame["Wrist"]
        shoulder = frame["Shoulder"]
        left_shoulder = frame["ShoulderL"]

        output["elbow_flexion"].append(
            calc_elbow_flex(shoulder, elbow, wrist) * RADIAN_TO_DEGREES
        )
        output["wrist_flexion"].append(
            calc_wrist_flex(elbow, wrist, hand) * RADIAN_TO_DEGREES
        )
        shoulder_flex, shoulder_abduct = calc_shoulder(
            left_shoulder, shoulder, hip_left, hip_right, elbow
        )
        output["shoulder_flexion"].append(shoulder_flex * RADIAN_TO_DEGREES)
        output["shoulder_abduction"].append(shoulder_abduct * RADIAN_TO_DEGREES)

    click.echo(output["elbow_flexion"][:5])
    return output


@cli.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2")
def calc_joints_write(file1, file2):
    data = []

    with open(file1, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_data = {"frame": int(row["frame"])}

            # Define body parts (without x/y/z suffixes)
            body_parts = [
                "Shoulder",
                "Elbow",
                "Wrist",
                "Index",
                "Thumb",
                "Pinky",
                "HipL",
                "HipR",
                "ShoulderL",
            ]

            # For each body part, create an array of [x, y, z]
            for part in body_parts:
                x_key = f"{part} x"
                y_key = f"{part} y"
                z_key = f"{part} z"

                # Handle potential whitespace in column names
                x_key_alt = f"__{part} x"
                y_key_alt = f"__{part} y"
                z_key_alt = f"__{part} z"

                # Try to get values, handling both with and without __ prefix
                x = row.get(x_key) or row.get(x_key_alt)
                y = row.get(y_key) or row.get(y_key_alt)
                z = row.get(z_key) or row.get(z_key_alt)

                # Convert to float and create array
                if x and y and z:
                    frame_data[part] = [float(x), float(y), float(z)]

            data.append(frame_data)

    joint_angles = calc_joint_angles_from_data_dict(data)

    with open(file2, "w") as outfile:
        writer = csv.writer(outfile, lineterminator="\n")
        writer.writerow(
            ["frame", "shoulder_flex", "shoulder_abduct", "elbow_flex", "wrist_flex"]
        )
        for index, num in enumerate(joint_angles["frame"]):
            if index == 0:
                click.echo(num)
            row = [
                joint_angles["frame"][num],
                joint_angles["shoulder_flexion"][num],
                joint_angles["shoulder_abduction"][num],
                joint_angles["elbow_flexion"][num],
                joint_angles["wrist_flexion"][num],
            ]
            writer.writerow(row)


def remove_outliers(
    data: np.ndarray, method: str = "iqr", iqr_multiplier: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from data using IQR or Z-score method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array
    method : str
        'iqr' for Interquartile Range or 'zscore' for Z-score method
    iqr_multiplier : float
        Multiplier for IQR method (default 1.5 for standard outlier detection)

    Returns:
    --------
    cleaned_data : np.ndarray
        Data with outliers replaced by interpolated values
    outlier_mask : np.ndarray
        Boolean mask indicating outlier positions
    """
    if method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > 3
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")

    # Interpolate outliers
    cleaned_data = data.copy()
    if np.any(outlier_mask):
        # Get indices of non-outliers
        valid_indices = np.where(~outlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]

        # Interpolate outlier values
        if len(valid_indices) > 1:
            cleaned_data[outlier_indices] = np.interp(
                outlier_indices, valid_indices, data[valid_indices]
            )

    return cleaned_data, outlier_mask


def detect_motion_phases(
    data: np.ndarray, window: int = 5
) -> Tuple[int, int, np.ndarray]:
    """
    Detect the two endpoints and movement phase in the data.

    Returns:
    --------
    endpoint1_idx : int
        Index where first endpoint ends and movement begins
    endpoint2_idx : int
        Index where movement ends and second endpoint begins
    movement_phase : np.ndarray
        Boolean array indicating movement phase
    """

    # Calculate rolling velocity (rate of change)
    velocity = np.abs(np.diff(data, prepend=data[0]))
    smoothed_velocity = np.convolve(velocity, np.ones(window) / window, mode="same")

    # Threshold for detecting movement (adaptive based on data)
    velocity_threshold = np.percentile(smoothed_velocity, 75) * 0.3

    # Find where significant movement occurs
    is_moving = smoothed_velocity > velocity_threshold

    # Find the first sustained movement start (endpoint1 end)
    endpoint1_idx = 0
    for i in range(len(is_moving) - window):
        if np.sum(is_moving[i : i + window]) >= window * 0.7:
            endpoint1_idx = i
            break

    # Find the last sustained movement end (endpoint2 start)
    endpoint2_idx = len(data) - 1
    for i in range(len(is_moving) - 1, window, -1):
        if np.sum(is_moving[i - window : i]) >= window * 0.7:
            endpoint2_idx = i
            break

    return endpoint1_idx, endpoint2_idx, is_moving


def transform_motion_pattern(
    original_data: np.ndarray,
    endpoint1_idx: int,
    endpoint2_idx: int,
    original_endpoint1: float,
    original_endpoint2: float,
    new_endpoint1: float,
    new_endpoint2: float,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Transform the motion pattern to new endpoints while preserving the movement characteristics.

    Parameters:
    -----------
    scale_factor : float
        Controls the aggressiveness of scaling (1.0 = match original range, >1.0 = more aggressive, <1.0 = less aggressive)
    """

    transformed = np.zeros_like(original_data)

    # Endpoint 1 phase: constant value
    transformed[: endpoint1_idx + 1] = new_endpoint1

    # Movement phase: scale and shift to match new endpoints
    movement_data = original_data[endpoint1_idx : endpoint2_idx + 1]

    # Use the actual range of the movement data for more conservative scaling
    actual_min = np.min(movement_data)
    actual_max = np.max(movement_data)
    actual_range = actual_max - actual_min

    if abs(actual_range) > 1e-6:
        # Normalize based on actual data range (0 to 1)
        normalized_movement = (movement_data - actual_min) / actual_range

        # Apply scale factor
        # Center around 0.5, then scale, then shift back
        centered = normalized_movement - 0.5
        scaled_centered = centered * scale_factor
        normalized_movement = scaled_centered + 0.5

        # Scale to new range
        new_range = new_endpoint2 - new_endpoint1
        transformed[endpoint1_idx : endpoint2_idx + 1] = (
            new_endpoint1 + normalized_movement * new_range
        )
    else:
        # If no movement in original, create linear interpolation
        normalized_movement = np.linspace(0, 1, len(movement_data))
        new_range = new_endpoint2 - new_endpoint1
        transformed[endpoint1_idx : endpoint2_idx + 1] = (
            new_endpoint1 + normalized_movement * new_range
        )

    # Clip to stay within endpoints
    min_endpoint = min(new_endpoint1, new_endpoint2)
    max_endpoint = max(new_endpoint1, new_endpoint2)
    transformed[endpoint1_idx : endpoint2_idx + 1] = np.clip(
        transformed[endpoint1_idx : endpoint2_idx + 1], min_endpoint, max_endpoint
    )

    # Endpoint 2 phase: constant value
    transformed[endpoint2_idx + 1 :] = new_endpoint2

    return transformed


def calculate_rmse(difference: np.ndarray, outlier_mask: np.ndarray = None) -> float:
    """
    Calculate Root Mean Square Error, optionally excluding outliers.

    Parameters:
    -----------
    difference : np.ndarray
        Array of differences between transformed and original
    outlier_mask : np.ndarray, optional
        Boolean mask indicating outlier positions to exclude

    Returns:
    --------
    rmse : float
        Root Mean Square Error
    """
    if outlier_mask is not None:
        # Exclude outliers from RMSE calculation
        valid_difference = difference[~outlier_mask]
    else:
        valid_difference = difference

    if len(valid_difference) == 0:
        return 0.0

    mse = np.mean(valid_difference**2)
    rmse = np.sqrt(mse)
    return rmse


def plot_comparison(
    result_df: pd.DataFrame, joint_column: str, rmse: float, outlier_count: int
):
    """
    Plot the original and transformed motion patterns with difference visualization.
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot original with outliers marked
        ax1.plot(
            result_df["frame"],
            result_df[f"{joint_column}_original"],
            label="Original",
            linewidth=2,
            color="blue",
        )
        if "is_outlier" in result_df.columns:
            outliers = result_df[result_df["is_outlier"]]
            if len(outliers) > 0:
                ax1.scatter(
                    outliers["frame"],
                    outliers[f"{joint_column}_original"],
                    color="red",
                    s=50,
                    marker="x",
                    label=f"Outliers ({outlier_count})",
                    zorder=5,
                )
        ax1.set_ylabel("Angle (degrees)")
        ax1.set_title(f"Original {joint_column} Motion")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot transformed
        ax2.plot(
            result_df["frame"],
            result_df[f"{joint_column}_transformed"],
            label="Transformed",
            linewidth=2,
            color="orange",
        )
        ax2.set_ylabel("Angle (degrees)")
        ax2.set_title(f"Transformed {joint_column} Motion")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot difference
        difference = (
            result_df[f"{joint_column}_transformed"]
            - result_df[f"{joint_column}_original"]
        )
        ax3.plot(
            result_df["frame"],
            difference,
            label=f"Difference (RMSE: {rmse:.3f}°)",
            linewidth=2,
            color="red",
        )
        ax3.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax3.fill_between(result_df["frame"], 0, difference, alpha=0.3, color="red")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Angle Difference (degrees)")
        ax3.set_title("Difference Between Transformed and Original")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Color-code motion phases on all plots
        for ax in [ax1, ax2, ax3]:
            endpoint1_frames = result_df[result_df["motion_phase"] == "endpoint1"][
                "frame"
            ]
            endpoint2_frames = result_df[result_df["motion_phase"] == "endpoint2"][
                "frame"
            ]

            if len(endpoint1_frames) > 0:
                ax.axvspan(
                    endpoint1_frames.iloc[0],
                    endpoint1_frames.iloc[-1],
                    alpha=0.1,
                    color="green",
                    label="Endpoint 1",
                )
            if len(endpoint2_frames) > 0:
                ax.axvspan(
                    endpoint2_frames.iloc[0],
                    endpoint2_frames.iloc[-1],
                    alpha=0.1,
                    color="red",
                    label="Endpoint 2",
                )

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("joint_column")
@click.argument("endpoint1", type=float)
@click.argument("endpoint2", type=float)
@click.option(
    "--start-frame",
    type=int,
    default=None,
    help="Starting frame to analyze (inclusive)",
)
@click.option(
    "--end-frame", type=int, default=None, help="Ending frame to analyze (inclusive)"
)
@click.option(
    "--smoothing-window",
    type=int,
    default=5,
    help="Window size for detecting endpoints",
)
@click.option(
    "--scale-factor",
    type=float,
    default=1.0,
    help="Scaling aggressiveness (1.0=normal, <1.0=gentler, >1.0=more aggressive)",
)
@click.option(
    "--outlier-method",
    type=click.Choice(["iqr", "zscore", "none"]),
    default="iqr",
    help="Method for outlier detection (iqr, zscore, or none)",
)
@click.option(
    "--iqr-multiplier",
    type=float,
    default=1.5,
    help="IQR multiplier for outlier detection (default 1.5)",
)
def joint_analysis(
    csv_path: str,
    joint_column: str,
    endpoint1: float,
    endpoint2: float,
    start_frame: int,
    end_frame: int,
    smoothing_window: int,
    scale_factor: float,
    outlier_method: str,
    iqr_multiplier: float,
):
    """
    Analyze joint angle trends from CSV and create similar motion pattern with new endpoints.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing joint angle data
    joint_column : str
        Name of the joint angle column to analyze (e.g., 'shoulder_flex', 'elbow_flex')
    endpoint1 : float
        Desired value for the first endpoint (starting position)
    endpoint2 : float
        Desired value for the second endpoint (ending position)
    start_frame : int, optional
        Starting frame to analyze (if None, uses first frame)
    end_frame : int, optional
        Ending frame to analyze (if None, uses last frame)
    smoothing_window : int
        Window size for detecting endpoints (larger = more stable detection)
    outlier_method : str
        Method for outlier detection ('iqr', 'zscore', or 'none')
    iqr_multiplier : float
        Multiplier for IQR outlier detection
    """
    # Read the CSV file
    click.echo(f"Reading {csv_path}!")
    df = pd.read_csv(csv_path, header=0)

    # Convert all columns except 'frame' to numeric, handling any errors
    for col in df.columns:
        if col != "frame":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if joint_column not in df.columns:
        raise ValueError(
            f"Column '{joint_column}' not found in CSV. Available columns: {df.columns.tolist()}"
        )

    # Filter data based on frame range
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    if len(df) == 0:
        click.echo("Error: No data in specified frame range!")
        return

    click.echo(
        f"Analyzing frames {df['frame'].iloc[0]} to {df['frame'].iloc[-1]} ({len(df)} frames)"
    )

    # Extract the joint angle data as float array
    original_data = df[joint_column].astype(float).values
    frames = df["frame"].values

    # Remove outliers if requested
    outlier_mask = np.zeros(len(original_data), dtype=bool)
    if outlier_method != "none":
        cleaned_data, outlier_mask = remove_outliers(
            original_data, method=outlier_method, iqr_multiplier=iqr_multiplier
        )
        outlier_count = np.sum(outlier_mask)
        click.echo(
            f"Detected and removed {outlier_count} outliers using {outlier_method} method"
        )
    else:
        cleaned_data = original_data
        outlier_count = 0

    # Detect endpoints and movement phases using cleaned data
    endpoint1_idx, endpoint2_idx, movement_phase = detect_motion_phases(
        cleaned_data, smoothing_window
    )

    # Extract the original endpoints from cleaned data
    original_endpoint1 = float(cleaned_data[endpoint1_idx])
    original_endpoint2 = float(cleaned_data[endpoint2_idx])

    click.echo(
        f"Detected endpoint 1 at frame {frames[endpoint1_idx]}: {original_endpoint1:.2f}"
    )
    click.echo(
        f"Detected endpoint 2 at frame {frames[endpoint2_idx]}: {original_endpoint2:.2f}"
    )

    try:
        # Transform using cleaned data
        transformed_data = transform_motion_pattern(
            cleaned_data,
            endpoint1_idx,
            endpoint2_idx,
            original_endpoint1,
            original_endpoint2,
            endpoint1,
            endpoint2,
            scale_factor,
        )
    except Exception as ex:
        click.echo(f"Error in transformation: {ex}")
        return

    # Calculate RMSE excluding outliers
    difference = transformed_data - cleaned_data
    rmse = calculate_rmse(
        difference, outlier_mask if outlier_method != "none" else None
    )
    click.echo(f"\nRMSE (excluding outliers): {rmse:.3f} degrees")

    # Calculate statistics
    mean_diff = np.mean(
        difference[~outlier_mask] if outlier_method != "none" else difference
    )
    max_diff = np.max(np.abs(difference))
    click.echo(f"Mean difference: {mean_diff:.3f} degrees")
    click.echo(f"Max absolute difference: {max_diff:.3f} degrees")

    # Create output dataframe
    result_df = df.copy()
    result_df[f"{joint_column}_original"] = original_data
    result_df[f"{joint_column}_cleaned"] = cleaned_data
    result_df[f"{joint_column}_transformed"] = transformed_data
    result_df["is_outlier"] = outlier_mask
    result_df["motion_phase"] = [
        (
            "endpoint1"
            if i <= endpoint1_idx
            else "movement" if i < endpoint2_idx else "endpoint2"
        )
        for i in range(len(original_data))
    ]
    result_df["difference"] = difference
    result_df["rmse"] = rmse

    # Save results
    output_path = csv_path.replace(".csv", "_transformed.csv")
    result_df.to_csv(output_path, index=False)
    click.echo(f"\nSaved results to {output_path}")

    # Plot comparison
    plot_comparison(result_df, joint_column, rmse, outlier_count)


def rotation_vector_to_euler_angles(rvec):
    """
    Convert rotation vector to Euler angles (yaw, pitch, roll) in degrees.

    Args:
        rvec: Rotation vector from solvePnP

    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Calculate Euler angles from rotation matrix
    # Using the convention: rotation order is ZYX (yaw, pitch, roll)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return yaw_deg, pitch_deg, roll_deg


def estimate_distance_and_orientation(corners, marker_size, camera_matrix, dist_coeffs):
    """
    Estimate distance and orientation to ArUco marker using solvePnP.

    Args:
        corners: Detected marker corners
        marker_size: Real-world size of marker in meters
        camera_matrix: Camera calibration matrix
        dist_coeffs: Camera distortion coefficients

    Returns:
        Tuple of (distance, yaw, pitch, roll) or (None, None, None, None)
    """
    # Define 3D points of marker corners in marker coordinate system
    obj_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    # Solve PnP to get rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

    if success:
        # Distance is the norm of translation vector
        distance = np.linalg.norm(tvec)

        # Get orientation angles
        yaw, pitch, roll = rotation_vector_to_euler_angles(rvec)

        return distance, yaw, pitch, roll

    return None, None, None, None


def get_camera_calibration(frame_shape):
    """
    Get approximate camera calibration parameters.
    For accurate measurements, proper camera calibration is recommended.
    """
    h, w = frame_shape[:2]
    # Approximate focal length
    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    dist_coeffs = np.zeros((4, 1))
    return camera_matrix, dist_coeffs


@cli.command()
@click.option(
    "--marker-size",
    default=0.05,
    type=float,
    help="Size of ArUco marker in meters (default: 0.05m = 5cm)",
)
@click.option(
    "--output",
    default="distances.csv",
    type=str,
    help="Output CSV file (default: distances.csv)",
)
@click.option("--camera", default=0, type=int, help="Camera index (default: 0)")
@click.option(
    "--dict",
    "aruco_dict_name",
    default="DICT_4X4_50",
    type=click.Choice(
        [
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_6X6_50",
            "DICT_6X6_100",
        ],
        case_sensitive=False,
    ),
    help="ArUco dictionary (default: DICT_4X4_50)",
)
def aruco_marker_webcam(marker_size, output, camera, aruco_dict_name):
    """Detect ArUco markers from webcam and log their distance and orientation to a CSV file."""

    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Open webcam
    cap = cv2.VideoCapture(camera + cv2.CAP_DSHOW)
    if not cap.isOpened():
        click.echo(
            click.style(f"Error: Could not open camera {camera}", fg="red"), err=True
        )
        sys.exit(1)

    # Get camera calibration
    ret, frame = cap.read()
    if not ret:
        click.echo(click.style("Error: Could not read from camera", fg="red"), err=True)
        sys.exit(1)

    camera_matrix, dist_coeffs = get_camera_calibration(frame.shape)

    # Open CSV file
    csv_file = open(output, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame",
            "marker_id",
            "distance_m",
            "distance_cm",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
        ]
    )

    click.echo(click.style(f"Starting ArUco detection. Output: {output}", fg="green"))
    click.echo(f"Marker size: {marker_size}m")
    click.echo(f"Dictionary: {aruco_dict_name}")
    click.echo(click.style("Press 'q' to quit\n", fg="yellow"))

    frame_count = 0

    try:
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)

            # If markers detected
            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, marker_id in enumerate(ids):
                    # Get corners for this marker
                    marker_corners = corners[i][0]

                    # Estimate distance and orientation
                    distance, yaw, pitch, roll = estimate_distance_and_orientation(
                        marker_corners, marker_size, camera_matrix, dist_coeffs
                    )

                    if distance is not None:
                        # Log to CSV
                        timestamp = datetime.now().isoformat()
                        csv_writer.writerow(
                            [
                                frame_count,
                                marker_id[0],
                                f"{distance:.4f}",
                                f"{distance*100:.2f}",
                                f"{yaw:.2f}",
                                f"{pitch:.2f}",
                                f"{roll:.2f}",
                            ]
                        )
                        csv_file.flush()

                        # Display on frame
                        center = tuple(marker_corners.mean(axis=0).astype(int))

                        # Display distance
                        text1 = f"ID:{marker_id[0]} Dist:{distance*100:.1f}cm"
                        cv2.putText(
                            frame,
                            text1,
                            (center[0] - 80, center[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # Display orientation
                        text2 = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
                        cv2.putText(
                            frame,
                            text2,
                            (center[0] - 80, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2,
                        )

                        print(
                            f"Marker {marker_id[0]}: {distance*100:.2f} cm | Yaw: {yaw:.2f}° Pitch: {pitch:.2f}° Roll: {roll:.2f}°"
                        )

            # Display frame
            cv2.imshow("ArUco Distance & Orientation Detection", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        click.echo(click.style(f"\nData saved to {output}", fg="green"))
