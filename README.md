# Comparing Arm Tracking Teleoperation to Conventional Joystick Control of a Laterally Mounted Robot Arm

This is the repository containing all of the materials for Nicholas Teague's BEng Electronic Engineering project.

The project goal is to create a way to control a robot by moving a human hand.

This is achieved by using MediaPipe's pose estimation:

![](/imgs/pose_estimation.gif)

![](/imgs/armtrackingstuff.gif)

## Approach

Firstly experiments were carried out to characterise the accuracy difference between two tracking approaches:

ArUco marker tracking and AI pose estimation via MediaPipe:

![](/imgs/arucoing.gif)

![](/imgs/frontonflex.gif)

Then CoppeliaSim was used to simulate a 7 DoF robot.

This started with implementing simple robot control:

![](/imgs/yumirobot.gif)

![](/imgs/yumijoystickcontrol.gif)

Where the arm tracking was gradually applied:

![](/imgs/yumicontrol2.gif)

And then experiments were devised to measure performance against conventional joystick control methods:

![](/imgs/yumiposedatagathering.gif)

![](/imgs/yumiposetransport.gif)

![](/imgs/yumijoystickdatagathering.gif)

## Repo Structure

The structure for this repository is as follows.

Good luck. I made this repo with the knowledge that I'll never have to touch any of this Terry Davis - forsaken code ever again.

```
├─assets                        // Miscellaneous assets used for meeting presentations
├─coppeliasimScenes             // CoppeliaSim related scenes
├─datasets                      // Source videos for tracking
├─deliverables                  // Assorted source text for initial proposal document + risk register   
├─document                      // LaTex ource for final project report
├─imgs                          // Assorted images used throughout for meeting presentations
├─Meeting Logs                  // Meeting notes
└─python_sketches               // Source code
    ├─coppeliacontrol           // Final source code used for experiments
    ├─cv_experiments            // Early implementations of computer vision
    ├─data_gathering_api        // Python package used for early data gathering + processing
    ├─pose_estimation_basics    // Initial computer vision + pose estimation code example
    └─random_misc               // Unsorted scripts used throughout
```