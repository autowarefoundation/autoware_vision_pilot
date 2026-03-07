# Production Releases

## Download
The current latest release and past releases of VisionPilot can be downloaded from here: https://github.com/autowarefoundation/autoware.privately-owned-vehicles/releases

## Descriptions

### Vision Pilot 0.9
Vision Pilot 0.9 processes images from a single front-facing camera to enable both ADAS features and highway autopilot within a single driving lane. Compared to Vision Pilot 0.5, this version of system incorporates both lateral perception and planning modules alongside longitudinal perception and planning modules in two parallel streams, through the additional integration of the AutoSpeed neural network for closest-in-path-object detection. This enables features such as Autonomous Cruise Control, Forward Collision Warning, and Automatic Emergency braking. In order to estimate the distance of the closest-in-path-object, a homography transform is utilized which maps image pixels to road coordinates, providing an estimate of real-world distances in metres. A Kalman filter is used to track the distance of the closest-in-path-object and estimate its velocity. To maintain a safe following distance to the lead vehicle, the system complies with [Mobileye's Responsibility Sensitive Safety framework](https://www.mobileye.com/technology/responsibility-sensitive-safety/).

**System Architecture**

<img src="../../Media/VisionPilot_0.9.png" width="100%">


### Vision Pilot 0.5
Vision Pilot 0.5 is an autonomous lateral steering system which processes images from a single front-facing camera for autonomous steering control features such as lane centering and lane-keep assist. Vision Pilot 0.5 has been tested with an automotive OEM on their private test track and the outputs of the system were compared with ground-truth data from human drivers. The system was able to achieve over 90% accuracy compared to human driven steering at highway driving speeds in high curvature turns (20+ degree steering angle) and high bank angles. 

Vision Pilot 0.5 uses two neural networks, EgoLanes for segmenting and classifying lane lines, and AutoSteer for estimating the steering angle to follow the road. In parallel, a traditional tracking pipeline is utilized to measure the cross-track error and yaw error and a feedback plus feedforward controller is utilized to remain centered in-lane. The system can run headless or a visualization can be shown which overlays the detected lanes and predicted steering angle on the input image. Although EgoLanes runs on each image individually, the AutoSteer network requires the current and previous images as input to capture spatio-temporal features. A moving average smoothing is applied to the output of AutoSteer to ensure that steering signal commands do not suffer from noise. 

**System Architecture**

<img src="../../Media/VisionPilot_0.5.png" width="100%">

