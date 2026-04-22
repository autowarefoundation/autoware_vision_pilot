## AutoSteer 2.0 - driving path prediction

AutoSteer 2.0 is a neural network which processes camera image frames and outputs future waypoints which define the driving path of the ego-car to 'follow the road'. The network does not perform explicit lane detection and processes the entire image frame to perform end-to-end spatial path prediction. The network is designed for in-lane driving and predicts the future waypoints to continue in the ego-cars existing driving corridor.

### Demo Video

[![Watch the Video](../../../Media/AutoSteer_2_thumbnail.jpg)](https://drive.google.com/file/d/13suX8urFenOyQhYpIbiEvl5POuoP7cTq/preview)

## Get Started
To easily try out the model on your own images and videos, please follow the steps in the [tutorial](tutorial.ipynb). For the best results, please ensure that your input video matches the aspect ratio of the model.

### Performance Results

AutoSteer 2.0 network is trained on combined dataset from three distinct datasets [TUSimple](https://www.kaggle.com/datasets/manideep1108/tusimple), [OpenLane](https://github.com/OpenDriveLab/OpenLane) and [CurveLanes](https://github.com/SoulmateB/CurveLanes). The ataset is prepared with 90:10 ratio for train:val split, and we achieve a mAP@50 score of 0.9691 and a mAP score of 0.9546 on validation data.

## Model variants

AutoSteer 2.0 processess camera frames in a 2:1 aspect ratio with size 1024px by 512px and predicts ego-path line in camera coordinate system. The original AutoSteer model takes as input the lane mask output from the EgoLanes network for the current and previous image. It outputs a probability vector which encodes fixed steering angles, and the argmax of this probability vector informs us of the steering angle predicted by the model. In practice, we add a moving average filter to the model output to ensure that the predicted steering values are smooth over time.

**AutoSteer 2.0 model weights - 2:1 aspect ratio, 1024px by 512px input image**

### [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/1-wfPyu7HId7YDSh_T_DkB3Ma1zPc7Fu_/view?usp=drive_link)
### [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1Rb05CScAfvl0OPRJz8S83Q4BWRG1pEKH/view?usp=drive_link)


**AutoSteer model weights  - 2:1 aspect ratio, 640px by 320px input image**

### [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/17yu0H81sFE6ZHuviT7SXH3iMjMmyyS0t/view?usp=sharing)
### [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1gxH6EM4HJ0rfnqt90cT1w7hgizW49jQe/view?usp=sharing)