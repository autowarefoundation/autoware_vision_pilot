# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import cv2
import sys
import time
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime, timedelta

from torch import get_file_path

sys.path.append('../..')
from Models.inference.auto_steer_infer import AutoSteerNetworkInfer


def make_visualization(frame, xp, h_vector):
    yp = np.linspace(0, 511, 64, dtype=int)

    # l_xp = xp[0] * 1024
    # l_h_vector = h_vector[0]
    # l_h_vector = (l_h_vector >= 0.5).astype(int)
    # l_xp = l_xp * l_h_vector

    e_xp = xp * 1024
    e_h_vector = h_vector
    e_h_vector = (e_h_vector >= 0.5).astype(int)
    e_xp = e_xp * e_h_vector

    # r_xp = xp[2] * 1024
    # r_h_vector = h_vector[2]
    # r_h_vector = (r_h_vector >= 0.5).astype(int)
    # r_xp = r_xp * r_h_vector

    # # Left
    # for x, y, h in zip(l_xp, yp, l_h_vector):
    #     if h == 1:
    #         cv2.circle(frame, (int(x), int(y)), 3, (228, 186, 20), thickness=-1)
    # Ego
    for x, y, h in zip(e_xp, yp, e_h_vector):
        if h == 1:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), thickness=-1)
    # Right
    # for x, y, h in zip(r_xp, yp, r_h_vector):
    #     if h == 1:
    #         cv2.circle(frame, (int(x), int(y)), 3, (180,105, 255), thickness=-1)


def main():
    parser = ArgumentParser()

    parser.add_argument("-a", "--autosteer_checkpoint_path", dest="autosteer_checkpoint_path",
                        help="path to pytorch AutoSteer checkpoint file to load model dict")
    parser.add_argument("-i", "--video_filepath", dest="video_filepath",
                        help="path to input video which will be processed by AutoSteer")
    parser.add_argument("-o", "--output_file", dest="output_file",
                        help="path to output video visualization file, must include output file name")
    parser.add_argument('-v', "--vis", action='store_true', default=False,
                        help="flag for whether to show frame by frame visualization while processing is occuring")
    parser.add_argument('-g', "--ground_truth",
                        help="json file containing ground truth steering angles for each frame")
    args = parser.parse_args()

    # Saved model checkpoint path
    autosteer_checkpoint_path = args.autosteer_checkpoint_path
    model = AutoSteerNetworkInfer(autosteer_checkpoint_path=autosteer_checkpoint_path)
    print('AutoSteer Model Loaded')

    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = args.video_filepath
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_datetime = datetime.now()  # or read metadata if available

    # Output filepath
    output_filepath_obj = args.output_file + '.avi'
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Video writer object
    writer_obj = cv2.VideoWriter(output_filepath_obj,
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (1024, 512))

    # Check if video catpure opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    else:
        print('Reading video frames')

    # Transparency factor
    alpha = 0.5

    # Read until video is completed
    print('Processing started')
    frame_index = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Running inference
            xp, h_vector = model.inference(frame)

            make_visualization(frame, xp, h_vector)
            if (args.vis):
                cv2.imshow('Prediction Objects', frame)
                cv2.waitKey(10)

            # Writing to video frame
            writer_obj.write(frame)

        else:
            print('Frame not read - ending processing')
            break
        frame_index += 1

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
# %%
