# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import os
import onnxruntime as ort

sys.path.append('../../..')
from Models.inference.auto_drive_infer import AutoDriveNetworkInfer

color_map = {
    1: (0, 0, 255),  # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)  # cyan
}


class AutoSpeedONNXInfer:
    def __init__(self, onnx_path):
        self.train_size = (640, 640)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f'ONNX model loaded with provider: {self.session.get_providers()[0]}')

    def resize_letterbox(self, img: Image.Image):
        target_w, target_h = self.train_size
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        padded_img = Image.new("RGB", self.train_size, (114, 114, 114))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(img_resized, (pad_x, pad_y))
        return padded_img, scale, pad_x, pad_y

    def image_to_array(self, image: Image.Image):
        img, scale, pad_x, pad_y = self.resize_letterbox(image)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
        return img_array, scale, pad_x, pad_y

    def xywh2xyxy(self, boxes):
        x = boxes.copy()
        x[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        x[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        x[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        x[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return x

    def nms(self, boxes, scores, iou_threshold=0.45):
        from torchvision import ops
        import torch
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores)
        keep = ops.nms(boxes_t, scores_t, iou_threshold)
        return keep.numpy()

    def post_process_predictions(self, raw_predictions, conf_thres=0.6, iou_thres=0.45):
        predictions = raw_predictions.squeeze(0).T
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:]
        
        # Sigmoid + confidence filter
        class_probs = 1 / (1 + np.exp(-class_probs))
        scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        mask = scores > conf_thres
        
        if mask.sum() == 0:
            return []
        
        boxes_xyxy = self.xywh2xyxy(boxes[mask])
        scores_filtered = scores[mask]
        class_ids_filtered = class_ids[mask]
        
        # NMS
        keep = self.nms(boxes_xyxy, scores_filtered, iou_thres)
        
        results = []
        for idx in keep:
            results.append([
                boxes_xyxy[idx, 0], boxes_xyxy[idx, 1],
                boxes_xyxy[idx, 2], boxes_xyxy[idx, 3],
                scores_filtered[idx], class_ids_filtered[idx]
            ])
        return results

    def inference(self, image: Image.Image):
        orig_w, orig_h = image.size
        img_array, scale, pad_x, pad_y = self.image_to_array(image)
        
        outputs = self.session.run(None, {'input': img_array})
        predictions = self.post_process_predictions(outputs[0])
        
        if len(predictions) == 0:
            return []
        
        # Adjust coordinates
        for pred in predictions:
            pred[0] = (pred[0] - pad_x) / scale
            pred[1] = (pred[1] - pad_y) / scale
            pred[2] = (pred[2] - pad_x) / scale
            pred[3] = (pred[3] - pad_y) / scale
            # Clamp to bounds
            pred[0] = max(0, min(orig_w, pred[0]))
            pred[1] = max(0, min(orig_h, pred[1]))
            pred[2] = max(0, min(orig_w, pred[2]))
            pred[3] = max(0, min(orig_h, pred[3]))
        
        return predictions


def make_visualization(prediction, image):
    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred

        # Pick color, fallback to white if unknown class
        color = color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint (.pt) or ONNX model (.onnx)")
    parser.add_argument("-i", "--video_filepath", dest="video_filepath",
                        help="path to input video which will be processed by AutoSpeed")
    parser.add_argument("-o", "--output_file", dest="output_file",
                        help="path to output video visualization file, must include output file name")
    parser.add_argument('-v', "--vis", action='store_true', default=False,
                        help="flag for whether to show frame by frame visualization while processing is occuring")
    args = parser.parse_args()

    # Detect model type and load
    model_path = args.model_checkpoint_path
    
    if model_path.endswith('.onnx'):
        print('Loading ONNX model...')
        model = AutoSpeedONNXInfer(onnx_path=model_path)
        print('ONNX Model Loaded')
    elif model_path.endswith('.pt') or os.path.isdir(model_path):
        print('Loading PyTorch model...')
        model = AutoDriveNetworkInfer(checkpoint_path=model_path)
        print('PyTorch Model Loaded')
    else:
        raise ValueError(f"Unsupported model format: {model_path}. Use .pt or .onnx")

    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = args.video_filepath
    cap = cv2.VideoCapture(video_filepath)

    # Output filepath
    output_filepath_obj = args.output_file + '.avi'

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer object
    writer_obj = cv2.VideoWriter(output_filepath_obj,
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))

    # Check if video catpure opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    else:
        print('Reading video frames')

    # Read until video is completed
    print('Processing started')
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            # image_pil = image_pil.resize((640, 640))

            # Running inference
            prediction = model.inference(image_pil)
            vis_obj = make_visualization(prediction, frame.copy())

            if (args.vis):
                # Resize for display to avoid a huge window
                display_w = 960
                h, w, _ = vis_obj.shape
                display_h = int(h * (display_w / w))
                vis_display = cv2.resize(vis_obj, (display_w, display_h))
                cv2.imshow('Prediction Objects', vis_display)
                cv2.waitKey(10)

            # Writing to video frame
            writer_obj.write(vis_obj)

        else:
            print('Frame not read - ending processing')
            break

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
# %%
