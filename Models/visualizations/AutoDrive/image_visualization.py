from argparse import ArgumentParser
import cv2
from PIL import Image
from Models.inference.auto_drive_infer import AutoDriveNetworkInfer

color_map = {
    1: (0, 0, 255),  # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)  # cyan
}


def make_visualization(prediction, input_image_filepath):
    img_cv = cv2.imread(input_image_filepath)
    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred

        # Pick color, fallback to white if unknown class
        color = color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        # label = f"Class: {int(cls)} | Score: {conf:.2f}"
        # cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Prediction Objects', img_cv)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath",
                        help="path to input image which will be processed by DomainSeg")
    args = parser.parse_args()
    model_checkpoint_path = args.model_checkpoint_path
    input_image_filepath = args.input_image_filepath

    model = AutoDriveNetworkInfer(model_checkpoint_path)
    img = Image.open(input_image_filepath).convert("RGB")

    prediction = model.inference(img)
    make_visualization(prediction, input_image_filepath)
