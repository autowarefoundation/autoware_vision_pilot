import os
import cv2
import numpy
import torch
import json
import numpy as np

from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class LoadDataAutoSteer(Dataset):
    def __init__(self, filenames):
        super().__init__()
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples

    def __getitem__(self, index):
        image, shape = self.load_image(index)
        label = self.labels[index].copy()

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        # xp
        e_xp = label["e_xp"]
        target_xp = np.stack([e_xp], axis=0)  # shape (2, 64)
        target_xp = torch.from_numpy(target_xp[:, :, None])  # shape (2, 64, 1)

        # h_vector
        e_h_vector  = label["e_h_vector"]
        target_h_vector = np.stack([e_h_vector], axis=0)  # shape (2, 64)
        target_h_vector = torch.from_numpy(target_h_vector[:, :, None])  # shape (2, 64, 1)

        return torch.from_numpy(sample), target_xp, target_h_vector

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]

        return image, (h, w)

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        x = {}
        for filename in filenames:
            zero_label = {
                "l_xp": np.zeros(0, dtype=np.float32),
                "r_xp": np.zeros(0, dtype=np.float32),
                "l_h_vector": np.zeros(0, dtype=np.float32),
                "r_h_vector": np.zeros(0, dtype=np.float32)
            }
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        data = json.load(f)
                        l_xp, e_xp, r_xp = [], [], []
                        l_h_vector, e_h_vector, r_h_vector = [], [], []
                        for item in data:
                            # if item['class'] == "left":
                            #     l_xp = np.array(item['xp'], dtype=np.float32)
                            #     l_h_vector = np.array(item.get('h_vector', []), dtype=np.float32)
                            if item['class'] == "ego":
                                e_xp = np.array(item['xp'], dtype=np.float32)
                                e_h_vector = np.array(item.get('h_vector', []), dtype=np.float32)
                            # elif item['class'] == "right":
                            #     r_xp = np.array(item['xp'], dtype=np.float32)
                            #     r_h_vector = np.array(item.get('h_vector', []), dtype=np.float32)
                        label = {
                            # "l_xp": l_xp,
                            "e_xp": e_xp,
                            # "r_xp": r_xp,
                            # "l_h_vector": l_h_vector,
                            "e_h_vector": e_h_vector,
                            # "r_h_vector": r_h_vector
                        }
                else:
                    label = zero_label
            except FileNotFoundError:
                label = zero_label
            except AssertionError:
                continue
            x[filename] = label
        torch.save(x, path)

        return x
