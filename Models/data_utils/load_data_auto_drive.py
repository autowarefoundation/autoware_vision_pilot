import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
import numpy as np

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class LoadDataAutoDrive(data.Dataset):
    def __init__(self, filenames, input_width, input_height, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_width = input_width
        self.input_height = input_height

        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        if self.mosaic and random.random() < self.params['mosaic']:
            # Load MOSAIC
            image, label = self.load_mosaic(index, self.params)
            # MixUp augmentation
            if random.random() < self.params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, self.params)

                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape, r = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_width, self.input_height, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, self.params, self.input_width, self.input_height)

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            augment_hsv(image, self.params)
            # Flip up-down
            if random.random() < self.params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < self.params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls).view(-1, 1)  # (N,1)
            target_box = torch.from_numpy(box).view(-1, 4)  # (N,4)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = min(self.input_width / w, self.input_height / h)

        if not self.augment:
            r = min(r, 1.0)  # only scale down
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=resample() if self.augment else cv2.INTER_LINEAR
            )

        return image, (h, w), r

    def load_mosaic(self, index, params):
        label4 = []

        # Mosaic canvas 2x input width & height
        canvas_w, canvas_h = self.input_width * 2, self.input_height * 2
        image4 = np.full((canvas_h, canvas_w, 3), 114, dtype=np.uint8)

        # Mosaic center (random within safe margins)
        xc = int(random.uniform(self.input_width // 2, canvas_w - self.input_width // 2))
        yc = int(random.uniform(self.input_height // 2, canvas_h - self.input_height // 2))

        # Four image indices: original + 3 random
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            # Load image
            image, (h0, w0), r = self.load_image(idx)
            h, w = image.shape[:2]

            # Determine mosaic placement (top-left, top-right, bottom-left, bottom-right)
            if i == 0:  # top-left
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x2a, y2a = xc, yc
            elif i == 1:  # top-right
                x1a, y1a = xc, max(yc - h, 0)
                x2a, y2a = min(xc + w, canvas_w), yc
            elif i == 2:  # bottom-left
                x1a, y1a = max(xc - w, 0), yc
                x2a, y2a = xc, min(yc + h, canvas_h)
            else:  # bottom-right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + w, canvas_w), min(yc + h, canvas_h)

            # Compute source patch coordinates
            x1b = max(0, - (x1a - (xc - w if i in [0, 2] else xc)))
            y1b = max(0, - (y1a - (yc - h if i in [0, 1] else yc)))
            x2b = x1b + (x2a - x1a)
            y2b = y1b + (y2a - y1a)

            # Clip source patch to image size
            x2b = min(w, x2b)
            y2b = min(h, y2b)
            x1b = max(0, x1b)
            y1b = max(0, y1b)

            # Recompute destination patch after clipping
            pw, ph = x2b - x1b, y2b - y1b
            x2a = x1a + pw
            y2a = y1a + ph

            # Paste patch safely
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # Adjust labels
            label = self.labels[idx].copy()
            if len(label):
                pad_w = x1a - x1b
                pad_h = y1a - y1b
                label[:, 1:] = wh2xy(label[:, 1:], w, h, pad_w, pad_h)
            label4.append(label)

        # Concatenate labels
        if len(label4):
            label4 = np.concatenate(label4, 0)
            # Clip to mosaic canvas
            np.clip(label4[:, 1], 0, canvas_w, out=label4[:, 1])
            np.clip(label4[:, 2], 0, canvas_h, out=label4[:, 2])
            np.clip(label4[:, 3], 0, canvas_w, out=label4[:, 3])
            np.clip(label4[:, 4], 0, canvas_h, out=label4[:, 4])
        else:
            label4 = np.zeros((0, 5), dtype=np.float32)

        # Optional perspective augmentation
        image4, label4 = random_perspective(image4, label4, params, self.input_width, self.input_height, border=(0, 0))

        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls_list, box_list, _ = zip(*batch)

        # Concatenate class and box tensors safely
        cls = torch.cat([c.view(-1, 1) for c in cls_list], dim=0)  # (total_targets,1)
        box = torch.cat([b.view(-1, 4) for b in box_list], dim=0)  # (total_targets,4)

        # Generate target image indices
        idx = []
        for i, c in enumerate(cls_list):
            idx.append(torch.full((c.shape[0],), i, dtype=torch.long))
        idx = torch.cat(idx, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': idx}

        return torch.stack(samples, dim=0), targets

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        x = {}
        for filename in filenames:
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
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            except AssertionError:
                continue
            x[filename] = label
        torch.save(x, path)
        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


import cv2

def resize(image, input_width, input_height, augment=False):
    # current shape
    h0, w0 = image.shape[:2]

    # scale ratio (keep aspect ratio)
    r = min(input_width / w0, input_height / h0)
    if not augment:
        r = min(r, 1.0)

    # new unpadded size
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    # resize
    if (w0, h0) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR if not augment else cv2.INTER_AREA)

    # padding to target size
    pad_w = input_width - new_w
    pad_h = input_height - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return image, (r, r), (left, top)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(image, label, params, input_width, input_height, border=(0, 0)):
    h = image.shape[0] + border[1] * 2  # height
    w = image.shape[1] + border[0] * 2  # width

    # Center translation to origin
    center = np.eye(3)
    center[0, 2] = -w / 2
    center[1, 2] = -h / 2

    # Perspective matrix (identity here, can be extended)
    perspective = np.eye(3)

    # Rotation + Scale
    rotate = np.eye(3)
    angle = random.uniform(-params['degrees'], params['degrees'])
    scale = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)

    # Shear
    shear = np.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation (relative to target width/height)
    translate = np.eye(3)
    translate[0, 2] = random.uniform(-params['translate'], params['translate']) * input_width
    translate[1, 2] = random.uniform(-params['translate'], params['translate']) * input_height

    # Compose final transformation matrix (order: right-to-left)
    M = translate @ shear @ rotate @ perspective @ center

    # Apply affine warp
    image_transformed = cv2.warpAffine(
        image,
        M[:2],
        dsize=(input_width, input_height),
        borderValue=(114, 114, 114)
    )

    # Transform labels
    if label.shape[0]:
        n = label.shape[0]
        # 8 points per box (x1y1, x2y2, x1y2, x2y1)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = label[:, [1,2,3,4,1,4,3,2]].reshape(n*4,2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        # Recreate boxes from transformed points
        x = xy[:, [0,2,4,6]]
        y = xy[:, [1,3,5,7]]
        new_boxes = np.zeros_like(label[:, 1:5])
        new_boxes[:, 0] = x.min(1)
        new_boxes[:, 1] = y.min(1)
        new_boxes[:, 2] = x.max(1)
        new_boxes[:, 3] = y.max(1)

        # Clip to final size
        new_boxes[:, [0,2]] = new_boxes[:, [0,2]].clip(0, input_width)
        new_boxes[:, [1,3]] = new_boxes[:, [1,3]].clip(0, input_height)

        # Keep only boxes with valid area
        keep = (new_boxes[:,2] - new_boxes[:,0] > 1) & (new_boxes[:,3] - new_boxes[:,1] > 1)
        label = label[keep].copy()
        label[:, 1:5] = new_boxes[keep]

    return image_transformed, label


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(a=32.0, b=32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations

            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]
            self.transform = albumentations.Compose(transforms,
                                                    albumentations.BboxParams('yolo', ['class_labels']))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = np.asarray(x["bboxes"], dtype=np.float32).reshape(-1, 4)  # (N,4)
            cls = np.asarray(x["class_labels"], dtype=np.int64).reshape(-1, 1)  # (N,1)
        return image, box, cls