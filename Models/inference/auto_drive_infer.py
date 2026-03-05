import torch
from torchvision import transforms, ops
from PIL import Image

class AutoDriveNetworkInfer():
    def __init__(self, checkpoint_path=''):
        self.train_size = (1024, 512)  # target width, height
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'Using {self.device} for inference')

        # Load model
        self.model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)['model']
        self.model = self.model.to(self.device).eval()

    def resize_letterbox(self, img: Image.Image):
        """
        Resize image maintaining aspect ratio with padding.
        Returns:
            padded_img: PIL.Image, resized + padded
            scale: float, scaling factor applied to original image
            pad_x: int, horizontal padding
            pad_y: int, vertical padding
        """
        target_w, target_h = self.train_size
        orig_w, orig_h = img.size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        padded_img = Image.new("RGB", self.train_size, (114, 114, 114))  # gray padding

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(img_resized, (pad_x, pad_y))

        return padded_img, scale, pad_x, pad_y

    def image_to_tensor(self, image: Image.Image):
        """Convert PIL image to tensor and keep scale/padding info."""
        img, scale, pad_x, pad_y = self.resize_letterbox(image)
        tensor = transforms.ToTensor()(img).to(self.device).half()
        return tensor.unsqueeze(0), scale, pad_x, pad_y

    def xywh2xyxy(self, x):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def nms(self, preds, iou_thres=0.45):
        """Apply NMS on predictions tensor [x1,y1,x2,y2,score,class]"""
        if preds.numel() == 0:
            return torch.empty(0, 6)
        boxes = preds[:, :4]
        scores = preds[:, 4]
        keep = ops.nms(boxes, scores, iou_thres)
        return preds[keep]

    def post_process_predictions(self, raw_predictions, conf_thres=0.6, iou_thres=0.45):
        predictions = raw_predictions.squeeze(0).permute(1, 0)
        boxes = predictions[:, :4]  # model gives cx,cy,w,h
        class_probs = predictions[:, 4:]

        # --- confidence filter ---
        scores, class_ids = torch.max(class_probs.sigmoid(), dim=1)
        mask = scores > conf_thres
        if mask.sum() == 0:
            return torch.empty(0, 6)

        # --- convert to xyxy before NMS ---
        boxes_xyxy = self.xywh2xyxy(boxes[mask])

        combined = torch.cat([
            boxes_xyxy,
            scores[mask].unsqueeze(1),
            class_ids[mask].float().unsqueeze(1)
        ], dim=1)

        return self.nms(combined, iou_thres)

    def inference(self, image: Image.Image):
        """
        Run inference on a single PIL image.
        Returns list of predictions: [[x1,y1,x2,y2,score,class], ...] in original image coordinates
        """
        orig_w, orig_h = image.size
        image_tensor, scale, pad_x, pad_y = self.image_to_tensor(image)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        predictions = self.post_process_predictions(predictions)
        if predictions.numel() == 0:
            return []

        # --- adjust from letterboxed to original coords ---
        predictions[:, [0, 2]] = (predictions[:, [0, 2]] - pad_x) / scale
        predictions[:, [1, 3]] = (predictions[:, [1, 3]] - pad_y) / scale

        # clamp to image bounds
        predictions[:, [0, 2]] = predictions[:, [0, 2]].clamp(0, orig_w)
        predictions[:, [1, 3]] = predictions[:, [1, 3]].clamp(0, orig_h)

        return predictions.tolist()
