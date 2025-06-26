from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torchvision
import numpy as np
from .base_detector import BaseDetector

class FasterRCNNDetector(BaseDetector):
    def __init__(self, confidence_threshold=0.5, device='cpu'):
        self.device = torch.device(device)

        # Загружаем backbone вручную
        self.model = fasterrcnn_resnet50_fpn(
            weights=None,                      # не скачивать weights
            weights_backbone=None              # не скачивать backbone отдельно
        )

        state_dict = torch.load("resources/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device).eval()
        self.conf_threshold = confidence_threshold

    def detect(self, frame: np.ndarray):
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        img = transform(frame).to(self.device)
        outputs = self.model([img])[0]
        boxes = outputs['boxes'].cpu().detach().numpy()
        scores = outputs['scores'].cpu().detach().numpy()
        labels = outputs['labels'].cpu().detach().numpy()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score >= self.conf_threshold:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append((x1, y1, w, h, score))
        return detections
