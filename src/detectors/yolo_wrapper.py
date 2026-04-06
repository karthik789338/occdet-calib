from __future__ import annotations

from typing import List

from ultralytics import YOLO

from src.detectors.base import BaseDetector, Detection


class YOLOWrapper(BaseDetector):
    def __init__(
        self,
        weights: str = "yolo8m.pt",
        device: str = "cpu",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        imgsz: int = 640,
        max_det: int = 300,
        model_name: str = "yolo_v8m",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.model = YOLO(weights)

    def predict(self, image_path: str) -> List[Detection]:
        self.validate_image_path(image_path)

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names if hasattr(result, "names") else {}

        if result.boxes is None:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_id = int(cls[i])
            score = float(conf[i])
            class_name = str(names.get(class_id, class_id))

            detections.append(
                Detection(
                    image_path=image_path,
                    model_name=self.model_name,
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        return detections