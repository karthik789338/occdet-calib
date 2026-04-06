from __future__ import annotations

from typing import List

from src.detectors.base import BaseDetector, Detection


class FCOSWrapper(BaseDetector):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        score_thr: float = 0.001,
        max_per_img: int = 300,
        model_name: str = "fcos_r50",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.score_thr = score_thr
        self.max_per_img = max_per_img
        self.model = None

    def _lazy_init(self) -> None:
        if self.model is not None:
            return

        from mmdet.apis import init_detector

        self.model = init_detector(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )

    def _get_class_names(self) -> List[str]:
        if self.model is None:
            return []
        meta = getattr(self.model, "dataset_meta", None)
        if meta is None:
            return []
        classes = meta.get("classes", None)
        if classes is None:
            return []
        return list(classes)

    def predict(self, image_path: str) -> List[Detection]:
        self.validate_image_path(image_path)
        self._lazy_init()

        from mmdet.apis import inference_detector

        result = inference_detector(self.model, image_path)
        pred_instances = result.pred_instances

        if pred_instances is None or len(pred_instances) == 0:
            return []

        bboxes = pred_instances.bboxes.detach().cpu().numpy()
        scores = pred_instances.scores.detach().cpu().numpy()
        labels = pred_instances.labels.detach().cpu().numpy()

        class_names = self._get_class_names()

        keep = scores >= self.score_thr
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(scores) == 0:
            return []

        order = scores.argsort()[::-1][: self.max_per_img]
        bboxes = bboxes[order]
        scores = scores[order]
        labels = labels[order]

        detections: List[Detection] = []
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox.tolist()
            class_id = int(label)
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else str(class_id)
            )

            detections.append(
                Detection(
                    image_path=image_path,
                    model_name=self.model_name,
                    class_id=class_id,
                    class_name=class_name,
                    score=float(score),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        return detections
