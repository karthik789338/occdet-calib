from __future__ import annotations

from typing import List

from src.detectors.base import BaseDetector, Detection


class DeformableDETRWrapper(BaseDetector):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        score_thr: float = 0.001,
        max_per_img: int = 300,
        model_name: str = "deformable_detr_r50",
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

        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                "MMDetection is not installed yet. Install it on the GCP VM before using DeformableDETRWrapper."
            ) from exc

        self.model = init_detector(
            self.config_path,
            self.checkpoint_path,
            device=self.device,
        )

    def predict(self, image_path: str) -> List[Detection]:
        self.validate_image_path(image_path)
        self._lazy_init()

        try:
            from mmdet.apis import inference_detector
        except ImportError as exc:
            raise ImportError("MMDetection is not installed correctly.") from exc

        result = inference_detector(self.model, image_path)

        detections: List[Detection] = []
        if result is None:
            return detections

        raise NotImplementedError(
            "Deformable DETR result parsing will be finalized on the GCP VM after MMDetection is installed."
        )