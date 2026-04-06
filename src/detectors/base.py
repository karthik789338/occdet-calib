from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    image_path: str
    model_name: str
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    raw_label: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDetector(ABC):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def predict(self, image_path: str) -> List[Detection]:
        raise NotImplementedError

    def predict_batch(self, image_paths: List[str]) -> List[Detection]:
        results: List[Detection] = []
        for image_path in image_paths:
            results.extend(self.predict(image_path))
        return results

    @staticmethod
    def validate_image_path(image_path: str) -> Path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return path