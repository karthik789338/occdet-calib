from __future__ import annotations

from src.data.load_coco_annotations import load_coco_detection_gt_from_dict


mock_coco = {
    "images": [
        {"id": 1, "file_name": "img1.jpg"},
        {"id": 2, "file_name": "img2.jpg"},
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 3, "name": "car"},
    ],
    "annotations": [
        {
            "id": 101,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 20, 30, 40],
            "area": 1200,
            "iscrowd": 0,
        },
        {
            "id": 102,
            "image_id": 1,
            "category_id": 3,
            "bbox": [100, 120, 50, 60],
            "area": 3000,
            "iscrowd": 0,
        },
        {
            "id": 103,
            "image_id": 2,
            "category_id": 1,
            "bbox": [5, 5, 10, 10],
            "area": 100,
            "iscrowd": 1,
        },
    ],
}

df = load_coco_detection_gt_from_dict(
    mock_coco,
    include_category_names=["person", "car"],
    drop_crowd=False,
)

print(df)
