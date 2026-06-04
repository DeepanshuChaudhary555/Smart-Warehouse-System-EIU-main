from ultralytics import YOLO
import torch
import torch.nn.functional as F
import numpy as np


# Load trained YOLO model
model = YOLO("runs/detect/runs/detect/retrain_active_learning-2/weights/best.pt")


# Global feature storage
feature_maps = []


# Hook function
def hook_fn(module, input, output):

    feature_maps.append(output)


# Register semantic hook layer
hook_layer = model.model.model[10]

hook_layer.register_forward_hook(hook_fn)


def extract_yolo_predictions(

    image_path,

    confidence_threshold=0.5
):

    global feature_maps

    feature_maps = []


    # Run YOLO inference
    results = model(image_path)


    predictions = []


    # No features captured
    if len(feature_maps) == 0:

        return []


    fmap = feature_maps[0]


    # Global Average Pooling
    pooled = F.adaptive_avg_pool2d(

        fmap,

        (1, 1)
    )

    pooled = pooled.squeeze().detach().cpu().numpy()


    for result in results:

        boxes = result.boxes


        for box in boxes:

            # Confidence
            confidence = float(box.conf[0])


            # Confidence filtering
            if confidence < confidence_threshold:

                continue


            # Class info
            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            # Entropy probabilities
            probs = [

                confidence,

                1 - confidence
            ]


            # Semantic embedding
            embedding = pooled.tolist()


            # Bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy().tolist()


            predictions.append({

                "image_path": image_path,

                "probs": probs,

                "embedding": embedding,

                "confidence": confidence,

                "bbox": xyxy,

                "class_name": class_name
            })


    return predictions