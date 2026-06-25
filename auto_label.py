from ultralytics import YOLO
from pathlib import Path

# Load trained model
model = YOLO("train_Enpochs_200/weights/best_200.pt")

# Run prediction
model.predict(
    source="all_images",
    conf=0.75,
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs/detect",
    name="auto_label_conf_0.75",
    exist_ok=True
)

print("Auto-labeling complete!")