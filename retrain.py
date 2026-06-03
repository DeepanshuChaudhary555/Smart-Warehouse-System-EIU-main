from ultralytics import YOLO

# Load your previously trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Continue training
model.train(
    data="y_yolo.yaml",
    epochs=20,
    imgsz=640,
    batch=2,
    workers=2,
    project="runs/detect",
    name="retrain_active_learning"
)