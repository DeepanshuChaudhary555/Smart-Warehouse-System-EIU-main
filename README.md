# Smart Warehouse System using YOLOv11 and YOLO26

AI-powered warehouse monitoring and object detection system using YOLO-based deep learning models for detecting warehouse lifts and racks in industrial environments.

Repository: [Smart-Warehouse-System-EIU](https://github.com/Strovana/Smart-Warehouse-System-EIU?utm_source=chatgpt.com)

This project includes:

* YOLOv11 and YOLO26 training
* Custom warehouse dataset
* Lift and rack detection
* Distance deviation measurement
* Real-time inference
* Performance evaluation using mAP, Precision, and Recall

Based on the uploaded training pipeline and evaluation workflow. 

---

# Project Overview

The system was developed to automate warehouse monitoring using computer vision and deep learning techniques.

The model detects:

* Lift structures
* Warehouse racks

and measures:

* alignment
* distance deviation
* structural displacement

between detected components.

The project aims to improve:

* industrial inspection
* warehouse automation
* predictive maintenance
* safety monitoring

---

# Features

* YOLOv11 support
* YOLO26 support
* Custom object detection training
* Real-time prediction
* Distance deviation calculation
* Confusion matrix analysis
* Precision-Recall curve analysis
* Docker support
* CUDA GPU acceleration

---

# Technologies Used

| Technology  | Purpose                   |
| ----------- | ------------------------- |
| Python      | Core programming          |
| PyTorch     | Deep learning framework   |
| YOLOv11     | Object detection          |
| YOLO26      | Enhanced object detection |
| OpenCV      | Image processing          |
| Ultralytics | YOLO training/inference   |
| CUDA        | GPU acceleration          |
| Docker      | Containerized deployment  |

---

# Dataset Classes

The custom dataset contains 2 classes:

| Class ID | Class Name |
| -------- | ---------- |
| 0        | Lift       |
| 1        | rack       |

Dataset configuration used in training: 

```yaml
nc: 2
names: ['Lift', 'rack']
```

---

# Project Structure

```text
Smart-Warehouse-System-EIU/
│
├── 05_dataRack/
│   ├── train/
│   ├── test/
│
├── train_Enpochs_200/
│   ├── weights/
│   ├── train/
│
├── runs/
├── Dockerfile
├── requirements.txt
├── newtrying.py
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Strovana/Smart-Warehouse-System-EIU.git
cd Smart-Warehouse-System-EIU
```

---

# Create Virtual Environment

## Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## Linux / WSL

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install PyTorch

CUDA 12.1 version:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

# Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install pillow
pip install pandas
pip install matplotlib
pip install pyyaml
```

---

# YOLO Training

The project uses YOLO26 for training with:

* 200 epochs
* image size 1280
* batch size 8
* GPU acceleration

Training configuration from the uploaded script: 

```python
results = model.train(
    data="y_yolo.yaml",
    epochs=200,
    lr0=0.0005,
    imgsz=1280,
    batch=8,
    device=0,
    plots=True,
    workers=2
)
```

---

# Train YOLO26

```bash
yolo detect train data=y_yolo.yaml model=yolo26n.pt epochs=200 imgsz=1280 batch=8
```

---

# Train YOLOv11

```bash
yolo detect train data=y_yolo.yaml model=yolo11n.pt epochs=200 imgsz=1280 batch=8
```

---

# Running Inference

## Image Prediction

```bash
yolo predict model=best.pt source=test.jpg
```

## Folder Prediction

```bash
yolo predict model=best.pt source=05_dataRack/test/images/test
```

## Webcam Detection

```bash
yolo predict model=best.pt source=0
```

---

# Distance Measurement System

The trained YOLO model is integrated into a distance deviation detection system.

Workflow: 

```text
Input Image
     ↓
YOLO Detection
     ↓
Detect Lift and Rack
     ↓
Calculate Vertical Pixel Distance
     ↓
Convert Pixels → Millimeters
     ↓
Deviation Detection
     ↓
Alert Generation
```

The system calculates:

* vertical distance between Lift and rack
* structural deviation
* alignment errors

using an empirical conversion:

```text
10 pixels = 1 mm
```

---

# Performance Evaluation

The project evaluates:

* Precision
* Recall
* mAP50
* mAP50-95
* PR Curve
* Confusion Matrix

Training analysis and evaluation were included in the uploaded notebook workflow. 

---

# Results Analysis

The project includes:

* Precision-Confidence curves
* Recall-Confidence curves
* Precision-Recall curves
* Confusion matrices
* Normalized confusion matrices

to analyze model performance and classification quality.

---

# Applications

* Smart warehouse monitoring
* Industrial inspection
* Rack alignment analysis
* Lift detection
* Structural deviation monitoring
* Automated maintenance systems
* Industry 4.0 applications

---

# Future Improvements

* Real-time multi-camera monitoring
* IoT integration
* Automated warehouse analytics
* Segmentation models
* Object tracking
* Edge AI deployment
* Grounding DINO integration
* SAM2 segmentation

---

# References

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com)
* [PyTorch](https://pytorch.org?utm_source=chatgpt.com)

---

# License

This project is intended for educational and research purposes.
