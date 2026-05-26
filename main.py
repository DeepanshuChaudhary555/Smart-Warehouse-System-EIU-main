import os
import shutil
import cv2
import json
import uuid

from datetime import datetime

from active_learning.pipeline import active_learning_pipeline

from yolo_inference import extract_yolo_predictions


# ---------------------------------
# IMAGE FOLDER
# ---------------------------------
image_folder = r"all_images"


# ---------------------------------
# OUTPUT FOLDERS
# ---------------------------------
output_folder = "selected_samples"

annotated_folder = "annotated_samples"


# ---------------------------------
# REPLAY BUFFER FOLDERS
# ---------------------------------
replay_folder = "replay_buffer/images"

metadata_folder = "replay_buffer/metadata"


# ---------------------------------
# CLEAR TEMP OUTPUT FOLDERS
# ---------------------------------
if os.path.exists(output_folder):

    shutil.rmtree(output_folder)

os.makedirs(output_folder)


if os.path.exists(annotated_folder):

    shutil.rmtree(annotated_folder)

os.makedirs(annotated_folder)


# ---------------------------------
# CREATE REPLAY BUFFER FOLDERS
# ---------------------------------
os.makedirs(replay_folder, exist_ok=True)

os.makedirs(metadata_folder, exist_ok=True)


all_predictions = []


# ---------------------------------
# PROCESS ALL IMAGES
# ---------------------------------
for filename in os.listdir(image_folder):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(

            image_folder,

            filename
        )

        print(f"\nPROCESSING: {image_path}")

        print("Running YOLO...")


        predictions = extract_yolo_predictions(

            image_path,

            confidence_threshold=0.5
        )

        print(f"Detections: {len(predictions)}")


        all_predictions.extend(predictions)


# ---------------------------------
# TOTAL DETECTIONS
# ---------------------------------
print("\nTOTAL DETECTIONS:")

print(len(all_predictions))


# ---------------------------------
# DYNAMIC CLUSTER SCALING
# ---------------------------------
cluster_count = int(len(all_predictions) * 0.4)

print(f"\nDynamic Cluster Count: {cluster_count}")


# ---------------------------------
# ACTIVE LEARNING PIPELINE
# ---------------------------------
print("\nRUNNING ACTIVE LEARNING PIPELINE...")


final_samples = active_learning_pipeline(

    all_predictions,

    entropy_threshold=0.5,

    n_clusters=cluster_count
)


# ---------------------------------
# REMOVE DUPLICATE IMAGES
# ---------------------------------
unique_samples = {}


for sample in final_samples:

    image_path = sample["image_path"]

    if image_path not in unique_samples:

        unique_samples[image_path] = sample


print(
    f"\nUnique Samples Selected For Human Review: "
    f"{len(unique_samples)}"
)


# ---------------------------------
# PROCESS FINAL SAMPLES
# ---------------------------------
for sample in unique_samples.values():

    source_path = sample["image_path"]

    print(source_path)


    # ---------------------------------
    # SAVE RAW SELECTED IMAGE
    # ---------------------------------
    filename = os.path.basename(source_path)

    destination = os.path.join(

        output_folder,

        filename
    )

    shutil.copy(

        source_path,

        destination
    )

    print(f"Saved: {destination}")


    # ---------------------------------
    # SAVE TO REPLAY BUFFER
    # ---------------------------------
    replay_destination = os.path.join(

        replay_folder,

        filename
    )

    shutil.copy(

        source_path,

        replay_destination
    )

    print(f"Replay Saved: {replay_destination}")


    # ---------------------------------
    # LOAD IMAGE
    # ---------------------------------
    image = cv2.imread(source_path)


    # ---------------------------------
    # BOUNDING BOX
    # ---------------------------------
    x1, y1, x2, y2 = map(

        int,

        sample["bbox"]
    )


    # ---------------------------------
    # DRAW BOUNDING BOX
    # ---------------------------------
    cv2.rectangle(

        image,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        2
    )


    # ---------------------------------
    # CLASS LABEL
    # ---------------------------------
    class_text = sample["class_name"]

    cv2.putText(

        image,

        class_text,

        (x1, y1 - 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2
    )


    # ---------------------------------
    # CONFIDENCE SCORE
    # ---------------------------------
    confidence_text = (

        f"Conf: {sample['confidence']:.2f}"
    )

    cv2.putText(

        image,

        confidence_text,

        (x1, y1 - 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 255, 0),

        2
    )


    # ---------------------------------
    # ENTROPY SCORE
    # ---------------------------------
    entropy_text = (

        f"Entropy: {sample['entropy']:.2f}"
    )

    cv2.putText(

        image,

        entropy_text,

        (x1, y1 - 15),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 0, 255),

        2
    )


    # ---------------------------------
    # SAVE ANNOTATED IMAGE
    # ---------------------------------
    annotated_path = os.path.join(

        annotated_folder,

        filename
    )

    cv2.imwrite(

        annotated_path,

        image
    )

    print(f"Annotated Saved: {annotated_path}")


    # ---------------------------------
    # ADVANCED METADATA
    # ---------------------------------

    # Timestamp
    timestamp = str(datetime.now())


    # Unique Sample ID
    sample_id = str(uuid.uuid4())


    # Image Dimensions
    image_height = image.shape[0]

    image_width = image.shape[1]


    # Bounding Box Area
    bbox_area = (

        (x2 - x1)

        * (y2 - y1)
    )


    # ---------------------------------
    # METADATA DICTIONARY
    # ---------------------------------
    metadata = {

        # ---------------------------------
        # BASIC INFO
        # ---------------------------------
        "sample_id": sample_id,

        "timestamp": timestamp,

        "image": filename,

        "class": sample["class_name"],


        # ---------------------------------
        # MODEL PREDICTION INFO
        # ---------------------------------
        "confidence": sample["confidence"],

        "entropy": sample["entropy"],

        "bbox": sample["bbox"],

        "bbox_area": bbox_area,


        # ---------------------------------
        # IMAGE INFO
        # ---------------------------------
        "image_width": image_width,

        "image_height": image_height,


        # ---------------------------------
        # ACTIVE LEARNING INFO
        # ---------------------------------
        "cluster_id": sample.get(

            "cluster_id",

            -1
        ),

        "semantic_embeddings": sample["embedding"],


        # ---------------------------------
        # HUMAN REVIEW INFO
        # ---------------------------------
        "human_review_status": False,

        "false_positive": False,


        # ---------------------------------
        # DATASET INFO
        # ---------------------------------
        "source_split": "unknown",


        # ---------------------------------
        # MODEL VERSION
        # ---------------------------------
        "model_version": "YOLO26_v1"
    }


    # ---------------------------------
    # SAVE METADATA JSON
    # ---------------------------------
    metadata_filename = (

        filename.rsplit(".", 1)[0]

        + ".json"
    )

    metadata_path = os.path.join(

        metadata_folder,

        metadata_filename
    )

    with open(metadata_path, "w") as f:

        json.dump(

            metadata,

            f,

            indent=4
        )

    print(f"Metadata Saved: {metadata_path}")


    # ---------------------------------
    # DISPLAY IMAGE
    # ---------------------------------

    window_name = (

        f"Human Review Sample - {filename}"
    )

    cv2.setWindowTitle(

        "Human Review",

        window_name
    )

    cv2.namedWindow(

        "Human Review",

        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(

        "Human Review",

        cv2.WND_PROP_FULLSCREEN,

        cv2.WINDOW_NORMAL
    )

    cv2.imshow(

        "Human Review",

        image
    )

    # Wait for key press
    cv2.waitKey(500)


# ---------------------------------
# CLOSE WINDOWS
# ---------------------------------
cv2.destroyAllWindows()