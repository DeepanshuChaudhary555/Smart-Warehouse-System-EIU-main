import os
import shutil
import cv2

from active_learning.pipeline import active_learning_pipeline

from yolo_inference import extract_yolo_predictions


# Folder containing images
image_folder = r"all_images"


# Raw selected images folder
output_folder = "selected_samples"


# Annotated images folder
annotated_folder = "annotated_samples"


# Clear old selected samples
if os.path.exists(output_folder):

    shutil.rmtree(output_folder)

os.makedirs(output_folder)


# Clear old annotated samples
if os.path.exists(annotated_folder):

    shutil.rmtree(annotated_folder)

os.makedirs(annotated_folder)


all_predictions = []


# Process ALL images
for filename in os.listdir(image_folder):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(
            image_folder,
            filename
        )

        print(f"\nPROCESSING: {image_path}")

        print("Running YOLO...")

        predictions = extract_yolo_predictions(
            image_path
        )

        print(f"Detections: {len(predictions)}")

        all_predictions.extend(predictions)


print("\nTOTAL DETECTIONS:")
print(len(all_predictions))


# -----------------------------
# Dynamic Cluster Scaling
# -----------------------------
cluster_count = int(len(all_predictions) * 0.4)

print(f"\nDynamic Cluster Count: {cluster_count}")


print("\nRUNNING ACTIVE LEARNING PIPELINE...")


final_samples = active_learning_pipeline(

    all_predictions,

    entropy_threshold=0.5,

    n_clusters=cluster_count
)


# Remove duplicate image paths
unique_samples = {}

for sample in final_samples:

    image_path = sample["image_path"]

    if image_path not in unique_samples:

        unique_samples[image_path] = sample


print(f"\nUnique Selected Images: {len(unique_samples)}")


print("\nFINAL SELECTED SAMPLES:")


for sample in unique_samples.values():

    source_path = sample["image_path"]

    print(source_path)


    # Save raw selected image
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


    # Load image
    image = cv2.imread(source_path)


    # Bounding box coordinates
    x1, y1, x2, y2 = map(
        int,
        sample["bbox"]
    )


    # Draw bounding box
    cv2.rectangle(

        image,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        2
    )


    # -----------------------------
    # CLASS LABEL
    # -----------------------------
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


    # -----------------------------
    # CONFIDENCE SCORE
    # -----------------------------
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


    # -----------------------------
    # ENTROPY SCORE
    # -----------------------------
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


    # Save annotated image
    annotated_path = os.path.join(

        annotated_folder,

        filename
    )

    cv2.imwrite(

        annotated_path,

        image
    )

    print(f"Annotated Saved: {annotated_path}")


    # Display image
    cv2.imshow(

        "Selected Sample",

        image
    )


    # Wait for key press
    cv2.waitKey(0)


cv2.destroyAllWindows()