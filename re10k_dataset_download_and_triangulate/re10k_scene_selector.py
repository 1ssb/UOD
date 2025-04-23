import numpy as np
import open3d as o3d
from ultralytics import YOLO
import logging
import os
import sys
from functools import lru_cache
from typing import Dict, List
from matplotlib import pyplot as plt

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def input_scene_id() -> str:
    """Input the scene folder ID from the user."""
    scene_id = input("Enter the scene folder ID: ").strip()
    return scene_id

def check_file_exists(file_path: str) -> None:
    """Check if the specified file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file {file_path} does not exist.")

@lru_cache(maxsize=100)
def check_geodesic_from_txt(file_path: str) -> Dict[str, np.ndarray]:
    """
    Read pose data from a text file, transform poses relative to the base image, and return a dictionary of poses.
    """
    check_file_exists(file_path)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) <= 1:
        print("Reject: Not enough data.")
        sys.exit()

    if len(lines) - 1 <= 100:
        print("Reject: Not enough lines in the file.")
        sys.exit()

    poses = {}
    focal_length = None

    for i, line in enumerate(lines[1:], 1):
        parts = line.split()
        if len(parts) < 19:
            print(f"Skipping line {i} due to insufficient data.")
            continue

        timestamp, params = parts[0], list(map(float, parts[1:]))
        fx = params[0]
        matrix_values = params[6:18]
        if focal_length is None:
            focal_length = np.round(fx * 640, 3)

        matrix_3x4 = np.array(matrix_values).reshape((3, 4))
        w2c_matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])

        try:
            c2w_matrix = np.linalg.inv(w2c_matrix_4x4)
        except np.linalg.LinAlgError as e:
            print(f"Skipping line {i} due to non-invertible matrix: {e}")
            continue

        poses[timestamp] = c2w_matrix

    first_pose = list(poses.values())[0][:3, 3]
    last_pose = list(poses.values())[-1][:3, 3]
    total_geodesic_distance = np.linalg.norm(first_pose - last_pose)

    if total_geodesic_distance < 2:
        print("Reject: Total geodesic distance is less than 2 units.")
        sys.exit()

    return poses

def load_yolo_model(model_path: str) -> YOLO:
    """Load the YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
    except Exception as e:
        logging.error(f"Model could not be loaded: {e}")
        sys.exit()
    return model

def process_images(model: YOLO, image_folder: str, target_objects: List[str]) -> float:
    """Process images and calculate the detection rate of target objects."""
    total_images = 0
    target_objects_count = 0

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_file)
            image = plt.imread(image_path)

            results = model(image[:, :, :3], verbose=False, stream=True, conf=0.1)
            for result in results:
                objects = [result.names[int(cls)] for cls in result.boxes.cls.cpu()]
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for obj, box, conf in zip(objects, boxes, confs):
                    if obj in target_objects and round(conf, 2) > 0.5:
                        target_objects_count += 1
            total_images += 1

    if total_images == 0:
        logging.error("No images found in the specified directory.")
        sys.exit()

    detection_rate = (target_objects_count / total_images) * 100
    return detection_rate

def main():
    scene_id = input_scene_id()
    file_path = f"test_GT_poses/{scene_id}.txt"

    check_geodesic_from_txt(file_path)

    target_objects = [
        "tv", 
        "oven", 
        "sink", 
        "chair", 
        "bed", 
        "refrigerator", 
        "book", 
        "laptop", 
        "couch", 
        "door"
    ]  # Change to whatever COCO object you want to use.

    model_path = 'yolo_models/yolov8x.pt'
    model = load_yolo_model(model_path)

    image_folder = f"dataset/test/{scene_id}/"
    detection_rate = process_images(model, image_folder, target_objects)

    if detection_rate < 60:
        print("Reject: Detection rate is less than 60% of total images.")
    else:
        print("Accept")

if __name__ == "__main__":
    main()