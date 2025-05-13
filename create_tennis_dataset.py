#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path

# Configuration
SOURCE_DATA_DIR = "data"
DATASET_NAME = "tennis-court-pose"
DATASET_ROOT = f"datasets/{DATASET_NAME}"
TRAIN_JSON_FILE = os.path.join(SOURCE_DATA_DIR, "data_train.json")
VAL_JSON_FILE = os.path.join(SOURCE_DATA_DIR, "data_val.json")

# Create directory structure
def create_directory_structure():
    """Create the necessary directory structure for the dataset."""
    for directory in [
        f"{DATASET_ROOT}/images/train",
        f"{DATASET_ROOT}/images/val",
        f"{DATASET_ROOT}/labels/train",
        f"{DATASET_ROOT}/labels/val",
    ]:
        os.makedirs(directory, exist_ok=True)
    print(f"✅ Created directory structure in {DATASET_ROOT}")

# Generate YAML file
def create_yaml_file():
    """Create the YAML configuration file for the dataset."""
    yaml_content = f"""# Tennis Court Detection dataset (Ultralytics YOLO format)
# Documentation: https://docs.ultralytics.com/datasets/pose/
# Example usage: yolo train data={DATASET_NAME}.yaml

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/{DATASET_NAME} # dataset root dir (relative to YOLO)
train: images/train # train images
val: images/val # val images
test: # test images (optional)

# Keypoints
kpt_shape: [14, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [1, 0, 3, 2, 6, 4, 7, 5, 9, 8, 11, 10, 12, 13]  # no specific flip indices for tennis court keypoints

# Classes
names:
  0: tennis-court
"""
    
    with open(f"{DATASET_NAME}.yaml", "w") as f:
        f.write(yaml_content)
    print(f"✅ Created {DATASET_NAME}.yaml")
    
# Process JSON data and create label files
def process_json_data(json_file, dataset_type):
    """Process JSON data and create label files in YOLO format."""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images for {dataset_type} dataset...")
    
    for item in data:
        img_id = item["id"]
        src_img_path = os.path.join(SOURCE_DATA_DIR, "images", f"{img_id}.png")
        
        # Skip if source image doesn't exist
        if not os.path.exists(src_img_path):
            print(f"⚠️ Source image {src_img_path} not found, skipping...")
            continue
            
        # Create symbolic link to image
        dst_img_path = os.path.join(DATASET_ROOT, "images", dataset_type, f"{img_id}.jpg")
        if not os.path.exists(dst_img_path):
            os.symlink(os.path.abspath(src_img_path), dst_img_path)
        
        # Create label file in YOLO format
        dst_label_path = os.path.join(DATASET_ROOT, "labels", dataset_type, f"{img_id}.txt")
        
        # Get keypoints from JSON
        raw_kps = item["kps"]
        
        # Get image dimensions
        from PIL import Image # Keep import here for local scope as in original, or move to top
        try:
            with Image.open(src_img_path) as img:
                img_width, img_height = img.size
        except Exception as e: # Catch potential errors during image opening
            print(f"⚠️ Error opening or reading image {src_img_path}: {e}, skipping...")
            continue
            
        if img_width <= 0 or img_height <= 0:
            print(f"⚠️ Image {src_img_path} has invalid dimensions ({img_width}x{img_height}), skipping...")
            continue
            
        img_width_f = float(img_width)
        img_height_f = float(img_height)

        if not raw_kps: # Check if raw_kps list is empty
             print(f"⚠️ No keypoints provided for {img_id} in JSON, skipping...")
             continue

        # Clamp raw keypoints, determine visibility based on original position
        # Stores tuples of (x_clamped_pixel, y_clamped_pixel, visibility_flag)
        pixel_kps_with_visibility = [] 
        for x_raw_coord, y_raw_coord in raw_kps:
            x_r, y_r = float(x_raw_coord), float(y_raw_coord)
            
            originally_out_of_bounds = False
            
            # Clamp X coordinate
            if x_r < 0.0:
                x_clamped_px = 0.0
                originally_out_of_bounds = True
            elif x_r > img_width_f:
                x_clamped_px = img_width_f
                originally_out_of_bounds = True
            else:
                x_clamped_px = x_r
            
            # Clamp Y coordinate
            if y_r < 0.0:
                y_clamped_px = 0.0
                originally_out_of_bounds = True
            elif y_r > img_height_f:
                y_clamped_px = img_height_f
                originally_out_of_bounds = True
            else:
                y_clamped_px = y_r
                
            # Set visibility: 1.0 if originally OOB, 2.0 otherwise
            visibility_flag = 1.0 if originally_out_of_bounds else 2.0
            pixel_kps_with_visibility.append((x_clamped_px, y_clamped_px, visibility_flag))

        if not pixel_kps_with_visibility: 
             print(f"⚠️ No valid keypoints after processing for {img_id}, skipping...") # Should not be easily hit if raw_kps not empty
             continue

        # Determine the bounding box from CLAMPED keypoints (pixel coordinates)
        # Use only x,y from pixel_kps_with_visibility for bounding box
        x_coords_clamped_px = [p[0] for p in pixel_kps_with_visibility]
        y_coords_clamped_px = [p[1] for p in pixel_kps_with_visibility]
        
        min_x_px = min(x_coords_clamped_px)
        max_x_px = max(x_coords_clamped_px)
        min_y_px = min(y_coords_clamped_px)
        max_y_px = max(y_coords_clamped_px)
        
        # Calculate normalized bounding box values
        center_x_norm = (min_x_px + max_x_px) / (2 * img_width_f)
        center_y_norm = (min_y_px + max_y_px) / (2 * img_height_f)
        bbox_width_norm = (max_x_px - min_x_px) / img_width_f
        bbox_height_norm = (max_y_px - min_y_px) / img_height_f
        
        # Format CLAMPED keypoints into normalized YOLO format [0,1]
        normalized_yolo_kps = []
        for x_px_clamped, y_px_clamped, vis_flag in pixel_kps_with_visibility: # Unpack visibility
            norm_x = x_px_clamped / img_width_f
            norm_y = y_px_clamped / img_height_f
            
            # Apply final clamp to [0,1] for normalized values to guard against precision issues
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            normalized_yolo_kps.extend([norm_x, norm_y, vis_flag])  # Use determined visibility
        
        # Final clamp for bounding box values (robustness for floating point)
        center_x_norm = max(0.0, min(1.0, center_x_norm))
        center_y_norm = max(0.0, min(1.0, center_y_norm))
        bbox_width_norm = max(0.0, min(1.0, bbox_width_norm))
        bbox_height_norm = max(0.0, min(1.0, bbox_height_norm))
        
        # Combine all data into a single line
        # Format: class_id center_x_norm center_y_norm bbox_width_norm bbox_height_norm kp1_x_norm kp1_y_norm kp1_vis ...
        line = f"0 {center_x_norm} {center_y_norm} {bbox_width_norm} {bbox_height_norm} " + \
               " ".join([str(coord) for coord in normalized_yolo_kps])
        
        # Write the label file
        with open(dst_label_path, "w") as f:
            f.write(line)
    
    print(f"✅ Processed {dataset_type} dataset")

def main():
    """Main function to create the dataset."""
    print(f"Creating tennis court dataset in {DATASET_NAME}...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create YAML file
    create_yaml_file()
    
    # Process train and validation data
    process_json_data(TRAIN_JSON_FILE, "train")
    process_json_data(VAL_JSON_FILE, "val")
    
    print("✅ Dataset creation completed!")
    print(f"You can now use this dataset with: yolo train data={DATASET_NAME}.yaml")

if __name__ == "__main__":
    main() 