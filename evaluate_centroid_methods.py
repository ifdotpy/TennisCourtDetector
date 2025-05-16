import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm import tqdm
from scipy.spatial import distance
import time
import os

# Import different centroid finding methods
from inference import (
    find_centroid_max_value
)

# Import advanced methods
from advanced_centroid_methods import (
    find_centroid_adaptive_threshold,
)

from courtnetv2 import CourtFinderNetHeatmap
from dataset import preprocess_image

def load_validation_data(json_path):
    """Load validation data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def is_point_in_image(x, y, width, height):
    """Check if a point is inside the image boundaries"""
    return 0 <= x < width and 0 <= y < height

def evaluate_centroid_method(model, val_data, centroid_method, detection_threshold=0.0001,
                            max_dist=2, output_width=640, output_height=360, num_samples=None, img_dir=None):
    """
    Evaluate a centroid detection method on validation data
    
    Args:
        model: Neural network model
        val_data: Validation data list
        centroid_method: Function to find centroids from heatmaps
        detection_threshold: Threshold for detection
        centroid_threshold: Threshold for centroid method
        max_dist: Maximum acceptable distance for true positive
        output_width: Output width
        output_height: Output height
        num_samples: Number of samples to evaluate (None = all)
        img_dir: Directory with source images (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = next(model.parameters()).device
    
    if num_samples is not None:
        val_data = val_data[:num_samples]
    
    # Metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    all_distances = []
    processing_times = []
    
    # Default original_width/height to output_width/height for the case where img_dir is not provided
    # In that case, GT coords are assumed to be already in the output space for dummy heatmap generation.
    current_original_width, current_original_height = output_width, output_height

    # For each sample in validation data
    for idx, sample in enumerate(tqdm(val_data)):
        img_id = sample['id']
        ground_truth_kps_original_scale = sample['kps'] # These are from the JSON file
        
        # If image directory is provided, load and process the image
        if img_dir:
            try:
                img_path = f"{img_dir}/{img_id}.png"  # Ensure PNG extension
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Could not load image {img_path}. Skipping.")
                    continue
                
                current_original_height, current_original_width = frame.shape[:2] # Get actual original dimensions
                    
                # Process image following your inference pipeline
                img = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                inp, _ = preprocess_image(img)
                inp_tensor = torch.tensor(inp).unsqueeze(0).float().to(device)
                
                # Run inference
                with torch.no_grad():
                    out = model(inp_tensor)[0]
                
                pred_heatmaps = out.detach().cpu().numpy()
            except Exception as e:
                print(f"Error processing image {img_id}: {str(e)}")
                continue
        else:
            # Create dummy heatmaps for evaluation without inference
            # current_original_width/height are already output_width/height, so scaling GTs later will be 1:1.
            pred_heatmaps = []
            for kp_orig_scale in ground_truth_kps_original_scale:
                heatmap = np.zeros((output_height, output_width), dtype=np.float32)
                # For dummy heatmaps, assume kp_orig_scale is already in output space if no img_dir
                # (or scale them if a fixed original size for all JSON data was known, but it's safer this way)
                x_eff, y_eff = int(kp_orig_scale[0]), int(kp_orig_scale[1]) 
                if not (0 <= x_eff < output_width and 0 <= y_eff < output_height): # If GTs are out of assumed output space, adjust them or warn
                    # This branch assumes GTs in JSON are already scaled if no img_dir is used.
                    # If they are not, this dummy heatmap might be less useful.
                    pass # Or print a warning if desired

                if 0 <= x_eff < output_width and 0 <= y_eff < output_height:
                    sigma = 3.0
                    for i in range(max(0, y_eff-10), min(output_height, y_eff+11)):
                        for j in range(max(0, x_eff-10), min(output_width, x_eff+11)):
                            heatmap[i, j] = np.exp(-((i-y_eff)**2 + (j-x_eff)**2) / (2*sigma**2))
                pred_heatmaps.append(heatmap)
            pred_heatmaps = np.array(pred_heatmaps)
        
        # For each keypoint
        for kp_idx, gt_kp_orig in enumerate(ground_truth_kps_original_scale):
            if kp_idx >= len(pred_heatmaps):
                continue
                
            x_gt_original, y_gt_original = gt_kp_orig
            
            # Scale ground truth coordinates to model's output space (output_width, output_height)
            if x_gt_original is None or y_gt_original is None or current_original_width == 0 or current_original_height == 0:
                x_gt_scaled, y_gt_scaled = None, None
                gt_in_image = False
            else:
                x_gt_scaled = (x_gt_original / current_original_width) * output_width
                y_gt_scaled = (y_gt_original / current_original_height) * output_height
                gt_in_image = is_point_in_image(x_gt_scaled, y_gt_scaled, output_width, output_height)
                
            start_time = time.time()
            if centroid_method == find_centroid_max_value:
                _, (x_pred, y_pred) = centroid_method(pred_heatmaps[kp_idx], threshold=detection_threshold)
            elif centroid_method == find_centroid_adaptive_threshold:
                _, (x_pred, y_pred) = centroid_method(pred_heatmaps[kp_idx])
            else:
                _, (x_pred, y_pred) = centroid_method(pred_heatmaps[kp_idx])
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            pred_in_image = is_point_in_image(x_pred, y_pred, output_width, output_height) if x_pred is not None and y_pred is not None else False
            
            if pred_in_image and gt_in_image:
                dst = distance.euclidean((x_pred, y_pred), (x_gt_scaled, y_gt_scaled)) # Use scaled GT for distance
                all_distances.append(dst)
                if dst < max_dist:
                    total_tp += 1
                else:
                    total_fp += 1
            elif pred_in_image:
                total_fp += 1
            elif gt_in_image:
                total_fn += 1
            else:
                total_tn += 1
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp + 1e-15)
    recall = total_tp / (total_tp + total_fn + 1e-15)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-15)
    f1_score = 2 * precision * recall / (precision + recall + 1e-15)
    
    # Get distance statistics
    mean_dist = np.mean(all_distances) if all_distances else float('inf')
    median_dist = np.median(all_distances) if all_distances else float('inf')
    p95_dist = np.percentile(all_distances, 95) if len(all_distances) > 1 else float('inf')
    
    # Calculate average processing time
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    return {
        'tp': total_tp,
        'fp': total_fp, 
        'fn': total_fn,
        'tn': total_tn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'p95_distance': p95_dist,
        'avg_processing_time': avg_processing_time,
        'num_evaluated_kps': total_tp + total_fp + total_fn + total_tn
    }

def visualize_comparison(frame, heatmap, methods_dict, output_path="method_comparison.png", 
                        detection_threshold=0.0001, centroid_threshold=0.3):
    """
    Create a visual comparison of different centroid methods on one heatmap
    
    Args:
        frame: Original image frame
        heatmap: Single heatmap to process
        methods_dict: Dictionary of method_name: method_function
        output_path: Path to save the output image
    """
    # Calculate grid size
    num_methods = len(methods_dict)
    rows = (num_methods + 1) // 2  # +1 for the input frame
    cols = 2
    
    # Calculate single image dimensions
    img_height, img_width = frame.shape[:2]
    
    # Create a grid
    grid_height = rows * img_height
    grid_width = cols * img_width
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add original frame at top-left
    grid[:img_height, :img_width] = frame.copy()
    
    # Add a heatmap visualization in the first row, second column
    heatmap_vis = cv2.applyColorMap(
        cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    alpha = 0.7
    heatmap_overlay = cv2.addWeighted(frame.copy(), 1-alpha, heatmap_vis, alpha, 0)
    grid[:img_height, img_width:2*img_width] = heatmap_overlay
    
    # Add text labels
    cv2.putText(grid, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(grid, "Heatmap", (img_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Process with each method and add to grid
    method_idx = 0
    for method_name, method_func in methods_dict.items():
        # Calculate position in grid
        row = 1 + (method_idx // cols)
        col = method_idx % cols
        
        # Get the centroid using this method
        if method_func == find_centroid_max_value:
            _, (x_pred, y_pred) = method_func(heatmap, threshold=detection_threshold)
        elif method_func == find_centroid_adaptive_threshold:
            _, (x_pred, y_pred) = method_func(heatmap, base_threshold=detection_threshold)
        elif method_func == find_centroid_weighted_max:
            _, (x_pred, y_pred) = method_func(heatmap, threshold=centroid_threshold)
        elif method_func == find_centroid_ensemble:
            _, (x_pred, y_pred) = method_func(heatmap, threshold=centroid_threshold)
        else:
            _, (x_pred, y_pred) = method_func(heatmap, threshold=centroid_threshold)
        
        # Create a copy of the frame for drawing
        frame_with_centroid = frame.copy()
        
        # Draw the centroid if valid
        if x_pred is not None and y_pred is not None:
            cv2.circle(frame_with_centroid, (int(x_pred), int(y_pred)), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_centroid, (int(x_pred), int(y_pred)), 7, (255, 255, 255), 2)
        
        # Add to grid
        top = row * img_height
        left = col * img_width
        grid[top:top+img_height, left:left+img_width] = frame_with_centroid
        
        # Add method name
        cv2.putText(grid, method_name, (left + 10, top + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        method_idx += 1
    
    # Save the combined visualization
    cv2.imwrite(output_path, grid)
    return grid

def main():
    parser = argparse.ArgumentParser(description="Evaluate different centroid finding methods")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data JSON")
    parser.add_argument("--img_dir", type=str, help="Directory with source images (optional)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--detection_threshold", type=float, default=0.0001, help="Threshold for detection")
    parser.add_argument("--max_dist", type=float, default=7, help="Maximum distance for true positive")
    parser.add_argument("--output_width", type=int, default=640, help="Output width")
    parser.add_argument("--output_height", type=int, default=360, help="Output height")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of methods")
    parser.add_argument("--visualize_sample", type=str, help="Sample ID to visualize (if not provided, picks a random one)")
    parser.add_argument("--visualize_output", type=str, default="method_comparison.png", help="Output path for visualization")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model_instance = CourtFinderNetHeatmap()
    
    # Dynamic device selection
    if torch.backends.mps.is_available():
        device_selected = torch.device("mps")
    elif torch.cuda.is_available():
        device_selected = torch.device('cuda')
    else:
        device_selected = torch.device('cpu')
    print(f"Using device: {device_selected}")
    
    model_instance = model_instance.to(device_selected)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device_selected)
    if 'model_state_dict' in checkpoint: # Old format
        model_instance.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint: # Likely PyTorch Lightning format
        # Adjust keys if necessary
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
        model_instance.load_state_dict(state_dict)
    else: # Raw state dict
        model_instance.load_state_dict(checkpoint)
    
    model_instance.eval()
    
    # Define centroid methods to evaluate
    centroid_methods = {
        "Max Value": find_centroid_max_value,
        "Adaptive Threshold": find_centroid_adaptive_threshold,
    }
    
    # Additional methods we could implement
    # TODO: Add more centroid finding methods if needed
    
    # Load validation data
    print(f"Loading validation data from {args.val_data}")
    val_data = load_validation_data(args.val_data)
    print(f"Loaded {len(val_data)} validation samples")
    
    # Evaluate each method
    results = {}
    for method_name, method_func in centroid_methods.items():
        print(f"\nEvaluating: {method_name}")
        metrics = evaluate_centroid_method(
            model_instance, 
            val_data,
            method_func,
            detection_threshold=args.detection_threshold,
            max_dist=args.max_dist,
            output_width=args.output_width,
            output_height=args.output_height,
            num_samples=args.num_samples,
            img_dir=args.img_dir
        )
        results[method_name] = metrics
        
    # Print results table
    print("\n" + "="*100)
    print("METHOD COMPARISON RESULTS")
    print("="*100)
    print(f"{'Method':<30} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Mean Dist':<10} | {'Time (ms)':<10} | {'TP/Total':<15}")
    print("-"*100)
    
    for method_name, metrics in results.items():
        print(f"{method_name:<30} | "
              f"{metrics['precision']:.4f}     | "
              f"{metrics['recall']:.4f}     | "
              f"{metrics['f1_score']:.4f}     | "
              f"{metrics['mean_distance']:.4f}     | "
              f"{metrics['avg_processing_time']*1000:.4f}   | "
              f"{metrics['tp']}/{metrics['tp'] + metrics['fp'] + metrics['fn'] + metrics['tn']}")
    
    print("="*100)
    
    # Find the best method based on different metrics
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    best_distance = min(results.items(), key=lambda x: x[1]['mean_distance'])
    fastest = min(results.items(), key=lambda x: x[1]['avg_processing_time'])
    
    print("\nBEST METHODS:")
    print(f"Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
    print(f"Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")
    print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    print(f"Best Mean Distance: {best_distance[0]} ({best_distance[1]['mean_distance']:.4f})")
    print(f"Fastest: {fastest[0]} ({fastest[1]['avg_processing_time']*1000:.4f} ms)")

    # Generate visualization if requested
    if args.visualize and args.img_dir:
        print("\nGenerating visualization...")
        
        # Choose a sample to visualize
        sample_id = args.visualize_sample
        if not sample_id:
            # Pick a random sample
            random_idx = np.random.randint(0, len(val_data))
            sample_id = val_data[random_idx]["id"]
        
        # Find the sample in val_data
        sample = None
        for s in val_data:
            if s["id"] == sample_id:
                sample = s
                break
                
        if sample is None:
            print(f"Sample ID {sample_id} not found in validation data")
        else:
            print(f"Visualizing methods for sample {sample_id}")
            
            # Load and process the image
            img_path = f"{args.img_dir}/{sample_id}.jpg"  # Adjust extension if needed
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"Error loading image {img_path}")
            else:
                # Resize to expected dimensions
                img = cv2.resize(frame, (args.output_width, args.output_height), interpolation=cv2.INTER_AREA)
                
                # Preprocess image
                inp, _ = preprocess_image(img)
                inp_tensor = torch.tensor(inp).unsqueeze(0).float().to(device_selected)
                
                # Run inference
                with torch.no_grad():
                    out = model_instance(inp_tensor)[0]
                
                # Get heatmaps
                heatmaps = out.detach().cpu().numpy()
                
                # Create a directory for visualizations if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(args.visualize_output)) or ".", exist_ok=True)
                
                # Generate visualization for each keypoint
                ground_truth_kps = sample["kps"]
                
                # Create a combined visualization with all methods for each keypoint
                for kp_idx in range(min(len(heatmaps), len(ground_truth_kps))):
                    x_gt, y_gt = ground_truth_kps[kp_idx]
                    
                    # Draw ground truth on the image
                    img_with_gt = img.copy()
                    cv2.circle(img_with_gt, (int(x_gt), int(y_gt)), 5, (0, 0, 255), -1)
                    cv2.circle(img_with_gt, (int(x_gt), int(y_gt)), 7, (255, 255, 255), 2)
                    
                    # Generate visualization
                    output_path = f"{os.path.splitext(args.visualize_output)[0]}_kp{kp_idx}.png"
                    visualize_comparison(
                        img_with_gt, 
                        heatmaps[kp_idx], 
                        centroid_methods,
                        output_path=output_path,
                        detection_threshold=args.detection_threshold,
                    )
                    print(f"Saved visualization for keypoint {kp_idx} to {output_path}")

if __name__ == "__main__":
    main() 