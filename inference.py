import time
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter # Added gaussian_filter
import matplotlib.pyplot as plt # Added for heatmap plotting
from dataset import preprocess_image
from courtnetv2 import CourtFinderNetHeatmap
import argparse
import plotly.graph_objects as go
from advanced_centroid_methods import find_centroid_adaptive_threshold
from tqdm import tqdm # Import tqdm

import numpy as np
from ultralytics import YOLO

# --- Constants for Heatmap --- 
COURT_W_S = 8.23          # Singles court width (m)
COURT_W_D = 10.97         # Doubles court width (m)
HALF_LEN = 11.885       # Baseline to net distance (m)
COURT_LEN = 23.77         # Full court length (m)
SERVICE_LINE_FROM_NET = 5.49 # Distance from net to service line (m)
SAFE_ZONE = 3.0           # Depth of area behind baseline for heatmap (m)
HEATMAP_RES = 0.05        # Resolution of heatmap grid (m, e.g., 0.05 = 5cm)
POSE_CONF_THR = 0.4       # Confidence threshold for ankle keypoints
LEFT_ANKLE_IDX = 15       # Index of left ankle in COCO keypoints
RIGHT_ANKLE_IDX = 16      # Index of right ankle in COCO keypoints

# --- Constant for Court Averaging ---
NUM_FRAMES_TO_AVERAGE_COURT = 5

# ──────────────────────────────────────────────────────────────────────────────
# Fast, purely-algebraic court-alignment
# ──────────────────────────────────────────────────────────────────────────────
# Inputs
#   raw_pts :  • flat list/1-D np.ndarray of length 28  –– OR ––
#              • shape (14, 2) array of (x,y) pixel coordinates
#
#              The order **must** follow the label list you gave:
#              0  Top-left corner              7  Bottom-left singles
#              1  Top-right corner             8  Bottom-left service line
#              2  Bottom-left corner           9  Bottom-right service line
#              3  Bottom-right corner         10  Top-left  service line
#              4  Top-right  singles          11  Top-right service line
#              5  Bottom-right singles        12  Center line top
#              6  Top-left  singles           13  Center line bottom
#
# Returns
#   np.ndarray  shape (14, 2)  –– the "snapped" / corrected points
#
# Maths
#   • Each straight court line is built from the **two** raw points that lie on it.
#     (a, b, c) with ‖(a,b)‖ = 1 for numerical stability.
#   • Intersections are solved analytically:  one 2×2 determinant per point.
#   • Total FLOPs per frame ≈ 100.
# ──────────────────────────────────────────────────────────────────────────────
def correct_court_points(raw_pts):
    pts = np.asarray(raw_pts, dtype=np.float64)
    if pts.ndim == 1:          # flat → (14,2)
        pts = pts.reshape(14, 2)
    if pts.shape != (14, 2):
        raise ValueError("Expect 14×2 coordinates (or flat length-28 array).")

    # --- helpers --------------------------------------------------------------
    def line_from_two(p, q):
        """Return (a,b,c) for the line through p,q in ax+by+c=0 normalised form."""
        (x1, y1), (x2, y2) = p, q
        a, b = y1 - y2, x2 - x1
        c     = x1 * y2 - x2 * y1
        nrm   = np.hypot(a, b)
        return (a / nrm, b / nrm, c / nrm)

    def intersect(L1, L2):
        """Intersect two homogeneous lines (a,b,c)."""
        a1, b1, c1 = L1
        a2, b2, c2 = L2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return np.nan, np.nan          # (almost) parallel – shouldn't happen
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return x, y

    # --- build the ten straight lines we need ---------------------------------
    # vertical-ish                                                             # indices used
    L_v_left_dbl   = line_from_two(pts[0],  pts[2])    # left doubles sideline    0,2
    L_v_left_sng   = line_from_two(pts[6],  pts[7])    # left singles sideline    6,7
    L_v_center     = line_from_two(pts[12], pts[13])   # centre service line     12,13
    L_v_right_sng  = line_from_two(pts[4],  pts[5])    # right singles sideline   4,5
    L_v_right_dbl  = line_from_two(pts[1],  pts[3])    # right doubles sideline   1,3

    # horizontal-ish
    L_h_top_base   = line_from_two(pts[0],  pts[1])    # far baseline             0,1
    L_h_top_srv    = line_from_two(pts[10], pts[11])   # far service line        10,11
    L_h_bot_srv    = line_from_two(pts[8],  pts[9])    # near service line        8,9
    L_h_bot_base   = line_from_two(pts[2],  pts[3])    # near baseline            2,3

    # --- compute the 14 corrected points --------------------------------------
    corr = np.empty_like(pts)

    # corners
    corr[0] = intersect(L_v_left_dbl,  L_h_top_base)   # Top-left corner
    corr[1] = intersect(L_v_right_dbl, L_h_top_base)   # Top-right corner
    corr[2] = intersect(L_v_left_dbl,  L_h_bot_base)   # Bottom-left corner
    corr[3] = intersect(L_v_right_dbl, L_h_bot_base)   # Bottom-right corner

    # singles sidelines × baselines
    corr[4] = intersect(L_v_right_sng, L_h_top_base)   # Top-right  singles
    corr[5] = intersect(L_v_right_sng, L_h_bot_base)   # Bottom-right singles
    corr[6] = intersect(L_v_left_sng,  L_h_top_base)   # Top-left  singles
    corr[7] = intersect(L_v_left_sng,  L_h_bot_base)   # Bottom-left singles

    # singles sidelines × service lines
    corr[8]  = intersect(L_v_left_sng,  L_h_bot_srv)   # Bottom-left  service-line
    corr[9]  = intersect(L_v_right_sng, L_h_bot_srv)   # Bottom-right service-line
    corr[10] = intersect(L_v_left_sng,  L_h_top_srv)   # Top-left   service-line
    corr[11] = intersect(L_v_right_sng, L_h_top_srv)   # Top-right  service-line

    # centre service line × service lines
    corr[12] = intersect(L_v_center, L_h_top_srv)      # Centre line top
    corr[13] = intersect(L_v_center, L_h_bot_srv)      # Centre line bottom

    return corr

def draw_court_lines(image, court_points, color=(0,255,255), thickness=1):
    """
    Draws the main tennis court lines on the image using the 14 corrected points.
    Args:
        image: np.ndarray, the image to draw on (modified in place)
        court_points: np.ndarray of shape (14,2), output of correct_court_points
        color: tuple, BGR color for the lines
        thickness: int, line thickness
    """
    pts = np.asarray(court_points, dtype=np.float32)
    # Define the main lines as pairs of indices
    lines = [
        # Outer rectangle (doubles)
        (0,1), (1,3), (3,2), (2,0),
        # Singles rectangle
        (6,4), (4,5), (5,7), (7,6),
        # Service boxes
        (10,11), (8,9),
        # Service box horizontals
        (10,8), (11,9),
        # Center service line
        (12,13),
    ]
    for i,j in lines:
        if not (np.any(np.isnan(pts[i])) or np.any(np.isnan(pts[j]))):
            cv2.line(image, (int(pts[i][0]), int(pts[i][1])), (int(pts[j][0]), int(pts[j][1])), color, thickness, lineType=cv2.LINE_AA)
    return image

def visualize_heatmaps_on_image(inp, centroids, filtered_heatmaps, alpha=0.5):
    # Create a figure
    fig = go.Figure()

    colors = [
        "red",  # Pure red
        "lime",  # Bright lime green
        "blue",  # Pure blue
        "yellow",  # Bright yellow
        "cyan",  # Bright cyan
        "magenta",  # Bright magenta
        "orange",  # Bright orange
        "purple",  # Bright purple
        "pink",  # Bright pink
        "gold",  # Bright gold
        "chartreuse",  # Bright chartreuse
        "indigo",  # Bright indigo
        "turquoise",  # Bright turquoise
        "white"  # Bright spring green
    ]

    # Overlay the combined heatmap on the input image
    fig.add_trace(go.Image(z=inp))

    for i, (centroid, filtered_heatmap) in enumerate(zip(centroids, filtered_heatmaps)):
        x_pred, y_pred = centroid
        fig.add_trace(go.Heatmap(z=filtered_heatmap, opacity=alpha, colorscale='Reds', showscale=False))
        fig.add_trace(go.Scatter(x=[x_pred], y=[y_pred], mode='markers',
                                 marker=dict(color=colors[i % len(colors)], size=7),
                                 name=f'Centroid {i}'))

    fig.update_layout(title='Combined Heatmaps with Best Pixels Marked', xaxis=dict(visible=False),
                      yaxis=dict(visible=False))
    fig.show()

def visualize_heatmaps_on_image_cv2(inp, centroids, filtered_heatmaps, alpha=0.5, title=None, timing_dict=None):
    # Convert the input image to RGB if it's not already
    if len(inp.shape) == 2 or inp.shape[2] == 1:
        inp = cv2.cvtColor(inp, cv2.COLOR_GRAY2RGB)
    elif inp.shape[2] == 4:
        inp = cv2.cvtColor(inp, cv2.COLOR_BGRA2BGR)
    
    # Ensure inp is 8-bit for colormap application and blending
    if inp.dtype != np.uint8:
        inp = cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # Create a blank image for the combined heatmap overlay
    # Initialize with zeros, ensure it's the same size as inp but can be float for accumulation
    combined_heatmap_colored = np.zeros_like(inp, dtype=np.float32)

    for i, (centroid, filtered_heatmap) in enumerate(zip(centroids, filtered_heatmaps)):
        if filtered_heatmap is None: # Skip if no heatmap data
            continue

        # Normalize the single heatmap: handle potential NaNs by replacing with 0 before normalization
        current_heatmap_normalized = np.nan_to_num(filtered_heatmap)
        
        # Only proceed if there's actual data in the heatmap after NaN removal
        if np.max(current_heatmap_normalized) > 0:
            current_heatmap_normalized = cv2.normalize(current_heatmap_normalized, None, 0, 255, cv2.NORM_MINMAX)
            current_heatmap_normalized = current_heatmap_normalized.astype(np.uint8)
            
            # Apply a colormap (e.g., JET)
            # Ensure heatmap is 2D for applyColorMap if it became 3D (e.g. shape (H, W, 1))
            if len(current_heatmap_normalized.shape) == 3 and current_heatmap_normalized.shape[2] == 1:
                current_heatmap_normalized = current_heatmap_normalized.squeeze()
            
            heatmap_colored_single = cv2.applyColorMap(current_heatmap_normalized, cv2.COLORMAP_JET)
            
            # Add to the combined heatmap overlay
            # We add heatmaps together; could also blend or take max
            combined_heatmap_colored += heatmap_colored_single.astype(np.float32)

        # Draw the centroid on the input image
        if centroid is not None and centroid[0] is not None and centroid[1] is not None:
            x_pred, y_pred = centroid
            cv2.circle(inp, (int(x_pred), int(y_pred)), 3, (0, 255, 0), -1) # Green circle for centroids

    # Normalize the combined heatmap overlay to the range [0, 255] after accumulation
    if np.max(combined_heatmap_colored) > 0: # Avoid division by zero if all heatmaps were empty
        combined_heatmap_colored = cv2.normalize(combined_heatmap_colored, None, 0, 255, cv2.NORM_MINMAX)
    combined_heatmap_colored = combined_heatmap_colored.astype(np.uint8)
    
    # Blend the heatmap overlay with the input image
    # Ensure inp is uint8
    if inp.dtype != np.uint8:
        inp = inp.astype(np.uint8)
        
    blended_image = cv2.addWeighted(inp, 1 - alpha, combined_heatmap_colored, alpha, 0)

    # Plot original and corrected centroids
    centroids_np = np.array(centroids)
    correct_time_ms = None
    if centroids_np.shape == (14, 2):
        # Draw original centroids (red)
        for (x, y) in centroids_np:
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                cv2.circle(blended_image, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red
        # Draw corrected centroids (green)
        t0 = time.time()
        corrected = correct_court_points(centroids_np)
        correct_time_ms = (time.time() - t0) * 1000
        if timing_dict is not None:
            timing_dict['correct_court_points_ms'] = correct_time_ms
        for (x, y) in corrected:
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                cv2.circle(blended_image, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green outline
        # Draw court lines
        draw_court_lines(blended_image, corrected, color=(0,255,255), thickness=1)

    # Add title to the image if provided
    if title:
        cv2.putText(blended_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return blended_image


def calculate_centroids(hm_hp):
    """Calculates centroids from heatmaps using adaptive threshold method."""
    centroids = []
    for i in range(hm_hp.shape[0]):
        # Calculate centroid - adaptive threshold handles thresholding internally for finding the point
        _, centroid_coords = find_centroid_adaptive_threshold(hm_hp[i])
        centroids.append(centroid_coords)
    return centroids

# ----------------------------------------------------------------------
# Fast centroid finder --------------------------------------------------
# ----------------------------------------------------------------------
def _centroid_of_peak(mat, lbl_buf, base=0.1, ratio=0.3):
    import numpy as np
    from scipy.ndimage import label

    vmax = mat.max()
    if vmax <= 0:
        return None, (None, None)

    max_pos = np.unravel_index(np.argmax(mat), mat.shape)
    thr     = max(base, ratio * vmax)

    mask = mat >= thr
    if not mask[max_pos]:
        mask[max_pos] = True

    # ---- label in-place into lbl_buf ---------------------------------
    res = label(mask, output=lbl_buf)
    n_features = res[1] if isinstance(res, tuple) else res

    peak_id = lbl_buf[max_pos]
    if peak_id == 0 or n_features == 0:
        return None, (None, None)

    # ---- centroid ----------------------------------------------------
    pts_y, pts_x = np.where(lbl_buf == peak_id)
    vals = mat[pts_y, pts_x]
    wsum = vals.sum()
    if wsum == 0:
        return None, (None, None)

    cy = (pts_y * vals).sum() / wsum
    cx = (pts_x * vals).sum() / wsum
    return None, (cx, cy)


# ----------------------------------------------------------------------
# Public API – same name, same args ------------------------------------
# ----------------------------------------------------------------------
def find_centroid_adaptive_threshold(matrix,
                                     base_threshold=0.1,
                                     adaptive_ratio=0.3):
    """
    Direct, faster replacement for your original routine.
    Keeps the exact signature and return shape.
    """
    # Allocate a scratch label buffer exactly once per call.
    # (If you call this on many heat-maps one after another,
    #  calculate_centroids() will pass in one persistent buffer.)
    lbl_buf = np.empty_like(matrix, dtype=np.int32)
    return _centroid_of_peak(matrix, lbl_buf,
                             base=base_threshold,
                             ratio=adaptive_ratio)


def calculate_centroids(hm_hp,
                        base_threshold=0.1,
                        adaptive_ratio=0.3):
    """
    Faster drop-in replacement.
    Uses one shared label buffer for *all* heat-maps in the batch
    and avoids Python-level allocations inside the loop.
    """
    if hm_hp.ndim != 3:
        raise ValueError("hm_hp must be (N, H, W)")

    # one scratch buffer, re-used for every frame ----------------------
    lbl_buf = np.empty(hm_hp.shape[1:], dtype=np.int32)

    centroids = []
    for mat in hm_hp:
        _, (cx, cy) = _centroid_of_peak(mat, lbl_buf,
                                        base=base_threshold,
                                        ratio=adaptive_ratio)
        centroids.append((cx, cy))
    return centroids


def draw_results_on_image_cv2(image, centroids, raw_heatmaps, corrected_points, alpha=0.5, title=None, detection_threshold=0.0001):
    """Draws heatmaps, centroids (original and corrected), and court lines on the image."""
    vis_image = image.copy()
    if vis_image.dtype != np.uint8:
        vis_image = cv2.normalize(vis_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ensure vis_image is 3 channels for blending
    if len(vis_image.shape) == 2 or vis_image.shape[2] == 1:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    elif vis_image.shape[2] == 4:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGRA2BGR)

    combined_heatmap_colored = np.zeros_like(vis_image, dtype=np.float32)
    num_heatmaps = raw_heatmaps.shape[0] if raw_heatmaps is not None else 0

    for i in range(num_heatmaps):
        current_raw_heatmap = raw_heatmaps[i]
        mask = current_raw_heatmap > detection_threshold
        filtered_heatmap = np.where(mask, current_raw_heatmap, np.nan)

        # Handle potential NaNs before checking max
        heatmap_for_norm = np.nan_to_num(filtered_heatmap)

        # Only proceed if there's actual data > 0 in the heatmap
        if np.max(heatmap_for_norm) > 0:
            # Normalize the valid heatmap data
            normalized_heatmap = cv2.normalize(heatmap_for_norm, None, 0, 255, cv2.NORM_MINMAX)
            # Convert to uint8 *after* normalization
            heatmap_uint8 = normalized_heatmap.astype(np.uint8)

            # Ensure it's 2D for applyColorMap if needed
            if len(heatmap_uint8.shape) == 3 and heatmap_uint8.shape[2] == 1:
                heatmap_uint8 = heatmap_uint8.squeeze()

            # Apply colormap only to valid, converted heatmap
            heatmap_colored_single = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_heatmap_colored += heatmap_colored_single.astype(np.float32)

    # Normalize the combined heatmap overlay
    if np.max(combined_heatmap_colored) > 0: # Avoid division by zero if all heatmaps were empty
        combined_heatmap_colored = cv2.normalize(combined_heatmap_colored, None, 0, 255, cv2.NORM_MINMAX)
    combined_heatmap_colored = combined_heatmap_colored.astype(np.uint8)

    # Blend the heatmap overlay with the input image
    blended_image = cv2.addWeighted(vis_image, 1 - alpha, combined_heatmap_colored, alpha, 0)

    # Plot original centroids (red)
    centroids_np = np.array(centroids)
    if centroids_np.shape == (14, 2):
        for (x, y) in centroids_np:
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                cv2.circle(blended_image, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red

    # Plot corrected centroids (green) and draw lines if available
    if corrected_points is not None and corrected_points.shape == (14, 2):
        for (x, y) in corrected_points:
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                cv2.circle(blended_image, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green outline
        # Draw court lines using the corrected points
        draw_court_lines(blended_image, corrected_points, color=(0,255,255), thickness=1)

    # Add title to the image if provided
    if title:
        cv2.putText(blended_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return blended_image

def preprocess_frame_data(frame, output_width, output_height):
    """Resizes and preprocesses a single frame for model input."""
    t_start = time.time()
    img = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
    inp, _ = preprocess_image(img) # inp is typically the normalized image tensor for the model
    preprocess_time = (time.time() - t_start) * 1000
    return img, inp, preprocess_time

def run_court_model_inference(inp, model, device):
    """Runs the court detection model inference."""
    t_start = time.time()
    inp_tensor = torch.tensor(inp).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(inp_tensor)[0]
    inference_time = (time.time() - t_start) * 1000
    return out, inference_time

def run_court_model_inference_batch(inp_batch_tensor, model, device):
    """Runs the court detection model inference on a batch of inputs."""
    t_start = time.time()
    # inp_batch_tensor is already a batch, e.g., (B, C, H, W)
    with torch.no_grad():
        out_batch = model(inp_batch_tensor) # Model handles the batch
    inference_time = (time.time() - t_start) * 1000
    return out_batch, inference_time

def postprocess_court_model_output(out):
    """Calculates centroids and corrected points from model output."""
    t_start = time.time()
    pred = out.detach().cpu().numpy()[:14] # Raw heatmaps
    centroids = calculate_centroids(pred)
    centroid_calc_time = (time.time() - t_start) * 1000

    # Calculate corrected court points (timing included separately)
    # Note: correction logic itself might be moved or refactored further depending on needs
    t_start_correction = time.time()
    centroids_np = np.array(centroids)
    corrected_points = centroids_np # Default to original if correction fails or isn't applicable

    # The original commented-out block for correction is kept here for reference,
    # but it was commented out in the original script provided.
    # if centroids_np.shape == (14, 2) and not np.all(np.isnan(centroids_np)):
    #     try:
    #         corrected_points = correct_court_points(centroids_np)
    #     except Exception as e:
    #         print(f"Error during correct_court_points: {e}")
    point_correction_time = (time.time() - t_start_correction) * 1000

    return pred, centroids, corrected_points, centroid_calc_time, point_correction_time

def postprocess_court_model_output_batch(out_batch):
    """Calculates centroids and corrected points from a batch of model outputs."""
    t_start_batch = time.time()
    
    preds_list = []
    centroids_list = []
    corrected_points_list = []
    
    num_in_batch = out_batch.shape[0]

    total_centroid_calc_time_for_batch = 0
    total_point_correction_time_for_batch = 0

    for i in range(num_in_batch):
        out_single = out_batch[i] 
        # pred_single = out_single.detach().cpu().numpy()[:14]
        # Assuming out_single is already on CPU and detached if model output is directly usable for numpy conversion
        # Or, if out_single is part of a larger tensor that needs detaching/CPU transfer:
        if isinstance(out_single, torch.Tensor):
            pred_single = out_single.detach().cpu().numpy()[:14]
        else: # If already a numpy array (e.g., if model output was already processed)
            pred_single = out_single[:14]

        t_start_centroids = time.time()
        centroids_single = calculate_centroids(pred_single)
        total_centroid_calc_time_for_batch += (time.time() - t_start_centroids) * 1000
        
        centroids_np_single = np.array(centroids_single)
        corrected_points_single = centroids_np_single # Default

        t_start_correction = time.time()
        # Ensure not all elements are NaN before attempting correction, and shape is correct
        if centroids_np_single.shape == (14, 2) and not np.all(np.isnan(centroids_np_single)):
            try:
                corrected_points_single = correct_court_points(centroids_np_single)
            except Exception as e:
                print(f"Error during correct_court_points for item {i} in batch: {e}")
        total_point_correction_time_for_batch += (time.time() - t_start_correction) * 1000

        preds_list.append(pred_single)
        centroids_list.append(centroids_single)
        corrected_points_list.append(corrected_points_single)

    return preds_list, centroids_list, corrected_points_list, total_centroid_calc_time_for_batch, total_point_correction_time_for_batch

def is_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None, False  # Cannot open the file, assume it's not a video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_count > 1


def main(args):
    # --- Court Model Setup ---
    court_model = CourtFinderNetHeatmap()

    # Dynamic device selection
    if torch.backends.mps.is_available():
        device_selected = torch.device("mps")
    elif torch.cuda.is_available():
        device_selected = torch.device('cuda')
    else:
        device_selected = torch.device('cpu')
    print(f"Using device: {device_selected}")

    court_model = court_model.to(device_selected)

    checkpoint = torch.load(args.model_path, map_location=device_selected)
    if 'model_state_dict' in checkpoint:  # Old format
        court_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:  # Likely PyTorch Lightning format
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
        court_model.load_state_dict(state_dict)
    else:  # Raw state dict
        court_model.load_state_dict(checkpoint)

    court_model.eval()

    # --- Pose Model Setup ---
    print(f"Loading YOLO pose model from: {args.pose_model_path}")
    pose_model = YOLO(args.pose_model_path)
    # Move YOLO model to the same device if possible (YOLO handles device internally but good practice)
    pose_model.to(device_selected)
    print("YOLO pose model loaded.")

    # --- Initialize Heatmap Variables ---
    H_homography = None
    all_foot_xy = [] # Accumulate foot positions across all frames

    # --- Variables for Static Court Layout ---
    frames_processed_for_court_avg = 0
    collected_court_points_for_avg = []
    static_court_points = None
    static_homography = None

    file_path = args.input_path
    cap, is_vid = is_video(file_path)
    batch_size_pose_only = args.batch_size # For pose estimation after initial court averaging

    if is_vid:
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames to process: {total_frames_in_video}")
        print(f"Court layout will be averaged from the first {NUM_FRAMES_TO_AVERAGE_COURT} frames (processed as a single batch for court model).")
        print(f"Subsequent pose estimation will use batch size: {batch_size_pose_only}")

        output_video = cv2.VideoWriter('output_video.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30, # Use source FPS
                                     (args.output_width, args.output_height))

        total_timings = {
            'preprocess_ms': 0, 
            'court_model_inference_ms': 0, # Will be for the single initial batch
            'court_model_postprocess_ms': 0, # Will be for the single initial batch (centroids + correction)
            'pose_estimation_ms': 0, # Accumulated over all batches
            'batch_processing_overhead_ms': 0 # Accumulated
        }
        progress_bar = tqdm(total=total_frames_in_video, desc="Processing video", unit="frame")
        frames_written_to_video = 0

        # --- Phase 1: Process initial NUM_FRAMES_TO_AVERAGE_COURT for court detection --- 
        initial_frames_original = []
        initial_frames_resized_for_pose = []
        initial_frames_preprocessed_for_court = []
        
        batch_preprocess_time_acc = 0
        for _ in range(NUM_FRAMES_TO_AVERAGE_COURT):
            ret, frame = cap.read()
            if not ret: break
            initial_frames_original.append(frame)
            img_resized, inp_preprocessed, ppt = preprocess_frame_data(frame, args.output_width, args.output_height)
            initial_frames_resized_for_pose.append(img_resized.copy())
            initial_frames_preprocessed_for_court.append(inp_preprocessed)
            batch_preprocess_time_acc += ppt
        total_timings['preprocess_ms'] += batch_preprocess_time_acc

        if initial_frames_original:
            num_initial_frames_read = len(initial_frames_original)
            initial_batch_overhead_start_time = time.time()

            # 1.1 YOLO Pose Estimation for initial batch
            t_start_pose_initial_batch = time.time()
            pose_results_initial_batch = pose_model.predict(source=initial_frames_resized_for_pose, half=True, verbose=False, stream=False)
            time_pose_initial_batch_ms = (time.time() - t_start_pose_initial_batch) * 1000
            total_timings['pose_estimation_ms'] += time_pose_initial_batch_ms

            # 1.2 Court Model (single batch for all initial frames)
            inp_court_model_initial_batch_tensor = torch.stack(
                [torch.tensor(inp).float() for inp in initial_frames_preprocessed_for_court]
            ).to(device_selected)
            
            out_court_initial_batch, time_court_infer_initial_batch_ms = run_court_model_inference_batch(
                inp_court_model_initial_batch_tensor, court_model, device_selected
            )
            total_timings['court_model_inference_ms'] += time_court_infer_initial_batch_ms # This is for the whole initial batch

            # 1.3 Post-process Court Model Output for initial batch
            t_start_court_post_initial_batch = time.time()
            preds_list_initial, centroids_list_initial, corrected_points_list_initial, \
            time_centroid_calc_initial_batch_ms, time_point_correct_initial_batch_ms = \
                postprocess_court_model_output_batch(out_court_initial_batch)
            time_court_post_initial_batch_ms = (time.time() - t_start_court_post_initial_batch) * 1000
            total_timings['court_model_postprocess_ms'] += time_court_post_initial_batch_ms # Combines centroid and correction timing
            
            # Calculate overhead for this initial super-batch
            initial_batch_total_block_duration_ms = (time.time() - initial_batch_overhead_start_time) * 1000
            sum_of_initial_components_ms = time_pose_initial_batch_ms + time_court_infer_initial_batch_ms + time_court_post_initial_batch_ms
            total_timings['batch_processing_overhead_ms'] += initial_batch_total_block_duration_ms - sum_of_initial_components_ms

            # 1.4 Average court points from this initial batch
            valid_points_for_averaging = []
            for cp in corrected_points_list_initial:
                if cp is not None and cp.shape == (14,2) and not np.all(np.isnan(cp)):
                    valid_points_for_averaging.append(cp)
            
            print(f"\n--- Finished processing {num_initial_frames_read} frames for court averaging ---")
            if valid_points_for_averaging:
                try:
                    stacked_points = np.stack(valid_points_for_averaging, axis=0)
                    static_court_points = np.nanmean(stacked_points, axis=0)
                    if np.all(np.isnan(static_court_points)):
                        print("Warning: Averaging court points resulted in all NaNs."); static_court_points = None
                    else: 
                        print("Static court points successfully averaged.")
                        static_homography = calculate_homography(static_court_points)
                        if static_homography is not None: print("Static homography calculated.")
                        else: print("Warning: Failed to calculate static homography from averaged points.")
                except Exception as e: print(f"Error averaging/static H: {e}"); static_court_points=None; static_homography=None
            else: print(f"No valid court points in first {num_initial_frames_read} frames. No static layout.")
            print("Court model will not run on subsequent frames.")

            # 1.5 Process and write initial frames
            for i in range(num_initial_frames_read):
                img_with_poses = pose_results_initial_batch[i].plot() if pose_results_initial_batch and i < len(pose_results_initial_batch) else initial_frames_resized_for_pose[i].copy()
                
                current_pred_for_drawing = preds_list_initial[i]
                current_centroids_for_drawing = centroids_list_initial[i]
                current_corrected_points_for_drawing = corrected_points_list_initial[i]
                
                active_H_for_footprints_initial = static_homography # Prefer static if already computed
                if active_H_for_footprints_initial is None and current_corrected_points_for_drawing is not None and \
                   current_corrected_points_for_drawing.shape == (14,2) and not np.all(np.isnan(current_corrected_points_for_drawing)):
                    active_H_for_footprints_initial = calculate_homography(current_corrected_points_for_drawing)
                    if i == 0 and active_H_for_footprints_initial is not None : H_homography = active_H_for_footprints_initial # Legacy for heatmap check
                
                if active_H_for_footprints_initial is not None and pose_results_initial_batch and i < len(pose_results_initial_batch):
                    foot_xy_frame = collect_foot_positions([pose_results_initial_batch[i]], active_H_for_footprints_initial, POSE_CONF_THR)
                    all_foot_xy.extend(foot_xy_frame)

                result_frame = draw_results_on_image_cv2(img_with_poses, current_centroids_for_drawing, current_pred_for_drawing, current_corrected_points_for_drawing, title="Tennis Court Detection")
                output_video.write(result_frame)
                progress_bar.update(1)
                frames_written_to_video += 1

        # --- Phase 2: Process remaining frames (pose-only batches) ---
        while True:
            batch_original_frames = []
            batch_resized_imgs_for_pose = []
            current_batch_preprocess_time_acc = 0

            for _ in range(batch_size_pose_only):
                ret, frame = cap.read()
                if not ret: break
                batch_original_frames.append(frame)
                img_resized, _, ppt = preprocess_frame_data(frame, args.output_width, args.output_height) # inp not needed here
                batch_resized_imgs_for_pose.append(img_resized.copy())
                current_batch_preprocess_time_acc += ppt
            
            if not batch_original_frames: break # End of video
            total_timings['preprocess_ms'] += current_batch_preprocess_time_acc
            num_in_current_pose_batch = len(batch_original_frames)

            current_pose_batch_overhead_start_time = time.time()

            # 2.1 YOLO Pose Estimation for current batch
            t_start_pose_curr_batch = time.time()
            pose_results_current_batch = pose_model.predict(source=batch_resized_imgs_for_pose, half=True, verbose=False, stream=False)
            time_pose_curr_batch_ms = (time.time() - t_start_pose_curr_batch) * 1000
            total_timings['pose_estimation_ms'] += time_pose_curr_batch_ms

            # Calculate overhead for this pose-only batch
            # For pose-only batches, components are just pose estimation time
            current_pose_batch_total_block_duration_ms = (time.time() - current_pose_batch_overhead_start_time) * 1000
            total_timings['batch_processing_overhead_ms'] += current_pose_batch_total_block_duration_ms - time_pose_curr_batch_ms

            # 2.2 Process and write frames from current pose-only batch
            for i in range(num_in_current_pose_batch):
                img_with_poses = pose_results_current_batch[i].plot() if pose_results_current_batch and i < len(pose_results_current_batch) else batch_resized_imgs_for_pose[i].copy()
                
                # Court model not run, use static points, pred/centroids are None
                if static_homography is not None and pose_results_current_batch and i < len(pose_results_current_batch):
                    foot_xy_frame = collect_foot_positions([pose_results_current_batch[i]], static_homography, POSE_CONF_THR)
                    all_foot_xy.extend(foot_xy_frame)
                
                result_frame = draw_results_on_image_cv2(img_with_poses, None, None, static_court_points, title="Tennis Court Detection (Static Court)")
                output_video.write(result_frame)
                progress_bar.update(1)
                frames_written_to_video +=1

        progress_bar.close()
        cap.release()
        output_video.release()
        print(f"\nVideo processing complete. Output saved to output_video.mp4 ({frames_written_to_video} frames written)")
        if frames_written_to_video > 0:
            print("Average timings per frame (note: court model timings are for initial batch only):")
            # Adjust court model timings to be per-frame *for the frames they ran on*
            if num_initial_frames_read > 0:
                avg_court_infer_ms_over_initial_frames = total_timings['court_model_inference_ms'] / num_initial_frames_read
                avg_court_post_ms_over_initial_frames = total_timings['court_model_postprocess_ms'] / num_initial_frames_read
            else:
                avg_court_infer_ms_over_initial_frames = 0
                avg_court_post_ms_over_initial_frames = 0
            
            # For reporting per-frame amortized over the whole video:
            amortized_court_infer_ms = total_timings['court_model_inference_ms'] / frames_written_to_video
            amortized_court_post_ms = total_timings['court_model_postprocess_ms'] / frames_written_to_video

            print(f"  preprocess_ms: {total_timings['preprocess_ms'] / frames_written_to_video:.2f} ms")
            print(f"  pose_estimation_ms: {total_timings['pose_estimation_ms'] / frames_written_to_video:.2f} ms")
            print(f"  court_model_inference_ms (amortized avg over all frames): {amortized_court_infer_ms:.2f} ms")
            print(f"  court_model_postprocess_ms (amortized avg over all frames): {amortized_court_post_ms:.2f} ms")
            print(f"  batch_processing_overhead_ms: {total_timings['batch_processing_overhead_ms'] / frames_written_to_video:.2f} ms")

        # --- Generate and Save Heatmap (After processing all frames) ---
        if not all_foot_xy:
            print("\nNo valid foot positions collected, skipping heatmap generation.")
        elif static_homography is None: # Primary check is for the successfully established static homography
            print("\nStatic homography from averaged court points was not successfully established.")
            if H_homography is None: # Fallback: if even the first frame H wasn't good
                 print("Additionally, no valid per-frame homography was established during initial phase. Heatmap unreliable/skipped.")
                 if not all_foot_xy: print("No foot positions collected, definitely skipping heatmap.")
                 # else, it might proceed with inconsistently transformed points if any were collected
            else:
                print("Heatmap will be generated using foot positions, but consistency relies on early per-frame homographies and may not be ideal.")
            if not all_foot_xy: # Double check if no points were collected at all
                 print("No foot positions collected, skipping heatmap generation.")
            elif all_foot_xy : # If points were collected despite H issues
                print("Proceeding with heatmap generation using collected foot positions...")
                roi_foot_xy = filter_footprints_for_heatmap(all_foot_xy, COURT_W_S, SAFE_ZONE)
                print(f"Foot positions in heatmap ROI: {len(roi_foot_xy)}")
                if not roi_foot_xy:
                    print("No foot positions within the specified ROI, skipping heatmap generation.")
                else:
                    heat, _, _ = generate_heatmap_data(roi_foot_xy, COURT_W_S, SAFE_ZONE, HEATMAP_RES)
                    if heat is not None:
                        plot_save_heatmap(heat, COURT_W_S, SAFE_ZONE, output_filename="heatmap_from_net.png")
                    else:
                        print("Heatmap data generation failed.")
        else: # Static homography is available and assumed to have been used for all_foot_xy after avg period
            print(f"\nTotal foot positions collected for heatmap (using static homography where applicable): {len(all_foot_xy)}")
            roi_foot_xy = filter_footprints_for_heatmap(all_foot_xy, COURT_W_S, SAFE_ZONE)
            print(f"Foot positions in heatmap ROI: {len(roi_foot_xy)}")
            if not roi_foot_xy:
                print("No foot positions within the specified ROI, skipping heatmap generation.")
            else:
                heat, _, _ = generate_heatmap_data(roi_foot_xy, COURT_W_S, SAFE_ZONE, HEATMAP_RES)
                if heat is not None:
                    plot_save_heatmap(heat, COURT_W_S, SAFE_ZONE, output_filename="heatmap_from_net.png")
                else:
                    print("Heatmap data generation failed.")

    else: # Single image processing
        frame_img = cv2.imread(args.input_path)
        if frame_img is None:
            print(f"Error: Could not read image {args.input_path}")
        else:
            print("Processing image...")
            # 1. Preprocess frame
            img, inp, preprocess_time = preprocess_frame_data(frame_img, args.output_width, args.output_height)
            timings = {'preprocess_ms': preprocess_time}

            # 2. Run YOLO Pose Estimation
            t_start_pose = time.time()
            pose_results = pose_model.predict(source=img.copy(), verbose=False)
            timings['pose_estimation_ms'] = (time.time() - t_start_pose) * 1000
            img_with_poses = pose_results[0].plot()

            # 3. Run Court Model Inference
            out, timings['court_model_inference_ms'] = run_court_model_inference(inp, court_model, device_selected)

            # 4. Post-process Court Model Output
            pred, centroids, corrected_points, timings['court_model_centroid_calc_ms'], timings['court_model_point_correction_ms'] = postprocess_court_model_output(out)

            # --- Heatmap Calculation (Single Image) ---
            H_single_image = None # Local H for single image
            if corrected_points is not None and corrected_points.shape == (14, 2):
                print("Calculating homography for single image...")
                H_single_image = calculate_homography(corrected_points)
                if H_single_image is not None: print("Homography calculated for single image.")
                else: print("Homography failed for single image.")
            else:
                print("Warning: Cannot calculate homography for single image, court points invalid.")

            current_frame_foot_xy = []
            if H_single_image is not None and pose_results:
                current_frame_foot_xy = collect_foot_positions(pose_results, H_single_image, POSE_CONF_THR)
            # --- End Heatmap Calculation (Single Image) ---

            # 5. Draw Court Results (on top of image with poses)
            t_start_drawing = time.time()
            result_image = draw_results_on_image_cv2(
                img_with_poses, # Start drawing on the image with poses
                centroids,
                pred, # Pass raw heatmaps (pred)
                corrected_points,
                title="Tennis Court Detection"
            )
            cv2.imwrite('result.png', result_image)
            print("Image processing complete. Output saved to result.png")
            print("Timings:")
            for key, time_val in timings.items():
                print(f"  {key}: {time_val:.2f} ms")

            # --- Generate and Save Heatmap (After processing single image) ---
            if not current_frame_foot_xy:
                print("\nNo valid foot positions collected for single image, skipping heatmap generation.")
            elif H_single_image is None:
                 print("\nHomography not calculated for single image, skipping heatmap generation.")
            else:
                print(f"\nTotal foot positions collected for single image heatmap: {len(current_frame_foot_xy)}")
                roi_foot_xy_single = filter_footprints_for_heatmap(current_frame_foot_xy, COURT_W_S, SAFE_ZONE)
                print(f"Foot positions in heatmap ROI (single image): {len(roi_foot_xy_single)}")
                if not roi_foot_xy_single:
                    print("No foot positions within ROI for single image, skipping heatmap generation.")
                else:
                    heat_single, _, _ = generate_heatmap_data(roi_foot_xy_single, COURT_W_S, SAFE_ZONE, HEATMAP_RES)
                    if heat_single is not None:
                        plot_save_heatmap(heat_single, COURT_W_S, SAFE_ZONE, output_filename="heatmap_from_net_single_image.png")
                    else:
                        print("Heatmap data generation failed for single image.")

    cv2.destroyAllWindows()

# --- Heatmap Generation Functions (Y-axis relative to Net) ---

def calculate_homography(court_points):
    """Calculates homography from image court points to world coordinates.
    (Same as before - uses baseline as reference for world coords)
    """
    src_pts = np.float32([court_points[2], court_points[3], court_points[1], court_points[0]]) # BL, BR, TR, TL
    if np.isnan(src_pts).any() or src_pts.shape != (4, 2):
        print("Warning: Invalid source points for homography calculation.")
        return None
    dst_pts = np.float32([
        [0, 0],                  # Point 2 (BL)
        [COURT_W_D, 0],          # Point 3 (BR)
        [COURT_W_D, COURT_LEN],  # Point 1 (TR)
        [0, COURT_LEN]           # Point 0 (TL)
    ])
    try:
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    except Exception as e:
        print(f"Error calculating homography: {e}")
        return None

def collect_foot_positions(pose_results_batch, homography, conf_threshold):
    """Extracts, filters, and transforms foot keypoints for a batch of frames.
    (Same as before - returns world coords relative to baseline origin)
    """
    batch_foot_xy = []
    if homography is None: return batch_foot_xy
    for pose_result in pose_results_batch:
        if pose_result.keypoints is None: continue
        kpts = pose_result.keypoints.data.cpu().numpy()
        for person_kpts in kpts:
            ankles_uv_conf = []
            if person_kpts[LEFT_ANKLE_IDX][2] >= conf_threshold: ankles_uv_conf.append(person_kpts[LEFT_ANKLE_IDX])
            if person_kpts[RIGHT_ANKLE_IDX][2] >= conf_threshold: ankles_uv_conf.append(person_kpts[RIGHT_ANKLE_IDX])
            if not ankles_uv_conf: continue
            ankles_uv = np.array([[ankle[:2]] for ankle in ankles_uv_conf], dtype=np.float32)
            try:
                ankles_xy = cv2.perspectiveTransform(ankles_uv, homography)
                if ankles_xy is not None: batch_foot_xy.extend(ankles_xy.reshape(-1, 2).tolist())
            except Exception: continue
    return batch_foot_xy

def filter_footprints_for_heatmap(all_foot_xy, court_w_singles, safe_zone):
    """Filters foot positions to the ROI (half court + safe zone).
    (Same as before - filters based on world coords relative to baseline origin)
    """
    singles_x_min = (COURT_W_D - court_w_singles) / 2
    singles_x_max = singles_x_min + court_w_singles
    roi_y_min_world = -safe_zone # Relative to baseline
    roi_y_max_world = HALF_LEN   # Relative to baseline (net line)
    roi_foot_xy = [(x, y) for x, y in all_foot_xy
                   if singles_x_min <= x <= singles_x_max
                   and roi_y_min_world <= y <= roi_y_max_world]
    return roi_foot_xy

def generate_heatmap_data(roi_foot_xy, court_w_singles, safe_zone, resolution):
    """Generates a 2D histogram heatmap relative to the net line."""
    if not roi_foot_xy:
        return None, None, None

    # Calculate heatmap dimensions based on physical area
    total_y_span_heatmap = HALF_LEN + safe_zone # Physical height of the heatmap area
    nx = int(np.ceil(court_w_singles / resolution))
    ny = int(np.ceil(total_y_span_heatmap / resolution))

    # 1. Adjust X coordinates relative to the left singles sideline
    singles_x_min_offset = (COURT_W_D - court_w_singles) / 2
    xs = np.array([p[0] - singles_x_min_offset for p in roi_foot_xy])

    # 2. Adjust Y coordinates relative to the NET line
    # Original p[1] is distance from baseline. We want distance from net.
    ys_rel_net = np.array([p[1] - HALF_LEN for p in roi_foot_xy])
    # Net will be 0, baseline -HALF_LEN, back of safe_zone -(HALF_LEN + safe_zone)

    # 3. Shift Y coordinates for histogram indexing
    # We want the physical range [-(HALF_LEN + safe_zone), 0] to map to hist range [0, total_y_span_heatmap]
    # So, add (HALF_LEN + safe_zone) to ys_rel_net
    ys_for_hist = ys_rel_net + total_y_span_heatmap

    try:
        H2d, xedges, yedges = np.histogram2d(
            xs, ys_for_hist, bins=[nx, ny],
            range=[[0, court_w_singles], [0, total_y_span_heatmap]] # Range for histogram indexing
        )
        heat = gaussian_filter(H2d, sigma=1.5)
        if np.max(heat) > 0: heat /= np.max(heat)
        else: heat = np.zeros_like(heat)
        return heat, xedges, yedges
    except Exception as e:
        print(f"Error generating histogram: {e}")
        return None, None, None

def plot_save_heatmap(heat, court_w_singles, safe_zone, output_filename="heatmap_from_net.png"):
    """Plots the heatmap relative to the net and saves it."""
    if heat is None: print("No heatmap data to plot."); return

    plt.figure(figsize=(6, 10))
    # Define the extent for imshow based on distance from net
    # Y-axis: 0 is net, negative values go towards baseline and safe zone
    y_min_plot = -(HALF_LEN + safe_zone)
    y_max_plot = 0 # Net line
    extent = [0, court_w_singles, y_min_plot, y_max_plot]

    plt.imshow(heat.T, extent=extent, origin='lower', cmap='hot', alpha=0.8)

    # Draw court lines relative to net (y=0)
    center_x = court_w_singles / 2
    # Net Line (y=0)
    plt.plot([0, court_w_singles], [0, 0], 'w', linestyle='-', linewidth=1.5)
    # Baseline (y=-HALF_LEN)
    plt.plot([0, court_w_singles], [-HALF_LEN, -HALF_LEN], 'w', linewidth=1.5)
    # Service Line (y=-SERVICE_LINE_FROM_NET)
    plt.plot([0, court_w_singles], [-SERVICE_LINE_FROM_NET, -SERVICE_LINE_FROM_NET], 'w', linestyle='--', linewidth=1.0)
    # Center Service Line (T)
    plt.plot([center_x, center_x], [-SERVICE_LINE_FROM_NET, 0], 'w', linestyle='--', linewidth=1.0)
    # Singles Sidelines (full height)
    plt.plot([0, 0], [y_min_plot, y_max_plot], 'w', linewidth=1.5)
    plt.plot([court_w_singles, court_w_singles], [y_min_plot, y_max_plot], 'w', linewidth=1.5)

    plt.title("Foot Placement Heatmap (Relative to Net)")
    plt.xlabel("Across Singles Court (m)")
    plt.ylabel("Distance from Net (m)")
    plt.xlim(-0.5, court_w_singles + 0.5) # Padding
    plt.ylim(y_min_plot - 0.5, y_max_plot + 0.5) # Padding
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='Normalized Footfall Density')
    plt.savefig(output_filename)
    print(f"Heatmap saved to {output_filename}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to court detection model')
    parser.add_argument('--pose_model_path', type=str, required=True, help='path to YOLO pose estimation model')
    parser.add_argument('--input_path', type=str, required=True, help='path to input image or video')
    parser.add_argument('--output_width', type=int, default=640, help='Width for processing/display')
    parser.add_argument('--output_height', type=int, default=360, help='Height for processing/display')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for pose estimation after initial court averaging')
    args = parser.parse_args()
    main(args)
