import os
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
import math

class SaveBatchCallback(pl.Callback):
    def __init__(self, save_dir, num_batches_to_save=3):
        super().__init__()
        self.save_dir = save_dir
        self.num_batches_to_save = num_batches_to_save
        self.batches_saved_count = 0
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.keypoint_labels = [
            "TLc", "TRc", "BLc", "BRc",
            "TRs", "BRs", "TLs", "BLs",
            "BLsl", "BRsl", "TLsl", "TRsl",
            "Ct", "Cb"
        ]
        
        os.makedirs(self.save_dir, exist_ok=True)

    def denormalize_image(self, img_tensor):
        img_np = img_tensor.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0)) # C,H,W to H,W,C
        img_np = (img_np * self.std) + self.mean # Denormalize
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8) # Scale to 0-255
        # At this point, img_np is H,W,C and in RGB format because mean/std are RGB
        return img_np

    def create_image_grid(self, images, grid_cols=None):
        if not images:
            return None
        
        img_h, img_w = images[0].shape[:2]
        num_images = len(images)

        if grid_cols is None:
            grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)
        
        processed_images_bgr = []
        for img in images:
            # Ensure all images in the list are BGR before gridding
            if img.ndim == 2 or img.shape[2] == 1:
                processed_images_bgr.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            elif img.shape[2] == 4:
                processed_images_bgr.append(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
            else: # Assume it's RGB from denormalize_image or already BGR
                  # If it's RGB, convert to BGR. If already BGR, this won't harm.
                processed_images_bgr.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        images = processed_images_bgr
        
        grid_h = img_h * grid_rows
        grid_w = img_w * grid_cols
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, img_bgr in enumerate(images):
            row = i // grid_cols
            col = i % grid_cols
            grid_image[row*img_h : (row+1)*img_h, col*img_w : (col+1)*img_w, :] = img_bgr
            
        return grid_image # This grid is now BGR

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.batches_saved_count < self.num_batches_to_save:
            processed_images_for_grid = []
            try:
                inputs, _, keypoints_batch, *_ = batch
                imgs_tensor = inputs.cpu()
                kps_batch_cpu = keypoints_batch.cpu().numpy()

                for i in range(imgs_tensor.shape[0]):
                    img_tensor_single = imgs_tensor[i]
                    vis_img_rgb = self.denormalize_image(img_tensor_single)
                    
                    if vis_img_rgb.ndim == 2 or vis_img_rgb.shape[2] == 1:
                        vis_img_draw_bgr = cv2.cvtColor(vis_img_rgb, cv2.COLOR_GRAY2BGR)
                    else:
                        vis_img_draw_bgr = cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR)
                    
                    current_kps = kps_batch_cpu[i]
                    for kp_idx in range(current_kps.shape[0]):
                        x, y = int(current_kps[kp_idx, 0]), int(current_kps[kp_idx, 1])
                        
                        h_img, w_img, _ = vis_img_draw_bgr.shape
                        if x != -1 and y != -1 and 0 <= x < w_img and 0 <= y < h_img:
                            cv2.circle(vis_img_draw_bgr, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                            if kp_idx < len(self.keypoint_labels):
                                label = self.keypoint_labels[kp_idx]
                                cv2.putText(vis_img_draw_bgr, label, (x + 5, y + 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                            
                    processed_images_for_grid.append(vis_img_draw_bgr)
                
                if processed_images_for_grid:
                    num_cols_in_grid = min(len(processed_images_for_grid), 4)
                    grid_image = self.create_image_grid(processed_images_for_grid, grid_cols=num_cols_in_grid)
                    if grid_image is not None:
                        filename = os.path.join(self.save_dir, 
                                                f"epoch_{trainer.current_epoch}_batchidx_{batch_idx}_globalbatch_{self.batches_saved_count}.png")
                        cv2.imwrite(filename, grid_image)
                
                self.batches_saved_count += 1
                if self.batches_saved_count >= self.num_batches_to_save:
                    pass
            except Exception as e:
                print(f"Error during SaveBatchCallback on_train_batch_start: {e}") 