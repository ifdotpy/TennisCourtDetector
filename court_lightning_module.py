import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# Unused torchmetrics imports removed
import numpy as np
from scipy.spatial import distance # For distance.euclidean

# Import actual functions
from advanced_centroid_methods import find_centroid_adaptive_threshold as find_centroid
from utils import is_point_in_image

# Assuming CourtFinderNetHeatmap is defined in courtnetv2.py as per your main.py
from courtnetv2 import CourtFinderNetHeatmap


class CourtLitModule(pl.LightningModule):
    def __init__(self, lr=1e-2, val_max_dist=4, num_keypoints=14, output_width=640, output_height=360):
        super().__init__()
        self.save_hyperparameters("lr", "val_max_dist", "num_keypoints", "output_width", "output_height")
        self.model = CourtFinderNetHeatmap()
        self.criterion = nn.MSELoss()
        

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets_hm, *_, = batch
        inputs = inputs.float()
        targets_hm = targets_hm.float()

        out = self(inputs)
        # Apply sigmoid to output if it's logits and targets are probabilities
        loss = self.criterion(out, targets_hm)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, gt_hm, kps, *_ = batch
        inputs = inputs.float()
        gt_hm = gt_hm.float()

        out_activated = self(inputs)
        loss = self.criterion(out_activated, gt_hm)
        self.log('val_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        pred_probs = out_activated.detach().cpu().numpy()

        batch_tp = 0
        batch_fp = 0
        batch_fn = 0
        batch_tn = 0
        batch_dsts = []

        for bs_idx in range(inputs.shape[0]):
            for kp_idx in range(self.hparams.num_keypoints):
                heatmap_pred = pred_probs[bs_idx][kp_idx]
                _, (x_pred, y_pred) = find_centroid(heatmap_pred)

                x_gt = kps[bs_idx][kp_idx][0].item()
                y_gt = kps[bs_idx][kp_idx][1].item()

                pred_in_image = is_point_in_image(x_pred, y_pred, self.hparams.output_width, self.hparams.output_height)
                gt_in_image = is_point_in_image(x_gt, y_gt, self.hparams.output_width, self.hparams.output_height)

                if pred_in_image and gt_in_image:
                    dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                    if dst < self.hparams.val_max_dist:
                        batch_dsts.append(dst)
                        batch_tp += 1
                    else:
                        batch_fp += 1
                elif pred_in_image:
                    batch_fp += 1
                elif gt_in_image:
                    batch_fn += 1
                else:
                    batch_tn += 1
        
        output = {"val_loss_batch_agg": loss.item(), "tp": batch_tp, "fp": batch_fp, "fn": batch_fn, "tn": batch_tn, "dsts": batch_dsts}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
            
        avg_loss_manual = np.mean([x['val_loss_batch_agg'] for x in outputs])
        self.log('val_loss_epoch', avg_loss_manual, prog_bar=True)
        
        total_tp = sum(x['tp'] for x in outputs)
        total_fp = sum(x['fp'] for x in outputs)
        total_fn = sum(x['fn'] for x in outputs)
        total_tn = sum(x['tn'] for x in outputs)
        
        all_dsts = [d for x in outputs for d in x['dsts']]

        precision = total_tp / (total_tp + total_fp + 1e-15)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-15)
        recall = total_tp / (total_tp + total_fn + 1e-15)

        self.log('val_precision', precision, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_recall', recall, prog_bar=False)
        self.log('val_tp', float(total_tp))
        self.log('val_fp', float(total_fp))
        self.log('val_fn', float(total_fn))
        self.log('val_tn', float(total_tn))
        if all_dsts:
            self.log('val_dst_p95', np.quantile(all_dsts, 0.95))
        else:
            self.log('val_dst_p95', 0.0)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss_epoch',
            'interval': 'epoch',
            'frequency': 2
        }
        return [optimizer], [scheduler] 