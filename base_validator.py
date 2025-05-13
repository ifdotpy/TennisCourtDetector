from collections import OrderedDict

import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

# from TennisCourtDetector.aug_test import keypoints
from inference import find_centroid
from utils import is_point_in_image


def val(model, val_loader, criterion, device, epoch, writer, max_dist=7):
    model.eval()
    losses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")

    dsts = []
    for iter_id, batch in enumerate(progress_bar):
        with torch.no_grad():
            batch_size = batch[0].shape[0]

            inputs = batch[0].float().to(device)
            gt_hm = batch[1].float().to(device)
            # keypoints = batch[2].float().to(device)
            kps = batch[2]

            out = model(inputs)
            loss = criterion(torch.sigmoid(out), gt_hm)

            pred = torch.sigmoid(out).detach().cpu().numpy()

            for bs in range(batch_size):
                for kps_num in range(14):
                    heatmap = pred[bs][kps_num]

                    _, (x_pred, y_pred) = find_centroid(heatmap, 0.1)

                    x_gt = kps[bs][kps_num][0].item()
                    y_gt = kps[bs][kps_num][1].item()

                    if is_point_in_image(x_pred, y_pred) and is_point_in_image(x_gt, y_gt):
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < max_dist:
                            dsts.append(dst)
                            tp += 1
                        else:
                            fp += 1
                    elif is_point_in_image(x_pred, y_pred):
                        fp += 1
                    elif is_point_in_image(x_gt, y_gt):
                        fn += 1
                    else:
                        tn += 1

            precision = tp / (tp + fp + 1e-15)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-15)

            progress_bar.set_postfix(OrderedDict([
                ('dst', -1 if len(dsts) == 0 else np.quantile(dsts, 0.95)),
                ('loss', round(loss.item(), 7)),
                ('tp', tp),
                ('tn', tn),
                ('fp', fp),
                ('fn', fn),
                ('pr', round(precision, 5)),
                ('acc', round(accuracy, 5))
            ]))
            losses.append(loss.item())

    avg_loss = np.mean(losses)
    precision = tp / (tp + fp + 1e-15)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-15)

    # Log validation metrics
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Precision', precision, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    writer.add_scalar('Val/TP', tp, epoch)
    writer.add_scalar('Val/FP', fp, epoch)
    writer.add_scalar('Val/FN', fn, epoch)
    writer.add_scalar('Val/TN', tn, epoch)

    return avg_loss, tp, fp, fn, tn, precision, accuracy
