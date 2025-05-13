import numpy as np
import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device, epoch, writer, max_iters=1000, target_key='keypoints'):
    model.train()
    losses = []
    max_iters = min(max_iters, len(train_loader))
    progress_bar = tqdm(train_loader, total=max_iters, desc=f"Epoch {epoch}")

    for iter_id, batch in enumerate(progress_bar):
        inputs = batch[0].float().to(device)
        targets = batch[1].float().to(device) if target_key == 'heatmap' else batch[2].float().to(device)

        out = model(inputs)
        loss = criterion(torch.sigmoid(out), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix(avg_loss=np.mean(losses), p95_loss=np.quantile(losses, 0.95))

        # Log training loss per batch
        global_step = epoch * max_iters + iter_id
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

        if iter_id >= max_iters - 1:
            break

    avg_loss = np.mean(losses)
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    return avg_loss

def train_keypoints(model, train_loader, optimizer, criterion, device, epoch, writer, max_iters=1000):
    return train(model, train_loader, optimizer, criterion, device, epoch, writer, max_iters, target_key='keypoints')

def train_heatmap(model, train_loader, optimizer, criterion, device, epoch, writer, max_iters=1000):
    return train(model, train_loader, optimizer, criterion, device, epoch, writer, max_iters, target_key='heatmap')