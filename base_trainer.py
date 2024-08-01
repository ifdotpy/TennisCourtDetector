import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=1000):
    model.train()
    losses = []
    max_iters = min(max_iters, len(train_loader))
    progress_bar = tqdm(train_loader, total=max_iters, desc=f"Epoch {epoch}")

    for iter_id, batch in enumerate(progress_bar):
        out = model(batch[0].float().to(device))
        gt_hm_hp = batch[1].float().to(device)
        loss = criterion(F.sigmoid(out), gt_hm_hp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iter_id % 10 == 0:
            progress_bar.set_postfix(loss=loss.item())
        losses.append(loss.item())
        if iter_id > max_iters:
            break

    return np.mean(losses)