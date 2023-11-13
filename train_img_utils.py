import numpy as np
import torch
import tqdm

def run_epoch(model, dataloader, train, optim, device):
    total = 0.
    n = 0
    model.train(train)

    inter = 0
    union = 0
    inter_list = []
    union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (img,label) in dataloader:
                # Run prediction for diastolic frames and compute loss
                img = img.to(device)
                label = label.to(device)
                y_img = model(img)["out"]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(y_img[:,0, :, :], label, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                inter += np.logical_and(y_img[:, 0, :, :].detach().cpu().numpy() > 0., label[:, :, :].detach().cpu().numpy() > 0.).sum()
                union += np.logical_or(y_img[:, 0, :, :].detach().cpu().numpy() > 0., label[:, :, :].detach().cpu().numpy() > 0.).sum()
                inter_list.extend(np.logical_and(y_img[:,0, :, :].detach().cpu().numpy() > 0., label[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                union_list.extend(np.logical_or(y_img[:,0, :, :].detach().cpu().numpy() > 0., label[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += img.size(0)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) )".format(total / n / 112 / 112,  2 * inter / (union + inter)))
                pbar.update()

    inter_list = np.array(inter_list)
    union_list = np.array(union_list)

    return (total / n / 112 / 112,
            inter_list,
            union_list,

            )
