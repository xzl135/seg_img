import math
import os   
import torch
import torchvision.models.segmentation
import numpy as np
import datasets_img
import time
from train_img_utils import run_epoch
from numpy import array, float32


if __name__ == '__main__':
    # 参数设置
    weights = None
    lr_step_period = None
    lr = 1e-5
    weight_decay = 1e-5
    data_dir = 'rv_dataset'
    output = 'rv_dataset/output'
    batch_size = 20
    num_workers = 4
    num_epochs = 50
    device = torch.device('cpu')
    run_test = False
    save_video = True
    num_train_patients = None
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1,kernel_size=model.classifier[-1].kernel_size)
    model.to(device)
    # 学习率步长
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    # 求均值方差
    mean, std = array([32.366024, 32.50149 , 32.80471 ], dtype=float32),array([49.646393, 49.66005 , 49.909855], dtype=float32)
    kwargs = {"mean": mean,
              "std": std
              }
    # 加载数据集
    dataset = {"train": datasets_img.Echo(root=data_dir, split="train", **kwargs,pad=None, target_transform=None, noise=None),
               "val": datasets_img.Echo(root=data_dir, split="val", **kwargs,pad=None, target_transform=None, noise=None)}
    # 开始训练
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train"))

                loss, inter, union =run_epoch(model, dataloader,phase == "train", optim,device)

                dice = 2 * inter.sum() / (union.sum() + inter.sum())
                f.write("{},{},{},{},{},{}\n".format(epoch, phase, loss, dice, 
                                                              time.time() - start_time, batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

