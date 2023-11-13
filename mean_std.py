import numpy as np
import torch
import tqdm
import datasets_img

if __name__ == '__main__':
    data_dir = 'rv_dataset'
    samples = 128
    batch_size = 8
    num_workers = 4
    seed = 0
    np.random.seed(seed)

    dataset = datasets_img.Echo(root=data_dir, split="train")
    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    n = 0 
    s1 = 0.
    s2 = 0. 
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  
    std = np.sqrt(s2 / n - mean ** 2)  
    print(mean, std)
    # [14.54739562 14.72433021 16.68178077] [37.18475183 37.51768489 40.12321385]