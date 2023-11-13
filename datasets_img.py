import os
import numpy as np
import skimage.draw
import torchvision
import json
from PIL import Image

class Echo(torchvision.datasets.VisionDataset):
    def __init__(self,
                 root='dataset',
                 split="train",
                 mean=0., std=1.,
                 pad=None,  # 填充像素
                 target_transform=None,
                 noise=None,  # 添加噪声[0，1]
                 ):
        super().__init__(root, target_transform=target_transform)
        self.split = split
        self.mean = mean
        self.std = std
        self.pad = pad
        self.target_transform = target_transform
        self.noise = noise
        self.root = root
        self.fnames = []
        with open(os.path.join(self.root,f"Segmentation/{split}.txt"),"r") as f:
            self.fnames = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        img = os.path.join(self.root, "img", self.fnames[index])+'.jpg'
        img = Image.open(img)
        img = img.convert('RGB')
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)

        # 加噪声
        if self.noise is not None:
            n = img.shape[1] * img.shape[2] 
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            i = ind % img.shape[1]
            ind //= img.shape[1]
            j = ind 
            img[:, i, j] = 0

        # 标准化
        if isinstance(self.mean, (float, int)):
            img -= self.mean
        else:
            img-= self.mean.reshape(3, 1, 1)

        if isinstance(self.std, (float, int)):
            img /= self.std
        else:
            img /= self.std.reshape(3, 1, 1)

        # 读取标注
        label =os.path.join(self.root, "label", self.fnames[index])+'.json'
        with open(label, 'r') as m:
            data = json.load(m)
        shape = data['shapes'][0]  
        points = shape['points']  
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (img.shape[1], img.shape[2]))
        mask = np.zeros((img.shape[1], img.shape[2]), np.float32)
        mask[r, c] = 1

        if self.pad is not None:
            c, h, w = img.shape
            temp = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=img.dtype)
            temp[:, self.pad:-self.pad, self.pad:-self.pad] = img  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            img = temp[:, i:(i + h), j:(j + w)]
        return img, mask

    def __len__(self):
        return len(self.fnames)

    