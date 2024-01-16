import pydicom
import cv2
import numpy as np
import os

# 指定文件夹的路径
folder_path = 'C:/workspace/docu/华西数据/LGE阴性-张琳玲'
# 使用os.listdir()列出文件夹下的所有文件名
file_names = os.listdir(folder_path)

# 打印文件名
for file_name in file_names:
    filename = os.path.join(folder_path, file_name, '4CRV')  # 名字为举例
    dataset = pydicom.dcmread(filename)
    start = int(dataset.RWaveTimeVector[2]*dataset.CineRate/1000)
    end = int(dataset.RWaveTimeVector[3]*dataset.CineRate/1000)
    centers = []
    y_coordinates = []

    for i in range(start, end):
        img = dataset.pixel_array[i]
        img = np.array(img)  
        frame=str(i).zfill(3)
        cv2.imwrite(f'img/{file_name[:8]}_{frame}.jpg', cv2.cvtColor(dataset.pixel_array[i], cv2.COLOR_RGB2BGR))




