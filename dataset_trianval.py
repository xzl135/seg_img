import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

trainval_percent    = 1
train_percent       = 0.9
path      = 'rv_dataset'

if __name__ == "__main__":
    random.seed(0)

    seg = os.listdir(path+'/label')

    num     = len(seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  

    ftrainval   = open(os.path.join(path,'Segmentation','trainval.txt'), 'w')  
    ftest       = open(os.path.join(path,'Segmentation','test.txt'), 'w')  
    ftrain      = open(os.path.join(path,'Segmentation','train.txt'), 'w')  
    fval        = open(os.path.join(path,'Segmentation','val.txt'), 'w')  
    
    for i in list:  
        name = seg[i][:-5]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
 