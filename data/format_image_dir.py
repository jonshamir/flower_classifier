import numpy as np
import os
from scipy.io import loadmat

in_dir = 'data/flowers_all'
train_dir = 'data/flowers_train'
val_dir = 'data/flowers_val'
test_dir = 'data/flowers_test'

labels = loadmat('data/imagelabels.mat')['labels'][0]
split = loadmat('data/setid.mat')

train_ids = split['tstid'][0] # 75%
val_ids = split['valid'][0] # 12.5%
test_ids = split['trnid'][0] # 12.5%

for root, dirs, files in os.walk(in_dir):
    for name in files:
        path = os.path.join(root, name)
        if name.endswith('jpg'):
            id = int(name[6:11])
            label = str(labels[id-1]-1) # ids start at 1
            base_dir = root
            if id in train_ids: base_dir = train_dir
            elif id in val_ids: base_dir = val_dir
            elif id in test_ids: base_dir = test_dir
            new_dir = os.path.join(base_dir, label)

            if not os.path.exists(new_dir): os.makedirs(new_dir)
            new_path = os.path.join(new_dir, name)
            print(new_path)
            os.rename(path, new_path) # move file
