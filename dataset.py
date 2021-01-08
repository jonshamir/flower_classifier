import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


DATASETS = ['train', 'val', 'test']
BATCH_SIZE = 16

# pytorch pre-trained models expect input images normalized with mean and std
MEAN_VAL = [0.485, 0.456, 0.406]
STD_VAL = [0.229, 0.224, 0.225]
transforms = {
    'train': T.Compose([
        T.RandomRotation(degrees=30),
        T.RandomResizedCrop(size=224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN_VAL, STD_VAL)
    ]),
    'val': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(MEAN_VAL, STD_VAL)
    ]),
    'test': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(MEAN_VAL, STD_VAL)
    ]),
}



image_datasets = {
  d: datasets.ImageFolder(f'data/flowers_{d}', transforms[d]) for d in DATASETS
}

data_loaders = {
  d: DataLoader(image_datasets[d], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
  for d in DATASETS
}

dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}
class_labels = image_datasets['train'].classes


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([MEAN_VAL])
    std = np.array([STD_VAL])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None: plt.title(title)
    plt.axis('off')

inputs, classes = next(iter(data_loaders['train']))
out = torchvision.utils.make_grid(inputs, nrow=4)
imshow(out, title=[class_labels[x] for x in classes])
