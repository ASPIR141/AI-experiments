import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from lib.util import rearrange_dataset

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# dataset = torch.load('./dataset.pt')
# print(dataset.targets)
# imshow(torchvision.utils.make_grid(dataset.data))


dataset = datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
)


import os
from functools import partial

def save_dataset(root, images, labels):
    concat_path = partial(os.path.join, root)
    labels_list = list({str(label.item()) for label in labels})
    for folder in map(concat_path, labels_list):
        os.makedirs(folder, exist_ok=True)

    file_names = {}
    for idx, img in enumerate(images):
        label = str(labels[idx].item())
        value = file_names.get(label)
        file_names[label] = value + 1 if value is not None else 0
        save_image(img, f'images/{label}/{idx + file_names[label]}.png') # join path

dataset.targets = rearrange_dataset(dataset.targets, 1)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
for data, target in dataloader:
    print(target)
    # save_dataset('./images', data, target)
