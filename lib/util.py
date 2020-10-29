import os
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def show_image(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

def save_dataset(root: str, images: torch.Tensor, labels: torch.Tensor):
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

def rearrange_dataset(targets: torch.Tensor, label: int) -> torch.Tensor:
  return (targets == label).type(torch.uint8)