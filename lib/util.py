import os
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from torchvision.utils import save_image


def show_image(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

def save_dataset(root: str, images: List[torch.Tensor], labels: List[torch.Tensor], file_names: Dict):
    concat_path = partial(os.path.join, root)
    labels_list = list({label for label in labels}) # str(label.item())
    for folder in map(concat_path, labels_list):
        os.makedirs(folder, exist_ok=True)

    # file_names = {}
    for idx, img in enumerate(images):
        label = labels[idx] # str(labels[idx].item())
        value = file_names.get(label)
        file_names[label] = value + 1 if value is not None else 0
        save_image(img, f'images/{label}/{file_names[label]}.png') # join path

def rearrange_dataset(targets: torch.Tensor, label: int) -> torch.Tensor:
  return (targets == label).type(torch.uint8)