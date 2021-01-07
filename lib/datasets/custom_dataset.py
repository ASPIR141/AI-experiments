from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # https://pytorch.org/docs/1.1.0/_modules/torchvision/datasets/mnist.html

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets # labels

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return len(self.data)