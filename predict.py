import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from lib.networks.classifier import Net
from lib.modules.layers import confusion_layer
from lib.modules.activation import hardmax
from lib.util import save_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    args = parser.parse_args()
    print(args)

    # torch.manual_seed(args.seed)

    use_cuda = True if torch.cuda.is_available() else False

    if use_cuda:
        torch.cuda.set_device(0)
        print('CUDA support is enabled')

    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = datasets.MNIST("./data/mnist", train=True, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    results = []
    correct = 0

    with torch.no_grad():
        # if use_cuda:
        model = Net(ngpu).to(device)
        model = nn.DataParallel(model, list(range(ngpu)))

        model.load_state_dict(torch.load('./models/classifier.pt'))
        model.eval()

        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        file_names = {}
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, x = model(data)

            output = torch.exp(output)
            # x = hardmax(x)
            probabilities, labels = confusion_layer(output, classes, len(classes))
            print('Result ', target)
            print('Labels', labels)

            save_dataset('./images', data, labels, file_names)

            # pred = output.argmax(dim=1, keepdim=True)
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # print(correct)
            # imshow(torchvision.utils.make_grid(data.cpu()))

if __name__ == '__main__':
    main()