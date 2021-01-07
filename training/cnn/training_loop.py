import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .train import train, test
from lib.networks.classifier import Net

from tqdm import tqdm


def training_loop(batch_size, epochs, gamma, seed, log_interval, save_model):
    torch.manual_seed(seed)

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        torch.cuda.set_device(0)
        print('CUDA support is enabled')

    ngpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if (use_cuda and ngpu > 0) else 'cpu')

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST("./assets/data/mnist", train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST("./assets/data/mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    loss = nn.NLLLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    if use_cuda:
        model = nn.DataParallel(model, list(range(ngpu)))
        loss.cuda()

    optimizer = optim.Adam(model.parameters())

    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma) #  FIXME don't use sheduler with adam optimizer
    for epoch in tqdm(range(1, epochs + 1)):
        train(model, device, train_loader, loss, optimizer, epoch, log_interval)
        test(model, device, loss, test_loader, epoch)

    if save_model:
        torch.save(model.state_dict(), "./classifier.pt")
        print('Model saved')
