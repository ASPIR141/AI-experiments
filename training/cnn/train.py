import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./assets/logs/', max_queue=100)

def train(model, device, train_loader, criterion, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        
    writer.add_scalar('train/loss', loss.item(), global_step=epoch)
    writer.add_scalar('train/accuracy', 100. * correct / len(train_loader.dataset), global_step=epoch)
        # if batch_idx % log_interval == 0: 
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item())
            # )
            # if dry_run:
            #     break

def test(model, device, criterion, test_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # TODO
    writer.add_scalar('test/loss', test_loss, global_step=epoch)
    writer.add_scalar('test/accuracy', 100. * correct / len(test_loader.dataset), global_step=epoch)
    # XXX
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset))
    # )


# classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))