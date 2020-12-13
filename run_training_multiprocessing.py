import argparse
import torch.multiprocessing as mp

from training.cnn.training_loop import training_loop


def run(batch_size, epochs, gamma, seed, log_interval, save_model):
  training_loop(batch_size, epochs, gamma, seed, log_interval, save_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    # parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    # parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    args = parser.parse_args()
    print(args)

    run(**vars(args))

if __name__ == "__main__":
    main()