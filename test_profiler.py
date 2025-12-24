import os
import argparse
import torch
from models import vgg19
from memory_profile import measure_everything, extract_children_from_sequential
from torchvision import datasets, transforms
from memory import Memory
from pprint import pprint

def get_data_loader(args):
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    return train_loader

parser = argparse.ArgumentParser(description='PyTorch Imagenet Example')
parser.add_argument('data', metavar='DIR', nargs='?', default='/data/imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num-microbatches', type=int, default=8, metavar='M',
                    help='number of chunks to be split in a mini-batch (default: 8)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

model = vgg19()
device = torch.device('cuda')
model.to(device)
example_input_microbatch = torch.randn(
            args.batch_size // args.num_microbatches,
            3, 224, 224).cuda()

memory = Memory(model, example_input_microbatch, device)
stats = memory.profile_memory_stats()
pprint(stats)

# modules = []
# for _, m in model._modules.items():
#     modules.extend(extract_children_from_sequential(m))
# result_fwdTime, result_bwdTime, result_xbar, result_x, result_tmpFwd, result_tmpBwd = measure_everything(modules, example_input_microbatch)
# print(result_xbar)
# print(result_x)
# print(result_tmpFwd)
# print(result_tmpBwd)