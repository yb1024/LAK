import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data

source_train = torch.utils.data.DataLoader(
        datasets.SVHN('./dataset/SVHN/train', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                           transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                       ])),
        batch_size=512, shuffle=True,num_workers=8)
source_test = torch.utils.data.DataLoader(
        datasets.SVHN('./dataset/SVHN/test', download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                           transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                       ])),
        batch_size=512, shuffle=False,num_workers=8)