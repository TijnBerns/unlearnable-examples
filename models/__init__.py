import argument
from .NetMnist import *
from .ResNet import *

norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
args = argument.parser()

model_dict = {'mnist': NetMnist(),
              'cifar10': ResNet18(args.k_size, norm, num_classes=10),
              'poison_cifar10': ResNet18(args.k_size, norm, num_classes=10)}


def get_model(dataset):
    return model_dict[dataset]
