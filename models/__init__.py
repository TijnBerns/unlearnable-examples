import argument
from .NetMnist import *
from .ResNet import *
from .VGG import *
from .MLP import *

norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
args = argument.parser()


def get_model(model):
    if model == 'mnist':
        return NetMnist()
    elif model == 'cifar10' or model == 'poison_cifar10':
        return ResNet18(args.k_size, norm, num_classes=10)
    elif model == 'mlp':
        return NetCifar(norm)
    elif model == 'vgg':
        return VGG16(norm)
    else:
        raise NotImplementedError
