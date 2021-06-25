from .CustomTransforms import *
from torchvision import transforms
import imgaug.augmenters as iaa
import numpy as np


def get_transform(transform, trans_arg):
    transform_dict = {
        'tensor':
            [transforms.ToTensor()],
        'mnist':
            [transforms.ToTensor()],
        'cifar10':
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'poison_cifar10':
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'gray_cifar10':
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             np.array,
             iaa.Grayscale(alpha=1).augment_image,
             transforms.ToTensor()],
        'noise':
            [transforms.ToTensor(),
             GaussianNoise(mean=0, std=trans_arg / 100),
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()
             ],
        'gray_noise':
            [transforms.ToTensor(),
             GaussianNoise(mean=0, std=trans_arg / 100),
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             np.array,
             iaa.Grayscale(alpha=1).augment_image,
             transforms.ToTensor()
             ],
        'dropout':
            [np.array,
             iaa.CoarseDropout(per_channel=True, p=0.1, size_percent=trans_arg / 10).augment_image,
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'gray_dropout':
            [np.array,
             iaa.CoarseDropout(per_channel=True, p=0.1, size_percent=trans_arg / 10).augment_image,
             iaa.Grayscale(alpha=1).augment_image,
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'jpeg':
            [np.array,
             iaa.JpegCompression(compression=trans_arg).augment_image,
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'gray_jpeg':
            [np.array,
             iaa.JpegCompression(compression=trans_arg).augment_image,
             iaa.Grayscale(alpha=1).augment_image,
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
        'gray':
            [np.array,
             iaa.Grayscale(alpha=1).augment_image,
             transforms.ToTensor()],
        'median_blur':
            [np.array,
             iaa.MedianBlur(k=3).augment_image,
             transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()],
    }

    return transform_dict[transform]
