import torch
import random
from PIL import ImageFilter


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        """
        Callable class used to add Gaussian noise to a tensor

        :param mean: The mean of the noise
        :param std: The standard deviation of the noise
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor + torch.rand(tensor.shape) * self.std + self.mean).clamp(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class GaussianBlur(object):
    def __init__(self, radius=1):
        """
        Callable class used to apply a Gaussian blur to a PIL image

        :param radius: float: How much to blur the image.
                       Blur radius is chosen uniformly from [0, radius]
                       Radius should be non-negative
        """
        assert radius >= 0
        self.radius = radius

    def __call__(self, img):
        radius = random.uniform(0.5, max(0.5, self.radius))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __repr__(self):
        return self.__class__.__name__ + f"(radius={self.radius})"
