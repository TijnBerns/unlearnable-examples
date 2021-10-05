import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transforms import get_transform


class Attack(ABC):
    def __init__(self, device, epsilon):
        self.device = device
        self.epsilon = epsilon

    @abstractmethod
    def compute_noise(self, model, x, y, batch_noise=None):
        """

        :param model: Model used to generate noise
        :param x: Data
        :param y: Labels corresponding to data
        :param batch_noise: If specified, use batch_noise as initial starting point
        """
        pass

    def random_noise(self, x):
        """
        Generates random noise

        :param x: Data the random noise is matched to
        :return: Random noise of the same shape of input x
        """
        delta = torch.rand_like(x)
        delta.data = delta.data * 2 * self.epsilon - self.epsilon
        return delta


class FGSM(Attack, ABC):
    def __init__(self, device, epsilon):
        super(FGSM, self).__init__(device, epsilon)

    def compute_noise(self, model, x, y, batch_noise=None):
        """
        Generates error maximizing noise based on the fast gradient sign method

        :param model: Model which is used to generate noise
        :param x: Data on which the noise is based
        :param y: Labels corresponding to data
        :param batch_noise: If specified, use batch_noise as initial starting point
        :return: The generated perturbations
        """
        if batch_noise is None:
            delta = torch.zeros_like(x, requires_grad=True)
        else:
            delta = batch_noise.retain_grad()
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        return self.epsilon * delta.grad.detach().sign()


class PGDMax(Attack, ABC):
    def __init__(self, device, epsilon, step_size, iterations, restarts=1):
        super().__init__(device, epsilon)
        self.step_size = step_size
        self.iterations = iterations
        self.restarts = restarts

    def compute_noise(self, model, x, y, batch_noise=None):
        """
        Generates error maximizing noise based on the projected gradient sign method

        :param model: Model which is used to generate noise
        :param x: Data on which the noise is based
        :param y: Labels corresponding to data x
        :param batch_noise: If specified compute PGD noise from starting from this point
        :return: The generated perturbations
        """
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)

        for i in range(self.restarts):
            if batch_noise is None:
                delta = self.random_noise(x)
            else:
                delta = batch_noise
            delta.requires_grad = True

            for t in range(self.iterations):
                loss = nn.CrossEntropyLoss()(model(x + delta), y)
                loss.backward()
                delta.data = (delta + self.step_size * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x
                delta.grad.zero_()

            all_loss = nn.CrossEntropyLoss(reduction='none')(model(x + delta), y)
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_delta


class PGDMin(Attack, ABC):
    def __init__(self, device, epsilon, step_size, iterations):
        super().__init__(device, epsilon)
        self.step_size = step_size
        self.iterations = iterations

    def compute_noise(self, model, x, y, batch_noise=None):
        """
        Generates error minimizing noise based on the projected gradient sign method

        :param model: Model which is used to generate noise
        :param x: Data on which the noise is based
        :param y: Labels corresponding to data x
        :param batch_noise: Initial noise PGD is started from, if not specified start from random noise
        :return: The generated perturbations
        """
        if batch_noise is None:
            delta = self.random_noise(x)
        else:
            delta = batch_noise
        delta.requires_grad = True

        # transform_after_perturb = get_transform("gray_cifar10", None)

        for t in range(self.iterations):
            # loss = nn.CrossEntropyLoss()(model(transform_after_perturb(x + delta)), y)
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward()
            delta.data = (delta - self.step_size * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.data = (x + delta.data).clamp(0, 1) - x
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()

        return delta
