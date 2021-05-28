import torch
from transforms import get_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DatasetGenerator:
    def __init__(self, dataset, train_batch, test_batch, path, trans_arg, train_transform=None, noise_file=None):
        """
        Class used to generate dataset

        :param dataset: Name of the dataset that is used
        :param train_batch: Size of a training and validation set
        :param test_batch: Size of test batch
        :param path: Path to which dataset is saved/loaded
        :param train_transform: Name of transformation used on training set (if not set, use transform based on dataset)
        :param noise_file: Name of the file containing noise (only used in the poisoned CIFAR-10 dataset )
        """
        self.dataset = dataset
        self.path = path
        self.train_batch = train_batch
        self.test_batch = test_batch

        if train_transform is None:
            self.train_transform = transforms.Compose(get_transform(dataset, trans_arg))
        else:
            self.train_transform = transforms.Compose(get_transform(train_transform, trans_arg))

        self.test_transform = transforms.ToTensor()
        self.noise_file = noise_file
        self.train_set, self.validation_set, self.test_set = self._init_datasets()

    def _get_train_test_set(self):
        if self.dataset == 'cifar10':
            train_set = datasets.CIFAR10(self.path, train=True, download=True, transform=self.train_transform)
            test_set = datasets.CIFAR10(self.path, train=False, download=True, transform=self.test_transform)
        elif self.dataset == 'poison_cifar10':
            assert self.noise_file is not None
            clean_set = datasets.CIFAR10(self.path, train=True, download=True, transform=transforms.ToTensor())
            train_set = PerturbedDataset(clean_set, self.noise_file, self.train_transform)
            test_set = datasets.CIFAR10(self.path, train=False, download=True, transform=self.test_transform)
        elif self.dataset == 'mnist:':
            train_set = datasets.MNIST(self.path, train=True, download=True, transform=self.train_transform)
            test_set = datasets.MNIST(self.path, train=False, download=True, transform=self.test_transform)
        else:
            raise NotImplementedError

        return train_set, test_set

    def _init_datasets(self, validation_ratio=0.5):
        """
        Initializes train and test set. The test set is split into a validation and a test set

        :param validation_ratio: Ration of validation-, test set split
        :return: Train-, validation-, and test set
        """
        train_set, test_set = self._get_train_test_set()
        test_set, validation_set = torch.utils.data.random_split(test_set,
                                                                 [int((1 - validation_ratio) * len(test_set)),
                                                                  int(validation_ratio * len(test_set))])

        return train_set, validation_set, test_set

    def get_datasets(self):
        return self.train_set, self.validation_set, self.test_set

    def get_data_loaders(self):
        train_set, validation_set, test_set = self.get_datasets()

        train_loader = DataLoader(train_set, batch_size=self.train_batch, shuffle=False)
        validation_loader = DataLoader(validation_set, batch_size=self.test_batch, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, shuffle=False)

        return train_loader, validation_loader, test_loader


class PerturbedDataset(Dataset):
    def __init__(self, dataset, noise_file, transform=None):
        """
        Class representing a perturbed dataset, dataset is perturbed using a specified .pt file containing the noise.

        :param dataset: Dataset which is perturbed
        :param noise_file: Path to file containing the noise
        :param transform: Transform used (optional)
        """

        self.noise = torch.load(noise_file)
        self.transform = transform

        if dataset.data.shape[0] == len(self.noise):
            self.perturbed_data = self._addNoise(dataset, 'sample_wise')
        else:
            self.perturbed_data = self._addNoise(dataset, 'class_wise')

    def __len__(self):
        """
        Gets the number of samples in the dataset

        :return: The number of samples in the dataset
        """
        return len(self.perturbed_data)

    def __getitem__(self, idx):
        """
        Gets an augmented sample from the dataset

        :param idx: The index of the sample to retreive
        :return: Sample at specified index from the dataset
        """
        img, target = self.perturbed_data[idx]
        img = transforms.ToPILImage()(img.clamp(0, 1))
        img = self.transform(img)
        return img, target

    def _addNoise(self, dataset, noise_type):
        """
        Adds noise to the specified dataset

        :param dataset: The dataset to which the noise is added
        :param dataset: The type of noise which is added to the dataset (sample_wise or class_wise)
        """
        perturbed_images = []

        for i in range(len(dataset)):
            img, target = dataset[i]
            if noise_type == 'sample_wise':
                img += self.noise[i]
            elif noise_type == 'class_wise':
                img += self.noise[target]
            else:
                raise NotImplemented
            img = img.clamp(0, 1)

            perturbed_images.append((img, target))

        return perturbed_images

    def get_noise(self):
        return self.noise
