import torch
import trainer
import attacks
import torch.optim as optim
import logging


def add_sample_wise_noise(images, noise, index):
    """
    Adds sample wise noise to images

    :param images: Images on which to add noise
    :param noise: Tensor containing all noise
    :param index: Index to start selecting noise from in noise tensor
    :return: Sample-wise noise perturbed image, and index of noise tensor
    """
    for i, img in enumerate(images):
        images[i] = img + noise[index]
        index += 1

    return images, index


def add_class_wise_noise(images, labels, noise):
    """
    Adds class wise noise to images

    :param images: Images on which to add noise
    :param labels: Labels corresponding to the images
    :param noise: Tensor containing all class-wise oise
    :return: Class-wise noise perturbed image
    """
    for i, (img, label) in enumerate(zip(images, labels)):
        images[i] = img + noise[label.item()]
    return images


class NoiseGenerator:
    def __init__(self, model, epsilon, iterations, max_iterations, train_steps, stop_error):
        """
        Class used to generate sample wise noise

        :param model: The model which is used to generate perturbations
        :param epsilon: Noise bound
        :param iterations: Iterations of PGD attack (T)
        :param max_iterations: Maximum number of iterations of algorithm
        :param train_steps: Number of training steps (M)
        :param stop_error: Error at which algorithm stops executing (lambda)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.model = model
        self.attack = attacks.PGDMin(self.device, epsilon, epsilon / 10, iterations)
        self.trainer = trainer.Trainer(model)
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
        self.max_iter = max_iterations
        self.train_step = train_steps
        self.stop_error = stop_error

    def sample_wise(self, loader, save):
        """
        Generates error minimizing sample-wise noise

        :param loader: The dataloader to generate the perturbations for
        :param save: Path to which a file containing perturbations is stored
        :return: A list containing the perturbations for data in loader
        """
        # Setting up parameters for sample wise noise generation
        data_iter = iter(loader)
        condition = True
        train_idx, iteration = 0, 0

        # Generate random initial noise within l-ball
        noise = []

        for images, labels in loader:
            batch_noise = self.attack.random_noise(images)
            for random_noise in batch_noise:
                noise.append(random_noise)

        while condition and iteration < self.max_iter:
            # Train model for M steps
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

            for _ in range(self.train_step):
                try:
                    (images, labels) = next(data_iter)
                except StopIteration:
                    train_idx = 0
                    data_iter = iter(loader)
                    (images, labels) = next(data_iter)

                images, train_idx = add_sample_wise_noise(images, noise, train_idx)
                images, labels = images.to(self.device), labels.to(self.device)
                self.trainer.train_batch(images, labels, self.optimizer)

            # Optimize perturbations
            idx = 0
            for param in self.model.parameters():
                param.requires_grad = False

            for images, labels in loader:
                batch_start_idx, batch_noise = idx, []
                for i, _ in enumerate(images):
                    batch_noise.append(noise[idx])
                    idx += 1
                batch_noise = torch.stack(batch_noise).to(self.device)

                self.model.eval()
                images, labels = images.to(self.device), labels.to(self.device)
                eta = self.attack.compute_noise(self.model, images, labels, batch_noise)
                for i, delta in enumerate(eta):
                    noise[batch_start_idx + i] = delta.clone().detach().cpu()

            # Test stop condition
            error = self._evaluate_sample_wise(loader, noise)
            acc = 1 - error
            logging.info('Accuracy %.2f' % (acc * 100))
            condition = error > self.stop_error
        torch.save(noise, save + ".pt")
        return noise

    def _evaluate_sample_wise(self, loader, noise):
        """
        Evaluates the performance of self.model on sample-wise noise perturbed data

        :param loader: Dataloader used to test the performance on
        :param noise: Noise added to dataloader
        :return: Loss and error of self.model on the dataloader with added noise
        """
        total_err, idx = 0, 0
        for images, labels in loader:
            images, idx = add_sample_wise_noise(images, noise, idx)
            images, labels = images.to(self.device), labels.to(self.device)
            prediction, loss = self.trainer.train_batch(images, labels)
            total_err += (prediction.max(dim=1)[1] != labels).sum().item()

        return total_err / len(loader.dataset)

    def class_wise(self, loader, num_classes, save):
        """
        Generates error minimizing class-wise noise

        :param loader: The dataloader to generate the perturbations for
        :param num_classes: The number of classes to generate perturbations for
        :param save: Path to which a file containing perturbations is stored
        :return: A list containing the perturbations for data in loader
        """
        # Setting up parameters for class wise noise generation
        data_iter = iter(loader)
        condition = True
        iteration = 0

        # Generate random initial noise within l-ball
        images, _ = next(iter(loader))
        noise = []
        for i in range(num_classes):
            noise.append(self.attack.random_noise(images[0]))

        while condition and iteration < self.max_iter:
            # Train model for M steps
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

            for _ in range(self.train_step):
                try:
                    (images, labels) = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    (images, labels) = next(data_iter)

                add_class_wise_noise(images, labels, noise)
                images, labels = images.to(self.device), labels.to(self.device)
                self.trainer.train_batch(images, labels, self.optimizer)

            # Optimize perturbations
            for param in self.model.parameters():
                param.requires_grad = False

            for images, labels in loader:
                batch_noise = []
                for i, (img, label) in enumerate(zip(images, labels)):
                    batch_noise.append(noise[label])

                batch_noise = torch.stack(batch_noise).to(self.device)

                self.model.eval()
                images, labels = images.to(self.device), labels.to(self.device)
                eta = self.attack.compute_noise(self.model, images, labels, batch_noise)
                for (delta, label) in zip(eta, labels):
                    noise[label.item()] = (noise[label.item()] + delta.clone().detach().cpu()).clamp(-self.epsilon,
                                                                                                     self.epsilon)

            # Test stop condition
            error = self._evaluate_class_wise(loader, noise)
            acc = 1 - error
            logging.info('Accuracy %.2f' % (acc * 100))
            condition = error > self.stop_error
            torch.save(noise, save + ".pt")

        return noise

    def _evaluate_class_wise(self, loader, noise):
        """
        Evaluates the performance of self.model on class-wise noise perturbed data

        :param loader: Dataloader used to test the performance on
        :param noise: Noise added to dataloader
        :return: Loss and error of self.model on the dataloader with added noise
        """
        total_err = 0
        for images, labels in loader:
            images = add_class_wise_noise(images, labels, noise)
            images, labels = images.to(self.device), labels.to(self.device)
            prediction, _ = self.trainer.train_batch(images, labels)
            total_err += (prediction.max(dim=1)[1] != labels).sum().item()
        return total_err / len(loader.dataset)
