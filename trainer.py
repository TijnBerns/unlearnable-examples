import torch
import time
import torch.nn as nn
import torch.optim as optim
import logging


class Trainer:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

    def train_batch(self, x, y, opt=None, attack=None):
        """
        Forwards a single batch through the model, backpropagates if opt flag is set

        :param x: Data batch to be trained on
        :param y: Labels corresponding to data batch x
        :param opt: Optimization flag, if set the model parameters will be updated using backpropagation (optional)
        :param attack: Perturb x using specified attack (optional)
        :return: Predictions and loss on data
        """
        self.model.zero_grad()
        if attack is None:
            yp = self.model(x)
        else:
            noise = attack.compute_noise(self.model, x, y)
            yp = self.model(x + noise)

        loss = nn.CrossEntropyLoss()(yp, y)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()

        return yp, loss

    def epoch(self, loader, opt=None, attack=None):
        """
        Train self.model for one epoch

        :param loader: Data on which self.model is trained
        :param opt: Optimization flag, if set the model parameters will be updated using backpropagation
        :param attack: Optional, if specified, perturb x using specified attack
        :return: Average loss and error of self.model on dataloader
        """
        total_err, total_loss = 0., 0.

        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            yp, loss = self.train_batch(x, y, opt, attack)
            total_loss += loss.item() * x.shape[0]
            total_err += (yp.max(dim=1)[1] != y).sum().item()

        return total_loss / len(loader.dataset), total_err / len(loader.dataset)

    def train(self, train_loader, validation_loader, nr_epoch, save, attack=None):
        """
        Train self.model for nr_epoch epochs

        :param train_loader: Dataloader on which self.model is trained
        :param validation_loader: Dataloader on which the performance is validated
        :param nr_epoch: The number of epochs to train the model
        :param attack: Optional attack used to perturb data
        :param save: Name of file model is saved to
        """
        fields = ["Train loss", "Train error", "Validation loss", "Validation error", "Validation loss adv",
                  "Validation error adv"]

        logging.info(','.join(field for field in fields))

        opt = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nr_epoch, eta_min=0)

        start = time.time()
        best_error = 99.
        current_epoch = 0

        while current_epoch < nr_epoch:
            # Train the model for one epoch
            self.model.train()
            train_loss, train_err = self.epoch(train_loader, opt=opt, attack=attack)

            # Evaluate model performance on clean validation data
            validation_loss, validation_error = self.test(validation_loader)

            # Evaluate model performance on perturbed validation data
            validation_loss_adv, validation_error_adv = 0., 0.
            if attack is not None:
                validation_loss_adv, validation_error_adv = self.test(validation_loader, attack=attack)

            # Step scheduler
            scheduler.step()

            # Print results of epoch
            results = [train_loss, train_err, validation_loss, validation_error, validation_loss_adv,
                       validation_error_adv]
            logging.info(','.join(str(res) for res in results))

            current_epoch += 1

            # Save model
            if validation_error < best_error:
                best_error = validation_error
                torch.save(self.model.state_dict(), save + ".pt")

        print(f"Total training time: {time.time() - start} sec")

    def test(self, loader, attack=None):
        """
        Tests the performance of self.model on specified dataloader

        :param loader: Dataloader to train on
        :param attack: Optional argument specifying the attack used during testing
        :return: Test loss and error
        """
        self.model.eval()
        test_loss, test_err = self.epoch(loader, attack=attack)
        return test_loss, test_err
