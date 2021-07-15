import platform
import visual
import argument
import dataset
import attacks
import logging
import torch
import random
import numpy as np
from models import get_model
from trainer import Trainer
from unlearnable import NoiseGenerator


def test_trained_model(model, save, trainer, pgd, fgsm, test_loader):
    """
    Test the performance of a model on a given test set. Performance tested on clean data, FGSM perturbed data, and
    PGD perturbed data. Results are printed to console and logged to logging file

    :param model: The model which is used in the testing
    :param trainer: Train class
    :param pgd: PGD attack class
    :param fgsm: FGSM attack class
    :param test_loader: Dataloader on which performance is tested
    """
    model.load_state_dict(torch.load(save))
    test_loss, test_error = trainer.test(test_loader)
    test_loss_fgsm, test_error_fgsm = trainer.test(test_loader, attack=fgsm)
    test_loss_pgd, test_error_pgd = trainer.test(test_loader, attack=pgd)
    results = (f'Test loss,{test_loss},Test error,{test_error}\n' +
               f'Test loss FGSM,{test_loss_fgsm},Test error FGSM,{test_error_fgsm}\n' +
               f'Test loss PGD,{test_loss_pgd},Test error PGD,{test_error_pgd}\n')
    logging.info(results)
    print(results)


def main(args, device):
    # Initialize datasets and loaders
    datasetGenerator = dataset.DatasetGenerator(args.dataset, args.train_batch, args.test_batch, args.data_path,
                                                args.trans_arg, args.transform, args.delta_path + args.noise + '.pt')

    train_loader, validation_loader, test_loader = datasetGenerator.get_data_loaders()

    # Initialize model
    model = args.dataset if args.model is None else args.model
    net = get_model(model).to(device)
    if args.load_model is not None:
        net.load_state_dict(torch.load(args.model_path + args.load_model + '.pt'))

    # Initialize attacks and trainer
    trainer = Trainer(device, net)
    pgd_max = attacks.PGDMax(device, args.epsilon, args.epsilon / 10, args.iterations, args.restarts)
    fgsm = attacks.FGSM(device, args.epsilon)

    if args.todo == "train_nat":
        # Train the network in a natural manner
        trainer.train(train_loader, validation_loader, args.epoch, args.model_path + args.save)
        test_trained_model(net, args.model_path + args.save + '.pt', trainer, pgd_max, fgsm, test_loader)
        return

    elif args.todo == "train_pgd":
        # Train the network using adversarial training
        trainer.train(train_loader, validation_loader, args.epoch, args.model_path + args.save, attack=pgd_max)
        test_trained_model(net, args.model_path + args.save + '.pt', trainer, pgd_max, fgsm, test_loader)
        return

    elif args.todo == "train_fgsm":
        # Train the network using adversarial training
        trainer.train(train_loader, validation_loader, args.epoch, args.model_path + args.save, attack=fgsm)
        test_trained_model(net, args.model_path + args.save + '.pt', trainer, pgd_max, fgsm, test_loader)
        return

    elif args.todo == "test":
        # Test performance of trained model model
        test_trained_model(net, args.model_path + args.load_model + '.pt', trainer, pgd_max, fgsm, test_loader)
        return

    elif args.todo == "show_noise":
        index = 32  # Index of image showed in the visualization of data augmentation techniques
        images, _ = next(iter(train_loader))
        noise = torch.load(args.delta_path + args.noise + '.pt')

        # Multiply noise to ensure it is visible when plotted
        for i in range(len(noise)):
            noise[i] = noise[i].mul(100)

        visual.show_images(noise, 10, 10, save=args.result_path + 'noise')
        visual.show_images(images, 3, 3, save=args.result_path + 'images')

        visual.show_image(images[index], save=args.result_path + 'img_' + str(index))
        visual.show_image(noise[index], save=args.result_path + 'noise_' + str(index))
        return

    perturbation = NoiseGenerator(device, net, args.epsilon, args.iterations, args.max_iter, args.train_step,
                                  args.stop_error)

    if args.todo == "sample_wise":
        # Generate sample wise error minimizing noise
        perturbation.sample_wise(train_loader, save=args.delta_path + args.save)
        return

    elif args.todo == "class_wise":
        # Generate class wise error minimizing noise
        perturbation.class_wise(train_loader, num_classes=10, save=args.delta_path + args.save)
        return

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = argument.parser()
    argument.print_args(args)

    # Set seed and device
    seed = 0

    if torch.cuda.is_available() and args.gpu != 0:
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Print runtime information
    print('{:<20} : {}\n'.format('Python version', str(platform.python_version())) +
          '{:<20} : {}\n'.format('Device being used', str(device)) +
          '{:<20} : {}\n'.format('Pytorch version', str(torch.__version__)))

    # Set logging configuration
    logging.basicConfig(filename=args.result_path + args.save + ".csv",
                        level=logging.INFO,
                        format='%(message)s',
                        filemode='w')

    logging.info(device)

    # Run program with parsed arguments
    main(args, device)
