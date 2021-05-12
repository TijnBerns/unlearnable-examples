import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--todo', default='train_nat',
                        choices=['train_nat', 'train_fgsm', 'train_pgd', 'sample_wise', 'class_wise', 'show_noise'],
                        help='The task the program executes')
    parser.add_argument('--dataset', default='poison_cifar10',
                        choices=['mnist', 'cifar10', 'poison_cifar10'],
                        help='The dataset that is used')
    parser.add_argument('--noise', default='sample_wise_8',
                        help='The name of the file containing noise (must be a \'.pt\' file)')
    parser.add_argument('--model', default=None,
                        help='The name of a model that is loaded (must be a \'.pt\' file) (only used when testing)')
    parser.add_argument('--transform', default=None,
                        help='The transform that is used for the training set')
    parser.add_argument('--train_batch', default=128, type=int,
                        help='The batch size of the training and validation set')
    parser.add_argument('--test_batch', default=128, type=int,
                        help='The batch size of the testing set')
    parser.add_argument('--epoch', default=200, type=int,
                        help='The number of epochs the model is trained')
    parser.add_argument('--epsilon', default=8. / 255, type=float,
                        help="Epsilon used in perturbations")
    parser.add_argument('--iterations', default=20, type=int,
                        help="The number of iterations used in PGD")
    parser.add_argument('--restarts', default=1, type=int,
                        help="The number of restarts used in PGD")
    parser.add_argument('--stop_error', default=0.01, type=float,
                        help="Stopping error for generating unlearnable noise")
    parser.add_argument('--train_step', default=20, type=int,
                        help="The number of training steps for generating unlearnable noise")
    parser.add_argument('--max_iter', default=500, type=int,
                        help="maximum number of iterations for generating unlearnable noise")
    parser.add_argument('--model_path', default="models/", type=str,
                        help="Path to which the checkpoints are saved and loaded")
    parser.add_argument('--result_path', default="results/", type=str,
                        help="Path to which the training results are logged")
    parser.add_argument('--delta_path', default="/ceph/csedu-scratch/other/tberns/delta/", type=str,
                        help="Path to which the noise files are saved and loaded")
    parser.add_argument('--data_path', default="data/", type=str,
                        help="Path to which the dataset files are saved and loaded")
    parser.add_argument('--save', '-s', default='-', type=str,
                        help="Name used to save trained models, generated noise, and logging files")
    return parser.parse_args()


def print_args(args):
    print(10 * "=" + " ARGUMENTS " + 10 * "=")
    for (k, v) in vars(args).items():
        print('{:<20} : {}'.format(k, v))
    print(31 * "=" + "\n")