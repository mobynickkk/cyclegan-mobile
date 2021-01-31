from sys import argv
import argparse
from .modelapi import GANModelAPI
import pickle


def create_parser():
    prs = argparse.ArgumentParser()
    prs.add_argument('-ds1', '--train_dataset_a', nargs=1)
    prs.add_argument('-ds2', '--train_dataset_b', nargs=1)
    prs.add_argument('-glr', '--gen_lr', default=2e-4)
    prs.add_argument('-dlr', '--discr_lr', default=2e-4)
    prs.add_argument('-ep', '--epochs', default=50)
    prs.add_argument('-th', '--threshold', default=2)
    return prs


if __name__ == '__main__':
    """Will train and save in torch mode generator model with default params"""
    parser = create_parser()
    namespace = parser.parse_args(argv[1:])

    model = GANModelAPI(namespace.train_dataset_a, namespace.train_dataset_b,
                        gen_lr=namespace.gen_lr * 0.9 ** 5,
                        discr_lr=namespace.discr_lr * 0.9 ** 5)
    losses = model.train_models(max_epochs=namespace.epochs, threshold=namespace.th)
    model.save_models()
    with open('losses.pkl', 'wb') as file:
        pickle.dump(losses, file)
