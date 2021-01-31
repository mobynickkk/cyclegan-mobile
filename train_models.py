from sys import argv
from .modelapi import GANModelAPI
import pickle


if __name__ == '__main__':
    """Will train and save in torch mode generator model with default params"""
    model = GANModelAPI(argv[1], argv[2], gen_lr=2e-4 * 0.9 ** 5, discr_lr=2e-4 * 0.9 ** 5)
    losses = model.train_models(max_epochs=51, threshold=1.5)
    model.save_models()
    with open('losses.pkl', 'wb') as file:
        pickle.dump(losses, file)
