from sys import argv
from api.modelapi import GANModelAPI


if __name__ == '__main__':
    """Will train and save in torch mode generator model with default params"""
    model = GANModelAPI(argv[1], argv[2])
    model.train_models()
    model.save_models()
