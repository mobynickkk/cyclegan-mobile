from sys import argv
import argparse
from .modelapi import GANModelAPI
import pickle


def create_parser():
    prs = argparse.ArgumentParser()
    prs.add_argument('-ds1', '--train_dataset_a', nargs=1)
    prs.add_argument('-ds2', '--train_dataset_b', nargs=1)
    prs.add_argument('-sh', '--shift', default=True)
    prs.add_argument('-gopt', '--gen_optim', default='Adam')
    prs.add_argument('-dopt', '--discr_optim', default='Adam')
    prs.add_argument('-gsc', '--gen_sched', default='default')
    prs.add_argument('-dsc', '--discr_sched', default='default')
    prs.add_argument('-stp', '--step', default=25)
    prs.add_argument('-crt', '--criterion', default='bceloss')
    prs.add_argument('-glr', '--gen_lr', default=2e-4)
    prs.add_argument('-dlr', '--discr_lr', default=2e-4)
    prs.add_argument('-ep', '--epochs', default=75)
    prs.add_argument('-th', '--threshold', default=2)
    prs.add_argument('-is', '--image_size', default=256)
    return prs


if __name__ == '__main__':
    """Will train and save in torch mode generator model with default params"""
    parser = create_parser()
    namespace = parser.parse_args(argv[1:])

    model = GANModelAPI(namespace.train_dataset_a,
                        namespace.train_dataset_b,
                        shift=bool(namespace.shift),
                        gen_optimizer=namespace.gen_optim,
                        discr_optimizer=namespace.discr_optim,
                        gen_scheduler=namespace.gen_sched,
                        discr_scheduler=namespace.discr_sched,
                        step=int(namespace.step),
                        criterion=namespace.criterion,
                        gen_lr=float(namespace.gen_lr) * 0.9 ** 5,
                        discr_lr=float(namespace.discr_lr) * 0.9 ** 5,
                        image_size=int(namespace.image_size))

    losses = model.train_models(max_epochs=int(namespace.epochs), threshold=float(namespace.th))

    model.save_models()
    with open('losses.pkl', 'wb') as file:
        pickle.dump(losses, file)
