import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import chain
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import ImgDataset
from utils.train import train


class GANModelAPI:
    """Класс для упрощенного создания и обучения модели"""
    def __init__(self, files_a, files_b, gen_optimizer='Adam', discr_optimizer='Adam', gen_scheduler='default',
                 discr_scheduler='default', criterion='bceloss', epochs=200, hold_discr=True):
        self.dataloader = DataLoader(ImgDataset(files_a, files_b), batch_size=1, shuffle=True)
        self.generator_a2b = Generator()
        self.generator_b2a = Generator()
        self.discriminator_a = Discriminator()
        self.discriminator_b = Discriminator()
        if gen_optimizer == 'Adam':
            self.gen_optimizer = optim.Adam(chain(
                self.generator_a2b.parameters(),
                self.generator_b2a.parameters()
            ), lr=2e-4)
        else:
            raise NotImplemented(f'Optimizer {gen_optimizer} is not supported now')
        if discr_optimizer == 'Adam':
            self.discr_optimizer = optim.Adam(chain(
                self.discriminator_a.parameters(),
                self.discriminator_b.parameters()
            ), lr=2e-4)
        else:
            raise NotImplemented(f'Optimizer {discr_optimizer} is not supported now')
        if gen_scheduler == 'default':
            self.gen_sched = optim.lr_scheduler.LambdaLR(
                self.gen_optimizer,
                lr_lambda=lambda epoch: 0.9 ** (epoch - 99) if epoch > 99 else 1
            )
        else:
            raise NotImplemented(f'Generators lr scheduler {gen_scheduler} is not supported now')
        if discr_scheduler == 'default':
            self.discr_sched = optim.lr_scheduler.LambdaLR(
                self.discr_optimizer,
                lr_lambda=lambda epoch: 0.9 ** (epoch - 99) if epoch > 99 else 1
            )
        else:
            raise NotImplemented(f'Discriminators lr scheduler {discr_scheduler} is not supported now')
        if criterion == 'bceloss':
            self.criterion = nn.BCELoss()
        else:
            raise NotImplemented(f'Criterion {criterion} is not supported now')
        self.max_epochs = epochs
        self.hold_discr = hold_discr

    def train_models(self):
        train(self.generator_a2b, self.generator_b2a, self.discriminator_a, self.discriminator_b, self.gen_optimizer,
              self.discr_optimizer, self.gen_sched, self.discr_sched, self.criterion, self.dataloader, self.max_epochs,
              self.hold_discr)

    def save_models(self, mode='torch'):
        if mode == 'torch':
            torch.save(self.generator_a2b, 'gen_a2b.model')
            torch.save(self.generator_b2a, 'gen_b2a.model')
            torch.save(self.discriminator_a, 'discr_a.model')
            torch.save(self.discriminator_b, 'discr_b.model')
        elif mode == 'onnx':
            torch.onnx.export(torch.jit.script(self.generator_a2b), torch.ones((1, 1, 512, 512), dtype=torch.float32),
                              'gen_a2b.onnx', input_names=['img'])
            torch.onnx.export(torch.jit.script(self.generator_b2a), torch.ones((1, 1, 512, 512), dtype=torch.float32),
                              'gen_b2a.onnx', input_names=['img'])
        else:
            raise NotImplemented(f'Mode {mode} is not supported ')
