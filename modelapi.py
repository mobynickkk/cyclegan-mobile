import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import chain
from .models.generator import Generator
from .models.discriminator import Discriminator
from .utils.dataset import ImgDataset, ShiftDataset
from .utils.train import train, shift_train


class GANModelAPI:
    """Класс для упрощенного создания и обучения модели"""
    def __init__(self, files_a, files_b, shift=True, gen_optimizer='Adam', discr_optimizer='Adam', gen_scheduler='default',
                 discr_scheduler='default', criterion='bceloss', gen_lr=2e-4, discr_lr=2e-4):
        if not torch.cuda.is_available():
            raise BaseException('GPU is not available')
        device = torch.device('cuda')
        if shift:
            self.dataloader1 = DataLoader(ShiftDataset(files_a), batch_size=1, shuffle=True)
            self.dataloader2 = DataLoader(ShiftDataset(files_b), batch_size=1, shuffle=True)
        else:
            self.dataloader = DataLoader(ImgDataset(files_a, files_b), batch_size=1, shuffle=True)
        self.generator_a2b = Generator().to(device)
        self.generator_b2a = Generator().to(device)
        self.discriminator_a = Discriminator().to(device)
        self.discriminator_b = Discriminator().to(device)
        if gen_optimizer == 'Adam':
            self.gen_optimizer = optim.Adam(chain(
                self.generator_a2b.parameters(),
                self.generator_b2a.parameters()
            ), lr=gen_lr)
        elif gen_optimizer == 'AdamW':
            self.gen_optimizer = optim.AdamW(chain(
                self.generator_a2b.parameters(),
                self.generator_b2a.parameters()
            ), lr=gen_lr)
        else:
            raise NotImplemented(f'Optimizer {gen_optimizer} is not supported now')
        if discr_optimizer == 'Adam':
            self.discr_optimizer = optim.Adam(chain(
                self.discriminator_a.parameters(),
                self.discriminator_b.parameters()
            ), lr=discr_lr)
        elif discr_optimizer == 'AdamW':
            self.discr_optimizer = optim.AdamW(chain(
                self.discriminator_a.parameters(),
                self.discriminator_b.parameters()
            ), lr=discr_lr)
        else:
            raise NotImplemented(f'Optimizer {discr_optimizer} is not supported now')
        step = 100
        if gen_scheduler == 'default':
            self.gen_sched = optim.lr_scheduler.LambdaLR(
                self.gen_optimizer,
                lr_lambda=lambda epoch: 0.9 ** (epoch - step) if epoch > step else 1
            )
        elif gen_scheduler == 'step10warmup':
            self.gen_sched = optim.lr_scheduler.LambdaLR(
                self.gen_optimizer,
                lr_lambda=lambda epoch: (1/0.9) ** epoch if epoch < 5 else (1/0.9) ** 5 * 0.9 ** ((epoch - 5) // 10)
            )
        else:
            raise NotImplemented(f'Generators lr scheduler {gen_scheduler} is not supported now')
        if discr_scheduler == 'default':
            self.discr_sched = optim.lr_scheduler.LambdaLR(
                self.discr_optimizer,
                lr_lambda=lambda epoch: 0.9 ** (epoch - step) if epoch > step else 1
            )
        elif discr_scheduler == 'step10warmup':
            self.discr_sched = optim.lr_scheduler.LambdaLR(
                self.discr_optimizer,
                lr_lambda=lambda epoch: (1/0.9) ** epoch if epoch < 5 else (1/0.9) ** 5 * 0.9 ** ((epoch - 5) // 10)
            )
        else:
            raise NotImplemented(f'Discriminators lr scheduler {discr_scheduler} is not supported now')
        if criterion == 'bceloss':
            self.criterion = nn.BCELoss()
        else:
            raise NotImplemented(f'Criterion {criterion} is not supported now')
        self.shift = shift

    def train_models(self, max_epochs=200, hold_discr=True, threshold=0.5, intermediate_results=None):
        if self.shift:
            return shift_train(self.generator_a2b, self.generator_b2a, self.discriminator_a, self.discriminator_b,
                               self.gen_optimizer, self.discr_optimizer, self.gen_sched, self.discr_sched,
                               self.criterion, self.dataloader1, self.dataloader2, max_epochs, hold_discr, threshold,
                               intermediate_results=intermediate_results)
        else:
            return train(self.generator_a2b, self.generator_b2a, self.discriminator_a, self.discriminator_b,
                         self.gen_optimizer, self.discr_optimizer, self.gen_sched, self.discr_sched, self.criterion,
                         self.dataloader, max_epochs, hold_discr, threshold)

    def load_models(self, gen_a2b, gen_b2a, discr_a, discr_b):
        self.generator_a2b = gen_a2b
        self.generator_b2a = gen_b2a
        self.discriminator_a = discr_a
        self.discriminator_b = discr_b

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
