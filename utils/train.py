import torch
import torch.nn.functional as F
from time import time


def train(generator_a2b, generator_b2a, discriminator_a, discriminator_b, generator_optimizer, discriminator_optimizer,
          generator_scheduler, discriminator_scheduler, criterion, train_loader, max_epochs,
          hold_discriminators=False, threshold=0.5):
    """

    :param generator_a2b:
    :param generator_b2a:
    :param discriminator_a:
    :param discriminator_b:
    :param generator_optimizer:
    :param discriminator_optimizer:
    :param generator_scheduler:
    :param discriminator_scheduler:
    :param criterion:
    :param train_loader:
    :param max_epochs:
    :param hold_discriminators:
    :param threshold:
    :return: losses_lists
    """

    discriminators_losses = []
    generators_losses = []
    train_length = len(train_loader)

    for i in range(max_epochs):

        steps = 0
        total_loss_gen = 0
        total_loss_discr = 0
        avg_time = 0

        for j, t in enumerate(train_loader):
            a, b = t
            start = time()
            fake_b = generator_a2b(a)
            fake_a = generator_b2a(b)
            same_b = generator_a2b(b)
            same_a = generator_b2a(a)

            # Generators losses

            generator_optimizer.zero_grad()
            pred_fake_b = discriminator_b(fake_b)
            a2b_discriminator_loss = criterion(pred_fake_b, torch.ones_like(pred_fake_b))
            a2b_identity_loss = F.l1_loss(same_b, b)

            pred_fake_a = discriminator_a(fake_a)
            b2a_discriminator_loss = criterion(pred_fake_a, torch.ones_like(pred_fake_a))
            b2a_identity_loss = F.l1_loss(same_a, a)

            generators_loss = a2b_discriminator_loss + a2b_identity_loss + b2a_discriminator_loss + b2a_identity_loss
            generators_loss.backward()
            generator_optimizer.step()

            # Discriminators losses

            discriminator_optimizer.zero_grad()
            pred_fake_a = discriminator_a(fake_a.detach())
            a_fake_loss = criterion(pred_fake_a, torch.zeros_like(pred_fake_a))
            pred_real_a = discriminator_a(a)
            a_real_loss = criterion(pred_real_a, torch.ones_like(pred_real_a))

            pred_fake_b = discriminator_a(fake_b.detach())
            b_fake_loss = criterion(pred_fake_b, torch.zeros_like(pred_fake_b))
            pred_real_b = discriminator_b(b)
            b_real_loss = criterion(pred_real_b, torch.ones_like(pred_real_b))

            discriminators_loss = a_fake_loss + a_real_loss + b_fake_loss + b_real_loss

            if hold_discriminators and discriminators_loss.item() > threshold:
                discriminators_loss.backward()
                discriminator_optimizer.step()
            elif not hold_discriminators:
                discriminators_loss.backward()
                discriminator_optimizer.step()

            # Statistics
            avg_time += time() - start
            total_loss_discr += discriminators_loss.item()
            discriminators_losses.append(discriminators_loss.item())
            total_loss_gen += generators_loss.item()
            generators_losses.append(generators_loss.item())
            steps += 1

            if j % (train_length // 100) == 0:
                print(f'Epoch {i+1} completed {j/(train_length // 100):.3f}% ' +
                      f'gen_loss = {generators_loss.item():.3f} ' +
                      f'discriminators_loss = {discriminators_loss.item():.3f} ' +
                      f'avg_iter_duration = {avg_time / (train_length // 100)}')

                avg_time = 0

        print(f'Epoch {i+1}')
        print(f'Discriminator average train loss: {total_loss_discr/steps:.3f}')
        print(f'Generator average train loss: {total_loss_gen/steps:.3f}')

        torch.cuda.empty_cache()

        generator_scheduler.step()
        discriminator_scheduler.step()

    return generators_losses, discriminators_losses


def shift_train(generator_a2b, generator_b2a, discriminator_a, discriminator_b, generator_optimizer, discriminator_optimizer,
                generator_scheduler, discriminator_scheduler, criterion, loader_a, loader_b, max_epochs,
                hold_discriminators=False, threshold=0.5, intermediate_results=None):

    discriminators_losses = []
    generators_losses = []
    train_length = min(len(loader_a), len(loader_b))

    for i in range(max_epochs):

        steps = 0
        total_loss_gen = 0
        total_loss_discr = 0
        avg_time = 0

        for j, t in enumerate(zip(loader_a, loader_b)):
            a, b = t
            start = time()
            fake_b = generator_a2b(a)
            fake_a = generator_b2a(b)
            same_b = generator_a2b(b)
            same_a = generator_b2a(a)

            # Generators losses

            generator_optimizer.zero_grad()
            pred_fake_b = discriminator_b(fake_b)
            a2b_discriminator_loss = criterion(pred_fake_b, torch.ones_like(pred_fake_b))
            a2b_identity_loss = F.l1_loss(same_b, b)

            pred_fake_a = discriminator_a(fake_a)
            b2a_discriminator_loss = criterion(pred_fake_a, torch.ones_like(pred_fake_a))
            b2a_identity_loss = F.l1_loss(same_a, a)

            generators_loss = a2b_discriminator_loss + a2b_identity_loss + b2a_discriminator_loss + b2a_identity_loss
            generators_loss.backward()
            generator_optimizer.step()

            # Discriminators losses

            discriminator_optimizer.zero_grad()
            pred_fake_a = discriminator_a(fake_a.detach())
            a_fake_loss = criterion(pred_fake_a, torch.zeros_like(pred_fake_a))
            pred_real_a = discriminator_a(a)
            a_real_loss = criterion(pred_real_a, torch.ones_like(pred_real_a))

            pred_fake_b = discriminator_a(fake_b.detach())
            b_fake_loss = criterion(pred_fake_b, torch.zeros_like(pred_fake_b))
            pred_real_b = discriminator_b(b)
            b_real_loss = criterion(pred_real_b, torch.ones_like(pred_real_b))

            discriminators_loss = a_fake_loss + a_real_loss + b_fake_loss + b_real_loss

            if hold_discriminators and discriminators_loss.item() > threshold:
                discriminators_loss.backward()
                discriminator_optimizer.step()
            elif not hold_discriminators:
                discriminators_loss.backward()
                discriminator_optimizer.step()

            # Statistics
            avg_time += time() - start
            total_loss_discr += discriminators_loss.item()
            discriminators_losses.append(discriminators_loss.item())
            total_loss_gen += generators_loss.item()
            generators_losses.append(generators_loss.item())
            steps += 1

            if (j + 1) % (train_length // 100) == 0:
                print(f'Epoch {i + 1} completed {j // (train_length // 100)}% ' +
                      f'gen_loss = {generators_loss.item():.3f} ' +
                      f'discriminators_loss = {discriminators_loss.item():.3f} ' +
                      f'avg_iter_duration = {avg_time / (train_length // 100)}')

                avg_time = 0

        print(f'Epoch {i + 1}')
        print(f'Discriminator average train loss: {total_loss_discr / steps:.3f}')
        print(f'Generator average train loss: {total_loss_gen / steps:.3f}')

        if intermediate_results:
            a = next(iter(loader_a))
            intermediate_results(generator_a2b(a))

        torch.cuda.empty_cache()

        generator_scheduler.step()
        discriminator_scheduler.step()

    return generators_losses, discriminators_losses
