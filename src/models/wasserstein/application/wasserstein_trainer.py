import torch
import torch.nn as nn
from src.models.gan.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator
from src.utils.timer import Timer


class GANTrainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        # optimizer - updates weights
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.00002)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002)


    def train(self, train_loader, epochs):
        timer = Timer()
        timer.start()
        for epoch in range(epochs):
            loss_discriminator_total = 0.0
            loss_generator_total = 0.0
            for real_images, _ in train_loader:
                for _ in range(5):
                    # train discriminator
                    self.discriminator.train()
                    self.optimizer_discriminator.zero_grad()
                    fake_images = self.generator(real_images)
                    real_output = self.discriminator(real_images)
                    fake_output = self.discriminator(fake_images.detach())
                    #wasserstein loss
                    loss_discriminator = -torch.mean(real_output) + torch.mean(fake_output)
                    loss_discriminator.backward()
                    self.optimizer_discriminator.step()
                loss_discriminator_total += loss_discriminator.item()

                # train generator
                self.generator.train()
                self.optimizer_generator.zero_grad()
                # fake images - generator wants labels as 1
                fake_images = self.generator(real_images)
                fake_output = self.discriminator(fake_images)

                # total loss generator
                loss_generator = -torch.mean(fake_output)
                loss_generator.backward()
                self.optimizer_generator.step()
                loss_generator_total += loss_generator.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss D: {loss_discriminator_total:.4f}, Loss G: {loss_generator_total:.4f}')

        gan_time = timer.stop()
        print(f'GAN Wasserstein Training Time: {gan_time}')
        return gan_time