import torch
import torch.nn as nn
from src.models.pix2pix.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator
from src.utils.timer import Timer


class Pix2PixTrainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        # optimizer - updates weights
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.00002)
        # loss
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def train(self, train_loader, epochs):
        timer = Timer()
        timer.start()
        for epoch in range(epochs):
            loss_discriminator_total = 0.0
            loss_generator_total = 0.0
            for real_images, _ in train_loader:
                # train discriminator
                self.discriminator.train()
                self.optimizer_discriminator.zero_grad()

                #generate images
                fake_images = self.generator(real_images)

                # real images
                real_output = self.discriminator(real_images, real_images)
                real_labels = torch.ones_like(real_output) * 0.9
                loss_real = self.criterion(real_output, real_labels)

                # fake images - labeled as 0
                fake_output = self.discriminator(fake_images.detach())
                fake_labels = torch.zeros_like(fake_output)
                loss_fake = self.criterion(fake_output, fake_labels)

                # total loss discriminator
                loss_discriminator = loss_real + loss_fake
                loss_discriminator.backward()
                self.optimizer_discriminator.step()
                loss_discriminator_total += loss_discriminator.item()

                # train generator
                self.generator.train()
                self.optimizer_generator.zero_grad()

                # fake images - generator wants labels as 1
                fake_images = self.generator(real_images)
                fake_output = self.discriminator(real_images, fake_images)
                real_labels = torch.ones_like(fake_output)

                # total loss generator
                loss_gan = self.criterion(fake_output, real_labels)
                loss_l1 = self.l1_loss(fake_images, real_images) * 100
                loss_generator = loss_gan + loss_l1
                loss_generator.backward()
                self.optimizer_generator.step()
                loss_generator_total += loss_generator.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss D: {loss_discriminator_total:.4f}, Loss G: {loss_generator_total:.4f}')

        gan_time = timer.stop()
        print(f'Pix2Pix Training Time: {gan_time}')
        return gan_time