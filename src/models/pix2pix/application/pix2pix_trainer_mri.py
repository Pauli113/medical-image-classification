import torch
import torch.nn as nn
from src.models.pix2pix.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator
from src.utils.timer import Timer


class Pix2PixTrainerMRI:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        # lower discriminator lr to prevent mode collapse
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.00001)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def train(self, train_loader, epochs):
        timer = Timer()
        timer.start()
        for epoch in range(epochs):
            loss_discriminator_total = 0.0
            loss_generator_total = 0.0
            for real_images, _ in train_loader:

                # train discriminator 2x
                for _ in range(2):
                    self.discriminator.train()
                    self.optimizer_discriminator.zero_grad()

                    # add gaussian noise to input
                    noise = torch.randn_like(real_images) * 0.1
                    noisy_images = real_images + noise
                    fake_images = self.generator(noisy_images)

                    # real pair
                    real_output = self.discriminator(real_images, real_images)
                    real_labels = torch.ones_like(real_output) * 0.9
                    loss_real = self.criterion(real_output, real_labels)

                    # fake pair
                    fake_output = self.discriminator(real_images, fake_images.detach())
                    fake_labels = torch.zeros_like(fake_output)
                    loss_fake = self.criterion(fake_output, fake_labels)

                    loss_discriminator = loss_real + loss_fake
                    loss_discriminator.backward()
                    self.optimizer_discriminator.step()

                loss_discriminator_total += loss_discriminator.item()

                # train generator
                self.generator.train()
                self.optimizer_generator.zero_grad()

                noise = torch.randn_like(real_images) * 0.1
                noisy_images = real_images + noise
                fake_images = self.generator(noisy_images)
                fake_output = self.discriminator(real_images, fake_images)
                real_labels = torch.ones_like(fake_output)

                loss_gan = self.criterion(fake_output, real_labels)
                loss_l1 = self.l1_loss(fake_images, real_images) * 100
                loss_generator = loss_gan + loss_l1
                loss_generator.backward()
                self.optimizer_generator.step()
                loss_generator_total += loss_generator.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss D: {loss_discriminator_total:.4f}, Loss G: {loss_generator_total:.4f}')

        gan_time = timer.stop()
        print(f'Pix2Pix MRI Training Time: {gan_time}')
        return self.generator, gan_time