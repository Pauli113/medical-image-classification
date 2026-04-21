import torch
import torch.nn as nn
from src.models.gan.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator


class GANTrainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        #optimizer - updates weights
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        #loss
        self.criterion = nn.BCELoss()

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for real_images, _ in train_loader:
                #train discriminator
                self.discriminator.train()
                #reset weights
                self.optimizer_discriminator.zero_grad()
                #train generator
                #real images - get labeled as 1
                real_output = self.discriminator(real_images)
                real_labels = torch.ones_like(real_output)
                loss_real = self.criterion(real_output, real_labels)
                #fake images - get labeled as 0
                fake_images = self.generator(real_images)
                fake_output = self.discriminator(fake_images.detach())
                fake_labels = torch.zeros_like(fake_output)
                loss_fake = self.criterion(fake_output, fake_labels)
                #total loss
                loss_discriminator = loss_real + loss_fake
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                #train generator
                self.generator.train()
                #reset weights
                self.optimizer_generator.zero_grad()
                #train generator
                #fake images - get labeled as 1
                fake_images = self.generator(real_images)
                fake_output = self.discriminator(fake_images)
                real_labels = torch.ones_like(fake_output)
                #total loss
                loss_generator = self.criterion(fake_output, real_labels)
                loss_generator.backward()
                self.optimizer_generator.step()