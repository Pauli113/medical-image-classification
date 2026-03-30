from src.models.gan.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator


class GANTrainier:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        #optimizer - updates weights
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_discriminator = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        #loss
        self.criterion = nn.BCELoss()



