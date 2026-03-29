from src.models.gan.domain.discriminator import Discriminator
from src.models.gan.domain.generator import Generator


class GANTrainier:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
