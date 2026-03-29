
import torch
import torch.nn as nn



def encoder(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU()
    )


def decoder(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU()
    )


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # encoder
        self.enc1 = encoder(3, 64)
        self.enc2 = encoder(64, 128)
        self.enc3 = encoder(128, 256)
        self.enc4 = encoder(256, 512)
        self.pool = nn.MaxPool2d(2)



    def forward(self, x):
        return self.final(x)

