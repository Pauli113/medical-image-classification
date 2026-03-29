
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
    def __init__(self):
        super().__init__()
        # encoder
        self.enc1 = encoder(3, 64)
        self.enc2 = encoder(64, 128)
        self.enc3 = encoder(128, 256)
        self.enc4 = encoder(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.final = nn.Sequential(
            #reduce picture to 1x1
            nn.AdaptiveAvgPool2d(1),
            #reduce to vector
            nn.Flatten(),
            #reduce 512 to 1 number
            nn.Linear(512,1),
            #return 0 or 1, 0 is fake and 1 is real
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        return self.final(x)

