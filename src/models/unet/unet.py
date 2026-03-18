import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*112*112, 3)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x