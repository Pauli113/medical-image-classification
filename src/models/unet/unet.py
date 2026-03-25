
import torch
import torch.nn as nn

from src.utils.timer import Timer


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


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # encoder
        self.enc1 = encoder(3, 64)
        self.enc2 = encoder(64, 128)
        self.enc3 = encoder(128, 256)
        self.enc4 = encoder(256, 512)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = encoder(512, 1024)
        # decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = decoder(1536, 512)  # 1024 + 512 (skip connections)
        self.dec2 = decoder(768, 256)   # 512 + 256
        self.dec3 = decoder(384, 128)   # 256 + 128
        self.dec4 = decoder(192, 64)    # 128 + 64
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)                    # 224x224
        e2 = self.enc2(self.pool(e1))        # 112x112
        e3 = self.enc3(self.pool(e2))        # 56x56
        e4 = self.enc4(self.pool(e3))        # 28x28
        # bottleneck
        bottleneck = self.bottleneck(self.pool(e4))
        # decoder with skip connections
        d1 = self.dec1(torch.cat([self.up(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up(d3), e1], dim=1))
        # classification
        x = self.classifier(d4)
        x = x.mean(dim=[2, 3])
        return x

    def run_unet(self, train_loader, val_loader, epochs):
        timer = Timer()
        timer.start()
        # updates weights
        optimizer = torch.optim.Adam(self.parameters())
        # loss function
        criterion = nn.CrossEntropyLoss()

        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            # training
            self.train()
            correct = 0
            total = 0
            running_loss = 0.0
            for x, y in train_loader:
                # reset gradients
                optimizer.zero_grad()
                # prediction
                output = self(x)
                # calculate loss
                loss = criterion(output, y)
                running_loss += loss.item()
                # backpropagate loss
                loss.backward()
                # update weights
                optimizer.step()
                # return index of highest value - i.e prediction
                pred = torch.argmax(output, dim=1)
                # sum up correct predictions
                correct += (pred == y).sum().item()
                # total is equal to all pictures in dataloader
                total += y.size(0)
            train_accuracies.append(correct / total)

            # validation
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    # prediction
                    output = self(x)
                    # return index of highest value - i.e prediction
                    pred = torch.argmax(output, dim=1)
                    # sum up correct predictions
                    correct += (pred == y).sum().item()
                    # total is equal to all pictures in dataloader
                    total += y.size(0)
            val_accuracies.append(correct / total)

            # Print per Epoch
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracies[-1]*100:.2f}%')
            print(f'Validation Accuracy: {val_accuracies[-1]*100:.2f}%')

        #time -- hier mit timer arbeiten
        unet_time = timer.stop()


        print(f'U-Net Validation Accuracy: {val_accuracies[-1]*100:.2f}%')
        print(f'U-Net Validation Time: {unet_time}%')
        return train_accuracies, val_accuracies