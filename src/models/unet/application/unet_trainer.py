import torch
import torch.nn as nn
from src.models.unet.domain.unet import UNet
from src.utils.timer import Timer


class UNetTrainer:
    def __init__(self, num_classes):
        self.model = UNet(num_classes=num_classes)

    def run_unet(self, train_loader, val_loader, epochs):
        timer = Timer()
        timer.start()
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                pred = torch.argmax(output, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            train_accuracies.append(correct / total)

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    output = self.model(x)
                    pred = torch.argmax(output, dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            val_accuracies.append(correct / total)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracies[-1]*100:.2f}%')
            print(f'Validation Accuracy: {val_accuracies[-1]*100:.2f}%')

        unet_time = timer.stop()
        print(f'U-Net Validation Accuracy: {val_accuracies[-1]*100:.2f}%')
        print(f'U-Net Training Time: {unet_time}')
        return train_accuracies, val_accuracies, unet_time