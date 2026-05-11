import os

import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

from src.utils.runner import load_dataset, run_gan

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Directory paths for datasets
data_dirs = {
    'mri': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mri')),
    'xray': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray')),
    'xray_covid_19': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'xray_dataset_covid19'))
}


def save_generated_images(generator, train_loader, data_dir):
        gan_dir = os.path.join(data_dir, 'gan')
        os.makedirs(gan_dir, exist_ok=True)

        # Klassenordner aus originalem Datensatz holen
        original_dataset = datasets.ImageFolder(data_dir)
        classes = original_dataset.classes
        for cls in classes:
            os.makedirs(os.path.join(gan_dir, cls), exist_ok=True)

        img_count = 0
        with torch.no_grad():
            for images, labels in train_loader:
                fake_images = generator(images)
                for img, label in zip(fake_images, labels):
                    cls = classes[label.item()]
                    save_image(img, os.path.join(gan_dir, cls, f'fake_{img_count:05d}.png'))
                    img_count += 1
        print(f'Saved {img_count} images to {gan_dir}')
        return gan_dir


for dataset_name, data_dir in data_dirs.items():
    print(f'Training GAN on {dataset_name}...')
    train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
    generator, gan_time = run_gan(train_loader, epochs=50)
    save_generated_images(generator, train_loader, data_dir)