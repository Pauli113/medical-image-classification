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
    'xray_covid_19': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'xray_dataset_covid19')),
    'xray': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'xray'))
    #'mri': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'mri'))
}


def save_generated_images(generator, train_loader, data_dir):
    gan_dir = os.path.join(data_dir, 'gan')
    classes = train_loader.dataset.dataset.classes
    for cls in classes:
        os.makedirs(os.path.join(gan_dir, cls), exist_ok=True)

    # Normalisierung rückgängig machen
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_count = 0
    with torch.no_grad():
        for images, labels in train_loader:
            fake_images = generator(images)
            # Tanh Output von -1/1 auf 0/1
            fake_images = (fake_images + 1) / 2
            # Denormalisierung
            fake_images = fake_images * std + mean
            fake_images = torch.clamp(fake_images, 0, 1)
            for img, label in zip(fake_images, labels):
                cls = classes[label.item()]
                save_image(img, os.path.join(gan_dir, cls, f'fake_{img_count:05d}.png'))
                img_count += 1

    print(f'Saved {img_count} images to {gan_dir}')
    return gan_dir

if __name__ == "__main__":

    for dataset_name, data_dir in data_dirs.items():
        print(f'Training GAN on {dataset_name}...')
        train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
        generator, gan_time = run_gan(train_loader, epochs=30)
        save_generated_images(generator, train_loader, data_dir)

