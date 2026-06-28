import os
import torch
from torchvision import transforms, datasets as ds
from torchvision.utils import save_image
from src.utils.runner import load_dataset
from src.models.pix2pix.application.pix2pix_trainer_mri import Pix2PixTrainerMRI

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dirs = {
    'mri': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'mri')),
}


def save_generated_images(generator, train_loader, data_dir, classes):
    pix2pix_mri_dir = os.path.join(data_dir, 'pix2pix_mri')
    for cls in classes:
        os.makedirs(os.path.join(pix2pix_mri_dir, cls), exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_count = 0
    with torch.no_grad():
        for images, labels in train_loader:
            fake_images = generator(images)
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images * std + mean
            fake_images = torch.clamp(fake_images, 0, 1)
            for img, label in zip(fake_images, labels):
                cls = classes[label.item()]
                save_image(img, os.path.join(pix2pix_mri_dir, cls, f'fake_{img_count:05d}.png'))
                img_count += 1

    print(f'Saved {img_count} images to {pix2pix_mri_dir}')
    return pix2pix_mri_dir


if __name__ == "__main__":
    for dataset_name, data_dir in data_dirs.items():
        print(f'Training Pix2Pix MRI on {dataset_name}...')
        train_loader, val_loader, num_classes = load_dataset(data_dir, transform)

        # get classes directly from Training subfolder
        training_dir = os.path.join(data_dir, 'Training')
        if os.path.exists(training_dir):
            classes = ds.ImageFolder(training_dir).classes
        else:
            classes = train_loader.dataset.dataset.classes

        trainer = Pix2PixTrainerMRI()
        generator, pix2pix_time = trainer.train(train_loader, epochs=50)
        save_generated_images(generator, train_loader, data_dir, classes)