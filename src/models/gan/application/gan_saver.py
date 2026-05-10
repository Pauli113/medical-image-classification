import os

from torchvision import transforms
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
        pass



for dataset_name, data_dir in data_dirs.items():
    print(f'Training GAN on {dataset_name}...')
    train_loader, val_loader, num_classes = load_dataset(data_dir, transform)
    generator, gan_time = run_gan(train_loader, epochs=50)
    save_generated_images(generator, train_loader, data_dir)