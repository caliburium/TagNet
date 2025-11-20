import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from dataloader.ToBlackAndWhite import ToBlackAndWhite
from PIL import Image
import io
import numpy as np

# MNIST-M
class CustomParquetDataset(Dataset):
    """
    Custom Dataset for reading Parquet files (e.g., from Hugging Face).
    Assumes the 'image' column contains a dictionary {'bytes': ...}
    and the 'label' column contains the label.
    """

    def __init__(self, parquet_path, transform=None):
        self.transform = transform
        try:
            self.data_frame = pd.read_parquet(parquet_path, engine='pyarrow')
        except FileNotFoundError:
            print(f"Error: Parquet file not found at: {parquet_path}")
            print("Please run the downloader script first.")
            raise
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise

        if 'image' not in self.data_frame.columns or 'label' not in self.data_frame.columns:
            raise ValueError(f"Parquet file must contain 'image' and 'label' columns. (Path: {parquet_path})")

        self.images = self.data_frame['image']
        self.labels = self.data_frame['label']

    def __len__(self):
        # Return the total number of samples
        return len(self.data_frame)

    def __getitem__(self, index):
        image_data = self.images.iloc[index]
        label = self.labels.iloc[index]

        try:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")

        except (KeyError, TypeError, AttributeError):
            raise TypeError(f"Failed to process image data at index {index}. "
                            f"Expected dict with 'bytes' key, but got: {type(image_data)}")

        if self.transform:
            image = self.transform(image)

        return image, label


class FilteredCIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None, target_classes_list=None):
        self.transform = transform

        if target_classes_list is None:
            raise ValueError("target_classes_list must be provided")

        base_dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=None)

        self.class_map = {orig_label: new_label for new_label, orig_label in enumerate(target_classes_list)}
        self.data = []
        self.targets = []

        print(f"Filtering CIFAR100 (train={train}) for {len(target_classes_list)} classes...")
        for img_data, target_label in zip(base_dataset.data, base_dataset.targets):
            if target_label in self.class_map:
                self.data.append(img_data)
                new_label = self.class_map[target_label]
                self.targets.append(new_label)

        print(f"Filtering complete. Found {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, target


MAMMALS_CLASSES = [
    3, 42, 43, 88, 97,  # bear, leopard, lion, tiger, wolf
    34, 63, 64, 66, 75  # fox, porcupine, possum, raccoon, skunk
]

VEHICLES_CLASSES = [
    8, 13, 48, 58, 90,  # bicycle, bus, motorcycle, pickup_truck, train
    41, 69, 81, 85, 89  # lawn_mower, rocket, streetcar, tank, tractor
]

def data_loader(source, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform28 = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform32 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform28bw = transforms.Compose([
        ToBlackAndWhite(),
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_mnist = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist_rs = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if source == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif source == 'FMNIST':
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    elif source == 'KMNIST':
        dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    elif source == 'MNISTM':
        train_file_path = './data/mnistm-train.parquet'
        test_file_path = './data/mnistm-test.parquet'
        # MNISTMÀº 28x28 ÄÃ·¯ÀÌ¹Ç·Î transform28 »ç¿ëÀÌ ÀûÀýÇØ º¸ÀÔ´Ï´Ù.
        dataset = CustomParquetDataset(parquet_path=train_file_path, transform=transform28)
        dataset_test = CustomParquetDataset(parquet_path=test_file_path, transform=transform28)

    elif source == 'SVHN':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform32)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform32)

    elif source == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform32)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform32)

    elif source == 'CIFAR_M':
        dataset = FilteredCIFAR100(root='./data', train=True, transform=transform32, target_classes_list=MAMMALS_CLASSES)
        dataset_test = FilteredCIFAR100(root='./data', train=False, transform=transform32, target_classes_list=MAMMALS_CLASSES)

    elif source == 'CIFAR_V':  # Vehicles
        dataset = FilteredCIFAR100(root='./data', train=True, transform=transform32, target_classes_list=VEHICLES_CLASSES)
        dataset_test = FilteredCIFAR100(root='./data', train=False, transform=transform32, target_classes_list=VEHICLES_CLASSES)
    else:
        print("no source")
        raise ValueError(f"Unknown data source: {source}")

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)

    return loader, test_loader