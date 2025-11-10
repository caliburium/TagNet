import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from dataloader.ToBlackAndWhite import ToBlackAndWhite
from PIL import Image
import io


class CustomParquetDataset(Dataset):
    """
    Custom Dataset for reading Parquet files (e.g., from Hugging Face).
    Assumes the 'image' column contains a dictionary {'bytes': ...}
    and the 'label' column contains the label.
    """

    def __init__(self, parquet_path, transform=None):
        self.transform = transform
        try:
            # Load the Parquet file into a DataFrame
            self.data_frame = pd.read_parquet(parquet_path, engine='pyarrow')
        except FileNotFoundError:
            print(f"Error: Parquet file not found at: {parquet_path}")
            print("Please run the downloader script first.")
            raise
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise

        # Check if required columns exist
        if 'image' not in self.data_frame.columns or 'label' not in self.data_frame.columns:
            raise ValueError(f"Parquet file must contain 'image' and 'label' columns. (Path: {parquet_path})")

        self.images = self.data_frame['image']
        self.labels = self.data_frame['label']

    def __len__(self):
        # Return the total number of samples
        return len(self.data_frame)

    def __getitem__(self, index):
        # Retrieve the image data (which is a dict) and label
        image_data = self.images.iloc[index]
        label = self.labels.iloc[index]

        # --- THIS IS THE FIX ---
        # The 'image' column contains a dict: {'bytes': b'...'}
        # We need to extract the 'bytes', open them with PIL, and convert to RGB
        try:
            # 1. Get the bytes from the dictionary
            image_bytes = image_data['bytes']
            # 2. Open the bytes as a PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            # 3. Ensure the image is in RGB format (MNIST-M is color)
            image = image.convert("RGB")

        except (KeyError, TypeError, AttributeError):
            raise TypeError(f"Failed to process image data at index {index}. "
                            f"Expected dict with 'bytes' key, but got: {type(image_data)}")
        # --- END OF FIX ---

        # Apply transformations (e.g., ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)  # Now 'image' is a PIL.Image, so this works

        return image, label

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

    elif source == 'SVHN_BW':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform28bw)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform28bw)

    elif source == 'MNISTM':
        train_file_path = './data/mnistm-train.parquet'
        test_file_path = './data/mnistm-test.parquet'
        dataset = CustomParquetDataset(parquet_path=train_file_path, transform=transform28)
        dataset_test = CustomParquetDataset(parquet_path=test_file_path, transform=transform28)

    elif source == 'SVHN':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform28)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform28)

    elif source == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform28)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform28)

    else:
        print("no source")

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)

    return loader, test_loader
