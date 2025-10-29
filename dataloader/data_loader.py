from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from dataloader.ToBlackAndWhite import ToBlackAndWhite


# STL10 classes: 0: 'airplane', 1: 'bird', 2: 'car', 3: 'cat', 4: 'deer',
#                5: 'dog', 6: 'frog', 7: 'horse', 8: 'monkey', 9: 'ship'

# CIFAR-10 classes: 0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
#                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'

class STL10Mapped(datasets.STL10):
    def __init__(self, root, split='train', transform=None, download=False):
        super().__init__(root, split=split, transform=transform, download=download)

        self.label_map = {
            0: 0,  # airplane
            1: 2,  # bird -> CIFAR 2
            2: 1,  # car -> CIFAR 1 (automobile)
            3: 3,  # cat
            4: 4,  # deer
            5: 5,  # dog
            6: 7,  # frog
            7: 6,  # horse
            8: 8,  # ship
            9: 9   # truck
        }

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # 매핑된 새로운 레이블을 반환
        if target in self.label_map:
            new_target = self.label_map[target]
            return img, new_target
        else:
            return img, target

def data_loader(source, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_bw = transforms.Compose([
        ToBlackAndWhite(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if source == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    elif source == 'MNIST_RS':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist_rs)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist_rs)

    elif source == 'USPS':
        dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform_mnist_rs)
        dataset_test = datasets.USPS(root='./data', train=False, download=True, transform=transform_mnist_rs)

    elif source == 'SVHN':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    elif source == 'SVHN_BW':
        dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_bw)
        dataset_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_bw)

    elif source == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif source == 'STL10':
        dataset = STL10Mapped(root='./data', split='train', download=True, transform=transform_stl10)
        dataset_test = STL10Mapped(root='./data', split='test', download=True, transform=transform_stl10)
    else:
        print("no source")

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)

    return loader, test_loader
