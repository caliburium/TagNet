from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

class DomainNetDataset(Dataset):
    def __init__(self, root_dir, domain_name, split='train', selected_labels=None, transform=None):
        self.root_dir = Path(root_dir)
        self.domain_name = domain_name
        self.domain_path = self.root_dir / domain_name
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        if selected_labels is not None:
            if not isinstance(selected_labels, list) or not selected_labels:
                raise ValueError("selected_labels must be a non-empty list of integers or None.")
            self.selected_original_labels = sorted(list(set(selected_labels)))
            self.label_map = {orig_label: new_label for new_label, orig_label in
                              enumerate(self.selected_original_labels)}
            self.num_classes = len(self.selected_original_labels)
        else:
            print(
                f"Warning: selected_labels is None for {domain_name} {split}. All classes will be loaded and re-indexed.")
            all_original_labels = set()
            temp_label_file_path = self.domain_path / f"{self.domain_name}_{self.split}.txt"
            if not temp_label_file_path.exists():
                raise FileNotFoundError(f"Label file not found: {temp_label_file_path}")
            with open(temp_label_file_path, 'r') as f:
                for line in f:
                    _, label_str = line.strip().split()
                    all_original_labels.add(int(label_str))

            self.selected_original_labels = sorted(list(all_original_labels))
            self.label_map = {orig_label: new_label for new_label, orig_label in
                              enumerate(self.selected_original_labels)}
            self.num_classes = len(self.selected_original_labels)

        label_file_path = self.domain_path / f"{self.domain_name}_{self.split}.txt"
        if not label_file_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_file_path}")

        with open(label_file_path, 'r') as f:
            for line in f:
                img_relative_path, label_str = line.strip().split()
                original_label = int(label_str)

                if original_label in self.label_map:
                    self.image_paths.append(self.root_dir  / img_relative_path)
                    self.labels.append(self.label_map[original_label])

        if not self.image_paths:
            print(
                f"Warning: No images found for domain '{self.domain_name}', split '{self.split}' with selected_labels.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Skipping.")
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            raise RuntimeError(f"Error loading image {img_path}")

        if self.transform:
            image = self.transform(image)

        return image, label


class ResizeWithPad:
    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        img.thumbnail(self.size, Image.Resampling.LANCZOS)

        delta_w = self.size[1] - img.size[0]
        delta_h = self.size[0] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        return TF.pad(img, padding, self.fill, self.padding_mode)

def dn_loader(domain='real', selected_labels=None, batch_size=32, num_workers=8, image_size=224, root_dir="./data/DomainNet"):
    transform_train = transforms.Compose([
        ResizeWithPad(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        ResizeWithPad(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DomainNetDataset(root_dir=root_dir, domain_name=domain, split='train', selected_labels=selected_labels, transform=transform_train)
    test_dataset = DomainNetDataset(root_dir=root_dir, domain_name=domain, split='test', selected_labels=selected_labels, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
