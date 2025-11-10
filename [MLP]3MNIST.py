import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedMLP(nn.Module):
    def __init__(self, num_classes=10, hidden_size=384):
        super(SharedMLP, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.input_dim = 1 * 28 * 28
        # self.input_dim = 24 * 6 * 6

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=2, padding=0),  # 28 to 13
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 13 to 6
        )

        self.extractor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        features = self.extractor(x_flat)
        class_output = self.classifier(features)

        return class_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb.init(entity="hails",
               project="TagNet - 3MNIST",
               config=args.__dict__,
               name="[MLP]3MNIST_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_epoch:" + str(args.epoch)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    kmnist_loader, kmnist_loader_test = data_loader('KMNIST', args.batch_size)
    fmnist_loader, fmnist_loader_test = data_loader('FMNIST', args.batch_size)

    len_dataloader = max(len(mnist_loader), len(kmnist_loader), len(fmnist_loader))

    print("Data load complete, start training")

    model = SharedMLP(num_classes=10, hidden_size=args.hidden_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss_mnist, total_loss_kmnist, total_loss_fmnist = 0, 0, 0
        correct_mnist, correct_kmnist, correct_fmnist = 0, 0, 0
        samples_mnist, samples_kmnist, samples_fmnist = 0, 0, 0

        mnist_iter = iter(mnist_loader)
        kmnist_iter = iter(kmnist_loader)
        fmnist_iter = iter(fmnist_loader)

        for i in range(len_dataloader):
            try:
                mnist_images, mnist_labels = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(mnist_loader)
                mnist_images, mnist_labels = next(mnist_iter)

            try:
                kmnist_images, kmnist_labels = next(kmnist_iter)
            except StopIteration:
                kmnist_iter = iter(kmnist_loader)
                kmnist_images, kmnist_labels = next(kmnist_iter)

            try:
                fmnist_images, fmnist_labels = next(fmnist_iter)
            except StopIteration:
                fmnist_iter = iter(fmnist_loader)
                fmnist_images, fmnist_labels = next(fmnist_iter)

            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            kmnist_images, kmnist_labels = kmnist_images.to(device), kmnist_labels.to(device)
            fmnist_images, fmnist_labels = fmnist_images.to(device), fmnist_labels.to(device)

            optimizer.zero_grad()

            out_mnist = model(mnist_images)
            loss_mnist = criterion(out_mnist, mnist_labels)

            out_kmnist = model(kmnist_images)
            loss_kmnist = criterion(out_kmnist, kmnist_labels)

            out_fmnist = model(fmnist_images)
            loss_fmnist = criterion(out_fmnist, fmnist_labels)

            total_loss = loss_mnist + loss_kmnist + loss_fmnist
            total_loss.backward()
            optimizer.step()

            total_loss_mnist += loss_mnist.item()
            total_loss_kmnist += loss_kmnist.item()
            total_loss_fmnist += loss_fmnist.item()

            correct_mnist += (torch.argmax(out_mnist, dim=1) == mnist_labels).sum().item()
            correct_kmnist += (torch.argmax(out_kmnist, dim=1) == kmnist_labels).sum().item()
            correct_fmnist += (torch.argmax(out_fmnist, dim=1) == fmnist_labels).sum().item()

            samples_mnist += mnist_labels.size(0)
            samples_kmnist += kmnist_labels.size(0)
            samples_fmnist += fmnist_labels.size(0)

        avg_loss_mnist = total_loss_mnist / len_dataloader
        avg_loss_kmnist = total_loss_kmnist / len_dataloader
        avg_loss_fmnist = total_loss_fmnist / len_dataloader

        acc_mnist = (correct_mnist / samples_mnist) * 100
        acc_kmnist = (correct_kmnist / samples_kmnist) * 100
        acc_fmnist = (correct_fmnist / samples_fmnist) * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train MNIST  | Loss: {avg_loss_mnist:.6f} | Acc: {acc_mnist:.3f}%')
        print(f'  Train KMNIST | Loss: {avg_loss_kmnist:.6f} | Acc: {acc_kmnist:.3f}%')
        print(f'  Train FMNIST | Loss: {avg_loss_fmnist:.6f} | Acc: {acc_fmnist:.3f}%')

        wandb.log({
            'Train Loss/Label MNIST': avg_loss_mnist,
            'Train Loss/Label KMNIST': avg_loss_kmnist,
            'Train Loss/Label FMNIST': avg_loss_fmnist,
            'Train/Acc Label MNIST': acc_mnist,
            'Train/Acc Label KMNIST': acc_kmnist,
            'Train/Acc Label FMNIST': acc_fmnist,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group_name):
            label_correct, total_label_loss, total = 0, 0, 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                label_loss = criterion(class_output, labels)

                total_label_loss += label_loss.item() * images.size(0)
                label_correct += (torch.argmax(class_output, dim=1) == labels).sum().item()
                total += images.size(0)

            label_acc = label_correct / total * 100
            label_loss_avg = total_label_loss / total

            wandb.log({
                f'Test/Acc Label {group_name}': label_acc,
                f'Test Loss/Label {group_name}': label_loss_avg,
            }, step=epoch + 1)

            print(f'  Test {group_name:<6} | Loss: {label_loss_avg:.6f} | Acc: {label_acc:.3f}%')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST')
            tester(kmnist_loader_test, 'KMNIST')
            tester(fmnist_loader_test, 'FMNIST')


if __name__ == '__main__':
    main()