import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedMLP(nn.Module):
    def __init__(self, num_classes=10, hidden_size=768):
        super(SharedMLP, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.input_dim = 3 * 32 * 32

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
        x_flat = x.view(x.size(0), -1)
        features = self.extractor(x_flat)
        class_output = self.classifier(features)

        return class_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=3072)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb.init(entity="hails",
               project="TagNet - MSC",
               config=args.__dict__,
               name="[MLP]MSC_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_epoch:" + str(args.epoch)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)
    len_dataloader = max(len(mnist_loader), len(svhn_loader), len(cifar_loader))

    print("Data load complete, start training")

    model = SharedMLP(num_classes=10, hidden_size=args.hidden_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss_mnist, total_loss_svhn, total_loss_cifar = 0, 0, 0
        correct_mnist, correct_svhn, correct_cifar = 0, 0, 0
        samples_mnist, samples_svhn, samples_cifar = 0, 0, 0

        mnist_iter = iter(mnist_loader)
        svhn_iter = iter(svhn_loader)
        cifar_iter = iter(cifar_loader)

        for i in range(len_dataloader):
            try:
                mnist_images, mnist_labels = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(mnist_loader)
                mnist_images, mnist_labels = next(mnist_iter)

            try:
                svhn_images, svhn_labels = next(svhn_iter)
            except StopIteration:
                svhn_iter = iter(svhn_loader)
                svhn_images, svhn_labels = next(svhn_iter)

            try:
                cifar_images, cifar_labels = next(cifar_iter)
            except StopIteration:
                cifar_iter = iter(cifar_loader)
                cifar_images, cifar_labels = next(cifar_iter)

            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

            optimizer.zero_grad()

            out_mnist = model(mnist_images)
            loss_mnist = criterion(out_mnist, mnist_labels)

            out_svhn = model(svhn_images)
            loss_svhn = criterion(out_svhn, svhn_labels)

            out_cifar = model(cifar_images)
            loss_cifar = criterion(out_cifar, cifar_labels)

            total_loss = loss_mnist + loss_svhn + loss_cifar
            total_loss.backward()
            optimizer.step()

            total_loss_mnist += loss_mnist.item()
            total_loss_svhn += loss_svhn.item()
            total_loss_cifar += loss_cifar.item()

            correct_mnist += (torch.argmax(out_mnist, dim=1) == mnist_labels).sum().item()
            correct_svhn += (torch.argmax(out_svhn, dim=1) == svhn_labels).sum().item()
            correct_cifar += (torch.argmax(out_cifar, dim=1) == cifar_labels).sum().item()

            samples_mnist += mnist_labels.size(0)
            samples_svhn += svhn_labels.size(0)
            samples_cifar += cifar_labels.size(0)

        avg_loss_mnist = total_loss_mnist / len_dataloader
        avg_loss_svhn = total_loss_svhn / len_dataloader
        avg_loss_cifar = total_loss_cifar / len_dataloader

        acc_mnist = (correct_mnist / samples_mnist) * 100
        acc_svhn = (correct_svhn / samples_svhn) * 100
        acc_cifar = (correct_cifar / samples_cifar) * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train MNIST | Loss: {avg_loss_mnist:.6f} | Acc: {acc_mnist:.3f}%')
        print(f'  Train SVHN  | Loss: {avg_loss_svhn:.6f} | Acc: {acc_svhn:.3f}%')
        print(f'  Train CIFAR | Loss: {avg_loss_cifar:.6f} | Acc: {acc_cifar:.3f}%')

        wandb.log({
            'Train Loss/Label MNIST': avg_loss_mnist,
            'Train Loss/Label SVHN': avg_loss_svhn,
            'Train Loss/Label CIFAR': avg_loss_cifar,
            'Train/Acc Label MNIST': acc_mnist,
            'Train/Acc Label SVHN': acc_svhn,
            'Train/Acc Label CIFAR': acc_cifar,
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
                f'Test/Loss Label {group_name}': label_loss_avg,
            }, step=epoch + 1)

            print(f'  Test {group_name} | Acc: {label_acc:.3f}% | Loss: {label_loss_avg:.6f}')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST')
            tester(svhn_loader_test, 'SVHN')
            tester(cifar_loader_test, 'CIFAR')

if __name__ == '__main__':
    main()