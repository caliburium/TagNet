import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from dataloader.data_loader import data_loader
from functions.ReverseLayerF import ReverseLayerF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DomainMLP(nn.Module):
    def __init__(self, num_domains=3, hidden_size=768):
        super(DomainMLP, self).__init__()
        self.num_domains = num_domains
        self.hidden_size = hidden_size
        # self.input_dim = 32 * 6 * 6
        self.input_dim = 1 * 28 * 28

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.classifiers = nn.ModuleList()

        for _ in range(num_domains):
            classifier = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
            self.classifiers.append(classifier)

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_domains)
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        x_feat = self.pre_classifier(x_flat)

        reverse_feature = ReverseLayerF.apply(x_feat, 0)
        dom_out = self.discriminator(reverse_feature)
        dom_pred = torch.argmax(dom_out, dim=1)

        outs = torch.zeros(x_feat.size(0), 10, device=x.device)
        for dom_idx in range(self.num_domains):
            mask = (dom_pred == dom_idx)
            if mask.any():
                cls = self.classifiers[dom_idx]
                outs[mask] = cls(x_feat[mask])
        return outs, dom_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=192)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb.init(entity="hails",
               project="TagNet - 3MNIST",
               config=args.__dict__,
               name="[Debug] DomainClassifier_MLP"
               )

    numbers_loader, numbers_loader_test = data_loader('MNIST', args.batch_size)
    fashion_loader, fashion_loader_test = data_loader('FMNIST', args.batch_size)
    gana_loader, gana_loader_test = data_loader('KMNIST', args.batch_size)

    len_dataloader = min(len(numbers_loader), len(fashion_loader), len(gana_loader))

    print("Data load complete, start domain training")

    model = DomainMLP(num_domains=3, hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        p = epoch / num_epochs
        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
        model.train()
        total_loss, total_label_correct, total_domain_correct, total_samples = 0, 0, 0, 0

        mnist_iter = iter(numbers_loader)
        kmnist_iter = iter(gana_loader)
        fmnist_iter = iter(fashion_loader)

        for i in range(len_dataloader):
            mnist_images, mnist_labels = next(mnist_iter)
            kmnist_images, kmnist_labels = next(kmnist_iter)
            fmnist_images, fmnist_labels = next(fmnist_iter)

            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            kmnist_images, kmnist_labels = kmnist_images.to(device), kmnist_labels.to(device)
            fmnist_images, fmnist_labels = fmnist_images.to(device), fmnist_labels.to(device)

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            kmnist_dlabels = torch.full((kmnist_images.size(0),), 1, dtype=torch.long, device=device)
            fmnist_dlabels = torch.full((fmnist_images.size(0),), 2, dtype=torch.long, device=device)

            all_images = torch.cat([mnist_images, kmnist_images, fmnist_images], dim=0)
            all_labels = torch.cat([mnist_labels, kmnist_labels, fmnist_labels], dim=0)
            all_dlabels = torch.cat([mnist_dlabels, kmnist_dlabels, fmnist_dlabels], dim=0)

            optimizer.zero_grad()
            label_out, domain_out = model(all_images)
            label_loss = criterion(label_out, all_labels)
            domain_loss = criterion(domain_out, all_dlabels)
            loss = label_loss + domain_loss
            loss.backward()
            optimizer.step()

            total_label_loss = label_loss.item() * all_images.size(0)
            total_domain_loss = domain_loss.item() * all_images.size(0)
            total_loss += loss.item() * all_images.size(0)

            total_label_correct += (torch.argmax(label_out, dim=1) == all_labels).sum().item()
            total_domain_correct += (torch.argmax(domain_out, dim=1) == all_dlabels).sum().item()
            total_samples += all_images.size(0)

        avg_label_loss = total_label_loss / total_samples
        avg_domain_loss = total_domain_loss / total_samples
        label_acc = (total_label_correct / total_samples) * 100
        domain_acc = (total_domain_correct / total_samples) * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Lebel | Loss: {label_loss:.6f} | Acc: {label_acc:.3f}%')
        print(f'  Train Domain | Loss: {domain_loss:.6f} | Acc: {domain_acc:.3f}%')

        wandb.log({
            'Train/Acc Label': label_acc,
            'Train/Acc Domain': domain_acc,
            'Train Loss/Label': avg_label_loss,
            'Train Loss/Domain': avg_domain_loss,
        }, step=epoch + 1)

        def tester(model, loader, domain_id, criterion, group_name):
            label_correct, domain_correct, label_loss, domain_loss, total = 0, 0, 0, 0, 0

            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                dlabels = torch.full((images.size(0),), domain_id, dtype=torch.long, device=device)

                with torch.no_grad():
                    outputs, dom_out = model(images)
                    loss = criterion(outputs, labels)
                    dom_loss = criterion(dom_out, dlabels)

                label_loss += loss.item() * images.size(0)
                domain_loss += dom_loss.item() * images.size(0)

                label_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(dom_out, dim=1) == dlabels).sum().item()
                total += images.size(0)

            label_acc = (label_correct / total) * 100
            domain_acc = (domain_correct / total) * 100
            label_loss_avg = label_loss / total
            domain_loss_avg = domain_loss / total

            print(f'  Test {group_name:<6} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}%')
            wandb.log({
                f'Test/Acc Label {group_name}': label_acc,
                f'Test/Acc Domain {group_name}': domain_acc,
                f'Test Loss/Label {group_name}': label_loss_avg,
                f'Test Loss/Domain {group_name}': domain_loss_avg,
            }, step=epoch + 1)

        model.eval()
        tester(model, numbers_loader_test, 0, criterion, 'MNIST')
        tester(model, gana_loader_test, 1, criterion, 'KMNIST')
        tester(model, fashion_loader_test, 2, criterion, 'FMNIST')


if __name__ == '__main__':
    main()