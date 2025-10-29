import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from functions.lr_lambda import lr_lambda
import wandb
from dataloader.data_loader import data_loader
from model.DANN import DANN, dann_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    # parser.add_argument('--pre_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--feature_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=1.0)

    args = parser.parse_args()
    num_epochs = args.epoch
    # pre_epochs = args.pre_epochs

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="TagNet - MSC",
               config=args.__dict__,
               name="[DANN]S:SVHN/T:MNIST_512/64"
                    + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
                    + "_epoch:" + str(args.epoch)
                    # + "_fc:" + str(args.fc_lr) + "_disc:" + str(args.disc_lr)
               )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)

    print("Data load complete, start training")

    def lr_lambda(epoch):
        alpha = 10
        beta = 0.75
        p = epoch / num_epochs

        return (1 + alpha * p) ** (-beta)

    model = DANN().to(device)
    param = dann_weights(model, args.lr, args.feature_lr, args.fc_lr, args.disc_lr)
    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # optimizer_disc = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # pre_optimizer = optim.Adam(list(model.features.parameters()) + list(model.classifier.parameters()), lr=args.lr, weight_decay=args.opt_decay)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    """
    for epoch in range(pre_epochs):
        model.train()
        i = 0

        total_svhn_loss, total_svhn_correct = 0, 0
        total_samples = 0

        for svhn_images, svhn_labels in svhn_loader:
            lambda_p = 0.0

            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)

            pre_optimizer.zero_grad()

            svhn_out, _ = model(svhn_images, alpha=lambda_p)
            loss = criterion(svhn_out, svhn_labels)

            loss.backward()
            pre_optimizer.step()

            total_svhn_loss += loss.item()

            svhn_correct = (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()
            total_svhn_correct += svhn_correct
            total_samples += svhn_labels.size(0)

            i += 1

        svhn_acc_epoch = (total_svhn_correct / total_samples) * 100
        svhn_loss_epoch = total_svhn_loss / total_samples

        print(f"Pre Epoch {epoch + 1} | "
              f"SVHN Acc: {svhn_acc_epoch:.2f}%, Loss: {svhn_loss_epoch:.6f}"
              )

    print("Pretraining done")
    """

    for epoch in range(num_epochs):
        model.train()
        total_mnist_domain_loss, total_svhn_domain_loss, total_domain_loss = 0, 0, 0
        total_mnist_domain_correct, total_svhn_domain_correct = 0, 0
        total_label_loss = 0
        total_mnist_correct, total_svhn_correct = 0, 0
        total_loss, total_samples = 0, 0

        for i, (mnist_data, svhn_data) in enumerate(zip(mnist_loader, svhn_loader)):
            p = epoch / num_epochs
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
            # lambda_p = 0.01

            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 1, dtype=torch.long, device=device)

            optimizer.zero_grad()

            mnist_out, mnist_domain_out = model(mnist_images, alpha=lambda_p)
            svhn_out, svhn_domain_out = model(svhn_images, alpha=lambda_p)

            label_loss = criterion(svhn_out, svhn_labels)

            mnist_domain_loss = criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = criterion(svhn_domain_out, svhn_dlabels)
            domain_loss = mnist_domain_loss + svhn_domain_loss

            loss = label_loss + domain_loss

            loss.backward()
            optimizer.step()

            total_label_loss += label_loss.item()
            total_loss += loss.item()

            total_domain_loss += domain_loss.item()
            total_mnist_domain_loss += mnist_domain_loss.item()
            total_svhn_domain_loss += svhn_domain_loss.item()

            total_mnist_correct += (torch.argmax(mnist_out, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()

            total_samples += svhn_labels.size(0)

        # scheduler.step()

        current_lr_feature = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']
        current_lr_discriminator = optimizer.param_groups[2]['lr']

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples
        domain_avg_loss = total_domain_loss / (total_samples * 2)

        label_avg_loss = total_label_loss / total_samples
        total_avg_loss = total_loss / total_samples

        mnist_acc_epoch = total_mnist_correct / total_samples * 100
        svhn_acc_epoch = total_svhn_correct / total_samples * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Label Loss: {label_avg_loss:.8f} | '
              f'Domain Loss: {domain_avg_loss:.8f} | '
              f'Total Loss: {total_avg_loss:.8f} | '
        )
        print(f'MNIST Domain Loss: {mnist_domain_avg_loss:.8f} | '
              f'SVHN Domain Loss: {svhn_domain_avg_loss:.8f} | '
        )
        print(f'MNIST Acc: {mnist_acc_epoch:.3f}% | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
              f'MNIST Domain Acc: {mnist_domain_acc_epoch:.3f}% | '
              f'SVHN Domain Acc: {svhn_domain_acc_epoch:.3f}% | '
              )

        wandb.log({
            'Train/Label Loss': label_avg_loss,
            'Train/Domain MNIST Loss': mnist_domain_avg_loss,
            'Train/Domain SVHN Loss': svhn_domain_avg_loss,
            'Train/Domain Loss': domain_avg_loss,
            'Train/Total Loss': total_avg_loss,
            'Train/MNIST Label Accuracy': mnist_acc_epoch,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Train/MNIST Domain Accuracy': mnist_domain_acc_epoch,
            'Train/SVHN Domain Accuracy': svhn_domain_acc_epoch,
            'Parameter/LR_FeatureExtractor': current_lr_feature,
            'Parameter/LR_Classifier': current_lr_classifier,
            'Parameter/LR_Discriminator': current_lr_discriminator,
            'Parameter/lambda_p': lambda_p,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            total_label_loss, total_domain_loss = 0.0, 0.0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                batch_domain_labels = torch.full((images.size(0),), domain_label, dtype=torch.long, device=device)

                class_output, domain_output = model(images, alpha=0.0)

                label_loss = criterion(class_output, labels)
                domain_loss = criterion(domain_output, batch_domain_labels)

                total_label_loss += label_loss.item() * images.size(0)
                total_domain_loss += domain_loss.item() * images.size(0)

                label_correct += (torch.argmax(class_output, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == batch_domain_labels).sum().item()
                total += images.size(0)

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            label_loss_avg = total_label_loss / total
            domain_loss_avg = total_domain_loss / total

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
                f'Test/Label {group} Loss': label_loss_avg,
                f'Test/Domain {group} Loss': domain_loss_avg,
            }, step=epoch + 1)

            print(
                f'Test {group} | Label Acc: {label_acc:.3f}% | Label Loss: {label_loss_avg:.8f} | Domain Acc: {domain_acc:.3f}% | Domain Loss: {domain_loss_avg:.8f}')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST', 0)
            tester(svhn_loader_test, 'SVHN', 1)

if __name__ == '__main__':
    main()
