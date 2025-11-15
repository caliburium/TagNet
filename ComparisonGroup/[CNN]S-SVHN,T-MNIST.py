import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.data_loader import data_loader
from model.SimpleCNN import SimpleCNN, cnn_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--fc_weight', type=float, default=1.0)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb.init(entity="hails",
               project="TagNet - MSC",
               config=args.__dict__,
               name="[CNN]S:SVHN/T:MNIST_"
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_epoch:" + str(args.epoch)
               )

    _, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)

    print("Data load complete, start training")

    def lr_lambda(epoch):
        alpha = 10
        beta = 0.75
        p = epoch / num_epochs

        return (1 + alpha * p) ** (-beta)

    model = SimpleCNN(hidden_size=args.hidden_size).to(device)
    param = cnn_weights(model, args.lr, args.feature_weight, args.fc_weight)
    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    # optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.opt_decay)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_label_loss = 0
        total_svhn_correct = 0
        total_loss, total_samples = 0, 0

        for i, svhn_data in enumerate(svhn_loader):
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)

            optimizer.zero_grad()
            svhn_out = model(svhn_images)
            loss = criterion(svhn_out, svhn_labels)
            loss.backward()
            optimizer.step()

            total_label_loss += loss.item()
            total_loss += loss.item()

            total_svhn_correct += (torch.argmax(svhn_out, dim=1) == svhn_labels).sum().item()

            total_samples += svhn_labels.size(0)

        # scheduler.step()

        current_lr_feature = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']

        label_avg_loss = total_label_loss / total_samples
        total_avg_loss = total_loss / total_samples

        svhn_acc_epoch = total_svhn_correct / total_samples * 100


        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Label Loss: {label_avg_loss:.8f} | '
              f'SVHN Acc: {svhn_acc_epoch:.3f}% | '
        )

        wandb.log({
            'Train/Label Loss': label_avg_loss,
            'Train/Total Loss': total_avg_loss,
            'Train/SVHN Label Accuracy': svhn_acc_epoch,
            'Parameter/LR_FeatureExtractor': current_lr_feature,
            'Parameter/LR_Classifier': current_lr_classifier,
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group):
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
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Label {group} Loss': label_loss_avg,
            }, step=epoch + 1)

            print(
                f'Test {group} | Label Acc: {label_acc:.3f}% | Label Loss: {label_loss_avg:.8f}')

        with torch.no_grad():
            tester(mnist_loader_test, 'MNIST')
            tester(svhn_loader_test, 'SVHN')

if __name__ == '__main__':
    main()
