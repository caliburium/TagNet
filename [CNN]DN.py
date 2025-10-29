import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.DomainNetLoader import dn_loader
from model.SimpleCNN import SimpleCNN224, cnn_weights

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_beta', type=float, default=0.25)
    args = parser.parse_args()

    num_epochs = args.epoch
    wandb.init(entity="hails",
               project="TagNet - DomainNet",
               config=args.__dict__,
               name="[CNN]DN_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_Adam"
               )

    # quickdraw, real, painting, sketch, clipart, infograph
    # 1. Furniture Domain Load
    furn_classes = [246, 110, 15, 213, 139, 299, 58, 102, 47, 48]
    furn_real_train, furn_real_test = dn_loader('real', furn_classes, args.batch_size)

    # 2. Mammal Domain Load
    mammal_classes = [61, 292, 81, 148, 319, 157, 83, 188, 312, 89]
    mammal_real_train, mammal_real_test = dn_loader('real', mammal_classes, args.batch_size)

    # 3. Tool Domain Load
    tool_classes = [314, 131, 227, 12, 40, 249, 10, 237, 207, 276]
    tool_real_train, tool_real_test = dn_loader('real', tool_classes, args.batch_size)

    # mammal_paint_train, mammal_paint_test = dn_loader('painting', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)
    # tool_painting_train, tool_painting_test = dn_loader('painting', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)


    print("Data load complete, start training")

    def lr_lambda(progress):
        alpha = args.lr_alpha
        beta = args.lr_beta
        return (1 + alpha * progress) ** (-beta)

    model = SimpleCNN224(num_classes=10, hidden_size=args.hidden_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_furn_loss, total_mammal_loss, total_tool_loss, total_loss = 0, 0, 0, 0
        total_furn_correct, total_mammal_correct, total_tool_correct = 0, 0, 0
        total_samples = 0

        for furn_data, mammal_data, tool_data in zip(furn_real_train, mammal_real_train, tool_real_train):
            furn_images, furn_labels = furn_data
            furn_images, furn_labels = furn_images.to(device), furn_labels.to(device)

            mammal_images, mammal_labels = mammal_data
            mammal_images, mammal_labels = mammal_images.to(device), mammal_labels.to(device)

            tool_images, tool_labels = tool_data
            tool_images, tool_labels = tool_images.to(device), tool_labels.to(device)

            furn_outputs = model(furn_images)
            mammal_outputs = model(mammal_images)
            tool_outputs = model(tool_images)

            furn_loss = criterion(furn_outputs, furn_labels)
            mammal_loss = criterion(mammal_outputs, mammal_labels)
            tool_loss = criterion(tool_outputs, tool_labels)

            loss = furn_loss + mammal_loss + tool_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_furn_loss += furn_loss.item() * furn_labels.size(0)
            total_mammal_loss += mammal_loss.item() * mammal_labels.size(0)
            total_tool_loss += tool_loss.item() * tool_labels.size(0)
            total_loss += loss.item() * (furn_labels.size(0) + mammal_labels.size(0) + tool_labels.size(0))

            furn_correct = (torch.argmax(furn_outputs, dim=1) == furn_labels).sum().item()
            mammal_correct = (torch.argmax(mammal_outputs, dim=1) == mammal_labels).sum().item()
            tool_correct = (torch.argmax(tool_outputs, dim=1) == tool_labels).sum().item()

            total_furn_correct += furn_correct
            total_mammal_correct += mammal_correct
            total_tool_correct += tool_correct

            total_samples += furn_labels.size(0)

            i += 1

        # scheduler.step()

        furn_acc_epoch = total_furn_correct / total_samples * 100
        mammal_acc_epoch = total_mammal_correct / total_samples * 100
        tool_acc_epoch = total_tool_correct / total_samples * 100

        furn_avg_loss = total_furn_loss / total_samples
        mammal_avg_loss = total_mammal_loss / total_samples
        tool_avg_loss = total_tool_loss / total_samples
        total_avg_loss = total_loss / (total_samples * 3)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], | '
            f'Total Loss: {total_avg_loss:.4f}, | '
            f'Furn Loss: {furn_avg_loss:.4f}, | '
            f'Mammal Loss: {mammal_avg_loss:.4f}, | '
            f'Tool Loss: {tool_avg_loss:.4f} | '
        )

        print(
            f'Furn Acc: {furn_acc_epoch:.3f}% | '
            f'Mammal Acc: {mammal_acc_epoch:.3f}% | '
            f'Tool Acc: {tool_acc_epoch:.3f}% | '
        )

        wandb.log({
            'Train/Label Loss': total_avg_loss,
            'Train/Furniture Label Loss': furn_avg_loss,
            'Train/Mammal Label Loss': mammal_avg_loss,
            'Train/Tool Label Loss': tool_avg_loss,
            'Train/Furniture Label Accuracy': furn_acc_epoch,
            'Train/Mammal Label Accuracy': mammal_acc_epoch,
            'Train/Tool Label Accuracy': tool_acc_epoch,
        }, step=epoch)

        model.eval()

        def tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                total += labels.size(0)
                correct += (torch.argmax(class_output, dim=1) == labels).sum().item()

            accuracy = correct / total * 100
            wandb.log({f'Test/Label {group} Accuracy': accuracy}, step=epoch)
            print(f'Test {group} | Label Acc: {accuracy:.3f}%')

        with torch.no_grad():
            tester(furn_real_test, 'Furniture')
            tester(mammal_real_test, 'Mammal')
            tester(tool_real_test, 'Tool')


if __name__ == '__main__':
    main()