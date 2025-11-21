import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR, LinearLR
from torchvision.models import alexnet, AlexNet_Weights
from thop import profile
from dataloader.DomainNetLoader import dn_loader
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineAlexNet(nn.Module):
    def __init__(self, num_classes=10, fc_hidden=512):
        super(BaselineAlexNet, self).__init__()

        base = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = base.features

        feat_dim = 256 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_epoch(mode, model, loaders, criterion, optimizer_c, optimizer_f, epoch, args):
    if mode == 'train':
        model.train()
        prefix = "Train"
        context = torch.enable_grad()
    else:  # 'test'
        model.eval()
        prefix = "Test"
        context = torch.no_grad()

    stats = {domain: {'loss': 0, 'correct': 0, 'samples': 0}
             for domain in ['human', 'mammal', 'tool', 'cloth', 'electricity']}

    total_loss_sum = 0
    total_samples_sum = 0

    with context:
        if mode == 'train':
            for i, (human_data, mammal_data, tool_data, cloth_data, elec_data) in enumerate(
                    zip(loaders['human'], loaders['mammal'], loaders['tool'], loaders['cloth'],
                        loaders['electricity'])):

                data_map = {
                    'human': human_data, 'mammal': mammal_data, 'tool': tool_data,
                    'cloth': cloth_data, 'electricity': elec_data
                }

                loss_batch_total = 0
                optimizer_c.zero_grad()
                optimizer_f.zero_grad()

                for key, val in data_map.items():
                    images, labels = val[0].to(device), val[1].to(device)
                    bs = images.size(0)

                    out = model(images)
                    loss = criterion(out, labels)
                    loss_batch_total += loss

                    stats[key]['loss'] += loss.item() * bs
                    stats[key]['correct'] += (torch.argmax(out, dim=1) == labels).sum().item()
                    stats[key]['samples'] += bs

                loss_batch_total.backward()
                optimizer_c.step()
                optimizer_f.step()

                total_loss_sum += loss_batch_total.item()

                if i % 10 == 0:
                    print(f"--- [Epoch {epoch + 1}, Batch {i}] Train Stats ---")

        else:
            for key, loader in loaders.items():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    bs = images.size(0)

                    out = model(images)
                    loss = criterion(out, labels)

                    stats[key]['loss'] += loss.item() * bs
                    stats[key]['correct'] += (torch.argmax(out, dim=1) == labels).sum().item()
                    stats[key]['samples'] += bs

                    total_loss_sum += loss.item()

    log_data = {}
    total_samples_all = sum([stats[k]['samples'] for k in stats])
    total_correct_all = 0

    for key in ['human', 'mammal', 'tool', 'cloth', 'electricity']:
        n_samples = stats[key]['samples']
        if n_samples == 0: continue

        acc = stats[key]['correct'] / n_samples * 100
        loss_avg = stats[key]['loss'] / n_samples

        log_data[f'{prefix}/Acc Label {key.upper()}'] = acc
        log_data[f'{prefix} Loss/Label {key.upper()}'] = loss_avg

        total_correct_all += stats[key]['correct']


    avg_acc = total_correct_all / total_samples_all * 100
    avg_total_loss = total_loss_sum / (total_samples_all if mode == 'test' else (total_samples_all / 5))

    log_data[f'{prefix} Loss/Total Avg'] = avg_total_loss
    log_data[f'Parameters/Acc Label Avg'] = avg_acc

    print(f'Epoch [{epoch + 1}/{args.epoch}] ({prefix})')
    print(f'  [LabelAcc] Avg: {avg_acc:<6.2f}% | H: {log_data[f"{prefix}/Acc Label HUMAN"]:<5.1f}% | M: {log_data[f"{prefix}/Acc Label MAMMAL"]:<5.1f}% | T: {log_data[f"{prefix}/Acc Label TOOL"]:<5.1f}%')

    ind_accs = {
        'avg': avg_acc
    }

    return log_data, ind_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--anneal_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--fc_hidden', type=int, default=512)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb_run = wandb.init(project="TagNet - DomainNet5",
                           config=args.__dict__,
                           entity="hails",
                           name="[AlexNet]DN5_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_FCL:" + str(args.fc_hidden)
                           )

    print("Loading DomainNet datasets...")

    # face eye nose mouth brain skull foot leg arm hand
    humanbody_real_train, humanbody_real_test \
        = dn_loader('real', [108, 106, 198, 193, 40, 264, 123, 168, 9, 137], args.batch_size)

    # dog, tiger, sheep, elephant, horse, cat, monkey, lion, pig
    mammal_paint_train, mammal_paint_test \
        = dn_loader('painting', [91, 311, 343, 258, 103, 147, 64, 186, 174, 222], args.batch_size)

    # nail, sword, bottlecap, basket, rifle, bandage, pliers, axe, paintcan, anvil
    tool_paint_train, tool_paint_test \
        = dn_loader('painting', [196, 299, 37, 18, 243, 14, 226, 11, 206, 7], args.batch_size)

    # shoe, sock, bracelet, wristwatch, bowtie, hat, eyeglasses, sweater, pants, underwear
    cloth_quickdraw_train, cloth_quickdraw_test \
        = dn_loader('quickdraw', [259, 274, 39, 341, 38, 139, 107, 297, 209, 329], args.batch_size)

    # toaster, headphones, washing machine, light bulb, television, telephone, keyboard, laptop, stereo, camera
    electricity_real_train, electricity_real_test \
        = dn_loader('real', [312, 140, 333, 169, 305, 304, 161, 166, 285, 55], args.batch_size)

    train_loaders = {
        'human': humanbody_real_train,
        'mammal': mammal_paint_train,
        'tool': tool_paint_train,
        'cloth': cloth_quickdraw_train,
        'electricity': electricity_real_train
    }
    test_loaders = {
        'human': humanbody_real_test,
        'mammal': mammal_paint_test,
        'tool': tool_paint_test,
        'cloth': cloth_quickdraw_test,
        'electricity': electricity_real_test
    }

    print("Data load complete, start training")

    model = BaselineAlexNet(num_classes=args.num_classes, fc_hidden=args.fc_hidden).to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print("Calculating FLOPs...")
    flops, thop_params = profile(model, inputs=(dummy_input,), verbose=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    inference_params = total_params

    print(f"Model Parameters (Total): {total_params / 1e3:.3f}K")
    print(f"Model Parameters (Trainable): {trainable_params / 1e3:.3f}K")
    print(f"Model Parameters (Inference Active): {inference_params / 1e3:.3f}K")
    print(f"Model FLOPs: {flops} FLOPs")

    wandb.log({
        "Parameters/FLOPs Train": flops,
        "Parameters/FLOPs Inference": flops,
        "Parameters/Total Params (K)": total_params / 1e3,
        "Parameters/Trainable Params (K)": trainable_params / 1e3,
        "Parameters/Inference Params (K)": inference_params / 1e3
    }, step=0)

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 1
    min_save_epoch = 1

    optimizer_c = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    optimizer_f = optim.Adam(model.features.parameters(), lr=args.lr, weight_decay=args.opt_decay)

    scheduler_cos_f = CosineAnnealingLR(optimizer_f, T_max=args.anneal_epochs, eta_min=1e-15)
    scheduler_const_f = ConstantLR(optimizer_f, factor=1e-9, total_iters=args.epoch)
    scheduler_f = SequentialLR(optimizer_f,
                               schedulers=[scheduler_cos_f, scheduler_const_f],
                               milestones=[args.anneal_epochs])

    scheduler_warmup_c = LinearLR(optimizer_c, start_factor=1e-15, end_factor=1.0, total_iters=args.anneal_epochs)
    scheduler_const_c = ConstantLR(optimizer_c, factor=1.0, total_iters=args.epoch)
    scheduler_c = SequentialLR(optimizer_c,
                               schedulers=[scheduler_warmup_c, scheduler_const_c],
                               milestones=[args.anneal_epochs])

    criterion = nn.CrossEntropyLoss()

    wandb.log({
        'Parameters/LR_feature': args.lr,
        'Parameters/LR_classifier': args.lr,
    }, step=0)

    for epoch in range(num_epochs):
        train_logs, _ = run_epoch(
            'train', model, train_loaders, criterion, optimizer_c, optimizer_f, epoch, args
        )

        test_logs, test_accs = run_epoch(
            'test', model, test_loaders, criterion, None, None, epoch, args
        )

        all_logs = {**train_logs, **test_logs}
        current_lr_f = scheduler_f.get_last_lr()[0]
        current_lr_c = scheduler_c.get_last_lr()[0]
        all_logs['Parameters/LR_feature'] = current_lr_f
        all_logs['Parameters/LR_classifier'] = current_lr_c

        wandb.log(all_logs, step=epoch + 1)

        current_avg_acc = test_accs['avg']
        if (epoch + 1) >= min_save_epoch and current_avg_acc > best_avg_acc:
            best_avg_acc = current_avg_acc
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"*** New best model saved to {save_path} (Epoch: {epoch + 1}, Avg Acc: {current_avg_acc:.2f}%) ***")
            wandb.log({"Parameters/Best Avg Accuracy": best_avg_acc}, step=epoch + 1)

        if (epoch + 1) % save_interval == 0:
            periodic_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), periodic_save_path)

        scheduler_c.step()
        scheduler_f.step()

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()