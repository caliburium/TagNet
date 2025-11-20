import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR, LinearLR
from functions.GumbelTauScheduler import GumbelTauScheduler
from thop import profile
from model.TagNet import TagNet_Alex
from dataloader.DomainNetLoader import dn_loader
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_label_partition_log_data(label_partition_counts, task_name, num_classes, num_tasks, prefix):
    log_data = {}
    total_counts_per_label = label_partition_counts.sum(dim=1)

    for label_idx in range(num_classes):
        total_for_label = total_counts_per_label[label_idx].item()
        for part_idx in range(num_tasks):
            count = label_partition_counts[label_idx, part_idx].item()

            if total_for_label > 0:
                percentage = (count / total_for_label) * 100
            else:
                percentage = 0.0

            log_key = f"Partition {prefix} {task_name}/Label:{label_idx}/Partition:{part_idx}"
            log_data[log_key] = percentage
    return log_data

def update_label_partition_counts(label_partition_counts, labels, partition_indices, num_classes):
    for l_idx in range(labels.size(0)):
        label_val = labels[l_idx].item()
        if 0 <= label_val < num_classes:
            label_partition_counts[label_val, partition_indices[l_idx].item()] += 1

def run_epoch(mode, model, loaders, criterion, optimizer_c, optimizer_f, epoch, args, tau):
    if mode == 'train':
        model.train()
        prefix = "Train"
        context = torch.enable_grad()
    else:  # 'test'
        model.eval()
        prefix = "Test"
        context = torch.no_grad()

    total_mnist_task_loss, total_mnist_task_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
    total_kmnist_task_loss, total_kmnist_task_correct, total_kmnist_loss, total_kmnist_correct = 0, 0, 0, 0
    total_fmnist_task_loss, total_fmnist_task_correct, total_fmnist_loss, total_fmnist_correct = 0, 0, 0, 0
    total_task_loss, total_label_loss, total_loss, single_task_samples = 0, 0, 0, 0

    mnist_partition_counts = torch.zeros(args.num_tasks, device=device)
    kmnist_partition_counts = torch.zeros(args.num_tasks, device=device)
    fmnist_partition_counts = torch.zeros(args.num_tasks, device=device)

    mnist_label_partition_counts = torch.zeros(args.num_classes, args.num_tasks, device=device)
    kmnist_label_partition_counts = torch.zeros(args.num_classes, args.num_tasks, device=device)
    fmnist_label_partition_counts = torch.zeros(args.num_classes, args.num_tasks, device=device)

    with context:
        for i, (mnist_data, kmnist_data, fmnist_data) in enumerate(zip(loaders['mnist'], loaders['kmnist'], loaders['fmnist'])):
            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            kmnist_images, kmnist_labels = kmnist_data
            kmnist_images, kmnist_labels = kmnist_images.to(device), kmnist_labels.to(device)
            fmnist_images, fmnist_labels = fmnist_data
            fmnist_images, fmnist_labels = fmnist_images.to(device), fmnist_labels.to(device)

            bs = mnist_images.size(0)
            bs_all = bs * 3

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            kmnist_dlabels = torch.full((kmnist_images.size(0),), 1, dtype=torch.long, device=device)
            fmnist_dlabels = torch.full((fmnist_images.size(0),), 2, dtype=torch.long, device=device)

            if mode == 'train':
                optimizer_c.zero_grad()
                optimizer_f.zero_grad()

            is_inference = (mode == 'test')

            out_label_mnist, out_task_mnist, prob_mnist = model(mnist_images, tau=tau, inference=is_inference)
            out_label_kmnist, out_task_kmnist, prob_kmnist = model(kmnist_images, tau=tau, inference=is_inference)
            out_label_fmnist, out_task_fmnist, prob_fmnist = model(fmnist_images,tau=tau, inference=is_inference)

            part_idx_mnist = torch.argmax(prob_mnist, dim=1)
            part_idx_kmnist = torch.argmax(prob_kmnist, dim=1)
            part_idx_fmnist = torch.argmax(prob_fmnist, dim=1)
            if i % 10 == 0 and mode == 'train':
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(part_idx_mnist, minlength=args.num_tasks)
                kmnist_counts = torch.bincount(part_idx_kmnist, minlength=args.num_tasks)
                fmnist_counts = torch.bincount(part_idx_fmnist, minlength=args.num_tasks)
                print(f"MNIST : {mnist_counts.cpu().numpy()} / KMNIST : {kmnist_counts.cpu().numpy()} / FMNIST : {fmnist_counts.cpu().numpy()}")

            update_label_partition_counts(mnist_label_partition_counts, mnist_labels, part_idx_mnist, args.num_classes)
            update_label_partition_counts(kmnist_label_partition_counts, kmnist_labels, part_idx_kmnist, args.num_classes)
            update_label_partition_counts(fmnist_label_partition_counts, fmnist_labels, part_idx_fmnist, args.num_classes)

            label_loss_mnist = criterion(out_label_mnist, mnist_labels)
            label_loss_kmnist = criterion(out_label_kmnist, kmnist_labels)
            label_loss_fmnist = criterion(out_label_fmnist, fmnist_labels)
            label_loss = label_loss_mnist + label_loss_kmnist + label_loss_fmnist

            mnist_task_loss = criterion(out_task_mnist, mnist_dlabels)
            kmnist_task_loss = criterion(out_task_kmnist, kmnist_dlabels)
            fmnist_task_loss = criterion(out_task_fmnist, fmnist_dlabels)
            task_loss = mnist_task_loss + kmnist_task_loss + fmnist_task_loss

            loss = task_loss + label_loss

            if mode == 'train':
                loss.backward()
                optimizer_c.step()
                optimizer_f.step()

            mnist_partition_counts += torch.bincount(part_idx_mnist, minlength=args.num_tasks).to(device)
            kmnist_partition_counts += torch.bincount(part_idx_kmnist, minlength=args.num_tasks).to(device)
            fmnist_partition_counts += torch.bincount(part_idx_fmnist, minlength=args.num_tasks).to(device)

            total_label_loss += label_loss.item() * bs_all
            total_mnist_loss += label_loss_mnist.item() * bs
            total_kmnist_loss += label_loss_kmnist.item() * bs
            total_fmnist_loss += label_loss_fmnist.item() * bs

            total_task_loss += task_loss.item() * bs_all
            total_mnist_task_loss += mnist_task_loss.item() * bs
            total_kmnist_task_loss += kmnist_task_loss.item() * bs
            total_fmnist_task_loss += fmnist_task_loss.item() * bs

            total_loss += loss.item() * bs_all

            total_mnist_correct += (torch.argmax(out_label_mnist, dim=1) == mnist_labels).sum().item()
            total_kmnist_correct += (torch.argmax(out_label_kmnist, dim=1) == kmnist_labels).sum().item()
            total_fmnist_correct += (torch.argmax(out_label_fmnist, dim=1) == fmnist_labels).sum().item()

            total_mnist_task_correct += (torch.argmax(out_task_mnist, dim=1) == mnist_dlabels).sum().item()
            total_kmnist_task_correct += (torch.argmax(out_task_kmnist, dim=1) == kmnist_dlabels).sum().item()
            total_fmnist_task_correct += (torch.argmax(out_task_fmnist, dim=1) == fmnist_dlabels).sum().item()

            single_task_samples += bs

    mnist_train_partition_log = get_label_partition_log_data(
        mnist_label_partition_counts, 'MNIST', args.num_classes, args.num_tasks, prefix=prefix)
    kmnist_train_partition_log = get_label_partition_log_data(
        kmnist_label_partition_counts, 'KMNIST', args.num_classes, args.num_tasks, prefix=prefix)
    fmnist_train_partition_log = get_label_partition_log_data(
        fmnist_label_partition_counts, 'FMNIST', args.num_classes, args.num_tasks, prefix=prefix)

    mnist_partition_ratios = mnist_partition_counts / single_task_samples * 100
    kmnist_partition_ratios = kmnist_partition_counts / single_task_samples * 100
    fmnist_partition_ratios = fmnist_partition_counts / single_task_samples * 100

    mnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_tasks)])
    kmnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {kmnist_partition_ratios[p]:.2f}%" for p in range(args.num_tasks)])
    fmnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {fmnist_partition_ratios[p]:.2f}%" for p in range(args.num_tasks)])

    mnist_task_avg_loss = total_mnist_task_loss / single_task_samples
    kmnist_task_avg_loss = total_kmnist_task_loss / single_task_samples
    fmnist_task_avg_loss = total_fmnist_task_loss / single_task_samples
    task_avg_loss = total_task_loss / (single_task_samples * 3)

    mnist_avg_loss = total_mnist_loss / single_task_samples
    kmnist_avg_loss = total_kmnist_loss / single_task_samples
    fmnist_avg_loss = total_fmnist_loss / single_task_samples
    label_avg_loss = total_label_loss / (single_task_samples * 3)

    total_avg_loss = total_loss / (single_task_samples * 3)

    mnist_acc_epoch = total_mnist_correct / single_task_samples * 100
    kmnist_acc_epoch = total_kmnist_correct / single_task_samples * 100
    fmnist_acc_epoch = total_fmnist_correct / single_task_samples * 100

    mnist_task_acc_epoch = total_mnist_task_correct / single_task_samples * 100
    kmnist_task_acc_epoch = total_kmnist_task_correct / single_task_samples * 100
    fmnist_task_acc_epoch = total_fmnist_task_correct / single_task_samples * 100

    avg_label_acc = (mnist_acc_epoch + kmnist_acc_epoch + fmnist_acc_epoch) / 3.0

    print(f'Epoch [{epoch + 1}/{args.epoch}] ({prefix}) | Tau: {tau:<5.3f}')
    print(f'  [Ratios] MNIST: [{mnist_partition_ratio_str}] | KMNIST: [{kmnist_partition_ratio_str}] | FMNIST: [{fmnist_partition_ratio_str}]')
    print(f'  [LabelAcc]MNIST: {mnist_acc_epoch:<6.2f}% | KMNIST: {kmnist_acc_epoch:<6.2f}% | FMNIST: {fmnist_acc_epoch:<6.2f}% | Avg: {avg_label_acc:<6.2f}%')
    print(f'  [TaskAcc] MNIST: {mnist_task_acc_epoch:<6.2f}% | KMNIST: {kmnist_task_acc_epoch:<6.2f}% | FMNIST: {fmnist_task_acc_epoch:<6.2f}%')
    print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {task_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
    print(f'  [LabelLoss]MNIST: {mnist_avg_loss:<6.4f} | KMNIST: {kmnist_avg_loss:<6.4f} | FMNIST: {fmnist_avg_loss:<6.4f}')
    print(f'  [TaskLoss] MNIST: {mnist_task_avg_loss:<6.4f} | KMNIST: {kmnist_task_avg_loss:<6.4f} | FMNIST: {fmnist_task_avg_loss:<6.4f}')

    log_data = {
        **{f"{prefix}/Partition {p} MNIST Ratio": mnist_partition_ratios[p].item() for p in range(args.num_tasks)},
        **{f"{prefix}/Partition {p} KMNIST Ratio": kmnist_partition_ratios[p].item() for p in range(args.num_tasks)},
        **{f"{prefix}/Partition {p} FMNIST Ratio": fmnist_partition_ratios[p].item() for p in range(args.num_tasks)},

        f'{prefix}/Acc Label MNIST': mnist_acc_epoch,   f'{prefix} Loss/Label MNIST': mnist_avg_loss,
        f'{prefix}/Acc Label KMNIST': kmnist_acc_epoch, f'{prefix} Loss/Label KMNIST': kmnist_avg_loss,
        f'{prefix}/Acc Label FMNIST': fmnist_acc_epoch, f'{prefix} Loss/Label FMNIST': fmnist_avg_loss,
        f'{prefix}/Acc Task MNIST': mnist_task_acc_epoch,   f'{prefix} Loss/Task MNIST': mnist_task_avg_loss,
        f'{prefix}/Acc Task KMNIST': kmnist_task_acc_epoch, f'{prefix} Loss/Task KMNIST': kmnist_task_avg_loss,
        f'{prefix}/Acc Task FMNIST': fmnist_task_acc_epoch, f'{prefix} Loss/Task FMNIST': fmnist_task_avg_loss,
        f'{prefix} Loss/Label Avg': label_avg_loss, f'{prefix} Loss/Task Avg': task_avg_loss,
        f'{prefix} Loss/Total': total_avg_loss, f'Parameters/Acc Label Avg': avg_label_acc,

        **mnist_train_partition_log,
        **kmnist_train_partition_log,
        **fmnist_train_partition_log,
    }

    ind_accs = {
        'mnist': mnist_acc_epoch,
        'kmnist': kmnist_acc_epoch,
        'fmnist': fmnist_acc_epoch,
        'avg': avg_label_acc
    }

    return log_data, ind_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--anneal_epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_tasks', type=int, default=3)
    parser.add_argument('--fc_hidden', type=int, default=128)
    parser.add_argument('--disc_hidden', type=int, default=128)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=3.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.98)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb_run = wandb.init(project="TagNet - 3MNIST",
                           config=args.__dict__,
                           entity="hails",
                           name="[Tagnet]3MNIST_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_FCL:" + str(args.fc_hidden)
                           )


    # face eye nose mouth brain skull foot leg arm hand
    humanbody_real_train, humanbody_real_test = dn_loader('real', [108, 106, 198, 193, 40, 264, 123, 168, 9, 137], args.batch_size)

    # dog, tiger, sheep, elephant, horse, cat, monkey, lion, pig
    mammal_paint_train, mammal_paint_test = dn_loader('painting', [91, 311, 343, 258, 103, 147, 64, 186, 174, 222], args.batch_size)

    # nail, sword, bottlecap, basket, rifle, bandage, pliers, axe, paintcan, anvil
    tool_paint_train, tool_paint_test = dn_loader('painting', [196, 299, 37, 18, 243, 14, 226, 11, 206, 7], args.batch_size)

    # shoe, sock, bracelet, wristwatch, bowtie, hat, eyeglasses, sweater, pants, underwear
    cloth_real_train, cloth_real_test = dn_loader('real', [259, 274, 39, 341, 38, 139, 107, 297, 209, 329], args.batch_size)

    # toaster, headphones, washing machine, light bulb, television, telephone, keyboard, laptop, stereo, camera
    electricity_quickdraw_train, electricity_quickdraw_test = dn_loader('quickdraw', [312, 140, 333, 169, 305, 304, 161, 166, 285, 55], args.batch_size)



    train_loaders = {
        'mnist': numbers_loader,
        'kmnist': gana_loader,
        'fmnist': fashion_loader
    }
    test_loaders = {
        'mnist': numbers_loader_test,
        'kmnist': gana_loader_test,
        'fmnist': fashion_loader_test
    }

    print("Data load complete, start training")

    model = TagNet_Alex(num_classes=args.num_classes,
                        num_tasks=args.num_tasks,
                        fc_hidden=args.fc_hidden,
                        disc_hidden=args.disc_hidden,
                        device=device
                        )

    model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops_train, thop_params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_infer, _ = profile(model, inputs=(dummy_input, 0.1, True), verbose=False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_features = sum(p.numel() for p in model.features.parameters())
    params_discriminator = sum(p.numel() for p in model.discriminator.parameters())
    params_single_classifier = sum(p.numel() for p in model.classifiers[0].parameters())
    inference_params = params_features + params_discriminator + params_single_classifier

    print(f"Model Parameters (Trainable): {trainable_params / 1e3:.3f}K")
    print(f"Model Parameters (Inference Active): {inference_params / 1e3:.3f}K")
    print(f"Model FLOPs (Train): {flops_train} FLOPs")
    print(f"Model FLOPs (Inference): {flops_infer} FLOPs")

    wandb.log({
        "Parameters/FLOPs Train": flops_train,
        "Parameters/FLOPs Inference": flops_infer,
        "Parameters/Trainable Params (K)": trainable_params / 1e3,
        "Parameters/Inference Params (K)": inference_params / 1e3
    }, step=0)

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 1
    min_save_epoch = 1

    optimizer_c = optim.Adam(model.classifiers.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    optimizer_f = optim.Adam(list(model.features.parameters()) + list(model.discriminator.parameters()),
                             lr=args.lr, weight_decay=args.opt_decay)

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

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        tau = tau_scheduler.get_tau()
        train_logs, _ = run_epoch(
            'train', model, train_loaders, criterion, optimizer_c, optimizer_f
            , epoch, args, tau
        )

        test_logs, test_accs = run_epoch(
            'test', model, test_loaders, criterion, None, None
            , epoch, args, 0.1
        )

        all_logs = {**train_logs, **test_logs}
        current_lr_f = scheduler_f.get_last_lr()[0]
        current_lr_c = scheduler_c.get_last_lr()[0]
        all_logs['Parameters/LR_feature'] = current_lr_f
        all_logs['Parameters/LR_classifier'] = current_lr_c
        all_logs['Parameters/Tau'] = tau

        wandb.log(all_logs, step=epoch + 1)

        current_avg_acc = test_accs['avg']
        if (epoch + 1) >= min_save_epoch and current_avg_acc > best_avg_acc:
            best_avg_acc = current_avg_acc
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(
                f"*** New best model saved to {save_path} (Epoch: {epoch + 1}, Avg Acc: {current_avg_acc:.2f}%) ***")
            wandb.log({"Parameters/Best Avg Accuracy": best_avg_acc}, step=epoch + 1)

        if (epoch + 1) % save_interval == 0:
            periodic_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), periodic_save_path)
            print(f"--- Periodic checkpoint saved to {periodic_save_path} ---")

        tau_scheduler.step()
        scheduler_c.step()
        scheduler_f.step()

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()