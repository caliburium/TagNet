import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNet, TagNet_weights
from dataloader.data_loader import data_loader
import math
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_label_partition_log_data(label_partition_counts, domain_name, num_classes, num_partition, prefix):
    log_data = {}
    total_counts_per_label = label_partition_counts.sum(dim=1)

    for label_idx in range(num_classes):
        total_for_label = total_counts_per_label[label_idx].item()
        for part_idx in range(num_partition):
            count = label_partition_counts[label_idx, part_idx].item()

            if total_for_label > 0:
                percentage = (count / total_for_label) * 100
            else:
                percentage = 0.0

            log_key = f"Partition {prefix} {domain_name}/Label:{label_idx}/Partition:{part_idx}"
            log_data[log_key] = percentage
    return log_data


def run_epoch(mode, model, loaders, criterion, optimizer, epoch, args, tau, lambda_p):
    if mode == 'train':
        model.train()
        prefix = "Train"
        context = torch.enable_grad()
    else:  # 'test'
        model.eval()
        prefix = "Test"
        context = torch.no_grad()
        lambda_p = 0.0

    total_mnist_domain_loss, total_mnist_domain_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
    total_kmnist_domain_loss, total_kmnist_domain_correct, total_kmnist_loss, total_kmnist_correct = 0, 0, 0, 0
    total_fmnist_domain_loss, total_fmnist_domain_correct, total_fmnist_loss, total_fmnist_correct = 0, 0, 0, 0
    total_domain_loss, total_label_loss, total_loss = 0, 0, 0
    total_specialization_loss, total_diversity_loss = 0, 0

    mnist_partition_counts = torch.zeros(args.num_partition, device=device)
    kmnist_partition_counts = torch.zeros(args.num_partition, device=device)
    fmnist_partition_counts = torch.zeros(args.num_partition, device=device)

    total_samples_m, total_samples_k, total_samples_f = 0, 0, 0

    mnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    kmnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    fmnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

    mnist_iter = iter(loaders['mnist'])
    kmnist_iter = iter(loaders['kmnist'])
    fmnist_iter = iter(loaders['fmnist'])

    len_dataloader = min(len(loaders['mnist']), len(loaders['kmnist']), len(loaders['fmnist']))

    with context:
        for i in range(len_dataloader):
            try:
                mnist_images, mnist_labels = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(loaders['mnist'])
                mnist_images, mnist_labels = next(mnist_iter)

            try:
                kmnist_images, kmnist_labels = next(kmnist_iter)
            except StopIteration:
                kmnist_iter = iter(loaders['kmnist'])
                kmnist_images, kmnist_labels = next(kmnist_iter)
            try:
                fmnist_images, fmnist_labels = next(fmnist_iter)
            except StopIteration:
                fmnist_iter = iter(loaders['fmnist'])
                fmnist_images, fmnist_labels = next(fmnist_iter)

            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            kmnist_images, kmnist_labels = kmnist_images.to(device), kmnist_labels.to(device)
            fmnist_images, fmnist_labels = fmnist_images.to(device), fmnist_labels.to(device)

            bs_m = mnist_images.size(0)
            bs_k = kmnist_images.size(0)
            bs_f = fmnist_images.size(0)
            bs_all = bs_m + bs_k + bs_f

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            kmnist_dlabels = torch.full((kmnist_images.size(0),), 1, dtype=torch.long, device=device)
            fmnist_dlabels = torch.full((fmnist_images.size(0),), 2, dtype=torch.long, device=device)

            if mode == 'train':
                optimizer.zero_grad()

            is_inference = (mode == 'test')

            out_label_mnist, out_domain_mnist, part_idx_mnist, switcher_prob_mnist = model(
                mnist_images, grl_lambda=lambda_p, tau=tau, inference=is_inference)
            out_label_kmnist, out_domain_kmnist, part_idx_kmnist, switcher_prob_kmnist = model(
                kmnist_images, grl_lambda=lambda_p, tau=tau, inference=is_inference)
            out_label_fmnist, out_domain_fmnist, part_idx_fmnist, switcher_prob_fmnist = model(
                fmnist_images, grl_lambda=lambda_p, tau=tau, inference=is_inference)

            if i % 1 == 0 and mode == 'train':
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(part_idx_mnist, minlength=args.num_partition)
                kmnist_counts = torch.bincount(part_idx_kmnist, minlength=args.num_partition)
                fmnist_counts = torch.bincount(part_idx_fmnist, minlength=args.num_partition)
                print(
                    f"MNIST : {mnist_counts.cpu().numpy()} / KMNIST : {kmnist_counts.cpu().numpy()} / FMNIST : {fmnist_counts.cpu().numpy()}")
                print(
                    f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.8f}, Bias Mean: {model.partition_switcher.bias.data.mean():.8f}")

            for l_idx in range(mnist_labels.size(0)):
                label_val = mnist_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    mnist_label_partition_counts[label_val, part_idx_mnist[l_idx].item()] += 1

            for l_idx in range(kmnist_labels.size(0)):
                label_val = kmnist_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    kmnist_label_partition_counts[label_val, part_idx_kmnist[l_idx].item()] += 1

            for l_idx in range(fmnist_labels.size(0)):
                label_val = fmnist_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    fmnist_label_partition_counts[label_val, part_idx_fmnist[l_idx].item()] += 1

            label_loss_mnist = criterion(out_label_mnist, mnist_labels)
            label_loss_kmnist = criterion(out_label_kmnist, kmnist_labels)
            label_loss_fmnist = criterion(out_label_fmnist, fmnist_labels)
            label_loss = label_loss_mnist + label_loss_kmnist + label_loss_fmnist

            mnist_domain_loss = criterion(out_domain_mnist, mnist_dlabels)
            kmnist_domain_loss = criterion(out_domain_kmnist, kmnist_dlabels)
            fmnist_domain_loss = criterion(out_domain_fmnist, fmnist_dlabels)
            domain_loss = mnist_domain_loss + kmnist_domain_loss + fmnist_domain_loss

            epsilon = 1e-8
            # entropy list -> mean -> loss
            avg_prob_mnist = torch.mean(switcher_prob_mnist, dim=0)
            avg_prob_kmnist = torch.mean(switcher_prob_kmnist, dim=0)
            avg_prob_fmnist = torch.mean(switcher_prob_fmnist, dim=0)
            entropy_mnist = -torch.sum(avg_prob_mnist * torch.log(avg_prob_mnist + epsilon))
            entropy_kmnist = -torch.sum(avg_prob_kmnist * torch.log(avg_prob_kmnist + epsilon))
            entropy_fmnist = -torch.sum(avg_prob_fmnist * torch.log(avg_prob_fmnist + epsilon))

            loss_specialization = entropy_mnist + entropy_kmnist + entropy_fmnist

            if torch.isnan(loss_specialization):
                print('caution, specialization')

            all_probs = torch.cat((switcher_prob_mnist, switcher_prob_kmnist, switcher_prob_fmnist), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-9))

            if torch.isnan(loss_diversity):
                print('caution, diversity')

            entropy_loss = args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity

            loss = label_loss + domain_loss + entropy_loss
            # loss = domain_loss

            if mode == 'train':
                loss.backward()
                entries = []
                for name, param in model.partition_switcher.named_parameters():
                    if param.grad is None:
                        grad_str = 'None'
                    else:
                        grad_str = f"{torch.mean(torch.abs(param.grad)).item():.6f}"
                    data_str = f"{torch.mean(torch.abs(param.data)).item():.6f}"
                    entries.append(f"{name}: {grad_str}, {data_str}")
                print(" | ".join(entries) + f" | loss: {loss.item():.6f} ")

                optimizer.step()

            mnist_partition_counts += torch.bincount(part_idx_mnist, minlength=args.num_partition).to(device)
            kmnist_partition_counts += torch.bincount(part_idx_kmnist, minlength=args.num_partition).to(device)
            fmnist_partition_counts += torch.bincount(part_idx_fmnist, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item() * bs_all
            total_mnist_loss += label_loss_mnist.item() * bs_m
            total_kmnist_loss += label_loss_kmnist.item() * bs_k
            total_fmnist_loss += label_loss_fmnist.item() * bs_f

            total_domain_loss += domain_loss.item() * bs_all
            total_mnist_domain_loss += mnist_domain_loss.item() * bs_m
            total_kmnist_domain_loss += kmnist_domain_loss.item() * bs_k
            total_fmnist_domain_loss += fmnist_domain_loss.item() * bs_f

            total_specialization_loss += loss_specialization.item() * bs_all
            total_diversity_loss += loss_diversity.item() * bs_all
            total_loss += loss.item() * bs_all

            total_mnist_correct += (torch.argmax(out_label_mnist, dim=1) == mnist_labels).sum().item()
            total_kmnist_correct += (torch.argmax(out_label_kmnist, dim=1) == kmnist_labels).sum().item()
            total_fmnist_correct += (torch.argmax(out_label_fmnist, dim=1) == fmnist_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(out_domain_mnist, dim=1) == mnist_dlabels).sum().item()
            total_kmnist_domain_correct += (
                        torch.argmax(out_domain_kmnist, dim=1) == kmnist_dlabels).sum().item()
            total_fmnist_domain_correct += (
                        torch.argmax(out_domain_fmnist, dim=1) == fmnist_dlabels).sum().item()

            total_samples_m += bs_m
            total_samples_k += bs_k
            total_samples_f += bs_f

    if total_samples_m == 0: total_samples_m = 1
    if total_samples_k == 0: total_samples_k = 1
    if total_samples_f == 0: total_samples_f = 1

    total_samples_all = total_samples_m + total_samples_k + total_samples_f
    print(f'samples MNIST:{total_samples_m} / KMNIST:{total_samples_k} / FMNIST:{total_samples_f}')

    mnist_train_partition_log = get_label_partition_log_data(
        mnist_label_partition_counts, 'MNIST', args.num_classes, args.num_partition, prefix=prefix)
    kmnist_train_partition_log = get_label_partition_log_data(
        kmnist_label_partition_counts, 'KMNIST', args.num_classes, args.num_partition, prefix=prefix)
    fmnist_train_partition_log = get_label_partition_log_data(
        fmnist_label_partition_counts, 'FMNIST', args.num_classes, args.num_partition, prefix=prefix)

    mnist_partition_ratios = mnist_partition_counts / total_samples_m * 100
    kmnist_partition_ratios = kmnist_partition_counts / total_samples_k * 100
    fmnist_partition_ratios = fmnist_partition_counts / total_samples_f * 100

    mnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    kmnist_partition_ratio_str = " | ".join(  # º¯°æ
        [f"Partition {p}: {kmnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    fmnist_partition_ratio_str = " | ".join(  # º¯°æ
        [f"Partition {p}: {fmnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

    mnist_domain_avg_loss = total_mnist_domain_loss / total_samples_m
    kmnist_domain_avg_loss = total_kmnist_domain_loss / total_samples_k
    fmnist_domain_avg_loss = total_fmnist_domain_loss / total_samples_f
    domain_avg_loss = total_domain_loss / total_samples_all

    mnist_avg_loss = total_mnist_loss / total_samples_m
    kmnist_avg_loss = total_kmnist_loss / total_samples_k
    fmnist_avg_loss = total_fmnist_loss / total_samples_f
    label_avg_loss = total_label_loss / total_samples_all

    specialization_loss = total_specialization_loss / total_samples_all
    diversity_loss = total_diversity_loss / total_samples_all
    total_avg_loss = total_loss / total_samples_all

    mnist_acc_epoch = total_mnist_correct / total_samples_m * 100
    kmnist_acc_epoch = total_kmnist_correct / total_samples_k * 100
    fmnist_acc_epoch = total_fmnist_correct / total_samples_f * 100

    mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples_m * 100
    kmnist_domain_acc_epoch = total_kmnist_domain_correct / total_samples_k * 100
    fmnist_domain_acc_epoch = total_fmnist_domain_correct / total_samples_f * 100

    avg_label_acc = (mnist_acc_epoch + kmnist_acc_epoch + fmnist_acc_epoch) / 3.0

    print(f'Epoch [{epoch + 1}/{args.epoch}] ({prefix})')
    print(f'  [Ratios] MNIST: [{mnist_partition_ratio_str}] | KMNIST: [{kmnist_partition_ratio_str}] | FMNIST: [{fmnist_partition_ratio_str}]')
    print(f'  [Acc]    MNIST: {mnist_acc_epoch:<6.2f}% | KMNIST: {kmnist_acc_epoch:<6.2f}% | FMNIST: {fmnist_acc_epoch:<6.2f}% | Avg: {avg_label_acc:<6.2f}%')
    print(f'  [DomAcc] MNIST: {mnist_domain_acc_epoch:<6.2f}% | KMNIST: {kmnist_domain_acc_epoch:<6.2f}% | FMNIST: {fmnist_domain_acc_epoch:<6.2f}%')
    print(f'  [Reg]    Spec:  {specialization_loss:<8.4f} | Div:    {diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
    print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {domain_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
    print(f'  [Label]  MNIST: {mnist_avg_loss:<6.4f} | KMNIST: {kmnist_avg_loss:<6.4f} | FMNIST: {fmnist_avg_loss:<6.4f}')
    print(f'  [Domain] MNIST: {mnist_domain_avg_loss:<6.4f} | KMNIST: {kmnist_domain_avg_loss:<6.4f} | FMNIST: {fmnist_domain_avg_loss:<6.4f}')

    log_data = {
        **{f"{prefix}/Partition {p} MNIST Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
        **{f"{prefix}/Partition {p} KMNIST Ratio": kmnist_partition_ratios[p].item() for p in
           range(args.num_partition)},
        **{f"{prefix}/Partition {p} FMNIST Ratio": fmnist_partition_ratios[p].item() for p in
           range(args.num_partition)},
        f'Parameters/Acc Label Avg': avg_label_acc,
        f'{prefix}/Acc Label MNIST': mnist_acc_epoch,
        f'{prefix}/Acc Label KMNIST': kmnist_acc_epoch,
        f'{prefix}/Acc Label FMNIST': fmnist_acc_epoch,
        f'{prefix}/Acc Domain MNIST': mnist_domain_acc_epoch,
        f'{prefix}/Acc Domain KMNIST': kmnist_domain_acc_epoch,
        f'{prefix}/Acc Domain FMNIST': fmnist_domain_acc_epoch,
        f'{prefix} Loss/Label MNIST': mnist_avg_loss,
        f'{prefix} Loss/Label KMNIST': kmnist_avg_loss,
        f'{prefix} Loss/Label FMNIST': fmnist_avg_loss,
        f'{prefix} Loss/Label Avg': label_avg_loss,
        f'{prefix} Loss/Domain MNIST': mnist_domain_avg_loss,
        f'{prefix} Loss/Domain KMNIST': kmnist_domain_avg_loss,
        f'{prefix} Loss/Domain FMNIST': fmnist_domain_avg_loss,
        f'{prefix} Loss/Domain Avg': domain_avg_loss,
        f'{prefix} Loss/Specialization': specialization_loss,
        f'{prefix} Loss/Diversity': diversity_loss,
        f'{prefix} Loss/Total': total_avg_loss,
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
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_partition', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_domains', type=int, default=3)
    parser.add_argument('--fc_hidden', type=int, default=384)
    parser.add_argument('--disc_hidden', type=int, default=24)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.618)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # learning rate tune
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=1.0)
    parser.add_argument('--switcher_lr', type=float, default=1.0)

    # entropy tune
    parser.add_argument('--reg_alpha', type=float, default=0.3)
    parser.add_argument('--reg_beta', type=float, default=1.0)
    parser.add_argument('--lambda_p', type=float, default=-0.1)

    args = parser.parse_args()
    init_lambda = args.lambda_p
    num_epochs = args.epoch

    wandb_run = wandb.init(entity="hails",
                           project="TagNet - 3MNIST",
                           config=args.__dict__,
                           name="[Tagnet]3MNIST_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_FCL:" + str(args.fc_hidden)
                                + "_DiscL:" + str(args.disc_hidden)
                                + "_spe:" + str(args.reg_alpha)
                                + "_div:" + str(args.reg_beta)
                                + "_lr(d)" + str(args.disc_lr)
                                + "_lr(s)" + str(args.switcher_lr)
                                + "_lambda_p:" + str(args.lambda_p)
                           )

    numbers_loader, numbers_loader_test = data_loader('MNIST', args.batch_size)
    fashion_loader, fashion_loader_test = data_loader('FMNIST', args.batch_size)
    gana_loader, gana_loader_test = data_loader('KMNIST', args.batch_size)

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

    model = TagNet(num_classes=args.num_classes,
                   num_partition=args.num_partition,
                   num_domains=args.num_domains,
                   fc_hidden=args.fc_hidden,
                   disc_hidden=args.disc_hidden,
                   device=device
            )

    model.to(device)

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 50
    min_save_epoch = 50

    params = TagNet_weights(
        model,
        lr=args.lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    )

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.opt_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # phi = (1 + math.sqrt(5)) / 2
        # lambda_p = init_lambda / phi ** (epoch / 50)
        # p = epoch / num_epochs
        # lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
        lambda_p = init_lambda
        tau = tau_scheduler.get_tau()

        train_logs, _ = run_epoch(
            'train', model, train_loaders, criterion, optimizer, epoch, args, tau, lambda_p
        )

        test_logs, test_accs = run_epoch(
            'test', model, test_loaders, criterion, None, epoch, args, 0.1, lambda_p
        )

        all_logs = {**train_logs, **test_logs}
        all_logs['Parameters/Tau'] = tau
        all_logs['Parameters/Lambda_p'] = lambda_p

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

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()