import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNetMLP, TagNet_weights
from dataloader.data_loader import data_loader
import math
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
    total_svhn_domain_loss, total_svhn_domain_correct, total_svhn_loss, total_svhn_correct = 0, 0, 0, 0
    total_cifar_domain_loss, total_cifar_domain_correct, total_cifar_loss, total_cifar_correct = 0, 0, 0, 0
    total_domain_loss, total_label_loss, total_loss = 0, 0, 0
    total_specialization_loss, total_diversity_loss = 0, 0

    mnist_partition_counts = torch.zeros(args.num_partition, device=device)
    svhn_partition_counts = torch.zeros(args.num_partition, device=device)
    cifar_partition_counts = torch.zeros(args.num_partition, device=device)

    total_samples_m, total_samples_s, total_samples_c = 0, 0, 0

    mnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    svhn_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    cifar_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

    mnist_iter = iter(loaders['mnist'])
    svhn_iter = iter(loaders['svhn'])
    cifar_iter = iter(loaders['cifar'])

    len_dataloader = min(len(loaders['mnist']), len(loaders['svhn']), len(loaders['cifar']))

    with context:
        for i in range(len_dataloader):
            try:
                mnist_images, mnist_labels = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(loaders['mnist'])
                mnist_images, mnist_labels = next(mnist_iter)
            try:
                svhn_images, svhn_labels = next(svhn_iter)
            except StopIteration:
                svhn_iter = iter(loaders['svhn'])
                svhn_images, svhn_labels = next(svhn_iter)
            try:
                cifar_images, cifar_labels = next(cifar_iter)
            except StopIteration:
                cifar_iter = iter(loaders['cifar'])
                cifar_images, cifar_labels = next(cifar_iter)

            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

            bs_m = mnist_images.size(0)
            bs_s = svhn_images.size(0)
            bs_c = cifar_images.size(0)
            bs_all = bs_m + bs_s + bs_c

            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 1, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 2, dtype=torch.long, device=device)

            if mode == 'train':
                optimizer.zero_grad()

            is_inference = (mode == 'test')

            out_label_mnist, out_domain_mnist, part_idx_mnist, switcher_prob_mnist = model(
                mnist_images, alpha=lambda_p, tau=tau, inference=is_inference)
            out_label_svhn, out_domain_svhn, part_idx_svhn, switcher_prob_svhn = model(
                svhn_images, alpha=lambda_p, tau=tau, inference=is_inference)
            out_label_cifar, out_domain_cifar, part_idx_cifar, switcher_prob_cifar = model(
                cifar_images, alpha=lambda_p, tau=tau, inference=is_inference)

            if i % 1 == 0 and mode == 'train':
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(part_idx_mnist, minlength=args.num_partition)
                svhn_counts = torch.bincount(part_idx_svhn, minlength=args.num_partition)
                cifar_counts = torch.bincount(part_idx_cifar, minlength=args.num_partition)
                print(f"MNIST : {mnist_counts.cpu().numpy()} / SVHN  : {svhn_counts.cpu().numpy()} / CIFAR : {cifar_counts.cpu().numpy()}")
                print(f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.8f}, Bias Mean: {model.partition_switcher.bias.data.mean():.8f}")

            for l_idx in range(mnist_labels.size(0)):
                label_val = mnist_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    mnist_label_partition_counts[label_val, part_idx_mnist[l_idx].item()] += 1
            for l_idx in range(svhn_labels.size(0)):
                label_val = svhn_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    svhn_label_partition_counts[label_val, part_idx_svhn[l_idx].item()] += 1
            for l_idx in range(cifar_labels.size(0)):
                label_val = cifar_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    cifar_label_partition_counts[label_val, part_idx_cifar[l_idx].item()] += 1

            label_loss_mnist = criterion(out_label_mnist, mnist_labels)
            label_loss_svhn = criterion(out_label_svhn, svhn_labels)
            label_loss_cifar = criterion(out_label_cifar, cifar_labels)
            label_loss = label_loss_mnist + label_loss_svhn + label_loss_cifar

            mnist_domain_loss = criterion(out_domain_mnist, mnist_dlabels)
            svhn_domain_loss = criterion(out_domain_svhn, svhn_dlabels)
            cifar_domain_loss = criterion(out_domain_cifar, cifar_dlabels)
            domain_loss = mnist_domain_loss + svhn_domain_loss + cifar_domain_loss

            epsilon = 1e-9
            # entropy list -> mean -> loss
            entropy_mnist = -torch.sum(switcher_prob_mnist * torch.log(switcher_prob_mnist + epsilon), dim=1)
            loss_specialization_mnist = entropy_mnist.mean()
            entropy_svhn = -torch.sum(switcher_prob_svhn * torch.log(switcher_prob_svhn + epsilon), dim=1)
            loss_specialization_svhn = entropy_svhn.mean()
            entropy_cifar = -torch.sum(switcher_prob_cifar * torch.log(switcher_prob_cifar + epsilon), dim=1)
            loss_specialization_cifar = entropy_cifar.mean()

            loss_specialization = loss_specialization_mnist + loss_specialization_svhn + loss_specialization_cifar

            if torch.isnan(loss_specialization):
                print('caution, specialization')

            all_probs = torch.cat((switcher_prob_mnist, switcher_prob_svhn, switcher_prob_cifar), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-9))

            if torch.isnan(loss_diversity):
                print('caution, diversity')

            entropy_loss = args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity

            loss = label_loss + domain_loss + entropy_loss

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
            svhn_partition_counts += torch.bincount(part_idx_svhn, minlength=args.num_partition).to(device)
            cifar_partition_counts += torch.bincount(part_idx_cifar, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item() * bs_all
            total_mnist_loss += label_loss_mnist.item() * bs_m
            total_svhn_loss += label_loss_svhn.item() * bs_s
            total_cifar_loss += label_loss_cifar.item() * bs_c

            total_domain_loss += domain_loss.item() * bs_all
            total_mnist_domain_loss += mnist_domain_loss.item() * bs_m
            total_svhn_domain_loss += svhn_domain_loss.item() * bs_s
            total_cifar_domain_loss += cifar_domain_loss.item() * bs_c

            total_specialization_loss += loss_specialization.item() * bs_all
            total_diversity_loss += loss_diversity.item() * bs_all
            total_loss += loss.item() * bs_all

            total_mnist_correct += (torch.argmax(out_label_mnist, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(out_label_svhn, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += (torch.argmax(out_label_cifar, dim=1) == cifar_labels).sum().item()

            total_mnist_domain_correct += (torch.argmax(out_domain_mnist, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(out_domain_svhn, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += (torch.argmax(out_domain_cifar, dim=1) == cifar_dlabels).sum().item()

            total_samples_m += bs_m
            total_samples_s += bs_s
            total_samples_c += bs_c

    if total_samples_m == 0: total_samples_m = 1
    if total_samples_s == 0: total_samples_s = 1
    if total_samples_c == 0: total_samples_c = 1
    total_samples_all = total_samples_m + total_samples_s + total_samples_c
    print(f'samples MNIST:{total_samples_m} / SVHN{total_samples_s} / CIFAR {total_samples_c}')

    mnist_train_partition_log = get_label_partition_log_data(
        mnist_label_partition_counts, 'MNIST', args.num_classes, args.num_partition, prefix=prefix)
    svhn_train_partition_log = get_label_partition_log_data(
        svhn_label_partition_counts, 'SVHN', args.num_classes, args.num_partition, prefix=prefix)
    cifar_train_partition_log = get_label_partition_log_data(
        cifar_label_partition_counts, 'CIFAR', args.num_classes, args.num_partition, prefix=prefix)

    mnist_partition_ratios = mnist_partition_counts / total_samples_m * 100
    svhn_partition_ratios = svhn_partition_counts / total_samples_s * 100
    cifar_partition_ratios = cifar_partition_counts / total_samples_c * 100

    mnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    svhn_partition_ratio_str = " | ".join(
        [f"Partition {p}: {svhn_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    cifar_partition_ratio_str = " | ".join(
        [f"Partition {p}: {cifar_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

    mnist_domain_avg_loss = total_mnist_domain_loss / total_samples_m
    svhn_domain_avg_loss = total_svhn_domain_loss / total_samples_s
    cifar_domain_avg_loss = total_cifar_domain_loss / total_samples_c
    domain_avg_loss = total_domain_loss / total_samples_all

    mnist_avg_loss = total_mnist_loss / total_samples_m
    svhn_avg_loss = total_svhn_loss / total_samples_s
    cifar_avg_loss = total_cifar_loss / total_samples_c
    label_avg_loss = total_label_loss / total_samples_all

    specialization_loss = total_specialization_loss / total_samples_all
    diversity_loss = total_diversity_loss / total_samples_all
    total_avg_loss = total_loss / total_samples_all

    mnist_acc_epoch = total_mnist_correct / total_samples_m * 100
    svhn_acc_epoch = total_svhn_correct / total_samples_s * 100
    cifar_acc_epoch = total_cifar_correct / total_samples_c * 100

    mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples_m * 100
    svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples_s * 100
    cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples_c * 100

    avg_label_acc = (mnist_acc_epoch + svhn_acc_epoch + cifar_acc_epoch) / 3.0

    print(f'Epoch [{epoch + 1}/{args.epoch}] ({prefix})')
    print(f'  [Ratios] MNIST: [{mnist_partition_ratio_str}] | SVHN: [{svhn_partition_ratio_str}] | CIFAR: [{cifar_partition_ratio_str}]')
    print(f'  [Acc]    MNIST: {mnist_acc_epoch:<6.2f}% | SVHN: {svhn_acc_epoch:<6.2f}% | CIFAR: {cifar_acc_epoch:<6.2f}% | Avg: {avg_label_acc:<6.2f}%')
    print(f'  [DomAcc] MNIST: {mnist_domain_acc_epoch:<6.2f}% | SVHN: {svhn_domain_acc_epoch:<6.2f}% | CIFAR: {cifar_domain_acc_epoch:<6.2f}%')
    print(f'  [Reg]    Spec:  {specialization_loss:<8.4f} | Div:    {diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
    print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {domain_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
    print(f'  [Label]  MNIST: {mnist_avg_loss:<6.4f} | SVHN: {svhn_avg_loss:<6.4f} | CIFAR: {cifar_avg_loss:<6.4f}')
    print(f'  [Domain] MNIST: {mnist_domain_avg_loss:<6.4f} | SVHN: {svhn_domain_avg_loss:<6.4f} | CIFAR: {cifar_domain_avg_loss:<6.4f}')

    log_data = {
        **{f"{prefix}/Partition {p} MNIST Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
        **{f"{prefix}/Partition {p} SVHN Ratio": svhn_partition_ratios[p].item() for p in range(args.num_partition)},
        **{f"{prefix}/Partition {p} CIFAR Ratio": cifar_partition_ratios[p].item() for p in range(args.num_partition)},
        f'Parameters/Acc Label Avg': avg_label_acc,
        f'{prefix}/Acc Label MNIST': mnist_acc_epoch,
        f'{prefix}/Acc Label SVHN': svhn_acc_epoch,
        f'{prefix}/Acc Label CIFAR': cifar_acc_epoch,
        f'{prefix}/Acc Domain MNIST': mnist_domain_acc_epoch,
        f'{prefix}/Acc Domain SVHN': svhn_domain_acc_epoch,
        f'{prefix}/Acc Domain CIFAR': cifar_domain_acc_epoch,
        f'{prefix} Loss/Label MNIST': mnist_avg_loss,
        f'{prefix} Loss/Label SVHN': svhn_avg_loss,
        f'{prefix} Loss/Label CIFAR': cifar_avg_loss,
        f'{prefix} Loss/Label Avg': label_avg_loss,
        f'{prefix} Loss/Domain MNIST': mnist_domain_avg_loss,
        f'{prefix} Loss/Domain SVHN': svhn_domain_avg_loss,
        f'{prefix} Loss/Domain CIFAR': cifar_domain_avg_loss,
        f'{prefix} Loss/Domain Avg': domain_avg_loss,
        f'{prefix} Loss/Specialization': specialization_loss,
        f'{prefix} Loss/Diversity': diversity_loss,
        f'{prefix} Loss/Total': total_avg_loss,
        **mnist_train_partition_log,
        **svhn_train_partition_log,
        **cifar_train_partition_log,
    }

    ind_accs = {
        'mnist': mnist_acc_epoch,
        'svhn': svhn_acc_epoch,
        'cifar': cifar_acc_epoch,
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
    parser.add_argument('--fc_hidden', type=int, default=768)
    parser.add_argument('--disc_hidden', type=int, default=120)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.97)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # learning rate tune
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=0.1)
    parser.add_argument('--switcher_lr', type=float, default=0.1)

    # entropy tune
    parser.add_argument('--reg_alpha', type=float, default=0.25)
    parser.add_argument('--reg_beta', type=float, default=1.0)
    parser.add_argument('--lambda_p', type=float, default=1e-2)

    args = parser.parse_args()
    init_lambda = args.lambda_p
    num_epochs = args.epoch

    wandb_run = wandb.init(entity="hails",
                           project="TagNet - MSC",
                           config=args.__dict__,
                           name="[Tagnet]MSC_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_FCL:" + str(args.fc_hidden)
                                + "_DiscL:" + str(args.disc_hidden)
                                + "_spe:" + str(args.reg_alpha)
                                + "_div:" + str(args.reg_beta)
                                + "_lr(d)" + str(args.disc_lr)
                                + "_lr(s)" + str(args.switcher_lr)
                                + "_lambda_p:" + str(args.lambda_p)
                           )

    mnist_loader, mnist_loader_test = data_loader('MNISTM', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)

    train_loaders = {
        'mnist': mnist_loader,
        'svhn': svhn_loader,
        'cifar': cifar_loader
    }
    test_loaders = {
        'mnist': mnist_loader_test,
        'svhn': svhn_loader_test,
        'cifar': cifar_loader_test
    }

    print("Data load complete, start training")

    model = TagNetMLP(num_classes=args.num_classes,
                      num_partition=args.num_partition,
                      num_domains=args.num_domains,
                      fc_hidden=args.fc_hidden,
                      disc_hidden=args.disc_hidden,
                      device=device
                      )

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 50
    min_save_epoch = 50

    params = TagNet_weights(
        model,
        lr=args.lr,
        pre_weight=args.prefc_lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    )

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.opt_decay)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # phi = (1 + math.sqrt(5)) / 2
        # lambda_p = init_lambda / phi ** (epoch / 50)
        lambda_p = init_lambda
        tau = tau_scheduler.get_tau()

        train_logs, _ = run_epoch(
            'train', model, train_loaders, criterion, optimizer, epoch, args, tau, lambda_p
        )

        test_logs, test_accs = run_epoch(
            'test', model, test_loaders, criterion, None, epoch, args, tau, lambda_p
        )

        all_logs = {**train_logs, **test_logs}
        all_logs['Parameters/Tau'] = tau
        all_logs['Parameters/Learning Rate'] = optimizer.param_groups[0]['lr']
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