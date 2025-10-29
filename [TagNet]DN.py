import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from functions.lr_lambda import lr_lambda
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNet, TagNet_weights
from dataloader.DomainNetLoader import dn_loader
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_partition', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pre_classifier_out', type=int, default=1024)
    parser.add_argument('--part_layer', type=int, default=1024)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.97)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=1.0)
    parser.add_argument('--switcher_lr', type=float, default=1.0)

    # regularization
    parser.add_argument('--reg_alpha', type=float, default=0.2)
    parser.add_argument('--reg_beta', type=float, default=1.0)

    args = parser.parse_args()
    num_epochs = args.epoch

    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="TagNet - DomainNet",
               config=args.__dict__,
               name="[TagNet]DN_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_tau:" + str(args.init_tau)
               )

    # quickdraw, real, painting, sketch, clipart, infograph
    # furniture ~ 246 table, 110 teapot, 15 streetlight, 213 umbrella, 139 wine glass, 299 stairs, 58 toothbrush, 102 suitcase, 47 ladder, 48 picture frame
    furniture_real_train, furniture_real_test = dn_loader('real', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)
    # furniture_quickdraw_train, furniture_quickdraw_test = dn_loader('quickdraw', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)

    # tool ~ 314 nail, 131 sword, 227 bottlecap, 12 basket, 40 rifle, 249 bandage, 10 pliers, 237 axe, 207 paint can, 276 anvil
    tool_real_train, tool_real_test = dn_loader('real', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)
    # tool_painting_train, tool_painting_test = dn_loader('painting', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)

    # mammal ~ 61 squirrel, 292 dog, 81 whale, 148 tigger, 319 zebra, 157 sheep, 83 elephant, 188 horse, 312 cat, 89 monkey
    mammal_real_train, mammal_real_test = dn_loader('real', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)
    # mammal_paint_train, mammal_paint_test = dn_loader('painting', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)


    print("Data load complete, start training")

    model = TagNet(num_classes=args.num_classes,
                   pre_classifier_out=args.pre_classifier_out,
                   n_partition=args.num_partition,
                   part_layer=args.part_layer,
                   device=device
                   )

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    param_groups = TagNet_weights(model, args.lr,
                                  pre_weight=args.prefc_lr,
                                  fc_weight=args.fc_lr,
                                  disc_weight=args.disc_lr,
                                  switcher_weight=args.switcher_lr
                                  )
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.opt_decay)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # w_src, w_tgt = 1.0, 2.0
    # domain_criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_src, w_tgt], device=device))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        tau = tau_scheduler.get_tau()

        total_furniture_domain_loss, total_furniture_domain_correct, total_furniture_loss, total_furniture_correct = 0, 0, 0, 0
        total_tool_domain_loss, total_tool_domain_correct, total_tool_loss, total_tool_correct = 0, 0, 0, 0
        total_mammal_domain_loss, total_mammal_domain_correct, total_mammal_loss, total_mammal_correct = 0, 0, 0, 0
        total_domain_loss, total_label_loss = 0, 0
        total_specialization_loss, total_diversity_loss = 0, 0

        furniture_partition_counts = torch.zeros(args.num_partition, device=device)
        tool_partition_counts = torch.zeros(args.num_partition, device=device)
        mammal_partition_counts = torch.zeros(args.num_partition, device=device)
        total_samples = 0

        for i, (furniture_data, tool_data, mammal_data) in enumerate(zip(furniture_real_train, tool_real_train, mammal_real_train)):
            p = epoch / num_epochs
            lambda_p = 0.0001

            furniture_images, furniture_labels = furniture_data
            furniture_images, furniture_labels = furniture_images.to(device), furniture_labels.to(device)
            tool_images, tool_labels = tool_data
            tool_images, tool_labels = tool_images.to(device), tool_labels.to(device)
            mammal_images, mammal_labels = mammal_data
            mammal_images, mammal_labels = mammal_images.to(device), mammal_labels.to(device)

            furniture_dlabels = torch.full((furniture_images.size(0),), 0, dtype=torch.long, device=device)
            tool_dlabels = torch.full((tool_images.size(0),), 0, dtype=torch.long, device=device)
            mammal_dlabels = torch.full((mammal_images.size(0),), 1, dtype=torch.long, device=device)

            optimizer.zero_grad()

            furniture_out_part, furniture_domain_out, furniture_part_idx, furniture_part_gumbel = model(furniture_images, alpha=lambda_p, tau=tau, inference=False)
            tool_out_part, tool_domain_out, tool_part_idx, tool_part_gumbel = model(tool_images, alpha=lambda_p, tau=tau, inference=False)
            mammal_out_part, mammal_domain_out, mammal_part_idx, mammal_part_gumbel = model(mammal_images, alpha=lambda_p, tau=tau, inference=False)

            if i % 10 == 0: # Print 출력을 10배 줄임
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                furniture_counts = torch.bincount(furniture_part_idx, minlength=args.num_partition)
                tool_counts = torch.bincount(tool_part_idx, minlength=args.num_partition)
                mammal_counts = torch.bincount(mammal_part_idx, minlength=args.num_partition)

                print(f"Furniture: {furniture_counts.cpu().numpy()} / Tool: {tool_counts.cpu().numpy()} / Mammal: {mammal_counts.cpu().numpy()}")
                # print(f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.8f}, Bias Mean: {model.partition_switcher.bias.data.mean():.8f}")

            furniture_label_loss = criterion(furniture_out_part, furniture_labels)
            mammal_label_loss = criterion(mammal_out_part, mammal_labels)
            tool_label_loss = criterion(tool_out_part, tool_labels)

            source_part_gumbel = torch.cat((furniture_part_gumbel, tool_part_gumbel))
            avg_prob_source = torch.mean(source_part_gumbel, dim=0)
            avg_prob_target = torch.mean(mammal_part_gumbel, dim=0)

            epsilon = 1e-8
            loss_specialization_source = -torch.sum(avg_prob_source * torch.log(avg_prob_source + epsilon))
            loss_specialization_target = -torch.sum(avg_prob_target * torch.log(avg_prob_target + epsilon))

            all_probs = torch.cat((source_part_gumbel, mammal_part_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + epsilon))

            loss_specialization = loss_specialization_source + loss_specialization_target

            label_loss = (furniture_label_loss + tool_label_loss + mammal_label_loss
                          + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity)

            furniture_domain_loss = criterion(furniture_domain_out, furniture_dlabels)
            tool_domain_loss = criterion(tool_domain_out, tool_dlabels)
            mammal_domain_loss = criterion(mammal_domain_out, mammal_dlabels)

            domain_loss = (furniture_domain_loss + tool_domain_loss) / 2 + mammal_domain_loss

            loss = label_loss + domain_loss

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

            # count partition ratio
            furniture_partition_counts += torch.bincount(furniture_part_idx, minlength=args.num_partition).to(device)
            tool_partition_counts += torch.bincount(tool_part_idx, minlength=args.num_partition).to(device)
            mammal_partition_counts += torch.bincount(mammal_part_idx, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item()
            total_furniture_loss += furniture_label_loss.item()
            total_tool_loss += tool_label_loss.item()
            total_mammal_loss += mammal_label_loss.item()

            total_specialization_loss += loss_specialization.item()
            total_diversity_loss += loss_diversity.item()

            total_domain_loss += domain_loss.item()
            total_furniture_domain_loss += furniture_domain_loss.item()
            total_tool_domain_loss += tool_domain_loss.item()
            total_mammal_domain_loss += mammal_domain_loss.item()

            total_furniture_correct += (torch.argmax(furniture_out_part, dim=1) == furniture_labels).sum().item()
            total_tool_correct += ((torch.argmax(tool_out_part, dim=1) == tool_labels).sum().item())
            total_mammal_correct += (torch.argmax(mammal_out_part, dim=1) == mammal_labels).sum().item()

            total_furniture_domain_correct += (torch.argmax(furniture_domain_out, dim=1) == furniture_dlabels).sum().item()
            total_tool_domain_correct += ((torch.argmax(tool_domain_out, dim=1) == tool_dlabels).sum().item())
            total_mammal_domain_correct += (torch.argmax(mammal_domain_out, dim=1) == mammal_dlabels).sum().item()

            total_samples += furniture_labels.size(0)

        tau_scheduler.step()
        # scheduler.step()

        furniture_partition_ratios = furniture_partition_counts / total_samples * 100
        tool_partition_ratios = tool_partition_counts / total_samples * 100
        mammal_partition_ratios = mammal_partition_counts / total_samples * 100

        furniture_partition_ratio_str = " | ".join(
            [f"P{p}: {furniture_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        tool_partition_ratio_str = " | ".join(
            [f"P{p}: {tool_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        mammal_partition_ratio_str = " | ".join(
            [f"P{p}: {mammal_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

        furniture_domain_avg_loss = total_furniture_domain_loss / total_samples
        tool_domain_avg_loss = total_tool_domain_loss / total_samples
        mammal_domain_avg_loss = total_mammal_domain_loss / total_samples
        domain_avg_loss = total_domain_loss / (total_samples * 3)

        furniture_avg_loss = total_furniture_loss / total_samples
        tool_avg_loss = total_tool_loss / total_samples
        mammal_avg_loss = total_mammal_loss / total_samples
        label_avg_loss = total_label_loss / (total_samples * 3)

        specialization_loss = total_specialization_loss / total_samples
        diversity_loss = total_diversity_loss / total_samples

        furniture_acc_epoch = total_furniture_correct / total_samples * 100
        tool_acc_epoch = total_tool_correct / total_samples * 100
        mammal_acc_epoch = total_mammal_correct / total_samples * 100

        furniture_domain_acc_epoch = total_furniture_domain_correct / total_samples * 100
        tool_domain_acc_epoch = total_tool_domain_correct / total_samples * 100
        mammal_domain_acc_epoch = total_mammal_domain_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]--------------------------------------------')
        print(f'  Ratios | Furniture: [{furniture_partition_ratio_str}] | Tool: [{tool_partition_ratio_str}] | Mammal: [{mammal_partition_ratio_str}]')
        print(f'  Loss   | Label: {label_avg_loss:.4f} | Domain: {domain_avg_loss:.4f} | Total: {label_avg_loss + domain_avg_loss:.4f}')
        print(f'  Reg    | Specialization: {specialization_loss:.4f} | Diversity: {diversity_loss:.4f}')
        print(f'  Acc    | Furniture: {furniture_acc_epoch:.2f}% | Tool: {tool_acc_epoch:.2f}% | Mammal: {mammal_acc_epoch:.2f}%')
        print(f'  DomAcc | Furniture: {furniture_domain_acc_epoch:.2f}% | Tool: {tool_domain_acc_epoch:.2f}% | Mammal: {mammal_domain_acc_epoch:.2f}%')

        wandb.log({
            **{f"Train/Furniture Partition {p} Ratio": furniture_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/Mammal Partition {p} Ratio": mammal_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/Tool Partition {p} Ratio": tool_partition_ratios[p].item() for p in range(args.num_partition)},
            'Train/Furniture Label Loss': furniture_avg_loss,
            'Train/Mammal Label Loss': mammal_avg_loss,
            'Train/Tool Label Loss': tool_avg_loss,
            'Train/Label Loss': label_avg_loss,
            'Train/Specialization Loss': specialization_loss,
            'Train/Diversity Loss' : diversity_loss,
            'Train/Domain Furniture Loss': furniture_domain_avg_loss,
            'Train/Domain Mammal Loss': mammal_domain_avg_loss,
            'Train/Domain Tool Loss': tool_domain_avg_loss,
            'Train/Domain Loss': domain_avg_loss,
            'Train/Total Loss': (label_avg_loss + domain_avg_loss),
            'Train/Furniture Label Accuracy': furniture_acc_epoch,
            'Train/Mammal Label Accuracy': mammal_acc_epoch,
            'Train/Tool Label Accuracy': tool_acc_epoch,
            'Train/Furniture Domain Accuracy': furniture_domain_acc_epoch,
            'Train/Mammal Domain Accuracy': mammal_domain_acc_epoch,
            'Train/Tool Domain Accuracy': tool_domain_acc_epoch,
            'Parameter/Tau': tau,
            'Parameter/Lambda': lambda_p
        }, step=epoch + 1)

        model.eval()

        def tester(loader, group, domain_label):
            label_correct, domain_correct, total = 0, 0, 0
            partition_counts = torch.zeros(args.num_partition, device=device)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output_partitioned, domain_output, partition_idx, _ = model(images, alpha=0, tau=1e-5, inference=True)

                total += images.size(0)
                label_correct += (torch.argmax(class_output_partitioned, dim=1) == labels).sum().item()
                domain_correct += (torch.argmax(domain_output, dim=1) == domain_label).sum().item()
                partition_counts += torch.bincount(partition_idx, minlength=args.num_partition)

            label_acc = label_correct / total * 100
            domain_acc = domain_correct / total * 100
            partition_ratios = partition_counts / total * 100
            partition_ratio_str = " | ".join(
                [f"P{p}: {partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

            wandb.log({
                f'Test/Label {group} Accuracy': label_acc,
                f'Test/Domain {group} Accuracy': domain_acc,
                **{f"Test/{group} Partition {p} Ratio": partition_ratios[p].item() for p in range(args.num_partition)},
            }, step=epoch + 1)

            print(f'Test {group} | Label Acc: {label_acc:.3f}% | Domain Acc: {domain_acc:.3f}% | Partition Ratio: [{partition_ratio_str}]')

        with torch.no_grad():
            tester(furniture_real_test, 'Furniture', 0)
            tester(tool_real_test, 'Tool', 0)
            tester(mammal_real_test, 'Mammal', 1)


if __name__ == '__main__':
    main()
