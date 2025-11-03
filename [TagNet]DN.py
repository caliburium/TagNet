import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNet, TagNet_weights
from dataloader.DomainNetLoader import dn_loader
import math
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            log_key = f"Partition {domain_name} {prefix}/Partition:{part_idx}/Label:{label_idx}"
            log_data[log_key] = percentage
    return log_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_partition', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_domains', type=int, default=3)
    parser.add_argument('--pre_classifier_out', type=int, default=1536)
    parser.add_argument('--part_layer', type=int, default=1536)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.97)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=0.2)
    parser.add_argument('--switcher_lr', type=float, default=0.1)

    # regularization
    parser.add_argument('--reg_alpha', type=float, default=0.2)
    parser.add_argument('--reg_beta', type=float, default=1.0)
    parser.add_argument('--lambda_p', type=float, default=1e-2)

    args = parser.parse_args()
    init_lambda = args.lambda_p
    num_epochs = args.epoch

    save_dir = f"./checkpoints/{wandb.run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 50
    min_save_epoch = 150

    wandb.init(entity="hails",
               project="TagNet - DomainNet",
               config=args.__dict__,
               name="[Tagnet]DN_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
                    + "_PLayer:" + str(args.part_layer)
                    + "_spe:" + str(args.reg_alpha)
                    + "_div:" + str(args.reg_beta)
                    + "_lr(d)" + str(args.disc_lr)
                    + "_lr(s)" + str(args.switcher_lr)
                    + "_lambda_p:" + str(args.lambda_p)
                    + "_domain:" + str(args.num_domains)
                    + "_part:" + str(args.num_partition)
               )

    furniture_real_train, furniture_real_test = dn_loader('real', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)
    furniture_quickdraw_train, furniture_quickdraw_test = dn_loader('quickdraw',[246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)

    tool_real_train, tool_real_test = dn_loader('real', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)
    tool_painting_train, tool_painting_test = dn_loader('painting', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)

    mammal_real_train, mammal_real_test = dn_loader('real', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)
    mammal_paint_train, mammal_paint_test = dn_loader('painting', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)

    print("Data load complete, start training")

    model = TagNet(num_classes=args.num_classes,
                    pre_classifier_out=args.pre_classifier_out,
                    n_partition=args.num_partition,
                    part_layer=args.part_layer,
                    num_domains=args.num_domains,
                    device=device
                    )

    optimizer = optim.Adam(TagNet_weights(
        model,
        lr=args.lr,
        pre_weight=args.prefc_lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    ), lr=args.lr, weight_decay=args.opt_decay
    )

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)

    domain_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        phi = (1 + math.sqrt(5)) / 2
        lambda_p = init_lambda / phi ** (epoch / 20)
        tau = tau_scheduler.get_tau()

        total_fr_domain_loss, total_fr_domain_correct, total_fr_loss, total_fr_correct = 0, 0, 0, 0
        total_fq_domain_loss, total_fq_domain_correct, total_fq_loss, total_fq_correct = 0, 0, 0, 0

        total_tr_domain_loss, total_tr_domain_correct, total_tr_loss, total_tr_correct = 0, 0, 0, 0
        total_tp_domain_loss, total_tp_domain_correct, total_tp_loss, total_tp_correct = 0, 0, 0, 0

        total_mr_domain_loss, total_mr_domain_correct, total_mr_loss, total_mr_correct = 0, 0, 0, 0
        total_mp_domain_loss, total_mp_domain_correct, total_mp_loss, total_mp_correct = 0, 0, 0, 0

        total_domain_loss, total_label_loss, total_loss = 0, 0, 0
        total_specialization_loss, total_diversity_loss = 0, 0

        fr_partition_counts = torch.zeros(args.num_partition, device=device)
        fq_partition_counts = torch.zeros(args.num_partition, device=device)
        tr_partition_counts = torch.zeros(args.num_partition, device=device)
        tp_partition_counts = torch.zeros(args.num_partition, device=device)
        mr_partition_counts = torch.zeros(args.num_partition, device=device)
        mp_partition_counts = torch.zeros(args.num_partition, device=device)
        total_samples = 0

        # Label Partition Counts
        fr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        fq_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        tr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        tp_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        mr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        mp_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

        # --- [MODIFIED] --- 6개 로더를 zip으로 묶음
        train_loader_zip = zip(
            furniture_real_train, furniture_quickdraw_train,
            tool_real_train, tool_painting_train,
            mammal_real_train, mammal_paint_train
        )

        for i, (fr_data, fq_data, tr_data, tp_data, mr_data, mp_data) in enumerate(train_loader_zip):
            fr_images, fr_labels = fr_data
            fr_images, fr_labels = fr_images.to(device), fr_labels.to(device)
            fq_images, fq_labels = fq_data
            fq_images, fq_labels = fq_images.to(device), fq_labels.to(device)
            tr_images, tr_labels = tr_data
            tr_images, tr_labels = tr_images.to(device), tr_labels.to(device)
            tp_images, tp_labels = tp_data
            tp_images, tp_labels = tp_images.to(device), tp_labels.to(device)
            mr_images, mr_labels = mr_data
            mr_images, mr_labels = mr_images.to(device), mr_labels.to(device)
            mp_images, mp_labels = mp_data
            mp_images, mp_labels = mp_images.to(device), mp_labels.to(device)

            fr_dlabels = torch.full((fr_images.size(0),), 0, dtype=torch.long, device=device)
            fq_dlabels = torch.full((fq_images.size(0),), 0, dtype=torch.long, device=device)
            tr_dlabels = torch.full((tr_images.size(0),), 1, dtype=torch.long, device=device)
            tp_dlabels = torch.full((tp_images.size(0),), 1, dtype=torch.long, device=device)
            mr_dlabels = torch.full((mr_images.size(0),), 2, dtype=torch.long, device=device)
            mp_dlabels = torch.full((mp_images.size(0),), 2, dtype=torch.long, device=device)

            optimizer.zero_grad()

            bs_fr, bs_fq = fr_images.size(0), fq_images.size(0)
            bs_tr, bs_tp = tr_images.size(0), tp_images.size(0)
            bs_mr, bs_mp = mr_images.size(0), mp_images.size(0)

            all_images = torch.cat((fr_images, fq_images, tr_images, tp_images, mr_images, mp_images), dim=0)

            out_part, domain_out, part_idx, part_gumbel = model(all_images, alpha=lambda_p, tau=tau, inference=False)

            idx1 = bs_fr
            idx2 = idx1 + bs_fq
            idx3 = idx2 + bs_tr
            idx4 = idx3 + bs_tp
            idx5 = idx4 + bs_mr

            fr_out_part = out_part[:idx1]
            fq_out_part = out_part[idx1:idx2]
            tr_out_part = out_part[idx2:idx3]
            tp_out_part = out_part[idx3:idx4]
            mr_out_part = out_part[idx4:idx5]
            mp_out_part = out_part[idx5:]

            fr_domain_out = domain_out[:idx1]
            fq_domain_out = domain_out[idx1:idx2]
            tr_domain_out = domain_out[idx2:idx3]
            tp_domain_out = domain_out[idx3:idx4]
            mr_domain_out = domain_out[idx4:idx5]
            mp_domain_out = domain_out[idx5:]

            fr_part_idx = part_idx[:idx1]
            fq_part_idx = part_idx[idx1:idx2]
            tr_part_idx = part_idx[idx2:idx3]
            tp_part_idx = part_idx[idx3:idx4]
            mr_part_idx = part_idx[idx4:idx5]
            mp_part_idx = part_idx[idx5:]

            fr_part_gumbel = part_gumbel[:idx1]
            fq_part_gumbel = part_gumbel[idx1:idx2]
            tr_part_gumbel = part_gumbel[idx2:idx3]
            tp_part_gumbel = part_gumbel[idx3:idx4]
            mr_part_gumbel = part_gumbel[idx4:idx5]
            mp_part_gumbel = part_gumbel[idx5:]

            if i % 1 == 0:
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                fr_counts = torch.bincount(fr_part_idx, minlength=args.num_partition).cpu().numpy()
                fq_counts = torch.bincount(fq_part_idx, minlength=args.num_partition).cpu().numpy()
                tr_counts = torch.bincount(tr_part_idx, minlength=args.num_partition).cpu().numpy()
                tp_counts = torch.bincount(tp_part_idx, minlength=args.num_partition).cpu().numpy()
                mr_counts = torch.bincount(mr_part_idx, minlength=args.num_partition).cpu().numpy()
                mp_counts = torch.bincount(mp_part_idx, minlength=args.num_partition).cpu().numpy()
                print(f"  F-R: {fr_counts} / F-Q: {fq_counts}")
                print(f"  T-R: {tr_counts} / T-P: {tp_counts}")
                print(f"  M-R: {mr_counts} / M-P: {mp_counts}")
                print(f"  Switcher W: {model.partition_switcher.weight.data.mean():.8f}, B: {model.partition_switcher.bias.data.mean():.8f}")

            for l_idx in range(bs_fr): fr_label_partition_counts[
                fr_labels[l_idx].item(), fr_part_idx[l_idx].item()] += 1
            for l_idx in range(bs_fq): fq_label_partition_counts[
                fq_labels[l_idx].item(), fq_part_idx[l_idx].item()] += 1
            for l_idx in range(bs_tr): tr_label_partition_counts[
                tr_labels[l_idx].item(), tr_part_idx[l_idx].item()] += 1
            for l_idx in range(bs_tp): tp_label_partition_counts[
                tp_labels[l_idx].item(), tp_part_idx[l_idx].item()] += 1
            for l_idx in range(bs_mr): mr_label_partition_counts[
                mr_labels[l_idx].item(), mr_part_idx[l_idx].item()] += 1
            for l_idx in range(bs_mp): mp_label_partition_counts[
                mp_labels[l_idx].item(), mp_part_idx[l_idx].item()] += 1

            fr_label_loss = criterion(fr_out_part, fr_labels)
            fq_label_loss = criterion(fq_out_part, fq_labels)
            tr_label_loss = criterion(tr_out_part, tr_labels)
            tp_label_loss = criterion(tp_out_part, tp_labels)
            mr_label_loss = criterion(mr_out_part, mr_labels)
            mp_label_loss = criterion(mp_out_part, mp_labels)

            furniture_gumbel = torch.cat((fr_part_gumbel, fq_part_gumbel))
            tool_gumbel = torch.cat((tr_part_gumbel, tp_part_gumbel))
            mammal_gumbel = torch.cat((mr_part_gumbel, mp_part_gumbel))

            avg_prob_furniture = torch.mean(furniture_gumbel, dim=0)
            avg_prob_tool = torch.mean(tool_gumbel, dim=0)
            avg_prob_mammal = torch.mean(mammal_gumbel, dim=0)

            loss_spec_f = -torch.sum(avg_prob_furniture * torch.log(avg_prob_furniture + 1e-8))
            loss_spec_t = -torch.sum(avg_prob_tool * torch.log(avg_prob_tool + 1e-8))
            loss_spec_m = -torch.sum(avg_prob_mammal * torch.log(avg_prob_mammal + 1e-8))
            loss_specialization = loss_spec_f + loss_spec_t + loss_spec_m

            all_probs = torch.cat((furniture_gumbel, tool_gumbel, mammal_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))

            label_loss = (
                    (fr_label_loss + fq_label_loss) / 2 +
                    (tr_label_loss + tp_label_loss) / 2 +
                    (mr_label_loss + mp_label_loss) / 2 +
                    args.reg_alpha * loss_specialization +
                    args.reg_beta * loss_diversity
            )

            fr_domain_loss = domain_criterion(fr_domain_out, fr_dlabels)
            fq_domain_loss = domain_criterion(fq_domain_out, fq_dlabels)
            tr_domain_loss = domain_criterion(tr_domain_out, tr_dlabels)
            tp_domain_loss = domain_criterion(tp_domain_out, tp_dlabels)
            mr_domain_loss = domain_criterion(mr_domain_out, mr_dlabels)
            mp_domain_loss = domain_criterion(mp_domain_out, mp_dlabels)

            domain_loss = (
                    (fr_domain_loss + fq_domain_loss) / 2 +
                    (tr_domain_loss + tp_domain_loss) / 2 +
                    (mr_domain_loss + mp_domain_loss) / 2
            )

            loss = label_loss + domain_loss

            loss.backward()
            optimizer.step()

            fr_partition_counts += torch.bincount(fr_part_idx, minlength=args.num_partition).to(device)
            fq_partition_counts += torch.bincount(fq_part_idx, minlength=args.num_partition).to(device)
            tr_partition_counts += torch.bincount(tr_part_idx, minlength=args.num_partition).to(device)
            tp_partition_counts += torch.bincount(tp_part_idx, minlength=args.num_partition).to(device)
            mr_partition_counts += torch.bincount(mr_part_idx, minlength=args.num_partition).to(device)
            mp_partition_counts += torch.bincount(mp_part_idx, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item()
            total_fr_loss += fr_label_loss.item()
            total_fq_loss += fq_label_loss.item()
            total_tr_loss += tr_label_loss.item()
            total_tp_loss += tp_label_loss.item()
            total_mr_loss += mr_label_loss.item()
            total_mp_loss += mp_label_loss.item()

            total_domain_loss += domain_loss.item()
            total_fr_domain_loss += fr_domain_loss.item()
            total_fq_domain_loss += fq_domain_loss.item()
            total_tr_domain_loss += tr_domain_loss.item()
            total_tp_domain_loss += tp_domain_loss.item()
            total_mr_domain_loss += mr_domain_loss.item()
            total_mp_domain_loss += mp_domain_loss.item()

            total_specialization_loss += loss_specialization.item()
            total_diversity_loss += loss_diversity.item()
            total_loss += loss.item()

            total_fr_correct += (torch.argmax(fr_out_part, dim=1) == fr_labels).sum().item()
            total_fq_correct += (torch.argmax(fq_out_part, dim=1) == fq_labels).sum().item()
            total_tr_correct += (torch.argmax(tr_out_part, dim=1) == tr_labels).sum().item()
            total_tp_correct += (torch.argmax(tp_out_part, dim=1) == tp_labels).sum().item()
            total_mr_correct += (torch.argmax(mr_out_part, dim=1) == mr_labels).sum().item()
            total_mp_correct += (torch.argmax(mp_out_part, dim=1) == mp_labels).sum().item()

            total_fr_domain_correct += (torch.argmax(fr_domain_out, dim=1) == fr_dlabels).sum().item()
            total_fq_domain_correct += (torch.argmax(fq_domain_out, dim=1) == fq_dlabels).sum().item()
            total_tr_domain_correct += (torch.argmax(tr_domain_out, dim=1) == tr_dlabels).sum().item()
            total_tp_domain_correct += (torch.argmax(tp_domain_out, dim=1) == tp_dlabels).sum().item()
            total_mr_domain_correct += (torch.argmax(mr_domain_out, dim=1) == mr_dlabels).sum().item()
            total_mp_domain_correct += (torch.argmax(mp_domain_out, dim=1) == mp_dlabels).sum().item()

            total_samples += bs_fr

        if total_samples == 0: total_samples = 1
        fr_log = get_label_partition_log_data(fr_label_partition_counts, 'F-R', args.num_classes, args.num_partition,"Train")
        fq_log = get_label_partition_log_data(fq_label_partition_counts, 'F-Q', args.num_classes, args.num_partition,"Train")
        tr_log = get_label_partition_log_data(tr_label_partition_counts, 'T-R', args.num_classes, args.num_partition,"Train")
        tp_log = get_label_partition_log_data(tp_label_partition_counts, 'T-P', args.num_classes, args.num_partition,"Train")
        mr_log = get_label_partition_log_data(mr_label_partition_counts, 'M-R', args.num_classes, args.num_partition,"Train")
        mp_log = get_label_partition_log_data(mp_label_partition_counts, 'M-P', args.num_classes, args.num_partition,"Train")

        fr_ratios = fr_partition_counts / total_samples * 100
        fq_ratios = fq_partition_counts / total_samples * 100
        tr_ratios = tr_partition_counts / total_samples * 100
        tp_ratios = tp_partition_counts / total_samples * 100
        mr_ratios = mr_partition_counts / total_samples * 100
        mp_ratios = mp_partition_counts / total_samples * 100

        fr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(fr_ratios)])
        fq_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(fq_ratios)])
        tr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(tr_ratios)])
        tp_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(tp_ratios)])
        mr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(mr_ratios)])
        mp_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(mp_ratios)])

        # Domain Avg Loss
        fr_dom_avg_loss = total_fr_domain_loss / total_samples
        fq_dom_avg_loss = total_fq_domain_loss / total_samples
        tr_dom_avg_loss = total_tr_domain_loss / total_samples
        tp_dom_avg_loss = total_tp_domain_loss / total_samples
        mr_dom_avg_loss = total_mr_domain_loss / total_samples
        mp_dom_avg_loss = total_mp_domain_loss / total_samples

        # Label Avg Loss
        fr_avg_loss = total_fr_loss / total_samples
        fq_avg_loss = total_fq_loss / total_samples
        tr_avg_loss = total_tr_loss / total_samples
        tp_avg_loss = total_tp_loss / total_samples
        mr_avg_loss = total_mr_loss / total_samples
        mp_avg_loss = total_mp_loss / total_samples

        # Total Avg Loss
        total_samples_all = total_samples * 6
        domain_avg_loss = total_domain_loss / total_samples_all
        label_avg_loss = total_label_loss / total_samples_all
        specialization_loss = total_specialization_loss / total_samples_all
        diversity_loss = total_diversity_loss / total_samples_all
        total_avg_loss = total_loss / total_samples_all

        # Acc
        fr_acc = total_fr_correct / total_samples * 100
        fq_acc = total_fq_correct / total_samples * 100
        tr_acc = total_tr_correct / total_samples * 100
        tp_acc = total_tp_correct / total_samples * 100
        mr_acc = total_mr_correct / total_samples * 100
        mp_acc = total_mp_correct / total_samples * 100

        # Domain Acc
        fr_dom_acc = total_fr_domain_correct / total_samples * 100
        fq_dom_acc = total_fq_domain_correct / total_samples * 100
        tr_dom_acc = total_tr_domain_correct / total_samples * 100
        tp_dom_acc = total_tp_domain_correct / total_samples * 100
        mr_dom_acc = total_mr_domain_correct / total_samples * 100
        mp_dom_acc = total_mp_domain_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  [Ratios] F-R: [{fr_ratio_str}] | F-Q: [{fq_ratio_str}]')
        print(f'  [Ratios] T-R: [{tr_ratio_str}] | T-P: [{tp_ratio_str}]')
        print(f'  [Ratios] M-R: [{mr_ratio_str}] | M-P: [{mp_ratio_str}]')
        print(f'  [Acc]    F-R: {fr_acc:<6.2f}% | F-Q: {fq_acc:<6.2f}% | T-R: {tr_acc:<6.2f}% | T-P: {tp_acc:<6.2f}% | M-R: {mr_acc:<6.2f}% | M-P: {mp_acc:<6.2f}%')
        print(f'  [DomAcc] F-R: {fr_dom_acc:<6.2f}% | F-Q: {fq_dom_acc:<6.2f}% | T-R: {tr_dom_acc:<6.2f}% | T-P: {tp_dom_acc:<6.2f}% | M-R: {mr_dom_acc:<6.2f}% | M-P: {mp_dom_acc:<6.2f}%')
        print(f'  [Reg]    Spec:  {specialization_loss:<8.4f} | Div:    {diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
        print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {domain_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
        print(f'  [Label]  F-R: {fr_avg_loss:<6.4f} | F-Q: {fq_avg_loss:<6.4f} | T-R: {tr_avg_loss:<6.4f} | T-P: {tp_avg_loss:<6.4f} | M-R: {mr_avg_loss:<6.4f} | M-P: {mp_avg_loss:<6.4f}')
        print(f'  [Domain] F-R: {fr_dom_avg_loss:<6.4f} | F-Q: {fq_dom_avg_loss:<6.4f} | T-R: {tr_dom_avg_loss:<6.4f} | T-P: {tp_dom_avg_loss:<6.4f} | M-R: {mr_dom_avg_loss:<6.4f} | M-P: {mp_dom_avg_loss:<6.4f}')

        wandb.log({
            **{f"Train/Partition {p} F-R Ratio": r.item() for p, r in enumerate(fr_ratios)},
            **{f"Train/Partition {p} F-Q Ratio": r.item() for p, r in enumerate(fq_ratios)},
            **{f"Train/Partition {p} T-R Ratio": r.item() for p, r in enumerate(tr_ratios)},
            **{f"Train/Partition {p} T-P Ratio": r.item() for p, r in enumerate(tp_ratios)},
            **{f"Train/Partition {p} M-R Ratio": r.item() for p, r in enumerate(mr_ratios)},
            **{f"Train/Partition {p} M-P Ratio": r.item() for p, r in enumerate(mp_ratios)},
            'Train/Label F-R Acc': fr_acc, 'Train/Label F-Q Acc': fq_acc,
            'Train/Label T-R Acc': tr_acc, 'Train/Label T-P Acc': tp_acc,
            'Train/Label M-R Acc': mr_acc, 'Train/Label M-P Acc': mp_acc,
            'Train/Domain F-R Acc': fr_dom_acc, 'Train/Domain F-Q Acc': fq_dom_acc,
            'Train/Domain T-R Acc': tr_dom_acc, 'Train/Domain T-P Acc': tp_dom_acc,
            'Train/Domain M-R Acc': mr_dom_acc, 'Train/Domain M-P Acc': mp_dom_acc,
            'TrainLoss/Label F-R Loss': fr_avg_loss, 'TrainLoss/Label F-Q Loss': fq_avg_loss,
            'TrainLoss/Label T-R Loss': tr_avg_loss, 'TrainLoss/Label T-P Loss': tp_avg_loss,
            'TrainLoss/Label M-R Loss': mr_avg_loss, 'TrainLoss/Label M-P Loss': mp_avg_loss,
            'TrainLoss/Label Loss': label_avg_loss,
            'TrainLoss/Domain F-R Loss': fr_dom_avg_loss, 'TrainLoss/Domain F-Q Loss': fq_dom_avg_loss,
            'TrainLoss/Domain T-R Loss': tr_dom_avg_loss, 'TrainLoss/Domain T-P Loss': tp_dom_avg_loss,
            'TrainLoss/Domain M-R Loss': mr_dom_avg_loss, 'TrainLoss/Domain M-P Loss': mp_dom_avg_loss,
            'TrainLoss/Domain Loss': domain_avg_loss,
            'TrainLoss/Specialization Loss': specialization_loss,
            'TrainLoss/Diversity Loss': diversity_loss,
            'TrainLoss/Total Loss': total_avg_loss,
            'Parameters/Tau': tau,
            'Parameters/Learning Rate': optimizer.param_groups[0]['lr'],
            'Parameters/Lambda_p': lambda_p,
            **fr_log, **fq_log, **tr_log, **tp_log, **mr_log, **mp_log,
        }, step=epoch + 1)

        model.eval()

        with ((torch.no_grad())):
            test_total_fr_domain_loss, test_total_fr_domain_correct, test_total_fr_loss, test_total_fr_correct = 0, 0, 0, 0
            test_total_fq_domain_loss, test_total_fq_domain_correct, test_total_fq_loss, test_total_fq_correct = 0, 0, 0, 0
            test_total_tr_domain_loss, test_total_tr_domain_correct, test_total_tr_loss, test_total_tr_correct = 0, 0, 0, 0
            test_total_tp_domain_loss, test_total_tp_domain_correct, test_total_tp_loss, test_total_tp_correct = 0, 0, 0, 0
            test_total_mr_domain_loss, test_total_mr_domain_correct, test_total_mr_loss, test_total_mr_correct = 0, 0, 0, 0
            test_total_mp_domain_loss, test_total_mp_domain_correct, test_total_mp_loss, test_total_mp_correct = 0, 0, 0, 0

            test_total_domain_loss, test_total_label_loss, test_total_loss = 0, 0, 0
            test_total_specialization_loss, test_total_diversity_loss = 0, 0

            test_fr_partition_counts = torch.zeros(args.num_partition, device=device)
            test_fq_partition_counts = torch.zeros(args.num_partition, device=device)
            test_tr_partition_counts = torch.zeros(args.num_partition, device=device)
            test_tp_partition_counts = torch.zeros(args.num_partition, device=device)
            test_mr_partition_counts = torch.zeros(args.num_partition, device=device)
            test_mp_partition_counts = torch.zeros(args.num_partition, device=device)
            test_total_samples = 0

            test_fr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
            test_fq_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
            test_tr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
            test_tp_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
            test_mr_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
            test_mp_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

            test_loader_zip = zip(
                furniture_real_test, furniture_quickdraw_test,
                tool_real_test, tool_painting_test,
                mammal_real_test, mammal_paint_test
            )

            for i, (fr_data, fq_data, tr_data, tp_data, mr_data, mp_data) in enumerate(test_loader_zip):
                fr_images, fr_labels = fr_data
                fr_images, fr_labels = fr_images.to(device), fr_labels.to(device)
                fq_images, fq_labels = fq_data
                fq_images, fq_labels = fq_images.to(device), fq_labels.to(device)
                tr_images, tr_labels = tr_data
                tr_images, tr_labels = tr_images.to(device), tr_labels.to(device)
                tp_images, tp_labels = tp_data
                tp_images, tp_labels = tp_images.to(device), tp_labels.to(device)
                mr_images, mr_labels = mr_data
                mr_images, mr_labels = mr_images.to(device), mr_labels.to(device)
                mp_images, mp_labels = mp_data
                mp_images, mp_labels = mp_images.to(device), mp_labels.to(device)

                fr_dlabels = torch.full((fr_images.size(0),), 0, dtype=torch.long, device=device)
                fq_dlabels = torch.full((fq_images.size(0),), 0, dtype=torch.long, device=device)
                tr_dlabels = torch.full((tr_images.size(0),), 1, dtype=torch.long, device=device)
                tp_dlabels = torch.full((tp_images.size(0),), 1, dtype=torch.long, device=device)
                mr_dlabels = torch.full((mr_images.size(0),), 2, dtype=torch.long, device=device)
                mp_dlabels = torch.full((mp_images.size(0),), 2, dtype=torch.long, device=device)

                bs_fr, bs_fq = fr_images.size(0), fq_images.size(0)
                bs_tr, bs_tp = tr_images.size(0), tp_images.size(0)
                bs_mr, bs_mp = mr_images.size(0), mp_images.size(0)
                all_images = torch.cat((fr_images, fq_images, tr_images, tp_images, mr_images, mp_images), dim=0)

                out_part, domain_out, part_idx, part_gumbel = model(all_images, alpha=0, tau=tau, inference=False)

                idx1 = bs_fr
                idx2 = idx1 + bs_fq
                idx3 = idx2 + bs_tr
                idx4 = idx3 + bs_tp
                idx5 = idx4 + bs_mr

                fr_out_part = out_part[:idx1]
                fq_out_part = out_part[idx1:idx2]
                tr_out_part = out_part[idx2:idx3]
                tp_out_part = out_part[idx3:idx4]
                mr_out_part = out_part[idx4:idx5]
                mp_out_part = out_part[idx5:]

                fr_domain_out = domain_out[:idx1]
                fq_domain_out = domain_out[idx1:idx2]
                tr_domain_out = domain_out[idx2:idx3]
                tp_domain_out = domain_out[idx3:idx4]
                mr_domain_out = domain_out[idx4:idx5]
                mp_domain_out = domain_out[idx5:]

                fr_part_idx = part_idx[:idx1]
                fq_part_idx = part_idx[idx1:idx2]
                tr_part_idx = part_idx[idx2:idx3]
                tp_part_idx = part_idx[idx3:idx4]
                mr_part_idx = part_idx[idx4:idx5]
                mp_part_idx = part_idx[idx5:]

                fr_part_gumbel = part_gumbel[:idx1]
                fq_part_gumbel = part_gumbel[idx1:idx2]
                tr_part_gumbel = part_gumbel[idx2:idx3]
                tp_part_gumbel = part_gumbel[idx3:idx4]
                mr_part_gumbel = part_gumbel[idx4:idx5]
                mp_part_gumbel = part_gumbel[idx5:]

                for l_idx in range(bs_fr): test_fr_label_partition_counts[
                    fr_labels[l_idx].item(), fr_part_idx[l_idx].item()] += 1
                for l_idx in range(bs_fq): test_fq_label_partition_counts[
                    fq_labels[l_idx].item(), fq_part_idx[l_idx].item()] += 1
                for l_idx in range(bs_tr): test_tr_label_partition_counts[
                    tr_labels[l_idx].item(), tr_part_idx[l_idx].item()] += 1
                for l_idx in range(bs_tp): test_tp_label_partition_counts[
                    tp_labels[l_idx].item(), tp_part_idx[l_idx].item()] += 1
                for l_idx in range(bs_mr): test_mr_label_partition_counts[
                    mr_labels[l_idx].item(), mr_part_idx[l_idx].item()] += 1
                for l_idx in range(bs_mp): test_mp_label_partition_counts[
                    mp_labels[l_idx].item(), mp_part_idx[l_idx].item()] += 1

                fr_label_loss = criterion(fr_out_part, fr_labels)
                fq_label_loss = criterion(fq_out_part, fq_labels)
                tr_label_loss = criterion(tr_out_part, tr_labels)
                tp_label_loss = criterion(tp_out_part, tp_labels)
                mr_label_loss = criterion(mr_out_part, mr_labels)
                mp_label_loss = criterion(mp_out_part, mp_labels)

                furniture_gumbel = torch.cat((fr_part_gumbel, fq_part_gumbel))
                tool_gumbel = torch.cat((tr_part_gumbel, tp_part_gumbel))
                mammal_gumbel = torch.cat((mr_part_gumbel, mp_part_gumbel))

                avg_prob_furniture = torch.mean(furniture_gumbel, dim=0)
                avg_prob_tool = torch.mean(tool_gumbel, dim=0)
                avg_prob_mammal = torch.mean(mammal_gumbel, dim=0)

                loss_spec_f = -torch.sum(avg_prob_furniture * torch.log(avg_prob_furniture + 1e-8))
                loss_spec_t = -torch.sum(avg_prob_tool * torch.log(avg_prob_tool + 1e-8))
                loss_spec_m = -torch.sum(avg_prob_mammal * torch.log(avg_prob_mammal + 1e-8))
                loss_specialization = loss_spec_f + loss_spec_t + loss_spec_m

                all_probs = torch.cat((furniture_gumbel, tool_gumbel, mammal_gumbel), dim=0)
                avg_prob_global = torch.mean(all_probs, dim=0)
                loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))

                label_loss = ((fr_label_loss + fq_label_loss) / 2
                              + (tr_label_loss + tp_label_loss) / 2
                              + (mr_label_loss + mp_label_loss) / 2
                              + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity)

                fr_domain_loss = domain_criterion(fr_domain_out, fr_dlabels)
                fq_domain_loss = domain_criterion(fq_domain_out, fq_dlabels)
                tr_domain_loss = domain_criterion(tr_domain_out, tr_dlabels)
                tp_domain_loss = domain_criterion(tp_domain_out, tp_dlabels)
                mr_domain_loss = domain_criterion(mr_domain_out, mr_dlabels)
                mp_domain_loss = domain_criterion(mp_domain_out, mp_dlabels)

                domain_loss = ((fr_domain_loss + fq_domain_loss) / 2
                               + (tr_domain_loss + tp_domain_loss) / 2
                               + (mr_domain_loss + mp_domain_loss) / 2
                               )
                loss = label_loss + domain_loss

                test_fr_partition_counts += torch.bincount(fr_part_idx, minlength=args.num_partition).to(device)
                test_fq_partition_counts += torch.bincount(fq_part_idx, minlength=args.num_partition).to(device)
                test_tr_partition_counts += torch.bincount(tr_part_idx, minlength=args.num_partition).to(device)
                test_tp_partition_counts += torch.bincount(tp_part_idx, minlength=args.num_partition).to(device)
                test_mr_partition_counts += torch.bincount(mr_part_idx, minlength=args.num_partition).to(device)
                test_mp_partition_counts += torch.bincount(mp_part_idx, minlength=args.num_partition).to(device)

                test_total_label_loss += label_loss.item()
                test_total_fr_loss += fr_label_loss.item()
                test_total_fq_loss += fq_label_loss.item()
                test_total_tr_loss += tr_label_loss.item()
                test_total_tp_loss += tp_label_loss.item()
                test_total_mr_loss += mr_label_loss.item()
                test_total_mp_loss += mp_label_loss.item()

                test_total_domain_loss += domain_loss.item()
                test_total_fr_domain_loss += fr_domain_loss.item()
                test_total_fq_domain_loss += fq_domain_loss.item()
                test_total_tr_domain_loss += tr_domain_loss.item()
                test_total_tp_domain_loss += tp_domain_loss.item()
                test_total_mr_domain_loss += mr_domain_loss.item()
                test_total_mp_domain_loss += mp_domain_loss.item()

                test_total_specialization_loss += loss_specialization.item()
                test_total_diversity_loss += loss_diversity.item()
                test_total_loss += loss.item()

                test_total_fr_correct += (torch.argmax(fr_out_part, dim=1) == fr_labels).sum().item()
                test_total_fq_correct += (torch.argmax(fq_out_part, dim=1) == fq_labels).sum().item()
                test_total_tr_correct += (torch.argmax(tr_out_part, dim=1) == tr_labels).sum().item()
                test_total_tp_correct += (torch.argmax(tp_out_part, dim=1) == tp_labels).sum().item()
                test_total_mr_correct += (torch.argmax(mr_out_part, dim=1) == mr_labels).sum().item()
                test_total_mp_correct += (torch.argmax(mp_out_part, dim=1) == mp_labels).sum().item()

                test_total_fr_domain_correct += (torch.argmax(fr_domain_out, dim=1) == fr_dlabels).sum().item()
                test_total_fq_domain_correct += (torch.argmax(fq_domain_out, dim=1) == fq_dlabels).sum().item()
                test_total_tr_domain_correct += (torch.argmax(tr_domain_out, dim=1) == tr_dlabels).sum().item()
                test_total_tp_domain_correct += (torch.argmax(tp_domain_out, dim=1) == tp_dlabels).sum().item()
                test_total_mr_domain_correct += (torch.argmax(mr_domain_out, dim=1) == mr_dlabels).sum().item()
                test_total_mp_domain_correct += (torch.argmax(mp_domain_out, dim=1) == mp_dlabels).sum().item()

                test_total_samples += bs_fr

            if test_total_samples == 0: test_total_samples = 1

            test_fr_log = get_label_partition_log_data(test_fr_label_partition_counts, 'F-R', args.num_classes, args.num_partition, "Test")
            test_fq_log = get_label_partition_log_data(test_fq_label_partition_counts, 'F-Q', args.num_classes, args.num_partition, "Test")
            test_tr_log = get_label_partition_log_data(test_tr_label_partition_counts, 'T-R', args.num_classes, args.num_partition, "Test")
            test_tp_log = get_label_partition_log_data(test_tp_label_partition_counts, 'T-P', args.num_classes, args.num_partition, "Test")
            test_mr_log = get_label_partition_log_data(test_mr_label_partition_counts, 'M-R', args.num_classes, args.num_partition, "Test")
            test_mp_log = get_label_partition_log_data(test_mp_label_partition_counts, 'M-P', args.num_classes, args.num_partition, "Test")

            test_fr_ratios = test_fr_partition_counts / test_total_samples * 100
            test_fq_ratios = test_fq_partition_counts / test_total_samples * 100
            test_tr_ratios = test_tr_partition_counts / test_total_samples * 100
            test_tp_ratios = test_tp_partition_counts / test_total_samples * 100
            test_mr_ratios = test_mr_partition_counts / test_total_samples * 100
            test_mp_ratios = test_mp_partition_counts / test_total_samples * 100

            test_fr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_fr_ratios)])
            test_fq_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_fq_ratios)])
            test_tr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_tr_ratios)])
            test_tp_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_tp_ratios)])
            test_mr_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_mr_ratios)])
            test_mp_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_mp_ratios)])

            test_fr_dom_avg_loss = test_total_fr_domain_loss / test_total_samples
            test_fq_dom_avg_loss = test_total_fq_domain_loss / test_total_samples
            test_tr_dom_avg_loss = test_total_tr_domain_loss / test_total_samples
            test_tp_dom_avg_loss = test_total_tp_domain_loss / test_total_samples
            test_mr_dom_avg_loss = test_total_mr_domain_loss / test_total_samples
            test_mp_dom_avg_loss = test_total_mp_domain_loss / test_total_samples

            test_fr_avg_loss = test_total_fr_loss / test_total_samples
            test_fq_avg_loss = test_total_fq_loss / test_total_samples
            test_tr_avg_loss = test_total_tr_loss / test_total_samples
            test_tp_avg_loss = test_total_tp_loss / test_total_samples
            test_mr_avg_loss = test_total_mr_loss / test_total_samples
            test_mp_avg_loss = test_total_mp_loss / test_total_samples

            test_total_samples_all = test_total_samples * 6
            test_domain_avg_loss = test_total_domain_loss / test_total_samples_all
            test_label_avg_loss = test_total_label_loss / test_total_samples_all
            test_specialization_loss = test_total_specialization_loss / test_total_samples_all
            test_diversity_loss = test_total_diversity_loss / test_total_samples_all
            test_total_avg_loss = test_total_loss / test_total_samples_all

            test_fr_acc = test_total_fr_correct / test_total_samples * 100
            test_fq_acc = test_total_fq_correct / test_total_samples * 100
            test_tr_acc = test_total_tr_correct / test_total_samples * 100
            test_tp_acc = test_total_tp_correct / test_total_samples * 100
            test_mr_acc = test_total_mr_correct / test_total_samples * 100
            test_mp_acc = test_total_mp_correct / test_total_samples * 100

            current_avg_acc = (test_fr_acc + test_fq_acc + test_tr_acc + test_tp_acc + test_mr_acc + test_mp_acc) / 6.0

            test_fr_dom_acc = test_total_fr_domain_correct / test_total_samples * 100
            test_fq_dom_acc = test_total_fq_domain_correct / test_total_samples * 100
            test_tr_dom_acc = test_total_tr_domain_correct / test_total_samples * 100
            test_tp_dom_acc = test_total_tp_domain_correct / test_total_samples * 100
            test_mr_dom_acc = test_total_mr_domain_correct / test_total_samples * 100
            test_mp_dom_acc = test_total_mp_domain_correct / test_total_samples * 100

            if (epoch + 1) >= min_save_epoch and current_avg_acc > best_avg_acc:
                best_avg_acc = current_avg_acc
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved to {save_path} (Epoch: {epoch + 1}, Avg Acc: {current_avg_acc:.2f}%) ***")


            if (epoch + 1) % save_interval == 0:
                periodic_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), periodic_save_path)
                print(f"--- Periodic checkpoint saved to {periodic_save_path} ---")

            # ---------------------------
            print(f'Epoch [{epoch + 1}/{num_epochs}] (Test)')
            print(f'  [Ratios] F-R: [{test_fr_ratio_str}] | F-Q: [{test_fq_ratio_str}]')
            print(f'  [Ratios] T-R: [{test_tr_ratio_str}] | T-P: [{test_tp_ratio_str}]')
            print(f'  [Ratios] M-R: [{test_mr_ratio_str}] | M-P: [{test_mp_ratio_str}]')
            print(f'  [Acc]    F-R: {test_fr_acc:<6.2f}% | F-Q: {test_fq_acc:<6.2f}% | T-R: {test_tr_acc:<6.2f}% | T-P: {test_tp_acc:<6.2f}% | M-R: {test_mr_acc:<6.2f}% | M-P: {test_mp_acc:<6.2f}%')
            print(f'  [DomAcc] F-R: {test_fr_dom_acc:<6.2f}% | F-Q: {test_fq_dom_acc:<6.2f}% | T-R: {test_tr_dom_acc:<6.2f}% | T-P: {test_tp_dom_acc:<6.2f}% | M-R: {test_mr_dom_acc:<6.2f}% | M-P: {test_mp_dom_acc:<6.2f}%')
            print(f'  [Reg]    Spec:  {test_specialization_loss:<8.4f} | Div:    {test_diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
            print(f'  [Loss]   Label: {test_label_avg_loss:<8.4f} | Domain: {test_domain_avg_loss:<8.4f} | Total: {test_total_avg_loss:<8.4f}')
            print(f'  [Label]  F-R: {test_fr_avg_loss:<6.4f} | F-Q: {test_fq_avg_loss:<6.4f} | T-R: {test_tr_avg_loss:<6.4f} | T-P: {test_tp_avg_loss:<6.4f} | M-R: {test_mr_avg_loss:<6.4f} | M-P: {test_mp_avg_loss:<6.4f}')
            print(f'  [Domain] F-R: {test_fr_dom_avg_loss:<6.4f} | F-Q: {test_fq_dom_avg_loss:<6.4f} | T-R: {test_tr_dom_avg_loss:<6.4f} | T-P: {test_tp_dom_avg_loss:<6.4f} | M-R: {test_mr_dom_avg_loss:<6.4f} | M-P: {test_mp_dom_avg_loss:<6.4f}')

            wandb.log({
                **{f"Test/Partition {p} F-R Ratio": r.item() for p, r in enumerate(test_fr_ratios)},
                **{f"Test/Partition {p} F-Q Ratio": r.item() for p, r in enumerate(test_fq_ratios)},
                **{f"Test/Partition {p} T-R Ratio": r.item() for p, r in enumerate(test_tr_ratios)},
                **{f"Test/Partition {p} T-P Ratio": r.item() for p, r in enumerate(test_tp_ratios)},
                **{f"Test/Partition {p} M-R Ratio": r.item() for p, r in enumerate(test_mr_ratios)},
                **{f"Test/Partition {p} M-P Ratio": r.item() for p, r in enumerate(test_mp_ratios)},
                'Test/Label F-R Acc': test_fr_acc, 'Test/Label F-Q Acc': test_fq_acc,
                'Test/Label T-R Acc': test_tr_acc, 'Test/Label T-P Acc': test_tp_acc,
                'Test/Label M-R Acc': test_mr_acc, 'Test/Label M-P Acc': test_mp_acc,
                'Test/Domain F-R Acc': test_fr_dom_acc, 'Test/Domain F-Q Acc': test_fq_dom_acc,
                'Test/Domain T-R Acc': test_tr_dom_acc, 'Test/Domain T-P Acc': test_tp_dom_acc,
                'Test/Domain M-R Acc': test_mr_dom_acc, 'Test/Domain M-P Acc': test_mp_dom_acc,
                'TestLoss/Label F-R Loss': test_fr_avg_loss, 'TestLoss/Label F-Q Loss': test_fq_avg_loss,
                'TestLoss/Label T-R Loss': test_tr_avg_loss, 'TestLoss/Label T-P Loss': test_tp_avg_loss,
                'TestLoss/Label M-R Loss': test_mr_avg_loss, 'TestLoss/Label M-P Loss': test_mp_avg_loss,
                'TestLoss/Label Loss': test_label_avg_loss,
                'TestLoss/Domain F-R Loss': test_fr_dom_avg_loss, 'TestLoss/Domain F-Q Loss': test_fq_dom_avg_loss,
                'TestLoss/Domain T-R Loss': test_tr_dom_avg_loss, 'TestLoss/Domain T-P Loss': test_tp_dom_avg_loss,
                'TestLoss/Domain M-R Loss': test_mr_dom_avg_loss, 'TestLoss/Domain M-P Loss': test_mp_dom_avg_loss,
                'TestLoss/Domain Loss': test_domain_avg_loss,
                'TestLoss/Specialization Loss': test_specialization_loss,
                'TestLoss/Diversity Loss': test_diversity_loss,
                'TestLoss/Total Loss': test_total_avg_loss,
                **test_fr_log, **test_fq_log, **test_tr_log, **test_tp_log, **test_mr_log, **test_mp_log,
            }, step=epoch + 1)

        tau_scheduler.step()


if __name__ == '__main__':
    main()