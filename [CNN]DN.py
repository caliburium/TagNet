import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.DomainNetLoader import dn_loader
from model.SimpleCNN import SimpleCNN

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    args = parser.parse_args()

    num_epochs = args.epoch
    wandb.init(entity="hails",
               project="TagNet - DomainNet",
               config=args.__dict__,
               name="[CNN]DN_" + str(args.hidden_size)
                    + "_lr:" + str(args.lr)
                    + "_Batch:" + str(args.batch_size)
               )

    furniture_real_train, furniture_real_test = dn_loader('real', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)
    furniture_quickdraw_train, furniture_quickdraw_test = dn_loader('quickdraw', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)

    tool_real_train, tool_real_test = dn_loader('real', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)
    tool_painting_train, tool_painting_test = dn_loader('painting', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)

    mammal_real_train, mammal_real_test = dn_loader('real', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)
    mammal_paint_train, mammal_paint_test = dn_loader('painting', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)

    print("Data load complete, start training")

    model = SimpleCNN(num_classes=10, hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()

        total_fr_loss, total_fr_correct = 0, 0
        total_fq_loss, total_fq_correct = 0, 0
        total_tr_loss, total_tr_correct = 0, 0
        total_tp_loss, total_tp_correct = 0, 0
        total_mr_loss, total_mr_correct = 0, 0
        total_mp_loss, total_mp_correct = 0, 0

        total_label_loss, total_loss = 0, 0
        total_samples = 0

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

            optimizer.zero_grad()

            fr_outputs = model(fr_images)
            fq_outputs = model(fq_images)
            tr_outputs = model(tr_images)
            tp_outputs = model(tp_images)
            mr_outputs = model(mr_images)
            mp_outputs = model(mp_images)

            fr_label_loss = criterion(fr_outputs, fr_labels)
            fq_label_loss = criterion(fq_outputs, fq_labels)
            tr_label_loss = criterion(tr_outputs, tr_labels)
            tp_label_loss = criterion(tp_outputs, tp_labels)
            mr_label_loss = criterion(mr_outputs, mr_labels)
            mp_label_loss = criterion(mp_outputs, mp_labels)

            label_loss = (
                    (fr_label_loss + fq_label_loss) / 2 +
                    (tr_label_loss + tp_label_loss) / 2 +
                    (mr_label_loss + mp_label_loss) / 2
            )

            loss = label_loss

            loss.backward()
            optimizer.step()

            # --- [수정] --- 6개 통계 누적
            bs_fr = fr_labels.size(0)  # TagNet과 동일한 샘플 수 계산 기준
            total_samples += bs_fr

            total_label_loss += label_loss.item()
            total_loss += loss.item()

            total_fr_loss += fr_label_loss.item()
            total_fq_loss += fq_label_loss.item()
            total_tr_loss += tr_label_loss.item()
            total_tp_loss += tp_label_loss.item()
            total_mr_loss += mr_label_loss.item()
            total_mp_loss += mp_label_loss.item()

            total_fr_correct += (torch.argmax(fr_outputs, dim=1) == fr_labels).sum().item()
            total_fq_correct += (torch.argmax(fq_outputs, dim=1) == fq_labels).sum().item()
            total_tr_correct += (torch.argmax(tr_outputs, dim=1) == tr_labels).sum().item()
            total_tp_correct += (torch.argmax(tp_outputs, dim=1) == tp_labels).sum().item()
            total_mr_correct += (torch.argmax(mr_outputs, dim=1) == mr_labels).sum().item()
            total_mp_correct += (torch.argmax(mp_outputs, dim=1) == mp_labels).sum().item()

        if total_samples == 0: total_samples = 1

        fr_avg_loss = total_fr_loss / total_samples
        fq_avg_loss = total_fq_loss / total_samples
        tr_avg_loss = total_tr_loss / total_samples
        tp_avg_loss = total_tp_loss / total_samples
        mr_avg_loss = total_mr_loss / total_samples
        mp_avg_loss = total_mp_loss / total_samples

        total_samples_all = total_samples * 6
        label_avg_loss = total_label_loss / total_samples_all
        total_avg_loss = total_loss / total_samples_all

        fr_acc = total_fr_correct / total_samples * 100
        fq_acc = total_fq_correct / total_samples * 100
        tr_acc = total_tr_correct / total_samples * 100
        tp_acc = total_tp_correct / total_samples * 100
        mr_acc = total_mr_correct / total_samples * 100
        mp_acc = total_mp_correct / total_samples * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  [Acc]    F-R: {fr_acc:<6.2f}% | F-Q: {fq_acc:<6.2f}% | T-R: {tr_acc:<6.2f}% | T-P: {tp_acc:<6.2f}% | M-R: {mr_acc:<6.2f}% | M-P: {mp_acc:<6.2f}%')
        print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
        print(f'  [Label]  F-R: {fr_avg_loss:<6.4f} | F-Q: {fq_avg_loss:<6.4f} | T-R: {tr_avg_loss:<6.4f} | T-P: {tp_avg_loss:<6.4f} | M-R: {mr_avg_loss:<6.4f} | M-P: {mp_avg_loss:<6.4f}')

        wandb.log({
            'Train/Label F-R Acc': fr_acc, 'Train/Label F-Q Acc': fq_acc,
            'Train/Label T-R Acc': tr_acc, 'Train/Label T-P Acc': tp_acc,
            'Train/Label M-R Acc': mr_acc, 'Train/Label M-P Acc': mp_acc,

            'TrainLoss/Label F-R Loss': fr_avg_loss, 'TrainLoss/Label F-Q Loss': fq_avg_loss,
            'TrainLoss/Label T-R Loss': tr_avg_loss, 'TrainLoss/Label T-P Loss': tp_avg_loss,
            'TrainLoss/Label M-R Loss': mr_avg_loss, 'TrainLoss/Label M-P Loss': mp_avg_loss,

            'TrainLoss/Label Loss': label_avg_loss,
            'TrainLoss/Total Loss': total_avg_loss,
            'Parameters/Learning Rate': optimizer.param_groups[0]['lr'],
        }, step=epoch + 1)

        model.eval()

        with ((torch.no_grad())):
            test_total_fr_loss, test_total_fr_correct = 0, 0
            test_total_fq_loss, test_total_fq_correct = 0, 0
            test_total_tr_loss, test_total_tr_correct = 0, 0
            test_total_tp_loss, test_total_tp_correct = 0, 0
            test_total_mr_loss, test_total_mr_correct = 0, 0
            test_total_mp_loss, test_total_mp_correct = 0, 0

            test_total_label_loss, test_total_loss = 0, 0
            test_total_samples = 0

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

                bs_fr, bs_fq = fr_images.size(0), fq_images.size(0)
                bs_tr, bs_tp = tr_images.size(0), tp_images.size(0)
                bs_mr, bs_mp = mr_images.size(0), mp_images.size(0)

                all_images = torch.cat((fr_images, fq_images, tr_images, tp_images, mr_images, mp_images), dim=0)
                all_outputs = model(all_images)

                idx1 = bs_fr
                idx2 = idx1 + bs_fq
                idx3 = idx2 + bs_tr
                idx4 = idx3 + bs_tp
                idx5 = idx4 + bs_mr

                fr_out_part = all_outputs[:idx1]
                fq_out_part = all_outputs[idx1:idx2]
                tr_out_part = all_outputs[idx2:idx3]
                tp_out_part = all_outputs[idx3:idx4]
                mr_out_part = all_outputs[idx4:idx5]
                mp_out_part = all_outputs[idx5:]

                fr_label_loss = criterion(fr_out_part, fr_labels)
                fq_label_loss = criterion(fq_out_part, fq_labels)
                tr_label_loss = criterion(tr_out_part, tr_labels)
                tp_label_loss = criterion(tp_out_part, tp_labels)
                mr_label_loss = criterion(mr_out_part, mr_labels)
                mp_label_loss = criterion(mp_out_part, mp_labels)

                label_loss = ((fr_label_loss + fq_label_loss) / 2
                              + (tr_label_loss + tp_label_loss) / 2
                              + (mr_label_loss + mp_label_loss) / 2
                              )

                loss = label_loss

                test_total_label_loss += label_loss.item()
                test_total_fr_loss += fr_label_loss.item()
                test_total_fq_loss += fq_label_loss.item()
                test_total_tr_loss += tr_label_loss.item()
                test_total_tp_loss += tp_label_loss.item()
                test_total_mr_loss += mr_label_loss.item()
                test_total_mp_loss += mp_label_loss.item()

                test_total_loss += loss.item()

                test_total_fr_correct += (torch.argmax(fr_out_part, dim=1) == fr_labels).sum().item()
                test_total_fq_correct += (torch.argmax(fq_out_part, dim=1) == fq_labels).sum().item()
                test_total_tr_correct += (torch.argmax(tr_out_part, dim=1) == tr_labels).sum().item()
                test_total_tp_correct += (torch.argmax(tp_out_part, dim=1) == tp_labels).sum().item()
                test_total_mr_correct += (torch.argmax(mr_out_part, dim=1) == mr_labels).sum().item()
                test_total_mp_correct += (torch.argmax(mp_out_part, dim=1) == mp_labels).sum().item()

                test_total_samples += bs_fr

            if test_total_samples == 0: test_total_samples = 1

            test_fr_avg_loss = test_total_fr_loss / test_total_samples
            test_fq_avg_loss = test_total_fq_loss / test_total_samples
            test_tr_avg_loss = test_total_tr_loss / test_total_samples
            test_tp_avg_loss = test_total_tp_loss / test_total_samples
            test_mr_avg_loss = test_total_mr_loss / test_total_samples
            test_mp_avg_loss = test_total_mp_loss / test_total_samples

            test_total_samples_all = test_total_samples * 6
            test_label_avg_loss = test_total_label_loss / test_total_samples_all
            test_total_avg_loss = test_total_loss / test_total_samples_all

            test_fr_acc = test_total_fr_correct / test_total_samples * 100
            test_fq_acc = test_total_fq_correct / test_total_samples * 100
            test_tr_acc = test_total_tr_correct / test_total_samples * 100
            test_tp_acc = test_total_tp_correct / test_total_samples * 100
            test_mr_acc = test_total_mr_correct / test_total_samples * 100
            test_mp_acc = test_total_mp_correct / test_total_samples * 100

            print(f'Epoch [{epoch + 1}/{num_epochs}] (Test)')
            print(f'  [Acc]    F-R: {test_fr_acc:<6.2f}% | F-Q: {test_fq_acc:<6.2f}% | T-R: {test_tr_acc:<6.2f}% | T-P: {test_tp_acc:<6.2f}% | M-R: {test_mr_acc:<6.2f}% | M-P: {test_mp_acc:<6.2f}%')
            print(f'  [Loss]   Label: {test_label_avg_loss:<8.4f} | Total: {test_total_avg_loss:<8.4f}')
            print(f'  [Label]  F-R: {test_fr_avg_loss:<6.4f} | F-Q: {test_fq_avg_loss:<6.4f} | T-R: {test_tr_avg_loss:<6.4f} | T-P: {test_tp_avg_loss:<6.4f} | M-R: {test_mr_avg_loss:<6.4f} | M-P: {test_mp_avg_loss:<6.4f}')

            wandb.log({
                'Test/Label F-R Acc': test_fr_acc, 'Test/Label F-Q Acc': test_fq_acc,
                'Test/Label T-R Acc': test_tr_acc, 'Test/Label T-P Acc': test_tp_acc,
                'Test/Label M-R Acc': test_mr_acc, 'Test/Label M-P Acc': test_mp_acc,

                'TestLoss/Label F-R Loss': test_fr_avg_loss, 'TestLoss/Label F-Q Loss': test_fq_avg_loss,
                'TestLoss/Label T-R Loss': test_tr_avg_loss, 'TestLoss/Label T-P Loss': test_tp_avg_loss,
                'TestLoss/Label M-R Loss': test_mr_avg_loss, 'TestLoss/Label M-P Loss': test_mp_avg_loss,

                'TestLoss/Label Loss': test_label_avg_loss,
                'TestLoss/Total Loss': test_total_avg_loss,
            }, step=epoch + 1)


if __name__ == '__main__':
    main()