import torch
import torch.nn as nn
import torch.nn.functional as F
from functions.ReverseLayerF import ReverseLayerF


class TagNet(nn.Module):
    def __init__(self, num_classes=10, num_partition=3, num_domains=3,
                fc_hidden = 768, disc_hidden = 256,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet, self).__init__()
        self.device = device
        self.to(self.device)
        self.num_domains = num_domains
        self.partition_size = fc_hidden // num_domains
        self.input_dim = 16 * 7 * 7
        # self.input_dim = 1 * 28 * 28

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(self.input_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(fc_hidden, disc_hidden),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(disc_hidden, num_domains)
        self.partition_switcher = nn.Linear(disc_hidden, num_partition)

        self.classifiers = nn.ModuleList()

        for _ in range(num_domains):
            classifier = nn.Sequential(
                nn.Linear(fc_hidden, self.partition_size),
                nn.ReLU(),
                nn.Linear(self.partition_size, num_classes)
            )
            self.classifiers.append(classifier)


    def forward(self, x, grl_lambda=1.0, tau=0.1, inference=False):
        x = self.features(x)
        reverse_feature = ReverseLayerF.apply(x, grl_lambda)
        domain_penul = self.discriminator(reverse_feature)
        domain_out = self.discriminator_fc(domain_penul)
        switcher_out = self.partition_switcher(domain_penul)

        if not inference:  # mode == 'train'
            partition_prob = F.gumbel_softmax(switcher_out, tau=tau, hard=False)
            partition_idx = torch.argmax(partition_prob, dim=1)
            all_outs = []

            for classifier in self.classifiers:
                all_outs.append(classifier(x))

            stacked_outs = torch.stack(all_outs, dim=1)
            class_out = torch.bmm(partition_prob.unsqueeze(1), stacked_outs).squeeze(1)


        else:  # mode == 'test'
            # partition_prob = F.softmax(switcher_out, dim=1)
            partition_prob = F.gumbel_softmax(switcher_out, tau=tau, hard=True)
            partition_idx = torch.argmax(partition_prob, dim=1)

            all_outs = []

            for classifier in self.classifiers:
                all_outs.append(classifier(x))

            stacked_outs = torch.stack(all_outs, dim=1)
            class_out = torch.bmm(partition_prob.unsqueeze(1), stacked_outs).squeeze(1)
            #
            # class_out = torch.zeros(x.size(0), 10, device=x.device)
            # for part_i in range(self.num_domains):
            #     indices = torch.where(partition_idx == part_i)[0]
            #
            #     if len(indices) == 0:
            #         continue
            #
            #     feat_batch = x[indices]
            #     classifier = self.classifiers[part_i]
            #     output_batch = classifier(feat_batch)
            #     class_out[indices] = output_batch

        return class_out, domain_out, partition_idx, partition_prob

def TagNet_weights(model, lr, fc_weight=1.0, disc_weight=1.0, switcher_weight=1.0):
    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.classifiers.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr * switcher_weight},
    ]