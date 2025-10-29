import torch
import torch.nn as nn
from functions.ReverseLayerF import ReverseLayerF
from torch.nn.functional import gumbel_softmax


class TagNet(nn.Module):
    def __init__(self, num_classes=10, pre_classifier_out=128, n_partition=2, part_layer=128, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet, self).__init__()
        self.device = device
        self.n_partition = n_partition

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, pre_classifier_out),
            nn.LayerNorm(pre_classifier_out),
            nn.ReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(part_layer, num_domains)

        self.partition_switcher = nn.Linear(part_layer, n_partition)

        self.partitioned_classifier = nn.ModuleList()
        partition_hidden_size = part_layer // self.n_partition
        for _ in range(self.n_partition):
            sub_classifier = nn.Sequential(
                nn.Linear(pre_classifier_out, partition_hidden_size),
                nn.BatchNorm1d(partition_hidden_size),
                nn.ReLU(),
                nn.Linear(partition_hidden_size, num_classes)
            )
            self.partitioned_classifier.append(sub_classifier)

        self.to(self.device)


    def forward(self, input_data, alpha=1.0, tau=0.1, inference=False):
        feature = self.features(input_data)
        # feature = input_data
        feature = feature.view(feature.size(0), -1)
        feature = self.pre_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        partition_switcher_output = self.partition_switcher(domain_penul)

        if inference:
            partition_gumbel_or_probs = torch.softmax(partition_switcher_output, dim=1)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)
        else:
            partition_gumbel_or_probs = gumbel_softmax(partition_switcher_output, tau=tau, hard=True)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)

        # class_output_partitioned = torch.zeros(feature.size(0), self.classifier[-1].out_features, device=self.device)
        class_output_partitioned = torch.zeros(feature.size(0), self.partitioned_classifier[0][-1].out_features,
                                               device=self.device)
        for p_i in range(self.n_partition):
            indices = torch.where(partition_idx == p_i)[0]

            if len(indices) == 0:
                continue

            selected_features = feature[indices]

            xx = selected_features
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)

            class_output_partitioned[indices] = xx

        return class_output_partitioned, domain_output, partition_idx, partition_gumbel_or_probs


def TagNet_weights(model, lr, pre_weight=1.0, fc_weight=1.0, disc_weight=1.0, switcher_weight=1.0): # fc_weight 추가
    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.pre_classifier.parameters(), 'lr': lr * pre_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partitioned_classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr * switcher_weight},
    ]
