import torch
import torch.nn as nn
from functions.ReverseLayerF import ReverseLayerF


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0), # 32 to 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 28 to 13
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 13 to 13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 13 to 6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0), # 6 to 4
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 4 * 4, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.Linear(60, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, alpha=1.0):
        feature = self.features(x)
        feature = torch.flatten(feature, 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output


def dann_weights(model, lr, feature_weight=1.0, fc_weight=1.0, disc_weight=1.0):

    return [
        {'params': model.features.parameters(), 'lr': lr * feature_weight},
        {'params': model.classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
    ]
