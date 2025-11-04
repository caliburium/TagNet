import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=1536):
        super(SimpleCNN, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2), # 224 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 112-> 56
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 56 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 28 -> 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SimpleCNN32(nn.Module):
    def __init__(self, num_classes=10, hidden_size=1024):
        super(SimpleCNN32, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0), # 32 to 15
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 15 to 7
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def cnn_weights(model, lr, feature_weight=1.0, fc_weight=1.0):

    return [
        {'params': model.features.parameters(), 'lr': lr * feature_weight},
        {'params': model.classifier.parameters(), 'lr': lr * fc_weight},
    ]