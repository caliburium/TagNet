import torch
import torch.nn as nn
import torch.nn.functional as F


class TagNet(nn.Module):
    def __init__(self, num_classes=10, num_tasks=3,
                fc_hidden = 512, disc_hidden = 128,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet, self).__init__()
        self.device = device
        self.to(self.device)
        self.num_classes = num_classes
        self.num_tasks = num_tasks


        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(32 * 7 * 7, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, num_tasks)
        )

        self.classifiers = nn.ModuleList()

        for _ in range(num_tasks):
            classifier = nn.Sequential(
                nn.Linear(32 * 7 * 7, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, num_classes)
            )
            self.classifiers.append(classifier)


    def forward(self, x, tau=0.1, inference=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        task_out = self.discriminator(x)

        if not inference:  # mode == 'train'
            partition_prob = F.gumbel_softmax(task_out, tau=tau, hard=True)
            all_outs = []
            for classifier in self.classifiers:
                all_outs.append(classifier(x))

            stacked_outs = torch.stack(all_outs, dim=1)
            class_out = torch.bmm(partition_prob.unsqueeze(1), stacked_outs).squeeze(1)

        else :
            partition_prob = F.softmax(task_out, dim=1)
            partition_idx = torch.argmax(partition_prob,dim=1)

            class_out = torch.zeros(x.size(0), self.num_classes, device=x.device)

            for part_i in range(self.num_tasks):
                indices = torch.where(partition_idx == part_i)[0]

                if len(indices) == 0:
                    continue

                feat_batch = x[indices]
                classifier = self.classifiers[part_i]
                output_batch = classifier(feat_batch)
                class_out[indices] = output_batch

        return class_out, task_out, partition_prob


class TagNet_Alex(nn.Module):
    def __init__(self, num_classes=10, num_tasks=3,
                fc_hidden = 512, disc_hidden = 128,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet_Alex, self).__init__()
        self.device = device
        self.to(self.device)
        self.num_classes = num_classes
        self.num_tasks = num_tasks


        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(32 * 7 * 7, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, num_tasks)
        )

        self.classifiers = nn.ModuleList()

        for _ in range(num_tasks):
            classifier = nn.Sequential(
                nn.Linear(32 * 7 * 7, fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, num_classes)
            )
            self.classifiers.append(classifier)


    def forward(self, x, tau=0.1, inference=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        task_out = self.discriminator(x)

        if not inference:  # mode == 'train'
            partition_prob = F.gumbel_softmax(task_out, tau=tau, hard=True)
            all_outs = []
            for classifier in self.classifiers:
                all_outs.append(classifier(x))

            stacked_outs = torch.stack(all_outs, dim=1)
            class_out = torch.bmm(partition_prob.unsqueeze(1), stacked_outs).squeeze(1)

        else :
            partition_prob = F.softmax(task_out, dim=1)
            partition_idx = torch.argmax(partition_prob,dim=1)

            class_out = torch.zeros(x.size(0), self.num_classes, device=x.device)

            for part_i in range(self.num_tasks):
                indices = torch.where(partition_idx == part_i)[0]

                if len(indices) == 0:
                    continue

                feat_batch = x[indices]
                classifier = self.classifiers[part_i]
                output_batch = classifier(feat_batch)
                class_out[indices] = output_batch

        return class_out, task_out, partition_prob


def TagNet_weights(model, lr, fc_weight=1.0, disc_weight=1.0,):
    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.classifiers.parameters(), 'lr': lr * fc_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
    ]
