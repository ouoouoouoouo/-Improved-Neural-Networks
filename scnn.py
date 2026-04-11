import torch
import torch.nn as nn


class SCNN(nn.Module):
    def __init__(self, input_dim=34, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64,  kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)   # ← 最後是 128
        
    def forward(self, x):
        # x: (B, 100, 34)
        x = x.transpose(1, 2)         # -> (B, 34, 100)
        x = self.features(x)          # -> (B, 256, 50)
        x = self.gap(x).squeeze(-1)   # -> (B, 256)
        return self.classifier(x)     # -> (B, 4)