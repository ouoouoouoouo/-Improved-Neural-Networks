import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq, hidden)
        scores  = self.W(x)                              # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)           # (batch, seq, 1)
        context = (x * weights).sum(dim=1)              # (batch, hidden)
        return context


class CBLA(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=256, num_classes=4, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)


        # ── CNN channel ──────────────────────────────────────
        self.cnn = nn.Sequential(
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
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.cnn_dense = nn.Linear(128, 128)

        # ── Bi-LSTM + Attention channel ───────────────────────
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention  = Attention(hidden_dim * 2)
        self.lstm_dense = nn.Linear(hidden_dim * 2, 128)

        # ── Fusion ────────────────────────────────────────────
        # CAT: 128 + 128 = 256
        self.fusion_dense = nn.Linear(256, 256)   # nonlinear change
        self.classifier   = nn.Linear(256, num_classes)

    def get_features(self, x):
        """Returns 256-dim fused feature (before classifier)."""
        cnn_out = x.transpose(1, 2)
        cnn_out = self.cnn(cnn_out)
        cnn_out = self.gap(cnn_out).squeeze(-1)
        cnn_out = torch.relu(self.cnn_dense(cnn_out))

        lstm_out, _ = self.bilstm(x)
        lstm_out    = self.attention(lstm_out)
        lstm_out    = torch.relu(self.lstm_dense(lstm_out))

        feat = torch.cat([cnn_out, lstm_out], dim=1)
        feat = torch.relu(self.fusion_dense(feat))
        return self.dropout(feat)

    def forward(self, x):
        return self.classifier(self.get_features(x))

if __name__ == '__main__':
    model = CBLA()
    print(model)
    x   = torch.randn(64, 100, 34)
    out = model(x)
    print("output shape:", out.shape)   # 期望 (64, 4)