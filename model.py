import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextBiLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=4, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        self._init_weights()   # ← 最後呼叫

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)   # input-hidden weights
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)        # hidden-hidden weights（比 xavier 更適合 RNN）
            elif 'bias' in name:
                nn.init.zeros_(param)
        # classifier 也初始化
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def get_features(self, x):
        """Returns 512-dim bidirectional feature (before classifier)."""
        _, (h_n, _) = self.bilstm(x)
        h_n  = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)
        feat = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        return self.dropout(feat)

    def forward(self, x, lengths=None):
        return self.classifier(self.get_features(x))


if __name__ == '__main__':
    model = TextBiLSTM()
    print(model)
    x       = torch.randn(64, 500, 300)
    lengths = torch.randint(1, 500, (64,))
    out     = model(x, lengths)
    print("output shape:", out.shape)