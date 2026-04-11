"""
Multimodal models for Audio-Textual SER (IEMOCAP).

Paper section 4.2.3:
  Dense layer units: 1024, 512, 4
  M_CBLA_BL  : CBLA(256) + BiLSTM(512) → cat(768) → DNN(1024→512→4)
  M_CBLA_BLR : same + L2 regularization (weight_decay in optimizer)
               "R" = Regularization, NOT Residual
"""

import torch
import torch.nn as nn

from cbla  import CBLA
from model import TextBiLSTM

# shared DNN units as stated in paper
_DNN = (1024, 512)   # final layer is num_classes


class M_CBLA_BL(nn.Module):
    """CBLA (audio) + BiLSTM (text) → DNN(1024→512→4). No L2."""

    def __init__(self,
                 audio_dropout: float = 0.3,
                 text_dropout:  float = 0.3,
                 num_classes:   int   = 4):
        super().__init__()
        self.audio_enc = CBLA(dropout=audio_dropout)       # → 256-dim
        self.text_enc  = TextBiLSTM(dropout=text_dropout)  # → 512-dim

        in_dim = 256 + 512   # 768

        self.dnn = nn.Sequential(
            nn.Linear(in_dim,    _DNN[0]), nn.ReLU(),
            nn.Linear(_DNN[0],   _DNN[1]), nn.ReLU(),
            nn.Linear(_DNN[1],   num_classes),
        )

    def forward(self, audio, text):
        a = self.audio_enc.get_features(audio)   # (B, 256)
        t = self.text_enc.get_features(text)     # (B, 512)
        x = torch.cat([a, t], dim=1)            # (B, 768)
        return self.dnn(x)


class M_CBLA_BLR(nn.Module):
    """
    CBLA (audio) + BiLSTM (text) → DNN(1024→512→4) + L2 regularization.
    'R' = L2 Regularization via weight_decay in the optimizer.
    Architecture is identical to M_CBLA_BL.
    """

    def __init__(self,
                 audio_dropout: float = 0.3,
                 text_dropout:  float = 0.3,
                 num_classes:   int   = 4):
        super().__init__()
        self.audio_enc = CBLA(dropout=audio_dropout)       # → 256-dim
        self.text_enc  = TextBiLSTM(dropout=text_dropout)  # → 512-dim

        in_dim = 256 + 512   # 768

        self.dnn = nn.Sequential(
            nn.Linear(in_dim,    _DNN[0]), nn.ReLU(),
            nn.Linear(_DNN[0],   _DNN[1]), nn.ReLU(),
            nn.Linear(_DNN[1],   num_classes),
        )

    def forward(self, audio, text):
        a = self.audio_enc.get_features(audio)   # (B, 256)
        t = self.text_enc.get_features(text)     # (B, 512)
        x = torch.cat([a, t], dim=1)            # (B, 768)
        return self.dnn(x)


if __name__ == '__main__':
    model = M_CBLA_BLR()
    print(model)
    audio = torch.randn(8, 100, 34)
    text  = torch.randn(8, 500, 300)
    out   = model(audio, text)
    print("output shape:", out.shape)   # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"total params: {total:,}")
