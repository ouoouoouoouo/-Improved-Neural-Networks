import torch
import torch.nn as nn

from cbla  import CBLA
from model import TextBiLSTM


class MCBLABLR(nn.Module):
    """
    M-CBLA-BLR: Multimodal fusion of S-CBLA (audio) + T-BL (text).

    Feature dimensions:
        S-CBLA  → 256-dim  (CNN + Bi-LSTM + Attention, fusion_dense output)
        T-BL    → 512-dim  (Bi-LSTM, hidden_dim * 2)
        Fused V = [H, S]   → 768-dim
        DNN     768 → 1024 → 512 → num_classes
    L2 regularization is handled via weight_decay in the optimizer.
    """

    def __init__(self,
                 audio_input_dim=34,  audio_hidden_dim=256,
                 text_input_dim=300,  text_hidden_dim=256,
                 num_classes=4):
        super().__init__()

        self.scbla = CBLA(
            input_dim=audio_input_dim,
            hidden_dim=audio_hidden_dim,
            num_classes=num_classes,
        )
        self.tbl = TextBiLSTM(
            input_dim=text_input_dim,
            hidden_dim=text_hidden_dim,
            num_classes=num_classes,
        )

        # fused: 256 (audio) + 512 (text) = 768
        fused_dim = audio_hidden_dim * 2 + text_hidden_dim * 2  # 256*1 is CBLA's fixed 256

        # The CBLA fusion_dense always outputs 256, regardless of hidden_dim
        # T-BL outputs text_hidden_dim * 2
        audio_feat_dim = 256
        text_feat_dim  = text_hidden_dim * 2
        fused_dim      = audio_feat_dim + text_feat_dim  # 256 + 512 = 768

        self.dnn = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, audio, text):
        # audio: (B, 100, 34)
        # text:  (B, 500, 300)
        s_feat = self.scbla.get_features(audio)  # (B, 256)
        h_feat = self.tbl.get_features(text)     # (B, 512)

        # V = [H, S]  (paper notation: text first)
        fused = torch.cat([h_feat, s_feat], dim=1)  # (B, 768)
        return self.dnn(fused)


if __name__ == '__main__':
    model = MCBLABLR()
    print(model)
    audio = torch.randn(4, 100, 34)
    text  = torch.randn(4, 500, 300)
    out   = model(audio, text)
    print("output shape:", out.shape)   # (4, 4)
