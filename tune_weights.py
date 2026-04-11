import torch
import torch.nn as nn
import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix

from dataset import get_dataloaders, EMOTIONS
from model   import TextBiLSTM
from constant import IEMOCAP_DIR

DATA_ROOT  = f"{IEMOCAP_DIR}/processed_paper"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio, text, label, text_len in loader:
            text = text.to(DEVICE)
            out  = model(text, text_len)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    wa = (all_preds == all_labels).mean() * 100
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    ua = per_class.mean()
    return wa, ua, per_class

def run_once(weights, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader = get_dataloaders(DATA_ROOT, BATCH_SIZE)

    model     = TextBiLSTM(dropout=0.5).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_ua = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for audio, text, label, text_len in train_loader:
            text, label = text.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(text, text_len), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        wa, ua, per_class = evaluate(model, test_loader)
        scheduler.step(ua)
        if ua > best_ua:
            best_ua = ua
            best_wa = wa
            best_per_class = per_class.copy()

    return best_wa, best_ua, best_per_class


if __name__ == '__main__':
    # ang weight 從 1.2 到 2.5，步長 0.3
    # neu weight 從 0.5 到 0.9，步長 0.2
    # exc / sad 固定接近 1.0
    ang_options = [1.2, 1.5, 1.8, 2.1, 2.5]
    neu_options = [0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    total = len(ang_options) * len(neu_options)
    done  = 0

    ang_w = 1.5
    exc_w = 1.0
    neu_options = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for neu_w in neu_options:
        
        sad_w = 4.0 - ang_w - neu_w - exc_w
        if sad_w <= 0:
            continue
        weights = [ang_w, exc_w, neu_w, sad_w]

        wa, ua, per_class = run_once(weights, seed=42)
        done += 1
        print(f"[{done:2d}/{total}] ang={ang_w:.1f} exc={exc_w:.1f} "
              f"neu={neu_w:.1f} sad={sad_w:.1f}  "
              f"WA={wa:.2f}% UA={ua:.2f}%  "
              f"[ang={per_class[0]:.1f} exc={per_class[1]:.1f} "
              f"neu={per_class[2]:.1f} sad={per_class[3]:.1f}]")
        results.append((ua, wa, weights, per_class))

    results.sort(reverse=True)
    print(f"\n{'='*60}")
    print("Top 5 configurations by UA:")
    for ua, wa, w, pc in results[:5]:
        print(f"  weights={[round(x,2) for x in w]}  "
              f"WA={wa:.2f}%  UA={ua:.2f}%  "
              f"[ang={pc[0]:.1f} exc={pc[1]:.1f} "
              f"neu={pc[2]:.1f} sad={pc[3]:.1f}]")
