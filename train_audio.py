# train_audio.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

from dataset  import get_audio_dataloaders, EMOTIONS
#from scnn import SCNN
from cbla import CBLA

from constant import IEMOCAP_DIR

DATA_ROOT  = f"{IEMOCAP_DIR}/processed_paper"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 5e-4
IMPRO_ONLY = False   # True = improvised only（論文設定）; False = impro+script


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio, label in loader:
            audio = audio.to(DEVICE)
            out   = model(audio)
            pred  = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    wa = (all_preds == all_labels).mean() * 100
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    ua = per_class.mean()
    return wa, ua, per_class, all_preds, all_labels


def print_confusion_matrix(all_labels, all_preds, title="Confusion Matrix"):
    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print(f"\n{title}")
    print(f"{'':>6}", end="")
    for e in EMOTIONS:
        print(f"{e:>8}", end="")
    print()
    for i, e in enumerate(EMOTIONS):
        print(f"{e:>6}", end="")
        for j in range(4):
            print(f"{cm_norm[i][j]:>7.1f}%", end="")
        print(f"  (n={cm.sum(axis=1)[i]})")


def train(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_audio_dataloaders(DATA_ROOT, BATCH_SIZE,
                                                       impro_only=IMPRO_ONLY)

    model     = CBLA(dropout=0.3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    best_ua, best_wa = 0.0, 0.0
    best_preds = best_labels = best_per_class = None

    print(f"\n{'='*60}")
    print(f"  Seed={seed}  Device={DEVICE}  Epochs={EPOCHS}  LR={LR}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for audio, label in train_loader:
            audio, label = audio.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out  = model(audio)
            loss = criterion(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wa, ua, per_class, preds, labels = evaluate(model, test_loader)
        scheduler.step(ua)

        marker = ''
        if ua > best_ua:
            best_ua, best_wa = ua, wa
            best_per_class   = per_class.copy()
            best_preds       = preds.copy()
            best_labels      = labels.copy()
            torch.save(model.state_dict(), f'best_scbla_seed{seed}.pt')
            marker = '  ← best'

        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"WA={wa:.2f}%  UA={ua:.2f}%  "
              f"[ang={per_class[0]:.1f} exc={per_class[1]:.1f} "
              f"neu={per_class[2]:.1f} sad={per_class[3]:.1f}]{marker}")

    print(f"\nSeed {seed} Best  WA={best_wa:.2f}%  UA={best_ua:.2f}%")
    print_confusion_matrix(best_labels, best_preds,
                           title=f"Seed {seed} Best Confusion Matrix")
    return best_wa, best_ua


if __name__ == '__main__':
    seeds = [42, 123, 456]
    results = []
    for seed in seeds:
        wa, ua = train(seed)
        results.append((wa, ua))

    was = [r[0] for r in results]
    uas = [r[1] for r in results]
    print(f"\n{'='*60}")
    print(f"  3-run average  WA={sum(was)/3:.2f}%  UA={sum(uas)/3:.2f}%")
    print(f"  WA: {[f'{w:.2f}' for w in was]}")
    print(f"  UA: {[f'{u:.2f}' for u in uas]}")
    print(f"{'='*60}")
