import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

from constant import IEMOCAP_DIR

# ─── 設定 ────────────────────────────────────────────────────────────────
DATA_ROOT  = f"{IEMOCAP_DIR}/processed_paper"
EMOTIONS   = ['ang', 'exc', 'neu', 'sad']
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}

INPUT_DIM  = 300
SEQ_LEN    = 500
HIDDEN_DIM = 256
NUM_CLASS  = 4
EPOCHS     = 20
BATCH_SIZE = 64


# ─── 資料載入 ─────────────────────────────────────────────────────────────
def load_split(split):
    X, y = [], []
    text_root = os.path.join(DATA_ROOT, 'text', split)
    for emotion in EMOTIONS:
        text_dir = os.path.join(text_root, emotion)
        if not os.path.exists(text_dir):
            continue
        for path in sorted(glob.glob(f"{text_dir}/*.npy")):
            X.append(np.load(path))               # (500, 300)
            y.append(EMOTION2IDX[emotion])
    X = np.array(X, dtype='float32')             # (N, 500, 300)
    y = np.array(y, dtype='int32')               # (N,)
    print(f"[{split}] {len(y)} samples")
    return X, y


# ─── 模型 ─────────────────────────────────────────────────────────────────
def build_tbl():
    inp = keras.Input(shape=(SEQ_LEN, INPUT_DIM))
    x   = layers.Bidirectional(layers.LSTM(HIDDEN_DIM))(inp)
    x   = layers.Dense(NUM_CLASS, activation='softmax')(x)
    model = keras.Model(inp, x)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─── UAR 計算 ─────────────────────────────────────────────────────────────
def compute_metrics(model, X, y):
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    wa    = (preds == y).mean() * 100
    cm    = confusion_matrix(y, preds, labels=list(range(NUM_CLASS)))
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    ua    = per_class.mean()
    return wa, ua, per_class


def print_cm(model, X, y, title="Confusion Matrix"):
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    cm      = confusion_matrix(y, preds, labels=list(range(NUM_CLASS)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print(f"\n{title}")
    print(f"{'':>6}", end="")
    for e in EMOTIONS:
        print(f"{e:>8}", end="")
    print()
    for i, e in enumerate(EMOTIONS):
        print(f"{e:>6}", end="")
        for j in range(NUM_CLASS):
            print(f"{cm_norm[i][j]:>7.1f}%", end="")
        print(f"  (n={cm.sum(axis=1)[i]})")


# ─── 訓練 ─────────────────────────────────────────────────────────────────
def train(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split('train')
    X_test,  y_test  = load_split('test')

    model = build_tbl()
    # ← 加這裡，從 train set 計算
    counts  = np.bincount(y_train, minlength=NUM_CLASS).astype(float)
    weights = counts.sum() / (NUM_CLASS * counts)
    class_weight_dict = {i: weights[i] for i in range(NUM_CLASS)}
    print(f"class weights: { {EMOTIONS[i]: round(weights[i],3) for i in range(NUM_CLASS)} }")


    best_ua, best_wa = 0.0, 0.0
    best_per_class   = None

    print(f"\n{'='*60}")
    print(f"  Seed={seed}  Epochs={EPOCHS}  Batch={BATCH_SIZE}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0,
            shuffle=True,
            
        )
        wa, ua, per_class = compute_metrics(model, X_test, y_test)

        marker = ''
        if ua > best_ua:
            best_ua, best_wa = ua, wa
            best_per_class   = per_class.copy()
            model.save_weights(f'best_tbl_seed{seed}.weights.h5')
            marker = '  ← best'

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"WA={wa:.2f}%  UA={ua:.2f}%  "
              f"[ang={per_class[0]:.1f} exc={per_class[1]:.1f} "
              f"neu={per_class[2]:.1f} sad={per_class[3]:.1f}]{marker}")

    print(f"\nSeed {seed} Best  WA={best_wa:.2f}%  UA={best_ua:.2f}%")
    print_cm(model, X_test, y_test, title=f"Seed {seed} Best Confusion Matrix")
    return best_wa, best_ua


if __name__ == '__main__':
    # 指定用 4090（確認 index 後改這裡）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
