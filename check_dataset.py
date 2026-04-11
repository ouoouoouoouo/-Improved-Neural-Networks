"""
check_dataset.py
執行方式: CUDA_VISIBLE_DEVICES=2 python check_dataset.py
"""

import os
import re
import glob
import numpy as np
from collections import Counter

from constant import IEMOCAP_DIR

DATA_ROOT    = f"{IEMOCAP_DIR}/processed_paper"
IEMOCAP_PATH = f"{IEMOCAP_DIR}/IEMOCAP_full_release"

EMOTIONS   = ['ang', 'exc', 'neu', 'sad']
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}

FEATURE_DIM  = 34
MAX_FRAMES   = 100
GLOVE_DIM    = 300
MAX_TEXT_LEN = 500

SAMPLE_SHOW  = 5   # 要印幾筆樣本細節


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def get_transcript(utt_id):
    """從 IEMOCAP transcriptions 讀出逐字稿"""
    # utt_id e.g. Ses05M_impro03_M024
    m = re.match(r'(Ses\d+[MF])_(impro\d+[a-z]?)_([MF]\d+)', utt_id)
    if not m:
        return "<parse error>"
    session_str = m.group(1)          # Ses05M
    dialog_id   = f"{session_str}_{m.group(2)}"  # Ses05M_impro03
    session_num = int(re.search(r'\d+', session_str).group())
    session_dir = f"Session{session_num}"

    trans_file = os.path.join(
        IEMOCAP_PATH, session_dir,
        'dialog', 'transcriptions', f"{dialog_id}.txt"
    )
    if not os.path.exists(trans_file):
        return "<transcript not found>"
    with open(trans_file, 'r') as f:
        for line in f:
            if line.startswith(utt_id):
                text = re.sub(r'^.*\[.*?\]:\s*', '', line).strip()
                return text
    return "<not found in file>"


def collect_samples(split):
    """回傳該 split 所有樣本的 dict list"""
    samples = []
    audio_root = os.path.join(DATA_ROOT, 'audio', split)
    for emotion in EMOTIONS:
        audio_dir = os.path.join(audio_root, emotion)
        if not os.path.exists(audio_dir):
            continue
        for audio_path in sorted(glob.glob(f"{audio_dir}/*.npy")):
            text_path = audio_path.replace('/audio/', '/text/')
            utt_id    = os.path.basename(audio_path).replace('.npy', '')
            samples.append({
                'utt_id':      utt_id,
                'audio_path':  audio_path,
                'text_path':   text_path,
                'raw_label':   emotion,
                'mapped_label': EMOTION2IDX[emotion],
            })
    return samples


def nonzero_frames(arr):
    """計算音訊實際非 padding 的幀數（最後一個非全零 row + 1）"""
    mask = np.any(arr != 0, axis=1)
    nz   = np.where(mask)[0]
    return int(nz[-1] + 1) if len(nz) > 0 else 0


def nonzero_tokens(arr):
    """計算文字實際非 padding 的 token 數"""
    mask = np.any(arr != 0, axis=1)
    return int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Label count
# ─────────────────────────────────────────────────────────────────────────────

def check_label_count(samples, split):
    counts = Counter(s['raw_label'] for s in samples)
    total  = len(samples)
    print(f"\n{'='*55}")
    print(f"  [{split.upper()}] Label Distribution  (total={total})")
    print(f"{'='*55}")
    for emo in EMOTIONS:
        n   = counts.get(emo, 0)
        pct = n / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        print(f"  {emo} (idx={EMOTION2IDX[emo]})  {n:4d}  {pct:5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sample detail
# ─────────────────────────────────────────────────────────────────────────────

def check_sample_detail(samples, split, n=SAMPLE_SHOW):
    print(f"\n{'='*55}")
    print(f"  [{split.upper()}] Sample Detail  (first {n})")
    print(f"{'='*55}")
    for s in samples[:n]:
        transcript = get_transcript(s['utt_id'])
        print(f"  utt_id      : {s['utt_id']}")
        print(f"  transcript  : {transcript[:80]}")
        print(f"  raw_label   : {s['raw_label']}")
        print(f"  mapped_label: {s['mapped_label']}")
        print(f"  {'-'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. audio_len / text_len 統計
# ─────────────────────────────────────────────────────────────────────────────

def check_length_stats(samples, split):
    audio_lens = []
    text_lens  = []
    for s in samples:
        audio = np.load(s['audio_path'])   # (100, 34)
        text  = np.load(s['text_path'])    # (500, 300)
        audio_lens.append(nonzero_frames(audio))
        text_lens.append(nonzero_tokens(text))

    for name, lens in [('audio_len (frames)', audio_lens),
                        ('text_len  (tokens)', text_lens)]:
        arr = np.array(lens)
        print(f"\n  {name}")
        print(f"    min={arr.min():.0f}  max={arr.max():.0f}  "
              f"mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
              f"std={arr.std():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NaN / Inf 檢查
# ─────────────────────────────────────────────────────────────────────────────

def check_nan_inf(samples, split):
    nan_audio = inf_audio = nan_text = inf_text = 0
    for s in samples:
        audio = np.load(s['audio_path'])
        text  = np.load(s['text_path'])
        if np.isnan(audio).any():  nan_audio += 1
        if np.isinf(audio).any():  inf_audio += 1
        if np.isnan(text).any():   nan_text  += 1
        if np.isinf(text).any():   inf_text  += 1

    print(f"\n{'='*55}")
    print(f"  [{split.upper()}] NaN / Inf Check")
    print(f"{'='*55}")
    print(f"  audio NaN: {nan_audio}  Inf: {inf_audio}")
    print(f"  text  NaN: {nan_text}   Inf: {inf_text}")
    status = "✅ 全部乾淨" if not any([nan_audio, inf_audio, nan_text, inf_text]) \
             else "⚠️  有問題，需要處理"
    print(f"  → {status}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Padding 比例
# ─────────────────────────────────────────────────────────────────────────────

def check_padding_ratio(samples, split):
    audio_pad_ratios = []
    text_pad_ratios  = []
    for s in samples:
        audio = np.load(s['audio_path'])
        text  = np.load(s['text_path'])

        real_a = nonzero_frames(audio)
        real_t = nonzero_tokens(text)

        audio_pad_ratios.append(1 - real_a / MAX_FRAMES)
        text_pad_ratios.append(1 - real_t / MAX_TEXT_LEN)

    print(f"\n{'='*55}")
    print(f"  [{split.upper()}] Padding Ratio")
    print(f"{'='*55}")
    for name, ratios in [('audio (max_frames=100)',    audio_pad_ratios),
                          ('text  (max_len=500)',       text_pad_ratios)]:
        arr = np.array(ratios)
        print(f"  {name}")
        print(f"    mean={arr.mean()*100:.1f}%  "
              f"median={np.median(arr)*100:.1f}%  "
              f"max={arr.max()*100:.1f}%  "
              f"truncated={( arr == 0.0 ).sum()}")

def check_zero_text(samples, split):
    print(f"\n{'='*55}")
    print(f"  [{split.upper()}] text_len == 0 的樣本")
    print(f"{'='*55}")
    zero_count = 0
    for s in samples:
        text = np.load(s['text_path'])
        if nonzero_tokens(text) == 0:
            zero_count += 1
            transcript = get_transcript(s['utt_id'])
            print(f"  utt_id    : {s['utt_id']}")
            print(f"  transcript: [{transcript}]")   # 用 [] 包住，空字串一眼看出
            print(f"  raw_label : {s['raw_label']}")
            print()
    print(f"  共 {zero_count} 筆")

# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for split in ['train', 'test']:
        print(f"\n{'#'*55}")
        print(f"  SPLIT: {split.upper()}")
        print(f"{'#'*55}")

        samples = collect_samples(split)

        check_label_count(samples, split)
        check_sample_detail(samples, split)

        print(f"\n{'='*55}")
        print(f"  [{split.upper()}] Audio / Text Length Stats")
        print(f"{'='*55}")
        check_length_stats(samples, split)

        check_nan_inf(samples, split)
        check_padding_ratio(samples, split)
        check_zero_text(samples, split) 


if __name__ == '__main__':
    main()
