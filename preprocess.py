import glob
import os
import re
import string
import numpy as np
import tqdm
import librosa

from pyAudioAnalysis import ShortTermFeatures as aF

from constant import IEMOCAP_DIR

# ─── 論文設定 ────────────────────────────────────────────────────────────
# Section 5: "trained with Sessions 1–4, tested with Session 5"
TRAIN_SESSIONS = {1, 2, 3, 4}
TEST_SESSIONS  = {5}

PAPER_EMOTIONS = {'ang', 'exc', 'neu', 'sad'}

MAX_FRAMES  = 100
FEATURE_DIM = 34

MAX_TEXT_LEN = 500
GLOVE_DIM    = 300


# ─── 檔案發現 ─────────────────────────────────────────────────────────────

def get_impro_wav_folders(iemocap_path):
    wav_folders = glob.glob(iemocap_path + '/Session*/sentences/wav/*')
    return wav_folders   # ← 不再過濾 impro，全部包含

def get_wav_files(wav_folders):
    wav_files = []
    for f in wav_folders:
        wav_files.extend(glob.glob(f'{f}/*.wav'))
    return wav_files


def get_session(fname):
    """從路徑抽出 session 編號（int），e.g. Session1 → 1"""
    session_dir = fname.split('/')[-5]          # e.g. "Session1"
    return int(re.search(r'Session(\d+)', session_dir).group(1))


def get_class(fname):
    parts       = fname.split('/')
    session_id  = parts[-5]
    dialog_id   = parts[-2]
    sentence_id = os.path.basename(fname).split('.')[0]
    dataset_base = '/'.join(fname.split('/')[:-5])

    emo_file = f"{dataset_base}/{session_id}/dialog/EmoEvaluation/{dialog_id}.txt"
    with open(emo_file, 'r') as f:
        for line in f:
            if sentence_id in line:
                return line.split('\t')[2].strip()
    return None


# ─── 音訊特徵：(100, 34) ─────────────────────────────────────────────────

def extract_34dim_features(wav_path, max_frames=MAX_FRAMES):
    y, sr = librosa.load(wav_path, sr=16000)

    frame_size = int(sr * 0.025)   # 25ms
    frame_step = int(sr * 0.010)   # 10ms

    features, _ = aF.feature_extraction(y, sr, frame_size, frame_step)
    features = features.T                   # (num_frames, num_features)
    features = features[:, :FEATURE_DIM]   # 截到 34 維（避免版本差異）

    num_frames = features.shape[0]
    if num_frames >= max_frames:
        features = features[:max_frames, :]
    else:
        pad = np.zeros((max_frames - num_frames, FEATURE_DIM))
        features = np.vstack([features, pad])

    return features.astype('float32')       # (100, 34)


def save_audio_feature(wav_path, dest_base, label):
    utt_id   = os.path.basename(wav_path).replace('.wav', '')
    out_path = os.path.join(dest_base, label, f"{utt_id}.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, extract_34dim_features(wav_path))


# ─── 文字特徵：(500, 300) ─────────────────────────────────────────────────

def load_glove(glove_path):
    print("Loading GloVe embeddings...")
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            parts = line.rstrip().split(' ')
            if len(parts) == GLOVE_DIM + 1:
                glove[parts[0]] = np.array(parts[1:], dtype='float32')
    print(f"Loaded {len(glove)} words.")
    return glove


def get_transcript(wav_path):
    parts        = wav_path.split('/')
    session_id   = parts[-5]
    dialog_id    = parts[-2]
    sentence_id  = os.path.basename(wav_path).split('.')[0]
    dataset_base = '/'.join(wav_path.split('/')[:-5])

    trans_file = f"{dataset_base}/{session_id}/dialog/transcriptions/{dialog_id}.txt"
    with open(trans_file, 'r') as f:
        for line in f:
            if line.startswith(sentence_id):
                text = re.sub(r'^.*\[.*?\]:\s*', '', line).strip()
                return text
    return ""


def clean_transcript(text):
    # step1: 移除非語言標記
    text = re.sub(r'\[[A-Z_]+\]', '', text)
    # step2: 只移除「獨立的」標點（句尾 . , ! ?），保留縮寫中的 '
    text = re.sub(r"[^\w\s']", ' ', text)   # 保留字母、數字、空白、apostrophe
    text = re.sub(r"\s+", ' ', text)         # 多餘空白壓縮
    return text.strip()

def extract_glove_features(transcript, glove, max_len=MAX_TEXT_LEN, dim=GLOVE_DIM):
    transcript = clean_transcript(transcript)
    tokens     = transcript.lower().split()
    
    vectors = []
    for t in tokens:
        if t in glove:
            vectors.append(glove[t])
        else:
            # OOV：小隨機噪聲，讓模型能與 padding 區分
            vectors.append(np.random.uniform(-0.01, 0.01, dim).astype('float32'))
    
    # padding 維持 zero vector
    if len(vectors) < max_len:
        vectors += [np.zeros(dim, dtype='float32')] * (max_len - len(vectors))
    else:
        vectors = vectors[:max_len]
    
    return np.array(vectors, dtype='float32')

def save_text_feature(wav_path, dest_base, label, glove):
    utt_id   = os.path.basename(wav_path).replace('.wav', '')
    out_path = os.path.join(dest_base, label, f"{utt_id}.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    transcript = get_transcript(wav_path)
    np.save(out_path, extract_glove_features(transcript, glove))


# ─── 主流程 ───────────────────────────────────────────────────────────────

def main(iemocap_path, destination, glove_path):
    wav_folders = get_impro_wav_folders(iemocap_path)
    wav_files   = get_wav_files(wav_folders)

    # 依論文切分：Session 1-4 train, Session 5 test
    splits = {
        'train': [f for f in wav_files if get_session(f) in TRAIN_SESSIONS],
        'test':  [f for f in wav_files if get_session(f) in TEST_SESSIONS],
    }

    glove = load_glove(glove_path)

    for split_name, files in splits.items():
        print(f"\n=== {split_name} ({len(files)} files before emotion filter) ===")

        audio_dest = os.path.join(destination, 'audio', split_name)
        text_dest  = os.path.join(destination, 'text',  split_name)

        skipped = 0
        for f in tqdm.tqdm(files, desc=split_name):
            label = get_class(f)
            if label not in PAPER_EMOTIONS:
                skipped += 1
                continue
            save_audio_feature(f, audio_dest, label)
            save_text_feature(f, text_dest, label, glove)

        print(f"  skipped {skipped} utterances (non-target emotion)")


if __name__ == '__main__':
    IEMOCAP_PATH = f"{IEMOCAP_DIR}/IEMOCAP_full_release"
    DESTINATION  = f"{IEMOCAP_DIR}/processed_paper"   # 扁平結構，不再有 fold 層
    GLOVE_PATH   = f"{IEMOCAP_DIR}/glove.840B.300d.txt"
    main(IEMOCAP_PATH, DESTINATION, GLOVE_PATH)