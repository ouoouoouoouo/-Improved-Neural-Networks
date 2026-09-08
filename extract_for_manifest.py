"""Extract features for utterances a manifest asks for but preprocess.py skipped.

preprocess.py filters on PAPER_EMOTIONS = {ang, exc, neu, sad}, so every `hap`
utterance was dropped before any feature was written. The merits 4-class
convention merges hap and exc into a single `happy` class, so switching to that
convention needs those files extracted after the fact.

This reads a manifest (utt_id,label,split), finds which utt_ids have no .npy
under the existing tree, and extracts only those into a flat top-up directory:

    <out>/text/<utt_id>.npy    (500, 300) GloVe, identical settings
    <out>/audio/<utt_id>.npy   (100, 34)  handcrafted, identical settings

Flat, with no label or split directories, because in the manifest-driven setup
ground truth comes from the manifest and never from a path. Point the
complementarity tool at both roots:

    features:
      glove:
        kind: npy_dir
        roots: [.../processed_paper/text, .../processed_common/text]

Nothing already extracted is overwritten, so this costs only the missing
utterances rather than a full re-run.

Because every missing utterance belongs to one class, an environment whose
pyAudioAnalysis differs from the one that produced the original tree would give
that class a systematic feature offset -- a perfect predictor of the label that
would inflate every downstream number invisibly. --verify re-extracts a sample
of already-extracted files and refuses to continue unless they reproduce.

Usage:
    python extract_for_manifest.py \
        --manifest ~/merc-complementarity/manifests/iemocap_common.csv \
        --existing data/iemocap/processed_paper \
        --out      data/iemocap/processed_common
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tqdm

from constant import IEMOCAP_DIR
from preprocess import (extract_34dim_features, extract_glove_features,
                        get_impro_wav_folders, get_transcript, get_wav_files,
                        load_glove)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--existing", required=True, type=Path,
                    help="tree already extracted by preprocess.py; anything "
                         "found here is skipped")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iemocap", default=f"{IEMOCAP_DIR}/IEMOCAP_full_release", type=str)
    ap.add_argument("--glove", default=f"{IEMOCAP_DIR}/glove.840B.300d.txt", type=str)
    ap.add_argument("--seed", type=int, default=0,
                    help="OOV vectors are random; fixed so a re-run reproduces")
    ap.add_argument("--verify", type=int, default=20,
                    help="re-extract this many ALREADY extracted audio files and "
                         "compare against the stored .npy, to prove this "
                         "environment reproduces the original extraction (0 skips)")
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        wanted = [str(r["utt_id"]) for r in csv.DictReader(f)]
    have = {p.stem for p in args.existing.rglob("*.npy")}
    have |= {p.stem for p in args.out.rglob("*.npy")} if args.out.exists() else set()
    missing = [u for u in wanted if u not in have]
    print(f"manifest {len(wanted)} utts | already extracted {len(wanted) - len(missing)} "
          f"| to extract {len(missing)}")

    # utt_id -> wav path, over the whole corpus (labels are irrelevant here;
    # the manifest already decided which utterances matter).
    wavs = {Path(w).stem: w for w in get_wav_files(get_impro_wav_folders(args.iemocap))}
    found = [u for u in missing if u in wavs]
    if len(found) != len(missing):
        lost = [u for u in missing if u not in wavs][:5]
        print(f"[warn] {len(missing) - len(found)} manifest utts have no wav under "
              f"{args.iemocap}; first few: {lost}")
    if missing and not found:
        raise SystemExit("no wav found for any missing utterance — check --iemocap")

    # Every missing utterance is of one class (happy), so if this environment's
    # pyAudioAnalysis differs from the one that produced processed_paper, the
    # resulting systematic feature shift becomes a perfect predictor of that
    # class and silently inflates every downstream number. Prove it matches
    # before extracting anything.
    if args.verify:
        existing = sorted(p for p in (args.existing / "audio").rglob("*.npy")
                          if p.stem in wavs)
        sample = existing[:: max(1, len(existing) // args.verify)][:args.verify]
        if not sample:
            print("[warn] --verify found nothing to check against")
        else:
            worst, worst_utt = 0.0, ""
            for p in tqdm.tqdm(sample, desc="verify"):
                d = float(np.abs(np.load(p) - extract_34dim_features(wavs[p.stem])).max())
                if d > worst:
                    worst, worst_utt = d, p.stem
            print(f"verify: {len(sample)} files, max abs diff {worst:.3e} ({worst_utt})")
            if worst > 1e-4:
                raise SystemExit(
                    "*** This environment does NOT reproduce the stored features. "
                    "Extracting the 595 happy utterances here would give them a "
                    "systematic offset that a classifier can read off as the label. "
                    "Use the environment that ran preprocess.py, or re-extract "
                    "everything from scratch in this one. Pass --verify 0 only if "
                    "you have another reason to believe this is safe. ***")
            print("verify: OK, this environment matches the original extraction")

    if not missing:
        print("nothing to do")
        return

    np.random.seed(args.seed)          # extract_glove_features draws OOV vectors
    glove = load_glove(args.glove)

    (args.out / "text").mkdir(parents=True, exist_ok=True)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    failed = []
    for utt in tqdm.tqdm(found, desc="extract"):
        wav = wavs[utt]
        try:
            np.save(args.out / "audio" / f"{utt}.npy", extract_34dim_features(wav))
            np.save(args.out / "text" / f"{utt}.npy",
                    extract_glove_features(get_transcript(wav), glove))
        except Exception as e:                       # noqa: BLE001
            failed.append((utt, repr(e)))

    print(f"\nwrote {len(found) - len(failed)} utterances to {args.out}")
    if failed:
        print(f"[warn] {len(failed)} failed, first few:")
        for utt, err in failed[:5]:
            print(f"  {utt}: {err}")


if __name__ == "__main__":
    main()
