"""
Utterance-level TF-IDF embeddings for baseline models

CHANGES vs original:
1. ❌ Removed session-level aggregation (no groupby file_id)
2. ✅ Use utterance-level samples
3. ✅ Use predefined split (split_manifest_by_pid.csv)
4. ✅ Support 3 experiments:
   - embedding only
   - embedding + tokenization (disfluency)
   - embedding + START/END tokens
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -----------------------------
# Load + prepare utterance-level data
# -----------------------------
def load_utterance_data(
    utterances_path: Path,
    split_path: Path,
    text_column: str,
    add_special_tokens: bool,
):
    log.info("Loading utterances: %s", utterances_path)
    df = pd.read_csv(utterances_path, low_memory=False)

    # keep necessary columns only
    df = df[["file_id", text_column, "label_binary"]].dropna()

    # rename text column → unified name
    df = df.rename(columns={text_column: "text"})

    # optional: add START/END tokens
    if add_special_tokens:
        log.info("Adding START/END tokens")
        df["text"] = df["text"].apply(lambda x: f"<START> {x} <END>")

    # load predefined split
    split_df = pd.read_csv(split_path)[["file_id", "split"]]

    # merge
    df = df.merge(split_df, on="file_id", how="inner")

    log.info("Total utterances after merge: %d", len(df))
    log.info("Split counts:\n%s", df["split"].value_counts())

    return df


# -----------------------------
# Run TF-IDF pipeline
# -----------------------------
def run_pipeline(
    utterances_path: Path,
    split_path: Path,
    out_dir: Path,
    text_column: str,
    max_features: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    add_special_tokens: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_utterance_data(
        utterances_path,
        split_path,
        text_column,
        add_special_tokens,
    )

    # split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    log.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"]) if len(val_df) else None
    X_test = vectorizer.transform(test_df["text"])

    # save
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    sparse.save_npz(out_dir / "X_train_tfidf.npz", X_train)
    sparse.save_npz(out_dir / "X_test_tfidf.npz", X_test)

    if X_val is not None:
        sparse.save_npz(out_dir / "X_val_tfidf.npz", X_val)

    np.save(out_dir / "y_train.npy", train_df["label_binary"].to_numpy())
    np.save(out_dir / "y_test.npy", test_df["label_binary"].to_numpy())

    if len(val_df):
        np.save(out_dir / "y_val.npy", val_df["label_binary"].to_numpy())

    # metadata
    meta = {
        "level": "utterance",
        "text_column": text_column,
        "add_special_tokens": add_special_tokens,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "vocab_size": X_train.shape[1],
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    log.info("Done. Saved to %s", out_dir)


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--utterances_csv", type=Path, required=True)
    parser.add_argument("--split_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)

    parser.add_argument(
        "--text_column",
        default="utterance_clean",
        choices=[
            "utterance_clean",
            "utterance_surface",
            "utterance_disfluency_tagged",
            "utterance_raw",
        ],
    )

    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)

    parser.add_argument("--add_special_tokens", action="store_true")

    args = parser.parse_args()

    run_pipeline(
        utterances_path=args.utterances_csv,
        split_path=args.split_csv,
        out_dir=args.out_dir,
        text_column=args.text_column,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        add_special_tokens=args.add_special_tokens,
    )


if __name__ == "__main__":
    main()