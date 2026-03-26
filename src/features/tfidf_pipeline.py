"""
Session-level TF-IDF embeddings from processed child utterances.

Reads data/processed/child_utterances.csv (output of create_datasets.py),
aggregates CHI utterances into one document per file_id, fits TF-IDF on
train split only, saves sparse matrices and a split manifest for baselines.
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
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def build_session_table(
    utterances_df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    if text_column not in utterances_df.columns:
        raise ValueError(f"Missing column {text_column!r} in utterances CSV")

    df = utterances_df.copy()
    df["utterance_index"] = pd.to_numeric(
        df.get("utterance_index", 0), errors="coerce"
    ).fillna(0).astype(int)
    df = df.sort_values(["file_id", "utterance_index"])

    def _join_series(ser: pd.Series) -> str:
        parts: list[str] = []
        for raw in ser.fillna(""):
            t = str(raw).strip()
            if t:
                parts.append(t)
        return " ".join(parts)

    session_text = (
        df.groupby("file_id", sort=False)[text_column]
        .agg(_join_series)
        .reset_index(name="session_text")
    )

    meta = (
        df.groupby("file_id", sort=False)
        .agg(
            label_binary=("label_binary", "first"),
            corpus=("corpus", "first"),
        )
        .reset_index()
    )

    out = meta.merge(session_text, on="file_id", how="inner")
    out["label_binary"] = pd.to_numeric(out["label_binary"], errors="coerce")
    out = out.dropna(subset=["label_binary", "session_text"])
    out = out[out["session_text"].str.len() > 0]
    out["label_binary"] = out["label_binary"].astype(int)
    return out


def make_splits(
    sessions: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 <= val_size < 1:
        raise ValueError("test_size and val_size must be in valid ranges")

    y = sessions["label_binary"]
    if y.nunique() < 2:
        raise ValueError("Need at least two classes in label_binary for stratified split")

    train_val, test = train_test_split(
        sessions,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    if len(train_val) == 0 or len(test) == 0:
        raise ValueError("Split produced empty train_val or test set")

    if val_size > 0:
        rel_val = val_size / (1.0 - test_size)
        y_tv = train_val["label_binary"]
        train, val = train_test_split(
            train_val,
            test_size=rel_val,
            random_state=seed,
            stratify=y_tv,
        )
    else:
        train, val = train_val, pd.DataFrame(columns=sessions.columns)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run_pipeline(
    utterances_path: Path,
    out_dir: Path,
    text_column: str,
    test_size: float,
    val_size: float,
    seed: int,
    max_features: int | None,
    ngram_max: int,
    min_df: int,
    max_df: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", utterances_path)
    utt = pd.read_csv(utterances_path, low_memory=False)
    sessions = build_session_table(utt, text_column=text_column)
    log.info("Sessions with non-empty text: %d", len(sessions))
    log.info("Label counts:\n%s", sessions["label_binary"].value_counts().sort_index().to_string())

    train_df, val_df, test_df = make_splits(
        sessions, test_size=test_size, val_size=val_size, seed=seed
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["session_text"])
    X_val = vectorizer.transform(val_df["session_text"]) if len(val_df) else None
    X_test = vectorizer.transform(test_df["session_text"])

    vectorizer_path = out_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(out_dir / "X_train_tfidf.npz", X_train)
    if X_val is not None and X_val.shape[0] > 0:
        sparse.save_npz(out_dir / "X_val_tfidf.npz", X_val)
    sparse.save_npz(out_dir / "X_test_tfidf.npz", X_test)

    np.save(out_dir / "y_train.npy", train_df["label_binary"].to_numpy())
    if len(val_df):
        np.save(out_dir / "y_val.npy", val_df["label_binary"].to_numpy())
    np.save(out_dir / "y_test.npy", test_df["label_binary"].to_numpy())

    manifest = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )[["file_id", "split", "label_binary", "corpus"]]
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    meta = {
        "utterances_csv": str(utterances_path.resolve()),
        "text_column": text_column,
        "n_sessions_train": int(len(train_df)),
        "n_sessions_val": int(len(val_df)),
        "n_sessions_test": int(len(test_df)),
        "vocabulary_size": int(X_train.shape[1]),
        "test_size": test_size,
        "val_size": val_size,
        "random_seed": seed,
        "vectorizer_path": vectorizer_path.name,
    }
    (out_dir / "embedding_run.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    log.info("Saved TF-IDF artifacts under %s", out_dir.resolve())
    log.info("Train matrix shape: %s, Test matrix shape: %s", X_train.shape, X_test.shape)


def main(argv: list[str] | None = None) -> None:
    root = project_root_from_here()
    default_utt = root / "data" / "processed" / "child_utterances.csv"
    default_out = root / "data" / "features"

    p = argparse.ArgumentParser(description="Build session-level TF-IDF embeddings.")
    p.add_argument(
        "--utterances_csv",
        type=Path,
        default=default_utt,
        help="Path to child_utterances.csv from create_datasets.py",
    )
    p.add_argument("--out_dir", type=Path, default=default_out)
    p.add_argument(
        "--text_column",
        default="utterance_clean",
        choices=[
            "utterance_clean",
            "utterance_surface",
            "utterance_disfluency_tagged",
            "utterance_raw",
        ],
        help="Which utterance text field to concatenate per session",
    )
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=50_000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    args = p.parse_args(argv)

    utt_path = args.utterances_csv
    if not utt_path.is_file():
        log.error(
            "File not found: %s\n"
            "Step 1: From repo root, run the preprocessing pipeline to create it:\n"
            "  cd src/preprocessing\n"
            "  python create_datasets.py --registry ../../file_info/files_master.csv "
            "--raw_root ../.. --output ../../data/processed/",
            utt_path,
        )
        sys.exit(1)

    run_pipeline(
        utterances_path=utt_path,
        out_dir=args.out_dir,
        text_column=args.text_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_features=args.max_features or None,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
    )


if __name__ == "__main__":
    main()
