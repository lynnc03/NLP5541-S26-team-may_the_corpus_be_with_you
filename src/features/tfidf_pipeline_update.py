"""
TF-IDF embeddings from processed child utterances.

Reads data/processed/child_utterances.csv (output of create_datasets.py),
fits TF-IDF on train split only, saves sparse matrices and a split manifest.

Two modes (--level):
  session   — aggregate all utterances per file_id into one document (legacy)
  utterance — one row per utterance, matching the transformer's granularity
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


def build_utterance_table(
    utterances_df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """One row per utterance — matches the transformer's granularity."""
    if text_column not in utterances_df.columns:
        raise ValueError(f"Missing column {text_column!r} in utterances CSV")

    df = utterances_df.copy()
    df["label_binary"] = pd.to_numeric(df["label_binary"], errors="coerce")
    df = df.dropna(subset=["label_binary", text_column])
    df = df[df[text_column].astype(str).str.strip().str.len() > 0].copy()
    df["label_binary"] = df["label_binary"].astype(int)
    df["session_text"] = df[text_column].astype(str).str.strip()
    keep = [c for c in ["file_id", "label_binary", "corpus", "session_text"] if c in df.columns]
    return df[keep].reset_index(drop=True)


#def make_splits(
#    sessions: pd.DataFrame,
#    test_size: float,
#    val_size: float,
#    seed: int,
#) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#    if not 0 < test_size < 1 or not 0 <= val_size < 1:
#        raise ValueError("test_size and val_size must be in valid ranges")

#    y = sessions["label_binary"]
#    if y.nunique() < 2:
#        raise ValueError("Need at least two classes in label_binary for stratified split")

#    train_val, test = train_test_split(
#        sessions,
#        test_size=test_size,
#        random_state=seed,
#        stratify=y,
#    )

#    if len(train_val) == 0 or len(test) == 0:
#        raise ValueError("Split produced empty train_val or test set")

#    if val_size > 0:
#        rel_val = val_size / (1.0 - test_size)
#        y_tv = train_val["label_binary"]
#        train, val = train_test_split(
#            train_val,
#            test_size=rel_val,
#            random_state=seed,
#            stratify=y_tv,
#        )
#    else:
#        train, val = train_val, pd.DataFrame(columns=sessions.columns)

#    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def apply_predefined_split(
    sessions: pd.DataFrame,
    split_path: Path,
):
    log.info("Loading split file: %s", split_path)
    split_df = pd.read_csv(split_path)[["file_id", "split"]]

    # Merge split info
    sessions = sessions.merge(split_df, on="file_id", how="inner")
    log.info("Columns after merge: %s", sessions.columns)

    train_df = sessions[sessions["split"] == "train"]
    val_df = sessions[sessions["split"] == "val"]
    test_df = sessions[sessions["split"] == "test"]

    log.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

def run_pipeline(
    utterances_path: Path,
    split_path: Path,
    out_dir: Path,
    text_column: str,
    test_size: float,
    val_size: float,
    seed: int,
    max_features: int | None,
    ngram_max: int,
    min_df: int,
    max_df: float,
    add_special_tokens: bool = False,
    level: str = "session",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", utterances_path)
    utt = pd.read_csv(utterances_path, low_memory=False)

    if level == "utterance":
        table = build_utterance_table(utt, text_column=text_column)
        log.info("Utterances with non-empty text: %d", len(table))
    else:
        table = build_session_table(utt, text_column=text_column)
        log.info("Sessions with non-empty text: %d", len(table))

    log.info("Label counts:\n%s", table["label_binary"].value_counts().sort_index().to_string())

    if add_special_tokens:
        log.info("Adding START/END tokens")
        table["session_text"] = table["session_text"].apply(
            lambda x: f"<START> {x} <END>"
        )

    train_df, val_df, test_df = apply_predefined_split(table, split_path)


    #train_df, val_df, test_df = make_splits(
    #    sessions, test_size=test_size, val_size=val_size, seed=seed
    #)

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
        "level": level,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
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
    default_split = root / "split_manifest_by_pid.csv"

    p = argparse.ArgumentParser(description="Build TF-IDF embeddings (session or utterance level).")
    p.add_argument("--utterances_csv", type=Path, default=default_utt)
    p.add_argument("--split_csv", type=Path, default=default_split,
                   help="PID-controlled split manifest (split_manifest_by_pid.csv)")
    p.add_argument("--out_dir", type=Path, default=default_out)
    p.add_argument(
        "--text_column",
        default="utterance_clean",
        choices=["utterance_clean", "utterance_surface", "utterance_disfluency_tagged", "utterance_raw"],
    )
    p.add_argument("--level", default="utterance", choices=["session", "utterance"],
                   help="Granularity: 'utterance' matches the transformer; 'session' aggregates per recording")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=50_000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--add_special_tokens", action="store_true")

    args = p.parse_args(argv)

    if not args.utterances_csv.is_file():
        log.error(
            "File not found: %s\n"
            "Run the preprocessing pipeline first:\n"
            "  python src/preprocessing/create_datasets.py",
            args.utterances_csv,
        )
        sys.exit(1)

    run_pipeline(
        utterances_path=args.utterances_csv,
        split_path=args.split_csv,
        out_dir=args.out_dir,
        text_column=args.text_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        add_special_tokens=args.add_special_tokens,
        level=args.level,
    )


if __name__ == "__main__":
    main()
