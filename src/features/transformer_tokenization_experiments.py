from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def build_session_table(utterances_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
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
    return out.reset_index(drop=True)


def build_shared_splits(
    utterances_df: pd.DataFrame,
    split_column: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = build_session_table(utterances_df, text_column=split_column)
    y = sessions["label_binary"]

    train_val, test = train_test_split(
        sessions,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        random_state=seed,
        stratify=train_val["label_binary"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


@dataclass
class EncodedDataset(Dataset):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def encode_texts(
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: np.ndarray,
    max_length: int,
) -> EncodedDataset:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return EncodedDataset(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=torch.tensor(labels, dtype=torch.long),
    )


def run_epoch(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        total_loss += float(out.loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch["labels"].numpy()
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, float]:
    pred = (prob_pos >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)),
    }


def run_experiment(
    utterances_df: pd.DataFrame,
    train_base: pd.DataFrame,
    val_base: pd.DataFrame,
    test_base: pd.DataFrame,
    text_column: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    lr: float,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    lookup = build_session_table(utterances_df, text_column=text_column).set_index("file_id")

    def _attach_text(base_df: pd.DataFrame) -> pd.DataFrame:
        joined = base_df[["file_id", "label_binary"]].copy()
        joined["session_text"] = joined["file_id"].map(lookup["session_text"])
        joined = joined.dropna(subset=["session_text"])
        return joined

    train_df = _attach_text(train_base)
    val_df = _attach_text(val_base)
    test_df = _attach_text(test_base)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = encode_texts(
        tokenizer,
        train_df["session_text"].astype(str).tolist(),
        train_df["label_binary"].to_numpy(),
        max_length=max_length,
    )
    val_ds = encode_texts(
        tokenizer,
        val_df["session_text"].astype(str).tolist(),
        val_df["label_binary"].to_numpy(),
        max_length=max_length,
    )
    test_ds = encode_texts(
        tokenizer,
        test_df["session_text"].astype(str).tolist(),
        test_df["label_binary"].to_numpy(),
        max_length=max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None
    for _ in range(epochs):
        _ = run_epoch(model, train_loader, optimizer, device)
        y_val, p_val = predict(model, val_loader, device)
        val_metrics = compute_metrics(y_val, p_val)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = predict(model, test_loader, device)
    test_metrics = compute_metrics(y_test, p_test)
    test_metrics["n_train"] = int(len(train_df))
    test_metrics["n_val"] = int(len(val_df))
    test_metrics["n_test"] = int(len(test_df))
    test_metrics["seed"] = int(seed)
    return test_metrics


def main(argv: list[str] | None = None) -> None:
    root = project_root_from_here()
    parser = argparse.ArgumentParser(
        description="Run two transformer tokenization experiments and compare metrics."
    )
    parser.add_argument(
        "--utterances_csv",
        type=Path,
        default=root / "data" / "processed" / "child_utterances.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "data" / "transformer_experiments",
    )
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    args = parser.parse_args(argv)

    if not args.utterances_csv.exists():
        raise FileNotFoundError(f"Missing file: {args.utterances_csv}")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    utterances_df = pd.read_csv(args.utterances_csv, low_memory=False)

    train_base, val_base, test_base = build_shared_splits(
        utterances_df=utterances_df,
        split_column="utterance_clean",
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    experiments = [
        "utterance_clean",
        "utterance_disfluency_tagged",
    ]

    rows: list[dict[str, float | str]] = []
    for text_col in experiments:
        metrics = run_experiment(
            utterances_df=utterances_df,
            train_base=train_base,
            val_base=val_base,
            test_base=test_base,
            text_column=text_col,
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
        )
        row: dict[str, float | str] = {"experiment": text_col}
        row.update(metrics)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_dir / "tokenization_experiment_metrics.csv", index=False)
    (args.output_dir / "tokenization_experiment_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print("Done. Metrics saved to:")
    print(args.output_dir / "tokenization_experiment_metrics.csv")


if __name__ == "__main__":
    main()
