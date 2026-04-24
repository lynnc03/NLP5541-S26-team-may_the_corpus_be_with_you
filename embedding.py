#!/usr/bin/env python3
"""
Generate TF-IDF features for all three text variants.

Run from repo root:
    python embedding.py              # utterance-level (default, matches transformer)
    python embedding.py --level session  # session-level (legacy)

Outputs:
    data/features/utterance/clean/
    data/features/utterance/disfluency/
    data/features/utterance/special_tokens/
"""

from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src" / "features"))

from tfidf_pipeline_update import run_pipeline  # noqa: E402

import argparse

VARIANTS = [
    ("clean",          "utterance_clean",             False),
    ("disfluency",     "utterance_disfluency_tagged", False),
    ("special_tokens", "utterance_clean",             True),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="utterance", choices=["session", "utterance"])
    p.add_argument("--utterances_csv", type=Path,
                   default=root / "data" / "processed" / "child_utterances.csv")
    p.add_argument("--split_csv", type=Path,
                   default=root / "split_manifest_by_pid.csv")
    p.add_argument("--out_dir", type=Path, default=root / "data" / "features")
    p.add_argument("--max_features", type=int, default=50_000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.utterances_csv.is_file():
        print(f"ERROR: {args.utterances_csv} not found. Run create_datasets.py first.")
        sys.exit(1)

    for name, text_col, special_tokens in VARIANTS:
        print(f"\n===== {name} ({args.level}-level) =====")
        out = args.out_dir / args.level / name
        run_pipeline(
            utterances_path=args.utterances_csv,
            split_path=args.split_csv,
            out_dir=out,
            text_column=text_col,
            test_size=0.15,
            val_size=0.15,
            seed=args.seed,
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            max_df=args.max_df,
            add_special_tokens=special_tokens,
            level=args.level,
        )
