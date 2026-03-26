#!/usr/bin/env python3
"""
Entry point for baseline-friendly embeddings (README: python embedding.py).

Delegates to src/features/tfidf_pipeline.py. Run from the repository root:

    python embedding.py

Requires data/processed/child_utterances.csv from create_datasets.py first.
"""

from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src" / "features"))

    from tfidf_pipeline import main  # type: ignore  # noqa: E402

    main()
