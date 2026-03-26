#!/usr/bin/env python3
"""
Entry point for tokenization + transformer comparison experiments.

Runs two input variants:
1) utterance_clean
2) utterance_disfluency_tagged
"""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src" / "features"))

    from transformer_tokenization_experiments import main  # type: ignore  # noqa: E402

    main()
