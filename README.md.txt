Talkbank_project includes:

## NOTE
This project currently uses conversational transcript data (.cha) for modeling.
Raw files are expected under:

data/raw/

Processed files are generated under:

data/processed/

## Install dependencies

Run:

pip install -r requirements.txt.txt

## Core pipeline

1) Build processed datasets (if needed)

cd src/preprocessing
python create_datasets.py --registry ../../file_info/files_master.csv --raw_root ../.. --output ../../data/processed/

2) Build baseline-friendly embeddings (TF-IDF)

From project root:

python embedding.py

Outputs:

data/features/X_train_tfidf.npz
data/features/X_val_tfidf.npz
data/features/X_test_tfidf.npz
data/features/y_train.npy
data/features/y_val.npy
data/features/y_test.npy
data/features/tfidf_vectorizer.joblib
data/features/split_manifest.csv
data/features/embedding_run.json

3) Run transformer tokenization experiments (2 inputs)

From project root:

python tokenization.py --epochs 1

This script compares:
- utterance_clean
- utterance_disfluency_tagged

Transformer experiment outputs:

data/transformer_experiments/tokenization_experiment_metrics.csv
data/transformer_experiments/tokenization_experiment_metrics.json

## Processed data files

data/processed/child_utterances.csv
- One row per child utterance.
- Includes raw/cleaned text variants, CHAT feature counts, annotations, and metadata.

data/processed/all_utterances.csv
- Same structure as child_utterances, but includes all speakers.

data/processed/child_context_windows.csv
- Child utterances with context_before/context_after fields.

data/processed/session_level.csv
- One row per recording session with aggregated session features.

data/processed/pipeline_warnings.csv
- Parsing/data quality warnings (if any).

## Master file

file_info/files_master.csv
- One row per .cha file with metadata and inclusion flags.
- Includes columns like file_id, file_path, label, label_binary, age, sex, include_v1.

## Useful notes

label_binary:
- 0 = typically developing control
- 1 = SLI or language disorder
