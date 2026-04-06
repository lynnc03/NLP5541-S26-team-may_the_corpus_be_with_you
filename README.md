# Automatic Detection of Language Disorders from Conversational Speech


## Team Members

| Name | Email |
|------|-------|
| **Ray Amberg** | amber079@umn.edu | 
| **Ning-Shan Chang** | chan2497@umn.edu | 
| **Gretchen Corcoran** | gcorcora@umn.edu | 
| **Alan Yan** | yan00463@umn.edu | 

## Problem Definition
Speech language disorders, including specific language impairment (SLI)/developmental language disorder (DLD), diagnosis relies heavily on manual analysis of conversational language samples by speech-language pathologists. Time-intensive and subjective diagnosis underscores the need for automated screening.

## Goal: 
Develop NLP models that assist early screening of language disorders


## Table of Contents
- [Repository Structure](#repository-structure)
- [Dependencies/Installation](#dependencies)
- [Data](#data)
- [Step 1: Preprocessing Pipeline](#step-1-preprocessing-pipeline)
- [Step 2: Features and Embeddings](#step-2-features-and-embeddings)
- [Step 3: Baseline Models](#step-3-baseline-models)
- [Step 4: Transformer Model](#step-4-transformer-model)
- [Transformer Results](#results)
- [Performance Comparisons](#performance-comparisons)
- [References](#references)


## Repository Structure
```plaintext
project_root/
│
├── data/
│   ├── features/                         #Initial features and train/test split ****NOTE: redo due to leakage.
│   │   ├── X_train_tfidf.npz
│   │   ├── X_val_tfidf.npz
│   │   ├── X_test_tfidf.npz
│   │   ├── y_train.npy / y_val.npy / y_test.npy
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── split_manifest.csv            #Initial train/val/test split ***NOTE: Redo due to leakage. Don't use, use later split manifest
│   │   └── embedding_run.json
│   └── transformer_experiments/          #Tokenization experiment outputs **NOTE: Redo due to stratificaiton leakage
│
├── file_info/
│   └── files_master.csv                  #Master registry of all .cha files (manually annotated)
│
├── notebooks/
│   └── nlp_project_exploration.ipynb     #Initial .cha file exploration (not formal EDA)
│
├── preprocessing/                        #Core preprocessing pipeline
│   ├── create_master_csv.py              #Step 1a: Scans corpus dirs, generate file registry
│   ├── parse_data.py                     #Step 1b: Parse .cha files - ParsedSpeech objects
│   ├── data_classes.py                   #Dataclasses: ParsedSpeech, Utterance, Metadata, etc.
│   ├── clean_text.py                     #Step 1c: CHAT notation - clean/surface/tagged text
│   └── create_datasets.py                #Step 1d: Orchestrate pipeline, write output CSVs
│
├── src/
│   ├── features/
│   │   ├── tfidf_pipeline.py             #TF-IDF feature extraction pipeline **REDO due to leakage??
│   │   └── transformer_tokenization_ex.. #Tokenization experiments **REDO due to leakage?
│   └── models/
│       └── majority_classifier.py        #Majority class baseline **REDO?
│
├── LogisticR.py                          #Logistic regression baseline
├── embedding.py                          #TF-IDF embedding pipeline
├── majorityC.py                          #Majority class baseline
├── tokenization.py                       #TODO: if you made this, fill in info
├── split_manifest_by_pid.csv             #Corrected train/val/test split by participant ID 
│                                         #   — use this for the transformer, not split_manifest.csv
├── transformerB.py                       #Transformer model definition
├── transformerT.py                       #Transformer training script
├── load_data.py                          #Data loader for transformer pipeline
├── split_by_pid.py                       #Generates split_manifest_by_pid.csv
├── requirements.txt
└── README.md
```

Notes: data/features/split_manifest.csv contains a data split that does not separate participants by PID across train/val/test sets, instead using splits from session ID. This creates data leakage due to some participnats having multiple sessions. Instead, split_manifest_by_pid.csv (root directory) is the corrected split used for the transformer, which ensures no child appears in more than one split.
TODO: Duplicate files: Several files appear in both the root directory and src/ (e.g., majorityC.py and src/models/majority_classifier.py). Please update README + structure to specify which one is current version.

## Dependencies/Installation

```bash
pip install -r requirements.txt
```
Key dependencies: pylangacq, pandas, transformers, torch, numpy, scikit-learn

## Data

Source: SCLARIN TalkBank / CHILDES conversational transcripts (.cha format)

Processed data files are too large for GitHub and are available on Google Drive:
https://drive.google.com/drive/u/1/folders/1gi70tvEbzI7_IevzK3NQrI6mrYAdFrxi

Raw .cha files can be downloaded from https://talkbank.org/childes/access/Clinical-Eng/ and should be placed in data/raw/ before running the preprocessing pipeline. Specific files used are described in /file_info/files_master.csv

Labeling: 

label_binary = 0 — typically developing control
label_binary = 1 — SLI or language disorder

V1 includes only controls and children marked SLI or language disorder. Other diagnoses (Down syndrome, hearing loss, late talker, etc.) are excluded from V1 but can be added easily in future versions.

## Step 1: Preprocessing Pipeline

All scripts are in the preprocessing/ directory. 

### How the scripts work:

data_classes.py contains the core data structures used throughout the pipeline: ParsedSpeech, Metadata, Utterance, CHATFeatures, and Annotations.
create_master_csv.py traverses your corpus directories, finds all .cha files, and extracts metadata (file path, age, sex) from each file's @ID header. It writes file_info/files_master.csv, which were then manually annotated with. Manual annotation avoids errors with varying @ID header formatting, and errors due to differences in file availability over time. Manual labeling also helps deal with ambiguous or incomplete headers and labels.

	label: diagnostic category (e.g., "SLI", "control")
	label_binary: binary label (1/0)
	include_v1: whether to include in the current dataset (1/0)
	has_audio: whether paired audio exists

parse_data.py (CHAFileParser) parses each .cha file using pylangacq. It extracts utterances, CHAT notation features (pauses, disfluencies, errors, unintelligible speech, and so on), and speaker metadata.. It also attaches context windows before and after each child utterance.

clean_text.py (TextCleaner) takes raw CHAT-annotated text and produces three cleaned versions per utterance:

	text_clean: CHAT notation stripped, target forms substituted in place of errors
	text_surface: what the child actually said (surface forms preserved, not corrected)
	text_disfluency_tagged: disfluencies replaced with explicit tokens, such as [PAUSE], [REPEAT], [ERROR]

create_datasets.py orchestrates the full pipeline: reads files_master.csv, filters to include_v1 == 1, runs each file through the parser and cleaner, and writes four output CSVs.

### Running the initial preprocessing pipeline:

Step 1a. Generate master csv file:

```bash
python preprocessing/create_master_csv.py
```
Then manually fill in label, label_binary, include_v1, and has_audio in file_info/files_master.csv.

Step 1b. Run the full pipeline:

```bash
python preprocessing/create_datasets.py
```

Optional arguments:
```bash
python preprocessing/create_datasets.py \
  --registry file_info/files_master.csv \
  --raw_root data/raw/ \
  --output data/processed/
```

I advise testing the parser on a single file at first. To do so, run:

```bash
python preprocessing/parse_data.py path/to/file.cha
```

### Output files:

All outputs are written to data/processed/:

	1. child_utterances.csv
	
		One row per child utterance. Includes raw text, cleaned text variants, all CHAT feature counts (pauses, disfluencies, errors, etc.), morphological annotations, and metadata (age, sex, label).

	2. all_utterances.csv
		Same as above but includes all speakers — parents, examiners, siblings. 
		
	3. child_context_windows.csv
		Child utterances with surrounding context. 
		
	4. session_level.csv
		One row per recording session. CHAT features summed/averaged across child utterances.
		
	5. pipeline_warnings.csv
		Flags cases where age or sex in the file header doesn't match files_master.csv. Not data, used only for quality checking.

### Pipeline settings
Adjust context window settings at the top of create_datasets.py:
```python
CONTEXT_WINDOW_BEFORE = 2
CONTEXT_WINDOW_AFTER  = 1
```

Files that aren't parsed successfully are skipped without crashing the pipeline and logged to data/processed/failed_files.txt.

## Step 2: Features and Embeddings

**** ALAN add more here???

TF-IDF embeddings

```bash
python embedding.py
```

Generates TF-IDF feature matrices saved to data/features/ (X_train_tfidf.npz, etc.) along with the fitted vectorizer (tfidf_vectorizer.joblib).

### Train/val/test split
The split used for all models is stored in split_manifest_by_pid.csv. This uses the script split_by_pid.py and ensures that no individual child, identified by participant ID, appears in more than one of train, validation, and test. This presents data leakage across sessions from the same child, and prevents the transformer superficially learning to identify a given child instead of the differences between disordered and regular speech. 

The split_by_pid.py script is called when running the transformerB.py script, see below for further details. This generates the split from the child_utterances.csv file. 

```bash
python transformerB.py
```

## Step 3: Baseline Models

### Majority class 

```bash
python transformerB.py
```

### Logistic regression
```bash
python src/models/LogisticR.py
```

### Baseline Results

**Majority Classifier**

| | Precision | Recall | F1-score | Support |
|--|-----------|--------|----------|---------|
| 0 (control) | 0.64 | 1.00 | 0.78 | 270 |
| 1 (SLI) | 0.00 | 0.00 | 0.00 | 155 |

**Logistic Regression**

| | Precision | Recall | F1-score | Support |
|--|-----------|--------|----------|---------|
| 0 (control) | 0.82 | 0.93 | 0.87 | 270 |
| 1 (SLI) | 0.83 | 0.64 | 0.72 | 155 |

## Step 4: Transformer Model

**Build the model:**
```bash
python transformerB.py
```

Building the model requires the file child_utterances.csv and relies on files split_by_pid.py and load_data.py

**Train the model:**
```bash
python transformerT.py
```

transformerT.py calls load_data.py and split_by_pid.py to load, batch, and split the data. The script currently builds the dataset on child_utterances.csv, but can be edited to take in other files, including per session.

## Transformer Results

*(To be updated)*

## Performance Comparisons

## References

**Lammert, J. M., Roberts, A. C., McRae, K., Batterink, L. J., & Butler, B. E.** (2025). <br>
*Early Identification of Language Disorders Using Natural Language Processing and Machine Learning: Challenges and Emerging Approaches.* <br>
Journal of Speech, Language, and Hearing Research, 68(2), 705–718. <br>

**Malathi, P., Legapriyadharshini, N., Nair, S. S., Sujatha, M. P., Sadaieswaran, R., & Thirumalaikumari, T.** (2024). <br>
*Automated Detection of Language Disorders in Children Using NLP and Machine Learning.* <br>
In Proceedings of the International Conference on Recent Innovation in Smart and Sustainable Technology (ICRISST), pp. 1–6. <br>

**Georgiou, G. P.** (2025). <br>
*Enhancing Developmental Language Disorder Identification with Artificial Intelligence: Development of an Explainable Screening App Using Real and Synthetic Data.* <br>
Journal of Autism and Developmental Disorders. <br>

**Jones, S., Fox, C., Gillam, S., & Gillam, R. B.** (2019). <br>
*An Exploration of Automated Narrative Analysis via Machine Learning.* <br>
PLOS ONE, 14(10), 1–14. <br>

