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

-   [Dependencies](#dependencies)
-   [Data cleaning and preprocessing pipeline](#Data-cleaning-and-preprocessing-pipeline)
-   [Embeddings + Initial Features](#Embeddings+Initial-Features)
-   [Baseline model training](#Baseline-model-training)
-   [Baseline-Results](#Baseline-Results)
-   [Transformer](#Transformer)
-   [Performance Comparisons](#Performance-Comparisons)
-   [References](#references)


## Dependencies

* transformers
* torch
* numpy
*

## Data-cleaning-and-preprocessing-pipeline

### Dataset
SCLARIN TalkBank / CHILDES conversational transcripts

Initial preprocessing pipeline output is located on google drive, not on GitHub due to size constraints:

https://drive.google.com/drive/u/1/folders/1gi70tvEbzI7_IevzK3NQrI6mrYAdFrxi

Further Preprocessing Notes: 

#### NOTE: this is only for v1. V1 includes only controls and children marked SLI/language_disorders. (Some just said "language_disorder" which was unspecified. This was a small number. We can remove if needed)


### Requirements.txt

	Run pip install -r requirements.txt to install required packages

### Data files:

No sound level data included at this time as only a few datasets have this.

Processed files are located in talkbank_project/data/processed

	1. child_utterances.csv

		About: One row for child utterance. This is the raw text, with cleaned text variance, all CHAT feature counts like pauses, disfuencies, etc. Also includes morphological annotations and metadata like age, sex.
		
		Label indicates if disordered or not.

	2. all_utterances.csv
	
		About: The same as child_utterances but includes the parent, sibling, or other person talking with the child who is the focus. 
	
		Useful if you want conversational context or interactions beyond just how the child talks.

	3. child_context_windows.csv

		About: child utterances, but each row also has the context before the child spoke and after. 

		Useful for conversations, or just including broader context.

	4. session_level.csv
		
		About: One row per speech recording. Features are summed and averaged across the number of child utterences. 

	5. pipeline_warnings.csv

		About: Not actually data. Includes information on cases when the age or sex of a child doesn't match what is in files_master.csv. This could happen due to parsing errors. Isn't a huge number of cases, but these could be 		addressed later on, time permitting
		
Raw files can be found in talkbank_project/data/raw.

### Master File Info:

	1.file_info/files_master.csv

		A csv containing information about all files in the dataset. One row per .cha file. Columns include file_id, file_path, label, label_binary, age, sex, include_v1.
		include_v1 notes if a given file was included in the first version of the pipeline, which included only controls, and children marked "SLI" or "lang_disorder" (small number of files just said "lang disorder")

### Notebooks

	1. notebooks/nlp_project_exploration.ipynb

		Just mesing around inspecting the way the .cha files are set up. Not EDA. Just for basic data processing.

### USEFUL NOTES:

	label_binary = 0 if typically developing control
	1 = SLI or language disorder (other types of language disorders not included for V1 like down syndrome, hearing loss, late_talker, etc.) Will be very easy to include these in future.


## Embeddings+Initial-Features

### Embeddings:
```bash
python embedding.py
```
(Note:)

### Initial Features:
```bash
python IFeature.py
```
(Note: )

## Baseline-model-training

### Majority class
```bash
python majorityC.py
```
(Note:)

### Logistic regression
```bash
python src/models/LogisticR.py
```

## Baseline-Results

### Logistic-Regression


| Precision | Recall | f1-score | Support |
|-----------|--------|----------|---------|
| 0 |  0.82 | 0.94 | 0.88 | 271 |
| 1 |  0.85 | 0.65 | 0.74 | 155 |


### Evaluation
| Precision | Recall | f1-score | Support |
|-----------|--------|----------|---------|
| Accuracy |   |  | 0.83 | 426 |
| Macro avg |  0.84 | 0.79 | 0.81 | 426 |
| Weighted avg |  0.83 | 0.83 | 0.82 | 426 |
| ROC AUC |  0.88 |  |  |  |

## Transformer

### Transformer building:
```bash
python transformerB.py
```
(Note:)

### Transformer training:
```bash
python transformerT.py
```
(Note: )

## Performance-Comparisons

## Results


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

