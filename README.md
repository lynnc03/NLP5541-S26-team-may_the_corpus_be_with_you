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

##Goal: 
Develop NLP models that assist early screening of language disorders


## Table of Contents

-   [Dependencies](#dependencies)
-   [Data cleaning and preprocessing pipeline](#Data cleaning and preprocessing pipeline)
-   [Embeddings + Initial Features](#Embeddings + Initial Features)
-   [Baseline model training](#Baseline model training)
-   [Baseline-Results](#Baseline-Results)
-   [Transformer](#Transformer)
-   [Performance Comparisons](#Performance Comparisons)
-   [References](#references)


## Dependencies

* transformers
* torch
* numpy
*

## Data cleaning and preprocessing pipeline

### Dataset
SCLARIN TalkBank / CHILDES conversational transcripts


## Embeddings + Initial Features

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

## Baseline model training

### Majority class
```bash
python majorityC.py
```
(Note:)

### Logistic regression
```bash
python LogisticR.py
```
(Note:)

## Baseline-Results

### Example Results

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

## Performance Comparisons

### Results

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

