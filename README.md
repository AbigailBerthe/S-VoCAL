# S-VoCAL
This repository contains the dataset and code accompanying the paper:  **S-VoCAL: A Dataset and Evaluation Framework for Inferring Speaking Voice Character Attributes in Literature**

It provides the S-VoCAL dataset and the evaluation framework introduced in the paper, as well as the code used to validate the proposed methodology.

## Overview

S-VoCAL addresses the task of inferring speaking voice–related attributes of fictional characters, from literary texts, using Large Language Models (LLMs).

The main contributions of this repository are:
- a curated dataset of character–book pairs annotated with voice-relevant attributes,
- an evaluation framework adapted to heterogeneous attribute types.

A reference inference pipeline is provided to illustrate and test the evaluation methodology.

## Dataset: S-VoCAL

The S-VoCAL dataset contains **952 character–book pairs** extracted from **192 novels** available in the public domain (Gutenberg Project).

Each instance corresponds to a fictional character appearing in a specific book, and is annotated with a set of attributes relevant to speaking voice characterization.

### Attributes

The dataset includes 8 attributes:

- Age
- Gender
- Origin
- Residence
- Spoken languages
- Occupation
- Physical health
- Type (human / non-human)

Attributes are heterogeneous in nature. They include categorical, and open-class values.

### Data sources and construction

All character information (attributes) was primarily extracted from Wikidata and aligned with books available on Project Gutenberg.

The dataset construction process includes:
- retrieval of novels and associated characters available on Wikidata,
- extraction and normalization of Wikidata properties for each character,
- manual curation and annotation for selected attributes (notably Age).

The released dataset corresponds to the curated version of the dataset of character's attributes used in the experiments reported in the paper.

## Evaluation framework

The evaluation framework is designed to handle the heterogeneous nature of character attributes.
Evaluation is done by 'evaluation.py' which calls 'embeddings_eval.py' and 'evaluation_metrics.py' depending on the attribute to evaluate, to rely on attribute-specific metrics.

### Attribute-aware evaluation

Different evaluation strategies are applied depending on the attribute type:

- **Closed attributes** (e.g. gender, age): F1-score  
- **Ordinal attributes** (age): Soft F1-score with distance-based weighting  
- **List-based attributes** (spoken languages): multi-label micro-F1  
- **Open-class attributes** (origin, residence, occupation, physical health, type): semantic similarity using instruction-conditioned embeddings and BERTScore

## Reference inference pipeline

A reference Retrieval-Augmented Generation (RAG) pipeline is provided to illustrate the use of the dataset and to validate the evaluation framework.

## Repository contents

- 'S-VoCAL_dataset.jsonl' — curated dataset
- 'wikidata.py' — dataset construction and analysis from Wikidata
- 'pipeline.py' — reference inference pipeline
- 'cleaner.py' — post-processing of model outputs, called by 'pipeline.py'
- 'evaluation.py' — evaluation orchestration
- 'evaluation_metrics.py' — discrete and list-based metrics, called by 'evaluation.py'
- 'embeddings_eval.py' — semantic similarity evaluation for open attributes, called by 'evaluation.py'

## How to use this repository

The dataset and evaluation framework can be reused independently of the inference pipeline.

Typical usage:
1. Load 'S-VoCAL_dataset.jsonl' as a gold reference.
2. Evaluate predictions from any system using 'evaluation.py'.

Running the reference inference pipeline additionally requires:
- Ollama
- a locally available LLM (e.g. 'qwen3:latest')

## Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Output files and directory structure

The code expects a minimal directory structure to store predictions and evaluation results.
These directories are **not created automatically** and must exist before running the scripts.

Create them once from the root of the repository:

```bash
mkdir -p Data/raw
mkdir -p Data/evaluation/dataframes
```

### Run the reference inference pipeline
Running the reference pipeline requires Ollama and a local LLM available in Ollama (e.g. qwen3:latest).

```bash
# Inference with E5-based retrieval, of attributes origin, residence and spoken_languages
python pipeline.py origin,residence,spoken_languages e5 qwen3:latest

# Inference using all character mentions
python pipeline.py origin,residence,spoken_languages all_mentions qwen3:latest
```

The script prints an OUTPUT_TIME identifier, which is required for evaluation.

### Evaluate predictions

```bash
# Evaluate raw predictions (with automatic post-processing)
python evaluation.py origin,residence,spoken_languages e5 cleaned <OUTPUT_TIME> qwen3:latest
```

### Where are the outputs written?

-Pipeline predictions: 

  - *Data/raw/* ; Filenames include a timestamp (OUTPUT_TIME) printed when running pipeline.py.

-Evaluation results:

  - Text summary (appended):
  *Data/evaluation/<datatype>_evaluation_results.txt*
  
  - Final scores (CSV):
  *Data/evaluation/dataframes/*
