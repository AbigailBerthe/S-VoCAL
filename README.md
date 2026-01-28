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



