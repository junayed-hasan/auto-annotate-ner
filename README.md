<<<<<<< HEAD
# Bangla-HealthNER Annotation using LLMs

This project provides a framework for annotating Bengali health-related text data using OpenAI's LLMs for Named Entity Recognition (NER). The system implements in-context learning with few-shot examples to perform NER annotations.

## Project Structure

```
├── annotate.py             # Main annotation script
├── evaluate.py             # Evaluation script
├── env.example             # Example environment variables file
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── configs/            # Configuration files
    │   ├── prompts.py      # Prompt templates
    ├── data/               # Data handling utilities
    │   ├── data_loader.py  # Dataset loading and processing
    ├── evaluation/         # Evaluation metrics
    │   ├── metrics.py      # Evaluation metrics implementation
    ├── models/             # Model implementations
    │   ├── openai_annotator.py  # OpenAI API client for annotation
    └── utils/              # Utility functions
        ├── tokenizer.py    # Tokenization utilities
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your OpenAI API key (see `env.example`)

## Usage

### Annotation

To annotate a dataset:

```bash
python annotate.py --input dataset/valid.json --sample-size 10 --model gpt-4.1
```

Key options:
- `--input`: Path to input dataset file (required)
- `--output`: Path to output annotated dataset file (default: auto-generated path in `results/`)
- `--model`: OpenAI model to use (default: gpt-4.1)
- `--sample-size`: Number of samples to annotate (default: all)
- `--batch-size`: Number of samples to process before delaying (default: 5)
- `--delay`: Delay between batches in seconds (default: 1.0)
- `--few-shot`: Number of few-shot examples to include (default: 3)
- `--seed`: Random seed for sampling (default: 42)

### Evaluation

To evaluate annotations:

```bash
python evaluate.py --input results/annotated_valid_10.json
```

Key options:
- `--input`: Path to annotated dataset file (required)
- `--output`: Path to output evaluation results file (default: none)
- `--detailed`: Include detailed per-sample metrics (default: false)

## Annotation Pipeline

1. The system loads the dataset and verifies token-label alignment
2. It selects few-shot examples from the dataset
3. For each text to annotate:
   - Tokenizes the text using whitespace tokenization
   - Creates a prompt with the annotation guidelines and few-shot examples
   - Calls the OpenAI API to generate annotations
   - Verifies that the number of annotations matches the number of tokens
   - Adds the annotations to the sample
4. The annotated dataset is saved to disk

## Evaluation Metrics

The evaluation script calculates the following metrics:

- Token-level metrics:
  - Accuracy
  - Macro precision, recall, and F1 score
- Entity-level metrics:
  - Precision, recall, and F1 score
- Entity-type metrics:
  - Precision, recall, and F1 score for each entity type

## Entity Types

The system recognizes the following entity types:
- Symptom
- Health Condition
- Medicine
- Specialist
- Age
- Dosage
- Medical Procedure
=======
# Bangla HealthNER LLM Annotation Pipeline

This repository contains code for automatically annotating Bengali health text data using Large Language Models (LLMs). This is intended to be a baseline implementation that doesn't use chain-of-thought prompting, scoring mechanisms, or human-in-the-loop intervention.

## Overview

This project implements an LLM-based annotation pipeline for the Bangla-HealthNER dataset, which consists of health-related texts in Bengali with annotations for seven different entity types:
- Symptom
- Health Condition
- Age
- Medicine
- Dosage
- Medical Procedure
- Specialist

## Implementation Details

- Uses the GPT-4o-mini model for annotation
- Provides annotation guidelines from the original paper to the LLM
- Ensures the number of labels matches the original data
- Follows the IOB tagging format as in the original dataset
>>>>>>> origin/main
