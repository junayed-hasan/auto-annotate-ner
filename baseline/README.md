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

## Directory Structure

```
llm-ner-pipeline/
│
├── data/                 # Contains the dataset files
│
├── src/                  # Source code
│   ├── annotate.py       # Main annotation script
│   ├── evaluate.py       # Evaluation script
│   └── utils.py          # Utility functions
│
└── requirements.txt      # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-ner-pipeline
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your API key in a `.env` file:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Usage

1. To annotate data:
```bash
python src/annotate.py --model gpt-4o-mini --input data/input.json --output data/output.json
```

2. To evaluate the results:
```bash
python src/evaluate.py --gold data/gold.json --pred data/pred.json
```

## Implementation Details

- Uses the GPT-4o-mini model for annotation
- Provides annotation guidelines from the original paper to the LLM
- Ensures the number of labels matches the original data
- Follows the IOB tagging format as in the original dataset

## Limitations

This baseline approach doesn't include:
1. Chain-of-thought prompting
2. Scoring mechanisms for uncertainty
3. Human-in-the-loop intervention

These features will be added in future implementations to improve annotation quality. 