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
