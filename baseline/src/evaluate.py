#!/usr/bin/env python3
"""
Script for evaluating the performance of LLM-based annotations.
"""

import argparse
import json
import os
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from seqeval.metrics import classification_report as seq_classification_report

from utils import load_json_file, save_json_file, ENTITY_CLASSES

def load_datasets(gold_file: str, pred_file: str, max_samples: int = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load the gold and predicted datasets.
    
    Args:
        gold_file: Path to the gold standard file
        pred_file: Path to the predictions file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Tuple of (gold_data, pred_data)
    """
    gold_data = load_json_file(gold_file)
    pred_data = load_json_file(pred_file)
    
    # Limit to max_samples if specified
    if max_samples is not None:
        max_samples = min(max_samples, len(gold_data), len(pred_data))
        gold_data = gold_data[:max_samples]
        # Filter pred_data to match the gold samples
        pred_data_filtered = []
        gold_texts = set(item["text"] for item in gold_data)
        for item in pred_data:
            if item["text"] in gold_texts:
                pred_data_filtered.append(item)
                if len(pred_data_filtered) >= max_samples:
                    break
        pred_data = pred_data_filtered
        print(f"Evaluating on {max_samples} samples")
    else:
        print(f"Evaluating on all available samples")
    
    return gold_data, pred_data

def compute_metrics(gold_data: List[Dict[str, Any]], 
                   pred_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute evaluation metrics for the predictions.
    
    Args:
        gold_data: Gold standard data
        pred_data: Predicted data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Prepare data for evaluation
    gold_labels = []
    pred_labels = []
    
    # Match gold and pred data by text
    matches = 0
    for i, gold_sample in enumerate(gold_data):
        # Find matching predicted sample
        match_found = False
        for pred_sample in pred_data:
            if gold_sample["text"] == pred_sample["text"]:
                match_found = True
                # Skip samples with mismatched label lengths
                if len(gold_sample["labels"]) != len(pred_sample["labels"]):
                    print(f"Skipping sample with mismatched label lengths: {gold_sample['text'][:30]}...")
                    continue
                
                gold_labels.append(gold_sample["labels"])
                pred_labels.append(pred_sample["labels"])
                matches += 1
                break
        
        if not match_found:
            print(f"No matching prediction found for sample {i}")
    
    print(f"Found {matches} matching samples out of {len(gold_data)} gold samples")
    
    # Compute metrics directly using seqeval
    results = {
        "precision": precision_score(gold_labels, pred_labels, zero_division=0),
        "recall": recall_score(gold_labels, pred_labels, zero_division=0),
        "f1": f1_score(gold_labels, pred_labels, zero_division=0),
        "accuracy": accuracy_score(gold_labels, pred_labels)
    }
    
    return results

def per_entity_metrics(gold_data: List[Dict[str, Any]], 
                      pred_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-entity evaluation metrics.
    
    Args:
        gold_data: Gold standard data
        pred_data: Predicted data
        
    Returns:
        Dictionary of per-entity evaluation metrics
    """
    # Prepare data for evaluation
    gold_labels = []
    pred_labels = []
    
    # Match gold and pred data by text
    for i, gold_sample in enumerate(gold_data):
        # Find matching predicted sample
        for pred_sample in pred_data:
            if gold_sample["text"] == pred_sample["text"]:
                # Skip samples with mismatched label lengths
                if len(gold_sample["labels"]) != len(pred_sample["labels"]):
                    continue
                
                gold_labels.append(gold_sample["labels"])
                pred_labels.append(pred_sample["labels"])
                break
    
    # Compute classification report
    report = seq_classification_report(y_true=gold_labels, y_pred=pred_labels, zero_division=0)
    
    # Extract per-entity metrics
    per_entity = {}
    for entity in ENTITY_CLASSES:
        if entity in report:
            per_entity[entity] = {
                "precision": report[entity]["precision"],
                "recall": report[entity]["recall"],
                "f1-score": report[entity]["f1-score"],
                "support": report[entity]["support"]
            }
    
    return per_entity

def plot_confusion_matrix(gold_data: List[Dict[str, Any]], 
                         pred_data: List[Dict[str, Any]], 
                         output_file: str) -> None:
    """
    Plot a confusion matrix for entity labels.
    
    Args:
        gold_data: Gold standard data
        pred_data: Predicted data
        output_file: Path to save the plot
    """
    # Prepare data for confusion matrix
    gold_labels_flat = []
    pred_labels_flat = []
    
    # Match gold and pred data by text
    for i, gold_sample in enumerate(gold_data):
        # Find matching predicted sample
        for pred_sample in pred_data:
            if gold_sample["text"] == pred_sample["text"]:
                # Skip samples with mismatched label lengths
                if len(gold_sample["labels"]) != len(pred_sample["labels"]):
                    continue
                
                gold_labels_flat.extend(gold_sample["labels"])
                pred_labels_flat.extend(pred_sample["labels"])
                break
    
    # Create label mapping to handle large number of classes
    unique_labels = sorted(list(set(gold_labels_flat + pred_labels_flat)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    gold_indices = [label_to_id[label] for label in gold_labels_flat]
    pred_indices = [label_to_id[label] for label in pred_labels_flat]
    
    # Compute confusion matrix
    cm = confusion_matrix(gold_indices, pred_indices)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', xticklabels=unique_labels, 
               yticklabels=unique_labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")

def print_detailed_metrics(overall_metrics: Dict[str, float], entity_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print detailed metrics in a readable format.
    
    Args:
        overall_metrics: Overall evaluation metrics
        entity_metrics: Per-entity evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nOverall Metrics:")
    print("-"*50)
    for metric, value in overall_metrics.items():
        print(f"{metric.capitalize():15}: {value:.4f}")
    
    print("\nPer-Entity Metrics:")
    print("-"*70)
    print(f"{'Entity':20} {'Precision':12} {'Recall':12} {'F1-Score':12} {'Support':10}")
    print("-"*70)
    
    # Print entities in a specific order
    entity_order = [
        "O",
        "B-Symptom", "I-Symptom",
        "B-Health Condition", "I-Health Condition",
        "B-Age", "I-Age",
        "B-Medicine", "I-Medicine",
        "B-Dosage", "I-Dosage",
        "B-Medical Procedure", "I-Medical Procedure",
        "B-Specialist", "I-Specialist"
    ]
    
    for entity in entity_order:
        if entity in entity_metrics:
            metrics = entity_metrics[entity]
            print(f"{entity:20} {metrics['precision']:12.4f} {metrics['recall']:12.4f} {metrics['f1-score']:12.4f} {metrics['support']:10}")
    
    print("="*70)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate LLM-based annotations.")
    parser.add_argument("--gold", required=True, help="Path to the gold standard file")
    parser.add_argument("--pred", required=True, help="Path to the predictions file")
    parser.add_argument("--output", default="evaluation_results.json", 
                       help="Path to save the evaluation results")
    parser.add_argument("--plot", default="confusion_matrix.png", 
                       help="Path to save the confusion matrix plot")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (default: all)")
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.gold, args.pred]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return
    
    # Load datasets
    gold_data, pred_data = load_datasets(args.gold, args.pred, args.max_samples)
    
    # Compute metrics
    overall_metrics = compute_metrics(gold_data, pred_data)
    entity_metrics = per_entity_metrics(gold_data, pred_data)
    
    # Save results
    results = {
        "overall": overall_metrics,
        "per_entity": entity_metrics
    }
    save_json_file(results, args.output)
    print(f"Evaluation results saved to {args.output}")
    
    # Print detailed metrics
    print_detailed_metrics(overall_metrics, entity_metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(gold_data, pred_data, args.plot)

if __name__ == "__main__":
    main() 