#!/usr/bin/env python
"""
Script for evaluating annotated Bengali medical text data.
"""

import os
import argparse
import json
from typing import List, Dict, Any

from src.data.data_loader import load_dataset, save_dataset
from src.evaluation.metrics import evaluate_dataset, print_evaluation_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate annotated Bengali medical text")
    
    parser.add_argument("--input", type=str, required=True, help="Path to annotated dataset file")
    parser.add_argument("--output", type=str, help="Path to output evaluation results file")
    parser.add_argument("--detailed", action="store_true", help="Include detailed per-sample metrics")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load annotated dataset
    print(f"Loading annotated dataset from {args.input}...")
    dataset = load_dataset(args.input)
    print(f"Loaded {len(dataset)} samples.")
    
    # Count samples with predictions
    samples_with_predictions = sum(1 for sample in dataset if 'predicted_labels' in sample and sample['predicted_labels'])
    print(f"Found {samples_with_predictions} samples with predictions.")
    
    # Evaluate dataset
    print("Evaluating annotations...")
    results = evaluate_dataset(dataset)
    
    # Print results
    print_evaluation_results(results)
    
    # Setup output path
    if args.output:
        print(f"Saving evaluation results to {args.output}...")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Remove sample metrics if not requested
        if not args.detailed:
            if 'sample_metrics' in results:
                del results['sample_metrics']
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main() 