#!/usr/bin/env python
"""
Script for annotating Bengali medical text data using the OpenAI API.
"""

import os
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any

from src.data.data_loader import load_dataset, sample_dataset, save_dataset
from src.utils.tokenizer import tokenize
from src.models.openai_annotator import OpenAIAnnotator
from src.configs.prompts import get_few_shot_examples


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Annotate Bengali medical text using OpenAI API")
    
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset file")
    parser.add_argument("--output", type=str, help="Path to output annotated dataset file")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to annotate (default: all)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of samples to process before delaying")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between batches in seconds")
    parser.add_argument("--few-shot", type=int, default=3, help="Number of few-shot examples to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    dataset = load_dataset(args.input)
    print(f"Loaded {len(dataset)} samples.")
    
    # Sample dataset if requested
    if args.sample_size is not None:
        print(f"Sampling {args.sample_size} samples...")
        dataset = sample_dataset(dataset, n=args.sample_size, seed=args.seed)
        print(f"Sampled {len(dataset)} samples.")
    
    # Verify token-label alignment
    print("Verifying token-label alignment...")
    for i, sample in enumerate(dataset):
        tokens = tokenize(sample['text'])
        if len(tokens) != len(sample['labels']):
            print(f"Warning: Sample {i} has mismatched token-label lengths: {len(tokens)} tokens vs {len(sample['labels'])} labels")
    
    # Setup output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/annotations_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # If we're sampling, include that in the filename
        if args.sample_size is not None:
            args.output = f"{output_dir}/annotated_{os.path.basename(args.input).split('.')[0]}_{args.sample_size}.json"
        else:
            args.output = f"{output_dir}/annotated_{os.path.basename(args.input).split('.')[0]}.json"
    
    # Get few-shot examples
    print(f"Extracting {args.few_shot} few-shot examples...")
    few_shot_examples = get_few_shot_examples(dataset, num_examples=args.few_shot)
    
    # Initialize annotator
    print(f"Initializing annotator with model {args.model}...")
    annotator = OpenAIAnnotator(model=args.model)
    
    # Annotate dataset
    print("Starting annotation...")
    start_time = time.time()
    
    annotated_dataset = annotator.annotate_batch(
        dataset,
        few_shot_examples=few_shot_examples,
        batch_size=args.batch_size,
        delay=args.delay
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Annotation completed in {elapsed_time:.2f} seconds.")
    
    # Save annotated dataset
    print(f"Saving annotated dataset to {args.output}...")
    save_dataset(annotated_dataset, args.output)
    
    # Print summary
    errors = sum(1 for sample in annotated_dataset if 'error' in sample)
    print(f"Annotated {len(annotated_dataset)} samples with {errors} errors.")
    
    if errors > 0:
        print("Errors:")
        for i, sample in enumerate(annotated_dataset):
            if 'error' in sample:
                print(f"  Sample {i}: {sample['error']}")
    
    print("Done!")


if __name__ == "__main__":
    main() 