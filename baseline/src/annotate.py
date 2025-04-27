#!/usr/bin/env python3
"""
Main script for annotating Bengali health text data using LLMs.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import (
    load_json_file,
    save_json_file,
    get_annotation_guidelines,
    get_few_shot_examples,
    parse_llm_response,
    tokenize_text
)

# Import the new prompt creation function
from create_prompt import create_annotation_prompt

# Load environment variables
load_dotenv()

def create_prompt(text: str, num_labels: int) -> str:
    """
    Create a prompt for annotating a text with entity labels.
    
    Args:
        text: The text to annotate
        num_labels: The number of labels needed in the response
        
    Returns:
        A prompt string
    """
    guidelines = get_annotation_guidelines()
    examples = get_few_shot_examples()
    
    prompt = f"""You are an expert text annotator for Bengali medical texts. 
    
Your task is to identify and label medical entities in the given text.

{guidelines}

{examples}

Now, annotate the following text with appropriate entity labels. Make sure to label each word exactly once.
The text has exactly {num_labels} words, so you must provide {num_labels} labels.

Return ONLY a JSON array of labels like ["O", "B-Symptom", "I-Symptom", ...] with no additional text or explanation.

Text: "{text}"
Labels:"""
    
    return prompt

def annotate_text(client: OpenAI, text: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Annotate a single text using the OpenAI API.
    
    Args:
        client: OpenAI client
        text: The text to annotate
        model: The model to use
        max_retries: Maximum number of retries on failure
        
    Returns:
        Dictionary with the original text and the predicted labels
    """
    text_tokens = tokenize_text(text)
    num_labels = len(text_tokens)
    
    # Use the new prompt creation function
    prompt = create_annotation_prompt(text, text_tokens)
    
    # Try to get a response from the API with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert medical text annotator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Use low temperature for more deterministic outputs
                max_tokens=1024,
                n=1
            )
            
            # Get the generated text
            generated_text = response.choices[0].message.content
            
            # Debug: Print the raw response and tokens
            print(f"\nText tokens ({len(text_tokens)}): {text_tokens}")
            print(f"Raw LLM response: {generated_text}")
            
            # Parse the response to get labels
            labels = parse_llm_response(generated_text, text_tokens)
            
            # Debug: Print the parsed labels
            print(f"Parsed labels ({len(labels)}): {labels}")
            
            # Ensure we have the right number of labels
            if len(labels) != num_labels:
                print(f"Warning: Label count mismatch. Expected {num_labels}, got {len(labels)}. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retrying
                    continue
                else:
                    # If this is the last attempt, pad with "O" labels or truncate
                    if len(labels) < num_labels:
                        print(f"Final attempt: Padding labels with 'O' from {len(labels)} to {num_labels}")
                        labels.extend(["O"] * (num_labels - len(labels)))
                    else:
                        print(f"Final attempt: Truncating labels from {len(labels)} to {num_labels}")
                        labels = labels[:num_labels]
            
            return {
                "text": text,
                "labels": labels
            }
            
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(10)  # Wait longer for API errors
            else:
                print("Max retries reached, returning default labels.")
                return {
                    "text": text,
                    "labels": ["O"] * num_labels
                }

def annotate_dataset(input_file: str, output_file: str, model: str, batch_size: int = 10, max_samples: int = None) -> None:
    """
    Annotate a dataset using the OpenAI API.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the annotated dataset
        model: The model to use
        batch_size: Number of samples to process per batch
        max_samples: Maximum number of samples to process (None for all)
    """
    # Load the dataset
    data = load_json_file(input_file)
    
    # Limit to max_samples if specified
    if max_samples is not None:
        max_samples = min(max_samples, len(data))
        data = data[:max_samples]
        print(f"Processing {max_samples} samples out of {len(data)} total samples")
    else:
        print(f"Processing all {len(data)} samples")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Process the dataset
    results = []
    
    # If output file exists, load existing results
    if os.path.exists(output_file):
        existing_results = load_json_file(output_file)
        results = existing_results
        print(f"Loaded {len(existing_results)} existing results from {output_file}")
        start_idx = len(existing_results)
    else:
        start_idx = 0
    
    # Process samples
    for i, sample in enumerate(tqdm(data[start_idx:], desc="Annotating")):
        # Calculate actual index in the data
        actual_idx = i + start_idx
        
        # Stop if we've reached max_samples
        if max_samples is not None and actual_idx >= max_samples:
            break
        
        text = sample["text"]
        
        # Annotate the text
        annotation = annotate_text(client, text, model)
        
        # Add the annotation to the results
        results.append(annotation)
        
        # Save intermediate results every 10 samples or at the end
        if (actual_idx + 1) % 10 == 0 or actual_idx == len(data) - 1 or (max_samples is not None and actual_idx == max_samples - 1):
            save_json_file(results, output_file)
            print(f"Saved intermediate results after {actual_idx + 1} samples.")
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Save the final results
    save_json_file(results, output_file)
    print(f"Annotation complete. Results saved to {output_file}")
    print(f"Processed {len(results)} samples")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Annotate Bengali health text data using LLMs.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", required=True, help="Path to save the annotated dataset")
    parser.add_argument("--model", default="gpt-4o-mini", 
                      help="Model to use for annotation (default: gpt-4o-mini)")
    parser.add_argument("--batch-size", type=int, default=10, 
                      help="Number of samples to process per batch (default: 10)")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Maximum number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    # Annotate the dataset
    annotate_dataset(args.input, args.output, args.model, args.batch_size, args.max_samples)

if __name__ == "__main__":
    main() 