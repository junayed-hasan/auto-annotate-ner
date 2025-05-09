import json
import time
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.tokenizer import tokenize, verify_predictions_alignment
from src.configs.prompts import create_annotation_prompt, get_few_shot_examples


class OpenAIAnnotator:
    """Class for annotating text using the OpenAI API."""
    
    def __init__(self, model: str = "gpt-4.1", api_key: Optional[str] = None):
        """
        Initialize the OpenAI annotator.
        
        Args:
            model: The OpenAI model to use
            api_key: OpenAI API key (if None, loads from environment variable)
        """
        # Load API key from environment if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
        if api_key is None:
            raise ValueError("OpenAI API key not provided and not found in environment variables.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
    def annotate(self, text: str, few_shot_examples: str, max_retries: int = 3, retry_delay: float = 2.0) -> List[str]:
        """
        Annotate text using the OpenAI API.
        
        Args:
            text: Text to annotate
            few_shot_examples: Few-shot examples formatted as a string
            max_retries: Maximum number of retries in case of errors
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of annotated labels
        """
        tokens = tokenize(text)
        expected_token_count = len(tokens)
        prompt = create_annotation_prompt(text, tokens, few_shot_examples)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are a helpful named entity recognition annotation assistant. You MUST provide EXACTLY {expected_token_count} labels for the text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent results
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if "labels" not in result:
                    raise ValueError("Response does not contain 'labels' key.")
                
                labels = result["labels"]
                
                # Verify that the number of labels matches the number of tokens
                if not verify_predictions_alignment(tokens, labels):
                    # Fix mismatched labels by padding or truncating
                    if len(labels) < expected_token_count:
                        # Pad with "O" labels if we have fewer labels than tokens
                        print(f"Warning: Padding {expected_token_count - len(labels)} labels for text: {text[:50]}...")
                        labels.extend(["O"] * (expected_token_count - len(labels)))
                    elif len(labels) > expected_token_count:
                        # Truncate if we have more labels than tokens
                        print(f"Warning: Truncating {len(labels) - expected_token_count} labels for text: {text[:50]}...")
                        labels = labels[:expected_token_count]
                
                # Final verification
                if not verify_predictions_alignment(tokens, labels):
                    raise ValueError(f"Failed to fix label count mismatch for text: {text[:50]}...")
                
                return labels
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # If all retries failed, return a list of "O" labels with the correct length
                    print(f"All retry attempts failed for text: {text[:50]}... Returning default 'O' labels.")
                    return ["O"] * expected_token_count
    
    def annotate_batch(self, samples: List[Dict[str, Any]], few_shot_examples: str, 
                       batch_size: int = 5, delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Annotate a batch of samples.
        
        Args:
            samples: List of samples to annotate
            few_shot_examples: Few-shot examples formatted as a string
            batch_size: Number of samples to process before delaying
            delay: Delay between batches in seconds
            
        Returns:
            List of annotated samples
        """
        results = []
        
        for i, sample in enumerate(samples):
            text = sample["text"]
            tokens = tokenize(text)
            expected_token_count = len(tokens)
            print(f"Annotating sample {i+1}/{len(samples)} - Tokens: {expected_token_count}")
            
            try:
                labels = self.annotate(text, few_shot_examples)
                
                # Double-check alignment before adding to results
                if len(labels) != expected_token_count:
                    print(f"Warning: Label count mismatch after annotation. Fixing for sample {i+1}...")
                    if len(labels) < expected_token_count:
                        labels.extend(["O"] * (expected_token_count - len(labels)))
                    else:
                        labels = labels[:expected_token_count]
                
                annotated_sample = sample.copy()
                annotated_sample["predicted_labels"] = labels
                results.append(annotated_sample)
                
                # Delay between batches to avoid rate limiting
                if (i + 1) % batch_size == 0 and i < len(samples) - 1:
                    print(f"Processed {i+1} samples. Waiting {delay} seconds...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error annotating sample {i+1}: {str(e)}")
                # Add the sample with default "O" labels to ensure label count matches
                annotated_sample = sample.copy()
                annotated_sample["predicted_labels"] = ["O"] * expected_token_count
                annotated_sample["error"] = str(e)
                results.append(annotated_sample)
        
        return results 