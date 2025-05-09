import json
import os
from typing import List, Dict, Union, Tuple, Optional, Any


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of dictionaries containing the dataset samples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_dataset(dataset: List[Dict[str, Any]], n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Sample a subset of the dataset.
    
    Args:
        dataset: The dataset to sample from
        n: Number of samples to include
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing the sampled dataset
    """
    import random
    random.seed(seed)
    return random.sample(dataset, min(n, len(dataset)))


def save_dataset(dataset: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save the dataset to a JSON file.
    
    Args:
        dataset: The dataset to save
        file_path: Path to save the dataset to
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def get_token_label_pairs(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Get the token-label pairs from a sample.
    
    Args:
        sample: Sample dictionary with 'text' and 'labels' keys
        
    Returns:
        List of (token, label) tuples
    """
    tokens = sample['text'].split()
    labels = sample['labels']
    
    # Ensure tokens and labels have the same length
    assert len(tokens) == len(labels), f"Tokens and labels have different lengths: {len(tokens)} vs {len(labels)}"
    
    return list(zip(tokens, labels))


def add_annotations_to_sample(sample: Dict[str, Any], annotations: List[str]) -> Dict[str, Any]:
    """
    Add model-generated annotations to a sample.
    
    Args:
        sample: Original sample dictionary
        annotations: List of annotations from the model
        
    Returns:
        Sample dictionary with added annotations
    """
    # Create a copy of the sample to avoid modifying the original
    result = sample.copy()
    result['predicted_labels'] = annotations
    return result 