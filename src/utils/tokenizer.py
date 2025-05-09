from typing import List, Dict, Any


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using whitespace.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    return text.split()


def get_token_count(text: str) -> int:
    """
    Get the number of tokens in a text.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    return len(tokenize(text))


def verify_token_label_alignment(sample: Dict[str, Any]) -> bool:
    """
    Verify that the number of tokens matches the number of labels.
    
    Args:
        sample: Sample dictionary with 'text' and 'labels' keys
        
    Returns:
        True if the number of tokens matches the number of labels, False otherwise
    """
    tokens = tokenize(sample['text'])
    labels = sample['labels']
    return len(tokens) == len(labels)


def verify_predictions_alignment(tokens: List[str], predictions: List[str]) -> bool:
    """
    Verify that the number of tokens matches the number of predictions.
    
    Args:
        tokens: List of tokens
        predictions: List of predicted labels
        
    Returns:
        True if the number of tokens matches the number of predictions, False otherwise
    """
    return len(tokens) == len(predictions)


def format_tokens_for_prompt(tokens: List[str]) -> str:
    """
    Format tokens for inclusion in a prompt.
    
    Args:
        tokens: List of tokens
        
    Returns:
        String representation of tokens suitable for a prompt
    """
    return " ".join([f"{i+1}. '{token}'" for i, token in enumerate(tokens)])


def format_token_label_pairs(tokens: List[str], labels: List[str]) -> str:
    """
    Format token-label pairs for inclusion in a prompt.
    
    Args:
        tokens: List of tokens
        labels: List of labels
        
    Returns:
        String representation of token-label pairs suitable for a prompt
    """
    return "\n".join([f"{i+1}. Token: '{token}' - Label: '{label}'" for i, (token, label) in enumerate(zip(tokens, labels))]) 