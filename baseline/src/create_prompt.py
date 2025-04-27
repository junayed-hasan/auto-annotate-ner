#!/usr/bin/env python3
"""
Module for creating prompts for the LLM annotator.
"""

from typing import List

def create_annotation_prompt(text: str, text_tokens: List[str]) -> str:
    """
    Create a prompt for annotating a text with entity labels.
    Explicitly includes the list of tokens to ensure correct label counts.
    
    Args:
        text: The original text to annotate
        text_tokens: The tokenized text
        
    Returns:
        A prompt string
    """
    num_tokens = len(text_tokens)
    
    # Get the guidelines and examples from utils (assumed to be imported)
    from utils import get_annotation_guidelines, get_few_shot_examples
    
    guidelines = get_annotation_guidelines()
    examples = get_few_shot_examples()
    
    # Join tokens with visible separators for clarity
    token_display = " | ".join(text_tokens)
    
    prompt = f"""You are an expert text annotator for Bengali medical texts. 
    
Your task is to identify and label medical entities in the given text.

{guidelines}

{examples}

Now, annotate the following tokenized text with appropriate entity labels. I've tokenized the text for you.
You must provide EXACTLY {num_tokens} labels - one label per token.

Original text: "{text}"

Tokenized text ({num_tokens} tokens): {token_display}

Return ONLY a JSON array of labels (exactly {num_tokens} labels) with no explanation or additional text.
The number of labels MUST match the number of tokens exactly.

Labels:"""
    
    return prompt 