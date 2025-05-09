from typing import List, Dict, Any
from src.utils.tokenizer import tokenize, format_token_label_pairs


ANNOTATION_GUIDELINES = """
# Annotation Guidelines for Bengali Health Named Entity Recognition

You are tasked with annotating Bengali health-related text for Named Entity Recognition (NER) using the following entity types:

1. Symptom - Physical or mental features indicating a medical condition
2. Health Condition - Specific diseases or medical events
3. Medicine - Names of medications or medical substances
4. Specialist - Types of medical practitioners
5. Age - References to age of patients
6. Dose - Information about medicine dosage
7. Medical Procedure - Medical interventions or treatments

## Key Guidelines:

1. Entities should be annotated as specifically as possible.
2. A single entity should not be divided into two parts, even if this requires including some unnecessary information.
3. Symptoms are indicators of a health condition. A health condition is a specific disease or medical event.
4. Injuries are considered health conditions.
5. Unnecessary information, such as the severity of a symptom, should not be annotated.
6. Redundant information should not be annotated.
7. Pay attention to contextual information when identifying entities.
8. Multiple symptoms mentioned together should be annotated separately.
9. Medicine names can include the amount of chemical in each intake.
10. Doctors frequently write dosage in the format "0+0+1".
11. Medical specialists typically have uniquely identifiable names but are sometimes referred to by more generic names.

## Annotation Format:

Use the Inside-Outside-Beginning (IOB) format:
- B-Entity: Beginning of an entity
- I-Entity: Inside/continuation of an entity
- O: Outside any entity (not part of a named entity)

For each token in the text, assign exactly one label from:
- B-Symptom, I-Symptom
- B-Health Condition, I-Health Condition
- B-Medicine, I-Medicine
- B-Specialist, I-Specialist
- B-Age, I-Age
- B-Dose, I-Dose or B-Dosage, I-Dosage
- B-Medical Procedure, I-Medical Procedure
- O (Outside any entity)
"""

def get_few_shot_examples(dataset: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """
    Get few-shot examples from the dataset.
    
    Args:
        dataset: The dataset to extract examples from
        num_examples: Number of examples to include
        
    Returns:
        Formatted few-shot examples as a string
    """
    examples = []
    
    for i, sample in enumerate(dataset[:num_examples]):
        text = sample['text']
        tokens = tokenize(text)
        labels = sample['labels']
        
        example = f"Example {i+1}:\nText: {text}\n"
        example += f"Tokens ({len(tokens)}): {tokens}\n"
        example += f"Labels ({len(labels)}): {labels}\n"
        example += "Token-Label Pairs:\n"
        example += format_token_label_pairs(tokens, labels)
        examples.append(example)
    
    return "\n\n".join(examples)


def create_annotation_prompt(text: str, tokens: List[str], few_shot_examples: str) -> str:
    """
    Create a prompt for annotation.
    
    Args:
        text: The text to annotate
        tokens: The tokenized text
        few_shot_examples: Few-shot examples formatted as a string
        
    Returns:
        Completed annotation prompt
    """
    token_count = len(tokens)
    prompt = f"""
{ANNOTATION_GUIDELINES}

# Few-Shot Examples:
{few_shot_examples}

# Text to Annotate:
Full text: {text}
Tokens ({token_count}): {tokens}

# Your Task:
1. For EACH token in the text, assign ONE label from the following:
   - B-Symptom, I-Symptom
   - B-Health Condition, I-Health Condition
   - B-Medicine, I-Medicine
   - B-Specialist, I-Specialist
   - B-Age, I-Age
   - B-Dosage, I-Dosage
   - B-Medical Procedure, I-Medical Procedure
   - O (Outside any entity)

2. CRITICAL: Your response MUST contain EXACTLY {token_count} labels - one for each token shown above.
   Count carefully and ensure your labels array has {token_count} elements.

3. Follow the Inside-Outside-Beginning (IOB) format:
   - Use "B-" prefix for the first token of an entity
   - Use "I-" prefix for subsequent tokens of the same entity
   - Use "O" for tokens that are not part of any entity

4. Return your answer in the following JSON format ONLY:
{{
  "labels": [
    "label_for_token_1",
    "label_for_token_2",
    ...
    "label_for_token_{token_count}"
  ]
}}

5. Do NOT include any explanations or additional text in your response, ONLY the JSON object.
6. Verify that your "labels" array contains EXACTLY {token_count} elements before submitting.
"""
    return prompt 