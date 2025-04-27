import json
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import re

# Define entity classes
ENTITY_CLASSES = [
    "O", 
    "B-Symptom", "I-Symptom", 
    "B-Health Condition", "I-Health Condition", 
    "B-Age", "I-Age", 
    "B-Medicine", "I-Medicine", 
    "B-Dosage", "I-Dosage", 
    "B-Medical Procedure", "I-Medical Procedure", 
    "B-Specialist", "I-Specialist"
]

# Entity mapping for conversion
ENTITY_MAPPING = {
    "Symptom": ["B-Symptom", "I-Symptom"],
    "Health Condition": ["B-Health Condition", "I-Health Condition"],
    "Age": ["B-Age", "I-Age"],
    "Medicine": ["B-Medicine", "I-Medicine"],
    "Dosage": ["B-Dosage", "I-Dosage"],
    "Medical Procedure": ["B-Medical Procedure", "I-Medical Procedure"],
    "Specialist": ["B-Specialist", "I-Specialist"],
    "O": ["O"]
}

def load_json_file(file_path: str) -> Any:
    """Load a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_annotation_guidelines() -> str:
    """Get the annotation guidelines to be included in prompts."""
    return """
In addition to the basic instruction of labelling each word in the samples with one of the seven entity labels, the annotators were provided with extensive instructions on how to address a large variety of different scenarios that were seen to arise. These instructions are provided below:

1. Entities should be annotated as specifically as possible. If an entity contains a sub-entity, the top-level entity should be annotated.
   Example: 'গত ৪ দিন ধরে মাথাব্যথা' should be tagged as 'গত ৪ দিন ধরে মাথাব্যথা'

2. A single entity should not be divided into two parts, even if this requires including some unnecessary information.
   Example: 'আমার মাথা গত ২ দিন ধরে অল্প বেশা করছে' should be tagged as 'আমার মাথা গত ২ দিন ধরে অল্প বেশা করছে'

3. Symptoms are indicators of a health condition. A health condition is a specific disease or medical event.
   Example: 'প্রায় সময় মাথাব্যথা হয়', 'সাইনুসাইটিস হতে পারে'

4. Injuries are considered health conditions.
   Example: 'আমার পা ফেটে গেছে'

5. Unnecessary information, such as the severity of a symptom, should not be annotated.
   Example: 'মাথে খুব উঁচু আলতে বেশি'

6. Redundant information (such as the word 'ডাক্তার' following the word 'বিশেষজ্ঞ') should not be annotated.
   Example: 'নাক, কান, গলা ডাক্তার', 'নাক, কান, গলা বিশেষজ্ঞ ডাক্তার'

7. It is important to pay attention to contextual information when identifying entities. For example, the term 'Corona' can be used to describe both the COVID-19 disease and the COVID-19 pandemic in Bengali.
   Example: 'করোনা চলাকালীন ডাক্তার দেখতে পাব্লি না', 'এটা কি কোরোনার লক্ষণ?'

8. Multiple symptoms mentioned together should be annotated separately.
   Example: 'bad cold and fever'

9. Medicine names can include the amount of chemical in each intake.
   Example: 'মেটাপ ১০ মিগ্রা খান'

10. Doctors frequently write the dosage in the format '0+1+1'.
    Example: 'মেটাপ ১০ মিগ্রা 0+0+1'

11. Medical specialists typically have uniquely identifiable names (e.g., গাইনেকোলজিষ্ট) but are sometimes referred to by more generic names (e.g., নারীর ডাক্তার).
    Example: 'একজন মেডিসিনের ডাক্তার দেখান'
"""

def get_few_shot_examples() -> str:
    """Get a few examples for in-context learning."""
    return """
Here are a few examples to help you understand the annotation format:

Example 1:
Text: "ধন্যবাদ আপনাকে প্রশ্নের জন্য । জ্বর থাকলে ৬ ঘন্টা পর পর নাপা খাবেন । ১০২ এর উপরে স্যাপোজিটর দিবেন । ধন্যবাদ"
Labels: ["O", "O", "O", "O", "O", "O", "O", "B-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "B-Medicine", "O", "O", "B-Symptom", "I-Symptom", "I-Symptom", "B-Medicine", "O", "O", "O"]

Example 2:
Text: "Patient is 6+ years old & has been suffering from fever for 2 days. Fever is rising 102 degree. No other symptoms but when fever is coming head & neck pain are appearing with vomiting tendency. Fever disappearing by taken Paracetamol syrup every after 5 hrs. Should it take any other medicine? If so, pls suggest..."
Labels: ["O", "O", "B-Age", "I-Age", "I-Age", "O", "O", "O", "O", "O", "O", "B-Symptom", "O", "O", "O", "O", "B-Symptom", "I-Symptom", "I-Symptom", "I-Symptom", "I-Symptom", "O", "O", "O", "O", "O", "O", "B-Symptom", "O", "O", "B-Symptom", "I-Symptom", "I-Symptom", "I-Symptom", "O", "O", "O", "B-Symptom", "I-Symptom", "O", "B-Symptom", "O", "O", "O", "B-Medicine", "I-Medicine", "B-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]

Example 3:
Text: "ডাক্তার ভাইকে প্রশ্ন করার জন্য ধন্যবাদ । Tab Napa 500 . 1 + 1 + 1 , 3 - 5 days . Tab Fexo 120 , 0 + 0 + 1 , 5 days . সাথে প্রেসারের ওষুধ , যেটা রেগুলার খেয়ে থাকেন সেটা চালিয়ে যাবেন । ধন্যবাদ"
Labels: ["O", "O", "O", "O", "O", "O", "O", "B-Medicine", "I-Medicine", "I-Medicine", "O", "B-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "O", "O", "O", "O", "O", "O", "B-Medicine", "I-Medicine", "I-Medicine", "O", "B-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "I-Dosage", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
"""

def convert_to_iob_format(labels: List[str], words: List[str]) -> List[str]:
    """
    Convert simplified labels to IOB format.
    
    Args:
        labels: List of entity labels (without IOB prefixes)
        words: List of words being labeled
        
    Returns:
        List of labels in IOB format
    """
    iob_labels = []
    prev_label = "O"
    
    for i, (label, word) in enumerate(zip(labels, words)):
        if label == "O":
            iob_labels.append("O")
        elif prev_label == "O" or prev_label.split("-")[-1] != label:
            iob_labels.append(f"B-{label}")
        else:
            iob_labels.append(f"I-{label}")
        
        prev_label = "O" if label == "O" else f"B-{label}"
        
    return iob_labels

def parse_llm_response(response: str, text_tokens: List[str]) -> List[str]:
    """
    Parse the response from the LLM and convert it to IOB format.
    
    Args:
        response: The LLM response string
        text_tokens: The original text tokens
        
    Returns:
        List of labels in IOB format
    """
    try:
        # Clean up the response if needed (remove extra characters, etc.)
        clean_response = response.strip()
        
        # Handle potential JSON formatting
        if clean_response.startswith("[") and clean_response.endswith("]"):
            # Try to parse as JSON list
            import json
            try:
                parsed_labels = json.loads(clean_response)
                return parsed_labels
            except json.JSONDecodeError:
                pass
        
        # If not parseable as JSON, try to extract labels using string manipulation
        # This is a fallback approach that depends on the exact format of the LLM response
        # You might need to adapt this based on the actual responses you get
        
        # Assuming format is like: ["O", "B-Symptom", "I-Symptom", ...]
        labels = []
        for item in clean_response.strip("[]").split(","):
            label = item.strip().strip('"\'')
            labels.append(label)
        
        # Ensure we have the right number of labels
        if len(labels) != len(text_tokens):
            # If not, default to "O" labels
            return ["O"] * len(text_tokens)
        
        return labels
    
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        # Default to "O" labels in case of parsing error
        return ["O"] * len(text_tokens)

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words. For Bengali text, this is a bit more complex than 
    just splitting on spaces since some words might not have spaces between them.
    
    Args:
        text: The input text to tokenize
        
    Returns:
        List of tokens
    """
    # First, handle punctuation by adding spaces around it
    # This helps with proper tokenization
    
    # Print original text for debugging
    print(f"Original text: {text}")
    
    # Add spaces around punctuation marks and special characters
    text = re.sub(r'([।,.!?()।/\-:\'\"])', r' \1 ', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split on spaces
    tokens = text.split()
    
    # Print tokenized result for debugging
    print(f"Tokenized into {len(tokens)} tokens: {tokens}")
    
    return tokens 