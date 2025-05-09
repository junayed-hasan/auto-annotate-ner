import json
import os
from collections import Counter

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_tokenization(dataset, name, num_examples=5):
    print(f"\n===== Tokenization Analysis for {name} dataset =====")
    
    # Check alignment for different tokenization methods
    for i, sample in enumerate(dataset):
        if i >= num_examples:
            break
            
        text = sample['text']
        labels = sample['labels']
        
        # Simple whitespace tokenization
        tokens = text.split()
        
        if len(tokens) != len(labels):
            print(f"\nSample #{i+1} - MISMATCH:")
        else:
            print(f"\nSample #{i+1} - MATCH:")
            
        print(f"Text: {text}")
        print(f"Whitespace tokens ({len(tokens)}): {tokens}")
        print(f"Labels ({len(labels)}): {labels}")
        
        # Display token-label pairs
        print("\nToken-Label Pairs:")
        for j, (token, label) in enumerate(zip(tokens[:min(len(tokens), len(labels))], labels[:min(len(tokens), len(labels))])):
            print(f"  {j+1}. Token: '{token}' - Label: '{label}'")
            
            # For tokens with B- or I- prefixes, show the entity type
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label.split('-')[1]
                print(f"     Entity: {entity_type}")
        
        # Check for punctuation handling
        punctuation = [char for char in text if not char.isalnum() and not char.isspace()]
        if punctuation:
            print(f"\nPunctuation characters: {set(punctuation)}")

def check_all_datasets():
    print("\n===== Checking all datasets for token-label alignment =====")
    
    dataset_dir = "dataset"
    files = ["valid.json", "train.json", "test.json"]
    
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        dataset = load_dataset(file_path)
        
        mismatches = 0
        for i, sample in enumerate(dataset):
            text = sample['text']
            labels = sample['labels']
            
            # Simple whitespace tokenization
            tokens = text.split()
            
            if len(tokens) != len(labels):
                mismatches += 1
                if mismatches <= 3:  # Only show first 3 mismatches
                    print(f"\nMismatch in {file}, sample #{i+1}:")
                    print(f"Text: {text}")
                    print(f"Tokens ({len(tokens)}): {tokens}")
                    print(f"Labels ({len(labels)}): {labels}")
        
        match_percentage = 100 - (mismatches/len(dataset)*100)
        print(f"\n{file}: {match_percentage:.2f}% match ({mismatches} mismatches out of {len(dataset)} samples)")

if __name__ == "__main__":
    dataset_dir = "dataset"
    valid_path = os.path.join(dataset_dir, "valid.json")
    
    # Load and analyze valid dataset
    valid_data = load_dataset(valid_path)
    analyze_tokenization(valid_data, "Valid")
    
    # Check alignment across all datasets
    check_all_datasets() 