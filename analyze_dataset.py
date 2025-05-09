import json
import os
from collections import Counter

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(dataset, name):
    print(f"\n===== Analyzing {name} dataset =====")
    print(f"Number of samples: {len(dataset)}")
    
    # Count entity types
    label_counter = Counter()
    for sample in dataset:
        for label in sample['labels']:
            if label != 'O':
                label_type = label.split('-')[1] if '-' in label else label
                label_counter[label_type] += 1
    
    print("Entity distribution:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")
    
    # Check token-label alignment
    mismatches = 0
    for i, sample in enumerate(dataset):
        text = sample['text']
        labels = sample['labels']
        
        # Simple whitespace tokenization
        tokens = text.split()
        
        if len(tokens) != len(labels):
            mismatches += 1
            if mismatches <= 3:  # Only show first 3 mismatches
                print(f"\nMismatch example #{mismatches}:")
                print(f"Text: {text}")
                print(f"Tokens ({len(tokens)}): {tokens}")
                print(f"Labels ({len(labels)}): {labels}")
    
    print(f"\nToken-label mismatches: {mismatches}/{len(dataset)} ({mismatches/len(dataset)*100:.2f}%)")

def character_level_analysis(dataset, name, max_samples=5):
    print(f"\n===== Character-level analysis for {name} dataset =====")
    
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
            
        text = sample['text']
        labels = sample['labels']
        
        print(f"\nSample #{i+1}:")
        print(f"Text: {text}")
        print(f"Labels: {labels}")
        
        # Try to understand character by character
        chars = list(text)
        print("\nCharacter by character:")
        for j, char in enumerate(chars):
            print(f"Char: '{char}' - Ord: {ord(char)}")
            if j > 20:  # Limit to first 20 characters
                print("...")
                break

if __name__ == "__main__":
    dataset_dir = "dataset"
    
    # Analyze valid dataset first
    valid_path = os.path.join(dataset_dir, "valid.json")
    valid_data = load_dataset(valid_path)
    analyze_dataset(valid_data, "Valid")
    character_level_analysis(valid_data, "Valid")
    
    # Check other datasets 
    train_path = os.path.join(dataset_dir, "train.json")
    test_path = os.path.join(dataset_dir, "test.json")
    
    train_data = load_dataset(train_path)
    test_data = load_dataset(test_path)
    
    analyze_dataset(train_data, "Train")
    analyze_dataset(test_data, "Test") 