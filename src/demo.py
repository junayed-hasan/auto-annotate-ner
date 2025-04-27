#!/usr/bin/env python3
"""
Demo script to show how to use the annotation pipeline with a small example.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from utils import tokenize_text, parse_llm_response
from annotate import create_prompt, annotate_text

# Load environment variables
load_dotenv()

# Sample Bengali health text
sample_texts = [
    "ধন্যবাদ আপনাকে প্রশ্নের জন্য । জ্বর থাকলে ৬ ঘন্টা পর পর নাপা খাবেন । ১০২ এর উপরে স্যাপোজিটর দিবেন । ধন্যবাদ",
    "আমার বাবার বয়স 35 - 40 তার পেট ব্যাথা করে এবং কিছু খেলেই পেট ফুলে যায় এবং বুমি বুমি ভাব আছে পাতলা পায়খানা টা একটু কমেছে কিছু ভালো সাজেশন দিন এখন আর জ্বর নেই Napa extra খাওয়ানো হচ্ছে",
    "ডাক্তার ভাইকে প্রশ্ন করার জন্য ধন্যবাদ । Tab Napa 500 . 1 + 1 + 1 , 3 - 5 days . Tab Fexo 120 , 0 + 0 + 1 , 5 days . সাথে প্রেসারের ওষুধ , যেটা রেগুলার খেয়ে থাকেন সেটা চালিয়ে যাবেন । ধন্যবাদ"
]

def run_demo():
    """Run a demonstration of the annotation pipeline."""
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Create a .env file with your API key or set it in your environment.")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Process each sample text
    results = []
    
    for i, text in enumerate(sample_texts):
        print(f"\n=== Sample {i+1} ===")
        print(f"Text: {text}")
        
        # Tokenize the text
        tokens = tokenize_text(text)
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Create the annotation prompt
        prompt = create_prompt(text, len(tokens))
        
        # Annotate the text
        print("Annotating...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Change to your preferred model
                messages=[
                    {"role": "system", "content": "You are an expert medical text annotator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Get the generated text
            generated_text = response.choices[0].message.content
            print(f"Raw response: {generated_text}")
            
            # Parse the response to get labels
            labels = parse_llm_response(generated_text, tokens)
            
            # Display the results
            print("\nAnnotation results:")
            for token, label in zip(tokens, labels):
                print(f"{token}: {label}")
            
            # Store the results
            results.append({
                "text": text,
                "labels": labels
            })
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Save the results
    with open("demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nDemo results saved to demo_results.json")

if __name__ == "__main__":
    run_demo() 