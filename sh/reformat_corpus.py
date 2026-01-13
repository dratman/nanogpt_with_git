#!/usr/bin/env python3
"""
Reformat a corpus file to one sentence per line using NLTK.

Usage: python reformat_corpus.py input.txt output.txt
"""

import re
import sys
from nltk.tokenize import sent_tokenize

def clean_and_split(text):
    """Clean text and split into sentences using NLTK."""
    # Remove book header lines (start with #)
    text = re.sub(r'#[^\n]*\n?', '', text)

    # Remove standalone Roman numeral chapter markers
    text = re.sub(r'^\s*(I{1,3}|IV|VI{0,3}|IX|X{1,3}|XI{1,3}|XIV|XV|XVI{0,3}|XIX|XX|L|C)\s*$',
                  '', text, flags=re.MULTILINE)

    # Normalize whitespace (collapse multiple spaces/newlines to single space)
    text = re.sub(r'\s+', ' ', text).strip()

    # Use NLTK sentence tokenizer
    sentences = sent_tokenize(text)

    # Clean up each sentence
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 1:  # Skip empty or single-char "sentences"
            cleaned.append(s)

    return cleaned

def main():
    if len(sys.argv) != 3:
        print("Usage: python reformat_corpus.py input.txt output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Splitting into sentences with NLTK...")
    sentences = clean_and_split(text)

    print(f"Writing {len(sentences)} sentences to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    print("Done!")

if __name__ == '__main__':
    main()
