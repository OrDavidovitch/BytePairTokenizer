from tqdm import tqdm
import json
import time
import os
from datasets import load_dataset
import requests
from datasets import load_dataset
from tokenizer import mostCommonPairOneCore, updateTextOneCore, countPairs, updateTextChunk, bytePairTokenizer


# Parameters
TARGET_SIZE_MB = 20
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024
OUTPUT_FILE = "corpus.txt"
GUTENBERG_BOOK_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
]

def download_gutenberg_books():
    for url in GUTENBERG_BOOK_URLS:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Downloaded book from {url}")
                yield response.text
            else:
                print(f"Failed to download {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def stream_wikipedia_text():
    print("Streaming Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True, streaming=True)
    for entry in dataset:
        text = entry.get("text", "").strip()
        if text:
            yield text

def build_corpus():
    total_size = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f, tqdm(total=TARGET_SIZE_BYTES, unit='B', unit_scale=True, desc="Building corpus") as pbar:
        # Gutenberg books
        for text in download_gutenberg_books():
            encoded_text = (text + "\n").encode("utf-8")
            f.write(text + "\n")
            total_size += len(encoded_text)
            pbar.update(len(encoded_text))
            if total_size >= TARGET_SIZE_BYTES:
                print("Target size reached with Gutenberg books.")
                return

        # Wikipedia articles (streaming!)
        for text in stream_wikipedia_text():
            encoded_text = (text + "\n").encode("utf-8")
            f.write(text + "\n")
            total_size += len(encoded_text)
            pbar.update(len(encoded_text))
            if total_size >= TARGET_SIZE_BYTES:
                print("Target size reached with Wikipedia articles.")
                break

    print(f"Corpus created: {OUTPUT_FILE} ({total_size / (1024 * 1024):.2f} MB)")

numCores = 4
size = 500

def main():
    # Build the corpus
    build_corpus()
    
    # Read the generated corpus
    with open("corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Train the tokenizer
    tokenizer = bytePairTokenizer()
    tokenizer.train(text, numMerges=size, numWorkers=numCores)
    
    # Save the vocabulary dictionaries
    with open("IdToStrDict.json", "w") as f:
        json.dump(tokenizer.vocab, f)
    
    with open("StrToIdDict.json", "w") as f:
        json.dump(tokenizer.inverseVocab, f)

if __name__ == '__main__':
    start = time.time()
    mp.freeze_support()  # Optional: especially if you plan to freeze the program into an executable.
    main()
    end = time.time()
    print(f"Building vocabulary of {size} words using {numCores} cores on a corpus of {TARGET_SIZE_MB} MB took: {np.round((end- start)/3600, 2)} hours")





