# scripts/prepare_dataset.py
from datasets import load_dataset
import spacy
import os
from tqdm import tqdm
import csv

nlp = spacy.load("en_core_web_sm")

def load_and_preprocess_cnn_dataset():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    splits = ['train', 'validation', 'test']
    output_folder = "../data"
    os.makedirs(output_folder, exist_ok=True)

    for split in splits:
        cnn_split = dataset[split]
        output_file = os.path.join(output_folder, f"cnn_processed_{split}.csv")

        with open(output_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["article", "title", "num_adjectives_in_title"])

            for data in tqdm(cnn_split, desc=f"Processing {split} split"):
                article = data['article']
                title = data['highlights']  # Treat 'highlights' as the title
                num_adjectives_in_title = count_adjectives(title)
                writer.writerow([article, title, num_adjectives_in_title])


def count_adjectives(text):
    doc = nlp(text)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)


if __name__ == "__main__":
    load_and_preprocess_cnn_dataset()