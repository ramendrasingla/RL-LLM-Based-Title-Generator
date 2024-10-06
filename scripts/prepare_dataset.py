import os
import json
from tqdm import tqdm
from datasets import load_dataset
from metrics import count_adjectives, pos_distribution, word_diversity, get_sentiment

def load_and_preprocess_cnn_dataset():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    splits = ['train', 'validation', 'test']
    output_folder = "./data"
    os.makedirs(output_folder, exist_ok=True)

    for split in splits:
        cnn_split = dataset[split]
        output_file = os.path.join(output_folder, f"cnn_processed_{split}.json")
        
        processed_data = []
        
        for data in tqdm(cnn_split, desc=f"Processing {split} split"):
            id = data['id']
            article = data['article']
            title = data['highlights']  # Treat 'highlights' as the title

            # Apply all metrics
            num_adjectives_in_title = count_adjectives(title)
            title_pos_distribution = pos_distribution(title)
            title_diversity_score = word_diversity(title)
            title_sentiment = get_sentiment(title)

            # Store processed data
            processed_data.append({
                "id": id,
                "article": article,
                "title": title,
                "num_adjectives_in_title": num_adjectives_in_title,
                "pos_distribution": title_pos_distribution,
                "diversity_score": title_diversity_score,
                "sentiment": title_sentiment
            })
        
        # Save processed data to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        print(f"Processed {split} data saved to {output_file}")

if __name__ == "__main__":
    load_and_preprocess_cnn_dataset()