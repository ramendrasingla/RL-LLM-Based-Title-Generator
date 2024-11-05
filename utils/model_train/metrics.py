import spacy
import numpy as np
from textblob import TextBlob
from collections import Counter

# Load spaCy model only once
nlp = spacy.load("en_core_web_sm")

# Metric: Adjective Count using spaCy
def count_adjectives(title):
    doc = nlp(title)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

# Metric: POS Pattern Matching (POS distribution)
def pos_distribution(title):
    doc = nlp(title)
    pos_tags = [token.pos_ for token in doc]
    return dict(Counter(pos_tags))

# Metric: Calculate POS similarity between generated and human titles
def pos_pattern_matching(generated_title, human_title):
    gen_pos_dist = pos_distribution(generated_title)
    human_pos_dist = pos_distribution(human_title)
    
    # Calculate similarity as the sum of the minimum occurrences for each POS tag
    similarity = sum(min(gen_pos_dist.get(pos, 0), human_pos_dist.get(pos, 0)) 
                     for pos in set(gen_pos_dist.keys()).union(human_pos_dist.keys()))
    
    # Normalize by the number of POS tags in the shorter title
    max_possible_similarity = min(sum(gen_pos_dist.values()), sum(human_pos_dist.values()))
    normalized_similarity = similarity / max_possible_similarity if max_possible_similarity else 0
    
    return normalized_similarity

# Metric: Word Diversity
def word_diversity(title):
    words = title.lower().split()
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0
    return diversity_score

# Metric: Sentiment (Emotional Tone)
def get_sentiment(title):
    blob = TextBlob(title)
    return blob.sentiment.polarity

# Metric: Sentiment Match
def sentiment_match(generated_title, human_title):
    gen_sentiment = get_sentiment(generated_title)
    human_sentiment = get_sentiment(human_title)
    return gen_sentiment - human_sentiment

def get_all_metrics(test_dataset, expected_predictions):
    output = {}

    adj_count_list = []
    pos_distribution_list = []
    word_diversity_list = []
    sentiment_match_list = []
    for ix, iy in zip(test_dataset, expected_predictions):
        adj_count_list.append(count_adjectives(ix))
        pos_distribution_list.append(pos_pattern_matching(ix, iy))
        word_diversity_list.append(word_diversity(ix))
        sentiment_match_list.append(sentiment_match(ix, iy))
    
    output['mean_adjective_count'] = np.mean(adj_count_list)
    output['mean_pos_distribution'] = np.mean(pos_distribution_list)
    output['mean_word_diversity'] = np.mean(word_diversity_list)
    output['mean_sentiment_score'] = np.mean(sentiment_match_list)

    return output


