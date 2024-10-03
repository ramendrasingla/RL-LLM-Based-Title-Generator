import spacy
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
    similarity = 0
    for pos in set(gen_pos_dist.keys()).union(human_pos_dist.keys()):
        similarity += min(gen_pos_dist.get(pos, 0), human_pos_dist.get(pos, 0))
    
    return similarity

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
    return gen_sentiment - human_sentiment  # Reward for closeness in sentiment

# Reward Function
def reward_function(generated_title, human_title, weights=None):
    # Default weights if not provided
    if weights is None:
        weights = {
            'adjective_penalty': 1.0,
            'pos_similarity': 1.0,
            'diversity_penalty': 1.0,
            'sentiment_reward': 1.0
        }

    # Calculate metrics
    gen_adjectives = count_adjectives(generated_title)
    human_adjectives = count_adjectives(human_title)
    adjective_penalty = abs(gen_adjectives - human_adjectives)
    
    pos_similarity = pos_pattern_matching(generated_title, human_title)
    diversity_penalty = word_diversity(generated_title)
    sentiment_reward = sentiment_match(generated_title, human_title)

    # Calculate the reward with weights
    reward = (-adjective_penalty * weights['adjective_penalty'] +
              pos_similarity * weights['pos_similarity'] -
              diversity_penalty * weights['diversity_penalty'] +
              sentiment_reward * weights['sentiment_reward'])
    
    return reward
