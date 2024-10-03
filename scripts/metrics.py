import spacy
from textblob import TextBlob
from collections import Counter

# Load spaCy model only once
nlp = spacy.load("en_core_web_sm")

# Metric 1: Adjective Count
def count_adjectives(title):
    doc = nlp(title)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

# Metric 2: POS Pattern Matching (POS distribution)
def pos_distribution(title):
    doc = nlp(title)
    pos_tags = [token.pos_ for token in doc]
    return dict(Counter(pos_tags))

# Metric 3: Word Diversity
def word_diversity(title):
    words = title.lower().split()
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0
    return diversity_score

# Metric 4: Sentiment (Emotional Tone)
def get_sentiment(title):
    blob = TextBlob(title)
    return blob.sentiment.polarity
