import nltk
from nltk import pos_tag, word_tokenize
from transformers import pipeline
import spacy
from textblob import TextBlob
from collections import Counter

# Download NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load sentiment analysis pipeline from transformers
sentiment_analyzer = pipeline("sentiment-analysis")

# Load spaCy model only once
nlp = spacy.load("en_core_web_sm")

# Function to count adjectives using NLTK
def count_nltk_adjectives(text):
    pos_tags = pos_tag(word_tokenize(text))
    return len([word for word, pos in pos_tags if pos.startswith('JJ')])

# Function to count adjectives using spaCy
def count_spacy_adjectives(title):
    doc = nlp(title)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

# POS Pattern Matching using NLTK
def pos_pattern_matching(generated, human):
    gen_pos = [tag for _, tag in pos_tag(word_tokenize(generated))]
    human_pos = [tag for _, tag in pos_tag(word_tokenize(human))]
    matching_score = sum(1 for g, h in zip(gen_pos, human_pos) if g == h)
    return matching_score / len(human_pos) if human_pos else 0

# POS Distribution using spaCy
def pos_distribution(title):
    doc = nlp(title)
    pos_tags = [token.pos_ for token in doc]
    return dict(Counter(pos_tags))

# Function to calculate word diversity
def word_diversity(text):
    words = word_tokenize(text.lower())  # Use NLTK for tokenization
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

# Sentiment analysis using NLTK and TextBlob
def sentiment_match(generated, human):
    gen_sentiment = sentiment_analyzer(generated)[0]
    human_sentiment = sentiment_analyzer(human)[0]
    return 1 if gen_sentiment['label'] == human_sentiment['label'] else -1

# Emotional Tone using TextBlob
def get_sentiment(title):
    blob = TextBlob(title)
    return blob.sentiment.polarity

# Reward function integrating all metrics
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
    gen_adjectives_nltk = count_nltk_adjectives(generated_title)
    human_adjectives_nltk = count_nltk_adjectives(human_title)
    adjective_penalty_nltk = abs(gen_adjectives_nltk - human_adjectives_nltk)
    
    gen_adjectives_spacy = count_spacy_adjectives(generated_title)
    human_adjectives_spacy = count_spacy_adjectives(human_title)
    adjective_penalty_spacy = abs(gen_adjectives_spacy - human_adjectives_spacy)
    
    pos_similarity = pos_pattern_matching(generated_title, human_title)
    diversity_penalty = word_diversity(generated_title)
    sentiment_reward = sentiment_match(generated_title, human_title)

    # Combine both adjective penalties
    total_adjective_penalty = adjective_penalty_nltk + adjective_penalty_spacy

    # Calculate the reward with weights
    reward = (-total_adjective_penalty * weights['adjective_penalty'] +
              pos_similarity * weights['pos_similarity'] -
              diversity_penalty * weights['diversity_penalty'] +
              sentiment_reward * weights['sentiment_reward'])
    
    return reward

