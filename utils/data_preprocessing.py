import re
import spacy

# Load Spacy's small English model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess title and content
def preprocess_text(text, lemmatize = True):
    #  Lowercase the text
    text = text.lower()
    
    # Remove special characters and extra whitespaces
    text = re.sub(r'\n|\t|—', ' ', text)  # Replace newlines, tabs, em dashes with spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove all non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    
    if lemmatize:
        # Optional Lemmatization
        doc = nlp(text)
        text = ' '.join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords & lemmatize
    
    return text