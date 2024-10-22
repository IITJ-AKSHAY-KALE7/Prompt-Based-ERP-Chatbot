import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load NLP models
nlp_spacy = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('stopwords')

def process_query_with_spacy(query):
    doc = nlp_spacy(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return f"Entities found with spaCy: {entities}"

def process_query_with_nltk(query):
    tokens = word_tokenize(query)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return f"Processed tokens with NLTK: {filtered_tokens}"
