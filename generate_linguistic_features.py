import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import openai
import os

# Load SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Function to calculate Type-Token Ratio (TTR)
def lexical_diversity(text):
    tokens = [token.text.lower() for token in nlp(text) if token.is_alpha]
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
    return ttr

# Function to calculate Syntactic Complexity
def syntactic_complexity(text):
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_clauses = sum(1 for token in doc if token.dep_ in ["ccomp", "advcl", "acl", "relcl", "xcomp"])
    complexity = num_clauses / num_sentences if num_sentences > 0 else 0
    return complexity

# Function to calculate Semantic Coherence
def semantic_coherence(text, embedding_model):
    # initialize OpenAI library
    openai.api_key = os.getenv("OPENAI_API_KEY")

    sentences = [sent.text for sent in nlp(text).sents]
    if len(sentences) < 2:
        return 1  # Single sentence is perfectly "coherent" within itself

    # Get embeddings for each sentence
    sentence_embeddings = []
    for sentence in sentences:
        response = openai.embeddings.create(input=sentence, model=embedding_model)
        sentence_embeddings.append(response.data[0].embedding)

    # Calculate pairwise cosine similarity
    coherence_scores = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
        coherence_scores.append(sim)

    avg_coherence = np.mean(coherence_scores) if coherence_scores else 1
    return avg_coherence

# Function to detect Speech Errors (e.g., hesitations)
def speech_errors(text):
    fillers = ["um", "uh", "hmm", "like", "you know"]
    tokens = [token.text.lower() for token in nlp(text)]
    error_count = sum(tokens.count(filler) for filler in fillers)
    error_rate = error_count / len(tokens) if len(tokens) > 0 else 0
    return error_rate

# Function to extract all features
def extract_features(text, embedding_model):
    features = {
        "lexical_diversity": lexical_diversity(text),
        "syntactic_complexity": syntactic_complexity(text),
        "semantic_coherence": semantic_coherence(text, embedding_model),
        "speech_errors": speech_errors(text),
    }
    return features

# Example Usage
if __name__ == "__main__":
    # Sample transcript
    sample_text = """
    Well, um, yesterday I went to the store, and, uh, I bought some apples, you know. 
    They were, like, really fresh, and I, uh, cooked them into a pie.
    """
    
    # Mock embeddings model
    class MockEmbeddingsModel:
        def create(self, input, model):
            return {
                "data": [{"embedding": np.random.rand(768).tolist()}]  # Random embedding for illustration
            }
    
    # Initialize mock embeddings model
    embeddings_model = MockEmbeddingsModel()
    
    # Extract features
    features = extract_features(sample_text)
    
    # Print features
    print("Extracted Features:")
    for feature, value in features.items():
        print(f"{feature}: {value}")
