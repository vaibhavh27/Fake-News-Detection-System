import pickle # Changed from json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import tokenizer_from_json # No longer needed
from transformers import pipeline # AutoTokenizer, AutoModelForSequenceClassification not directly needed if pipeline handles local path
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import argparse # Added
import os       # Added

# ---------------------
# Default Configuration (can be overridden by command-line arguments)
# ---------------------
DEFAULT_MAX_LEN_FAKE_NEWS = 150
DEFAULT_PREDICTION_THRESHOLD_FAKE_NEWS = 0.6
DEFAULT_SENTIMENT_CACHE_DIR = "assets/sentiment_model_cache/distilbert_sentiment" # Align with download_hf_models.py

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Fake News Detection and Sentiment Analysis on input text.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True, # Make this required as it's essential
        help="Directory containing the pre-trained fake news model (e.g., fake_news_model.h5) "
             "and its tokenizer (e.g., tokenizer.pickle)."
    )
    parser.add_argument(
        "--sentiment_cache_dir",
        type=str,
        default=DEFAULT_SENTIMENT_CACHE_DIR,
        help=f"Directory of the cached DistilBERT sentiment model. Default: {DEFAULT_SENTIMENT_CACHE_DIR}"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN_FAKE_NEWS,
        help=f"Maximum sequence length for fake news model padding. Default: {DEFAULT_MAX_LEN_FAKE_NEWS}"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD_FAKE_NEWS,
        help=f"Prediction threshold for classifying fake news (0.0 to 1.0). Default: {DEFAULT_PREDICTION_THRESHOLD_FAKE_NEWS}"
    )
    return parser.parse_args()

# Global variable for SpaCy model, initialized once
nlp_spacy_global = None

def initialize_nlp_resources_inference():
    global nlp_spacy_global # To modify the global variable
    print("Initializing NLP resources for inference...")
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download('wordnet', quiet=True)

    if nlp_spacy_global is None: # Load SpaCy only if not already loaded
        try:
            nlp_spacy_global = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm for SpaCy...")
            spacy.cli.download("en_core_web_sm")
            nlp_spacy_global = spacy.load("en_core_web_sm")
    print("NLP resources initialized for inference.")

# ---------------------
# Text Preprocessing (Should be IDENTICAL to training's preprocess_text)
# ---------------------
def preprocess_text(text_input):
    if not isinstance(text_input, str):
        text_input = str(text_input)
    text_input = text_input.lower()
    text_input = re.sub(r"\d+", "", text_input)
    text_input = re.sub(f"[{re.escape(string.punctuation)}]", "", text_input)
    tokens = word_tokenize(text_input)
    lemmatizer = WordNetLemmatizer()
    stop_words_english = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_english and word.strip()]
    return " ".join(tokens)

# ---------------------
# Passive Voice Conversion (Optional, based on training)
# ---------------------
def convert_to_passive(text_input):
    if nlp_spacy_global is None:
        print("Error: SpaCy model not initialized. Cannot convert to passive voice.")
        return text_input # Return original text if SpaCy not ready

    doc = nlp_spacy_global(text_input) # Use the globally loaded SpaCy model
    passive_sentences = []
    for sent in doc.sents:
        subject, verb, obj = None, None, None
        has_verb = False
        for token in sent:
            if "subj" in token.dep_ and subject is None: subject = token
            if "obj" in token.dep_ and obj is None: obj = token
            if token.pos_ == "VERB" and not has_verb:
                verb = token
                has_verb = True
        if subject and verb and obj and verb.lemma_ and obj.text and subject.text:
            verb_lemma = verb.lemma_
            passive_verb_form = verb_lemma + "ed"
            if verb_lemma.endswith('e'): passive_verb_form = verb_lemma + "d"
            passive_sentence = f"{obj.text.capitalize()} was {passive_verb_form} by {subject.text}."
            passive_sentences.append(passive_sentence)
        else:
            passive_sentences.append(sent.text)
    return " ".join(passive_sentences)


def main_inference():
    args = parse_arguments()
    initialize_nlp_resources_inference() # Ensure resources are loaded

    # Construct full paths from the --model_dir argument
    # Assuming standard file names within the model directory as per README
    keras_tokenizer_filename = "tokenizer.pickle"
    fake_news_model_filename = "fake_news_model.h5"

    keras_tokenizer_path = os.path.join(args.model_dir, keras_tokenizer_filename)
    fake_news_model_path = os.path.join(args.model_dir, fake_news_model_filename)
    
    sentiment_model_dir_path = args.sentiment_cache_dir

    # Use max_len and threshold from command-line arguments
    max_len_for_padding = args.max_len
    prediction_threshold = args.threshold

    # ---------------------
    # Load Keras Tokenizer for Fake News (from .pickle)
    # ---------------------
    print(f"Loading Keras tokenizer from: {keras_tokenizer_path}")
    try:
        with open(keras_tokenizer_path, 'rb') as handle: # 'rb' for read binary
            tokenizer = pickle.load(handle)
        print("Keras tokenizer loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Keras tokenizer file '{keras_tokenizer_path}' not found.")
        print(f"Please ensure '{keras_tokenizer_filename}' exists in the directory: '{args.model_dir}'")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the Keras tokenizer: {e}")
        exit()

    # ---------------------
    # Load Fake News Detection Model
    # ---------------------
    print(f"Loading fake news detection model from: {fake_news_model_path}")
    try:
        model_fake = load_model(fake_news_model_path)
        print("Fake news model loaded successfully.")
        # model_fake.summary() # Optional: to verify model structure
    except (OSError, FileNotFoundError): # OSError can also indicate model not found for Keras
        print(f"Error: Fake news model file '{fake_news_model_path}' not found.")
        print(f"Please ensure '{fake_news_model_filename}' exists in the directory: '{args.model_dir}'")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the fake news model: {e}")
        exit()

    # ---------------------
    # Load Sentiment Analysis Pipeline
    # ---------------------
    print(f"Attempting to load sentiment analysis pipeline from: {sentiment_model_dir_path}")
    sentiment_pipeline_instance = None
    try:
        if os.path.exists(sentiment_model_dir_path) and os.path.isdir(sentiment_model_dir_path):
            sentiment_pipeline_instance = pipeline("sentiment-analysis", model=sentiment_model_dir_path, tokenizer=sentiment_model_dir_path)
            print("Sentiment analysis pipeline loaded successfully from local cache.")
        else:
            print(f"Warning: Sentiment model cache directory '{sentiment_model_dir_path}' not found or is not a directory.")
            print("Attempting to download and load the default sentiment model (this may take a moment)...")
            sentiment_pipeline_instance = pipeline("sentiment-analysis") # Fallback to download default
            print("Default sentiment model downloaded/loaded.")
    except Exception as e:
        print(f"Warning: Could not load sentiment model from '{sentiment_model_dir_path}' due to: {e}")
        print("Sentiment analysis will be unavailable if the fallback download also failed.")

    # ---------------------
    # Prediction Functions
    # ---------------------
    def predict_fake_news(text_to_predict):
        processed_text = preprocess_text(text_to_predict)
        seq = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(seq, maxlen=max_len_for_padding, padding='post', truncating='post')
        pred_proba = model_fake.predict(padded)[0][0]
        label = "True News" if pred_proba > prediction_threshold else "Fake News"
        return f"{label} (Raw score for 'True': {pred_proba:.4f}, Threshold: {prediction_threshold})"

    def predict_sentiment(text_to_predict):
        if sentiment_pipeline_instance is None:
            return "Sentiment analysis unavailable (model not loaded)."
        if not text_to_predict or not text_to_predict.strip():
            return "NEUTRAL (empty input)"
        try:
            # Truncate text for models like DistilBERT that have input length limits
            result = sentiment_pipeline_instance(text_to_predict[:512])[0]
            return f"{result['label']} (Score: {result['score']:.2f})"
        except Exception as e:
            return f"Sentiment analysis error: {e}"

    # ---------------------
    # User Interaction Looppip install -r requirements.txt
    # ---------------------
    print("\n--- Fake News & Sentiment Analysis System ---")
    print(f"Using fake news model from: {args.model_dir}")
    print(f"Using MAX_LEN={max_len_for_padding}, THRESHOLD={prediction_threshold}")

    while True:
        user_input_original = input("\nEnter news article text (or type 'exit' to quit):\n")
        if user_input_original.lower() == 'exit':
            print("Exiting program.")
            break
        if not user_input_original.strip():
            print("Input is empty. Please enter some text.")
            continue

        # --- Text for Fake News Prediction ---
        # Determine if passive voice conversion should be applied based on how the model was trained.
        # For now, assuming models are trained on non-passivized text.
        text_for_fake_news_model = user_input_original
        # Example: If you have a flag or config for a specific model indicating it was trained on passive voice:
        # if model_trained_on_passive_voice_flag:
        #    text_for_fake_news_model = convert_to_passive(user_input_original)

        print("\n--- Results ---")
        fake_news_result = predict_fake_news(text_for_fake_news_model)
        print(f"Fake News Detection: {fake_news_result}")

        # For sentiment analysis, generally use original or minimally processed text.
        sentiment_result = predict_sentiment(user_input_original)
        print(f"Sentiment Analysis: {sentiment_result}")
        print("---------------------\n")

if __name__ == "__main__":
    main_inference()