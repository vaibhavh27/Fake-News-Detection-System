import pandas as pd
import numpy as np
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline
import pickle # For saving tokenizer as .pickle
import argparse
import os

# --- Default Configuration (can be overridden by command-line arguments) ---
DEFAULT_FAKE_NEWS_CSV = "Fake.csv" # Matches common naming
DEFAULT_TRUE_NEWS_CSV = "True.csv" # Matches common naming
DEFAULT_OUTPUT_TOKENIZER_PATH = "trained_assets/tokenizer.pickle"
DEFAULT_OUTPUT_MODEL_PATH = "trained_assets/fake_news_model.h5"
DEFAULT_SENTIMENT_MODEL_CACHE_PATH = "./sentiment_model_cache/distilbert_sentiment"

MAX_WORDS = 5000
MAX_LEN = 150
EMBEDDING_DIM = 100
LSTM_UNITS = 128
CNN_FILTERS = 256
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 3
PREDICTION_THRESHOLD = 0.6 # For the interactive demo part

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Fake News Detection Model.")
    parser.add_argument("--fake_csv", type=str, default=DEFAULT_FAKE_NEWS_CSV,
                        help="Path to the CSV file containing fake news articles.")
    parser.add_argument("--true_csv", type=str, default=DEFAULT_TRUE_NEWS_CSV,
                        help="Path to the CSV file containing true news articles.")
    parser.add_argument("--output_tokenizer_path", type=str, default=DEFAULT_OUTPUT_TOKENIZER_PATH,
                        help="Path to save the trained Keras Tokenizer (e.g., tokenizer.pickle).")
    parser.add_argument("--output_model_path", type=str, default=DEFAULT_OUTPUT_MODEL_PATH,
                        help="Path to save the trained Keras Model (e.g., fake_news_model.h5).")
    parser.add_argument("--sentiment_model_cache", type=str, default=DEFAULT_SENTIMENT_MODEL_CACHE_PATH,
                        help="Path to the cached DistilBERT sentiment model directory.")
    # Add more arguments for hyperparameters if needed (MAX_WORDS, MAX_LEN, etc.)
    return parser.parse_args()

# --- Download NLTK resources and SpaCy model (if not already downloaded) ---
def initialize_nlp_resources():
    print("Initializing NLP resources...")
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

    try:
        nlp_spacy = spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading en_core_web_sm for SpaCy...")
        spacy.cli.download('en_core_web_sm')
        nlp_spacy = spacy.load('en_core_web_sm')
    print("NLP resources initialized.")
    return nlp_spacy

# --- Preprocessing Functions ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words_english = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_english and word.strip()]
    return " ".join(tokens)

def convert_to_passive(text_input, nlp_processor): # Added nlp_processor argument
    doc = nlp_processor(text_input)
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

# --- Main Training Logic ---
def main():
    args = parse_arguments()
    nlp_spacy_instance = initialize_nlp_resources() # Get the loaded SpaCy model

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_tokenizer_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)

    # --- Load and Prepare Data ---
    print(f"Loading data from {args.fake_csv} and {args.true_csv}...")
    try:
        fake_df = pd.read_csv(args.fake_csv)
        true_df = pd.read_csv(args.true_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure CSV files exist at the specified paths.")
        exit()

    fake_df["label"] = 0
    true_df["label"] = 1
    news_df = pd.concat([fake_df, true_df], ignore_index=True)
    news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Preprocessing text data...")
    # Assuming 'text' is the column name in your CSVs
    if 'text' not in news_df.columns:
        print("Error: 'text' column not found in the combined dataframe. Please check your CSVs.")
        # You might also check 'title' or other relevant columns
        # For now, we'll assume 'title' if 'text' is not present.
        if 'title' in news_df.columns:
            print("Using 'title' column as text input.")
            news_df['processed_text'] = news_df['title'].astype(str).apply(preprocess_text)
        else:
            print("Error: Neither 'text' nor 'title' column found. Exiting.")
            exit()
    else:
        news_df['processed_text'] = news_df['text'].astype(str).apply(preprocess_text)


    # --- Tokenize and Pad Sequences ---
    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(news_df["processed_text"]) # Use the processed text column
    sequences = tokenizer.texts_to_sequences(news_df["processed_text"])
    data = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    labels = news_df["label"].values

    # Save the tokenizer using pickle
    print(f"Saving tokenizer to {args.output_tokenizer_path}...")
    with open(args.output_tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved successfully as pickle file.")

    # --- Train-Test Split and Handle Imbalance (SMOTE) ---
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"Original training shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training shape: {X_train_resampled.shape}, Class distribution: {np.bincount(y_train_resampled)}")

    # --- Define CNN + LSTM Model ---
    model_fake = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(CNN_FILTERS, 5, activation='relu'),
        Dropout(DROPOUT_RATE),
        Conv1D(CNN_FILTERS, 5, activation='relu'),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(LSTM_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    model_fake.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_fake.summary()

    # --- Train Model ---
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    print("Training Fake News Model...")
    history = model_fake.fit(X_train_resampled, y_train_resampled, batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(X_test, y_test), callbacks=[early_stop])

    # Save the trained model
    print(f"Saving trained model to {args.output_model_path}...")
    model_fake.save(args.output_model_path)
    print(f"Model saved successfully.")

    # --- Evaluate the Model ---
    print("\nEvaluating Fake News Model:")
    loss, accuracy = model_fake.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_proba = model_fake.predict(X_test)
    y_pred_fake_eval = (y_pred_proba > PREDICTION_THRESHOLD).astype(int) # Renamed to avoid conflict

    print("Accuracy:", accuracy_score(y_test, y_pred_fake_eval))
    print("Precision:", precision_score(y_test, y_pred_fake_eval))
    print("Recall:", recall_score(y_test, y_pred_fake_eval))
    print("F1 Score:", f1_score(y_test, y_pred_fake_eval))

    # --- Interactive Prediction (Optional, for quick testing after training) ---
    print("\n--- Interactive Fake News Detection Test ---")
    print(f"Loading sentiment analysis model from {args.sentiment_model_cache} (or downloading if not found)...")
    sentiment_pipeline_instance = None
    try:
        # Check if the sentiment model cache path exists and is a directory
        if os.path.exists(args.sentiment_model_cache) and os.path.isdir(args.sentiment_model_cache):
            sentiment_pipeline_instance = pipeline("sentiment-analysis", model=args.sentiment_model_cache, tokenizer=args.sentiment_model_cache)
            print(f"Loaded sentiment model from local cache: {args.sentiment_model_cache}")
        else:
            print(f"Sentiment model cache not found at {args.sentiment_model_cache}. Attempting to download default.")
            sentiment_pipeline_instance = pipeline("sentiment-analysis") # Downloads default
            # Optionally save it after download for future use by this script:
            # sentiment_pipeline_instance.model.save_pretrained(args.sentiment_model_cache)
            # sentiment_pipeline_instance.tokenizer.save_pretrained(args.sentiment_model_cache)
            # print(f"Default sentiment model downloaded. Consider running download_hf_models.py to cache it at {args.sentiment_model_cache}")

    except Exception as e:
        print(f"Could not load/download sentiment model: {e}. Sentiment analysis will be skipped in demo.")

    while True:
        user_text = input("\nEnter news article text for demo (or type 'exit' to quit):\n")
        if user_text.lower() == 'exit':
            break
        if not user_text.strip():
            print("Please enter some text.")
            continue

        # For fake news prediction:
        processed_user_text = preprocess_text(user_text)
        seq = tokenizer.texts_to_sequences([processed_user_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        pred_proba_user = model_fake.predict(padded)[0][0]
        fake_news_label_user = "True News" if pred_proba_user > PREDICTION_THRESHOLD else "Fake News"
        
        print("\n--- Demo Predictions ---")
        print(f"Fake News Detection: {fake_news_label_user} (Raw Score for 'True': {pred_proba_user:.2f})")

        # For sentiment analysis:
        if sentiment_pipeline_instance:
            try:
                sentiment_result = sentiment_pipeline_instance(user_text[:512])[0] # Truncate
                sentiment_label_user = f"{sentiment_result['label']} (Score: {sentiment_result['score']:.2f})"
                print(f"Sentiment Analysis: {sentiment_label_user}")
            except Exception as e_sentiment:
                print(f"Sentiment analysis error for demo: {e_sentiment}")
        else:
            print("Sentiment analysis skipped (model not loaded).")
        print("---------------------\n")

if __name__ == "__main__":
    main()