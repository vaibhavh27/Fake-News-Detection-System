import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse # Good practice to add, even if just for one path

# --- Configuration ---
DEFAULT_SENTIMENT_MODEL_SAVE_DIR = "assets/sentiment_model_cache/distilbert_sentiment"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    parser = argparse.ArgumentParser(description="Download and cache Hugging Face Sentiment Analysis Model.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULT_SENTIMENT_MODEL_SAVE_DIR,
        help=f"Directory to save the downloaded model and tokenizer. Default: {DEFAULT_SENTIMENT_MODEL_SAVE_DIR}"
    )
    args = parser.parse_args()

    save_location = args.save_path

    # Create the target directory and any parent directories if they don't exist
    os.makedirs(save_location, exist_ok=True)
    print(f"Ensured directory exists: {save_location}")

    print(f"Downloading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(save_location)
    print(f"Tokenizer saved to '{save_location}'.")

    print(f"Downloading model '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(save_location)
    print(f"Model saved to '{save_location}'.")

    print(f"\nSentiment model and tokenizer downloaded and saved successfully to '{save_location}'.")

if __name__ == "__main__":
    main()