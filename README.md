<!-- # Fake News Detection Model
Made 2 Models for the detection of Fake News.  
The model uses CNN + biLSTM for news detection.  
Uses distilBERT for Sentiment Analysis.
## Download Links for 1st Model (Contains Databases, Model, and Tokenizer)
***[CLICK ME](https://drive.google.com/drive/folders/189UjfsBH5Ur6fOx6Q4VChhIJ3O4lQ-bf?usp=sharing)***
## Download Links for 2nd Model (Contains Databases, Model, and Tokenizer)
***[CLICK ME](https://drive.google.com/drive/folders/1Czjijig-OMXdrBfXNBii2d13Wfs3KmUh?usp=sharing)***

## Building Your Model
### Download Requirements
```
pip install pandas
pip install numpy
pip install nltk
pip install spacy
pip install imbalanced-learn
pip install scikit-learn
pip install tensorflow
pip install transformers
python -m spacy download en_core_web_sm
```
```
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
```
### Files In Repo and What They Keep
1. Code - Consists of Code for model building. Can also use your database for creating your model.
2. Download your model - Consists of code to download your model and tokenizer.
3. Quick Load Model - Code for quick running of the model ( load your model and tokenizer and run in a separate file ) -->


# Fake News Detection System

This project implements a system for detecting fake news articles using a hybrid Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) model. It also includes sentiment analysis of the input text using a pre-trained DistilBERT model from Hugging Face Transformers.

The system allows users to classify news text as "True" or "Fake" and to analyze its sentiment. Two distinct pre-trained fake news detection models and their associated data/tokenizers are provided via Google Drive links. Users can also train their own models using the provided scripts and datasets.

## Features
*   **Fake News Detection:** Classifies news articles or text snippets as "True News" or "Fake News".
*   **Sentiment Analysis:** Determines the sentiment of the input text (e.g., POSITIVE/NEGATIVE) using a fine-tuned DistilBERT model.
*   **Hybrid Deep Learning Model:** Utilizes a CNN for local feature extraction and a BiLSTM for capturing sequential dependencies in text, leading to robust classification.
*   **Modular Scripts:** Separate scripts for training, inference, and downloading external models.
*   **Configurable Training:** The training script allows for custom datasets and output paths for models and tokenizers.
*   **Pre-trained Models:** Includes links to download pre-trained models for immediate use.

## Pre-trained Models Provided

This project offers two pre-trained fake news detection models, each potentially differing in training data or focus:

*   **Model 1:** This model might be trained on a general news dataset covering a variety of topics, aiming for broad applicability.
*   **Model 2:** This model could be fine-tuned on a more specific domain (e.g., political news, health news) or use a slightly different variant of the training data or hyperparameters for comparative performance.

*(Please verify the actual differences between your Model 1 and Model 2 and update the descriptions above accordingly for clarity.)*

**Download Links for Pre-trained Models, Tokenizers, and Datasets:**
*   **Model 1 Assets:** ***[CLICK ME](https://drive.google.com/drive/folders/189UjfsBH5Ur6fOx6Q4VChhIJ3O4lQ-bf?usp=sharing)***
    *   Expected contents: `fake_news_model.h5`, `tokenizer.pickle`, `Fake.csv`, `True.csv`
*   **Model 2 Assets:** ***[CLICK ME](https://drive.google.com/drive/folders/1Czjijig-OMXdrBfXNBii2d13Wfs3KmUh?usp=sharing)***
    *   Expected contents: `fake_news_model.h5`, `tokenizer.pickle`, `Fake_v2.csv` (example name), `True_v2.csv` (example name)

## Project File Structure

Your project directory will be organized as follows. The `sentiment_model_cache` directory and its contents will be **created automatically when you run `download_hf_models.py`**.

your-project-root/
├── train_model.py             # Script to train a new fake news model
├── run_inference.py           # Script to run predictions using a pre-trained model
├── download_hf_models.py      # Script to download the DistilBERT sentiment model
├── README.md                  # This documentation file
├── requirements.txt           # Python package dependencies
└── assets/
    ├── model_1/               # Contains assets for pre-trained Model 1 (example)
    │   ├── fake_news_model.h5 # The trained Keras model
    │   ├── tokenizer.pickle   # The Keras tokenizer object
    │   ├── Fake.csv           # Example fake news dataset used for this model
    │   └── True.csv           # Example true news dataset used for this model
    ├── model_2/               # Contains assets for pre-trained Model 2 (example)
    │   └── ...                # (model, tokenizer, and data CSVs for model 2)
    └── sentiment_model_cache/ # **CREATED BY `download_hf_models.py`**
        └── distilbert_sentiment/  # Cache for the downloaded DistilBERT model
            ├── config.json
            ├── pytorch_model.bin (or model.safetensors)
            └── ... (other tokenizer/model files)

## Setup and Installation

Follow these steps to set up the project environment:

1.  **Clone the Repository (if applicable) or Download Scripts:**
    If this project is hosted on a Git platform (like GitHub):
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-name>
    ```
    Otherwise, ensure `train_model.py`, `run_inference.py`, `download_hf_models.py`, `README.md`, and `requirements.txt` are in your project's root directory.

2.  **Create a Virtual Environment (Highly Recommended):**
    This isolates project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    Ensure the `requirements.txt` file (content provided in a previous response or create one based on the list below) is in your project root. Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include: `pandas`, `numpy`, `nltk`, `spacy`, `imbalanced-learn`, `scikit-learn`, `tensorflow`, `transformers`, `torch`.*

4.  **Download NLTK and SpaCy Resources:**
    These are essential for text preprocessing. Run Python in your activated virtual environment and execute:
    ```python
    import nltk
    import spacy

    print("Downloading NLTK resources (stopwords, punkt, wordnet)...")
    nltk.download('stopwords', quiet=False)
    nltk.download('punkt', quiet=False)
    nltk.download('wordnet', quiet=False)
    print("NLTK resources downloaded.")

    print("Checking/Downloading SpaCy model (en_core_web_sm)...")
    try:
        spacy.load('en_core_web_sm')
        print("SpaCy model 'en_core_web_sm' already installed.")
    except OSError:
        print("SpaCy model not found. Downloading 'en_core_web_sm'...")
        spacy.cli.download('en_core_web_sm') # Alternatively: python -m spacy download en_core_web_sm
        print("SpaCy model 'en_core_web_sm' downloaded.")
    ```

5.  **Download and Cache Sentiment Analysis Model:**
    Run the `download_hf_models.py` script. This will download the DistilBERT model (specifically `distilbert-base-uncased-finetuned-sst-2-english`) for sentiment analysis and save it locally to avoid re-downloading during inference.
    ```bash
    python download_hf_models.py
    ```
    This script should be configured to create and populate a directory like `assets/sentiment_model_cache/distilbert_sentiment/`.

6.  **Download Pre-trained Fake News Model Assets:**
    *   If it doesn't already exist, create the main `assets/` directory.
        ```bash
        mkdir -p assets # Creates 'assets' if it doesn't exist (Linux/macOS)
                        # On Windows, you can manually create it.
        ```
    *   For **Model 1**:
        *   Download its assets (`fake_news_model.h5`, `tokenizer.pickle`, `Fake.csv`, `True.csv`) from the Google Drive link provided above.
        *   Create the directory `assets/model_1/`.
        *   Place all downloaded files for Model 1 into this `assets/model_1/` directory.
    *   Repeat this process for **Model 2** (or any other pre-trained models you provide), placing its files into a corresponding subdirectory (e.g., `assets/model_2/`).

## Usage

### 1. Running Inference with a Pre-trained Model

To classify new text using one of the pre-trained models:

Use the `run_inference.py` script. You need to specify the directory containing the assets of the model you wish to use and the location of the cached sentiment model.



**Example (using Model 1):**
```bash
python run_inference.py --model_dir assets/model_1/ --sentiment_cache_dir assets/sentiment_model_cache/distilbert_sentiment/

he script will then prompt you to enter news article text. It will output the fake news classification and the sentiment analysis result.
(Note for Developers: The run_inference.py script needs to be correctly configured to:
1. Accept --model_dir and --sentiment_cache_dir command-line arguments.
2. Load the Keras model (.h5) and the tokenizer.pickle from the specified --model_dir.
3. Load the sentiment analysis pipeline using the --sentiment_cache_dir.
4. Ensure consistency in MAX_LEN and PREDICTION_THRESHOLD parameters. These might be hardcoded if constant across models, or ideally, loaded from a small configuration file (e.g., config.json) within each model_dir, or passed as additional command-line arguments if they vary per pre-trained model.)
2. Training Your Own Fake News Model
To train a new fake news detection model from scratch using your own datasets:
Use the train_model.py script. This script is configurable via command-line arguments for input data paths, output paths for the trained model and tokenizer, and the sentiment model cache location for its interactive demo.
Example (training a new model using custom datasets and saving outputs):
Suppose your custom datasets MyFakeNews.csv and MyTrueNews.csv are located in assets/custom_data/. You want to save the trained model and tokenizer to assets/custom_model/.
python train_model.py \
    --fake_csv assets/custom_data/MyFakeNews.csv \
    --true_csv assets/custom_data/MyTrueNews.csv \
    --output_tokenizer_path assets/custom_model/tokenizer.pickle \
    --output_model_path assets/custom_model/fake_news_model.h5 \
    --sentiment_model_cache assets/sentiment_model_cache/distilbert_sentiment
Use code with caution.
Bash
This command will:
Load and preprocess data from the specified CSVs.
Train the CNN+BiLSTM model.
Save the trained Keras model to assets/custom_model/fake_news_model.h5.
Save the Keras tokenizer to assets/custom_model/tokenizer.pickle.
The script will create assets/custom_model/ if it doesn't already exist.
Using Default Paths for Training:
If your training data files are named Fake.csv and True.csv and are located in the project root directory, and you run the script without specifying output paths, it will use its default settings:
python train_model.py
Use code with caution.
Bash
This will typically save the outputs to a new directory named trained_assets/ in the project root.
File Descriptions
train_model.py: Python script for training a new fake news detection model. Includes data loading, preprocessing, model definition, training, evaluation, and saving the model/tokenizer.
run_inference.py: Python script for loading a pre-trained fake news model and its tokenizer to perform inference on new, unseen text provided by the user. Also integrates sentiment analysis.
download_hf_models.py: Utility script to download and locally cache the pre-trained DistilBERT sentiment analysis model and tokenizer from Hugging Face.
README.md: This documentation file, providing an overview and instructions for the project.
requirements.txt: A text file listing all Python package dependencies required to run the project.
assets/: A directory intended to store all model-related assets:
model_X/: Subdirectories for each pre-trained model (e.g., model_1/, model_2/). Each should contain the .h5 model file, the .pickle tokenizer file, and optionally the Fake.csv and True.csv files used for training that specific model.
sentiment_model_cache/: Recommended location for the cached DistilBERT sentiment model. This directory and its distilbert_sentiment/ subfolder are typically created by download_hf_models.py.
You can also place your raw training CSVs in other subdirectories within assets/ if you are training new models (e.g., assets/custom_data/).

License
---

This version is more descriptive and should be quite helpful for anyone looking at your project. Remember to replace the generic model descriptions with accurate ones for your specific Model 1 and Model 2. Also, ensure the "Note for Developers" in the inference section regarding `MAX_LEN` and `PREDICTION_THRESHOLD` is addressed in your `run_inference.py` script for robustness if those parameters change between your pre-trained models.
