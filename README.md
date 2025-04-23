# Sentiment Analysis on Moroccan Darija Tweets

This project focuses on performing sentiment analysis on Moroccan Darija (Moroccan Arabic) tweets. The goal is to classify tweets as either positive or negative using a pre-trained BERT model fine-tuned on a dataset of Moroccan Darija tweets. The project follows a structured approach, starting from data loading and preprocessing to model training, evaluation, and prediction.

## 🚀 Overview

The notebook demonstrates how to perform sentiment analysis in the Moroccan Darija language using BERT. It covers the following steps:
- Importing necessary libraries
- Loading and cleaning the dataset
- Tokenizing the text data
- Fine-tuning a BERT model
- Training, evaluating, and making predictions

---

## 📋 Steps

### 1. Importing Libraries
The necessary libraries for the analysis are installed and imported:
- `transformers` for the BERT model
- `torch` for PyTorch
- `pandas` for data manipulation

Additionally, Google Drive is mounted to access the dataset stored in the cloud.

---

### 2. Dataset Loading
The dataset, **ElecMorocco2016.xlsx**, contains 10,254 Arabic Facebook comments about the Moroccan elections of 2016. The comments are written in both standard Arabic and the Moroccan dialect (Darija). The dataset includes labels indicating whether the comments are positive or negative.

---

### 3. Dataset Cleaning and Visualization
The dataset undergoes the following preprocessing steps:
- Dropping unnecessary columns and renaming them for clarity
- Mapping the labels to binary values (0 for negative, 1 for positive)
- Cleaning the text data:
  - Removal of stop words in Darija, as well as domain-specific stop words (e.g., "الله", "بنكيران", "المغرب")
  - Removal of mentions, hashtags, URLs, non-Arabic characters, and short words (length ≤ 2)

After cleaning, the dataset is visualized by plotting the top 25 most common words to understand the most frequent terms in the cleaned text.

---

### 4. Data Splitting
The dataset is split into training and testing sets using an 80-20 split ratio.

---

### 5. Tokenization and Input Formatting
Using the BERT tokenizer, the text data is converted into a format suitable for model input:
- Sentences are padded to ensure uniform length
- Attention masks are created to differentiate between padding and actual data

The tokenized data and labels are converted into tensors and loaded into PyTorch DataLoaders for efficient batch processing.

---

### 6. Model Training
A custom BERT classifier is defined, which consists of:
- A pre-trained BERT model
- A feed-forward classifier on top of BERT

The model is trained for 2 epochs on the training dataset, with evaluation performed on the validation set after each epoch. Key aspects of training include:
- Loss calculation
- Gradient clipping
- Learning rate scheduling

---

### 7. Predicting & Evaluating the Model
The trained model is evaluated on the test set using the following metrics:
- **ROC-AUC**
- **Accuracy**

A ROC curve is plotted to visualize the model's performance.

---

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `transformers`
  - `torch`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`


