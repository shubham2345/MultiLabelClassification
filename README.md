# MultiLabelClassification

Kaggle Competition Link: https://www.kaggle.com/competitions/nlp-243-fall-24-homework-1-relation-extraction/leaderboard


## Overview

This project addresses a supervised multi-label classification task focused on predicting multiple attributes associated with movie descriptions based on textual input data. Each movie description can be linked to several attributes such as genres, actors, directors, and other relevant features. &#x20;

---

## Dataset Details

The dataset consists of two main files:

- **Training Data: hw1\_train.csv**

  - Number of samples: 2312
  - Number of features: 2 (input attributes)
  - Number of labels: 19 (output attributes)
  - Number of None values: 319

- **Test Data: hw1\_test.csv**

  - Number of samples: 981
  - Number of features: 1 (input attributes)

The training dataset includes various movie descriptions along with their associated attributes, which are essential for training a multi-label classification model that generalizes well to unseen data.

---

## Input and Output Format

### Input

- **Column:** `UTTERANCES`
- **Description:** Textual descriptions of movies, including plot summaries, keywords, or other relevant details.

### Output

- **Column:** `CORE RELATIONS`
- **Description:** A space-separated string of attributes associated with each movie description. Each movie can have multiple attributes, and a "none" value indicates no attributes apply.

Example:

```
Input: A romantic drama set in 19th-century France.
Output: movie.genre romance movie.country france
```

---

## Techniques to Explore

Here are some techniques you can explore to improve your model's performance:

- **Embedding Methods:**

  - Use Count Vectorizer with character-level n-grams.
  - Experiment with pretrained embeddings such as Word2Vec and FastText.

- **Model Architectures:**

  - Implement a Multi-Layer Perceptron (MLP) model with character-level embeddings.
  - Use a baseline model for comparison.

- **Hyperparameter Tuning:**

  - Adjust learning rates, batch sizes, dropout rates, and hidden layer sizes.

- **Regularization Techniques:**

  - Use dropout, weight decay, and early stopping to prevent overfitting.

---

## Model Implementation

### Embedding Method: Count Vectorizer (Character-Level N-grams)

- **What It Is:** Converts text into a matrix of character counts.
- **Why Use It:** Handles diverse movie descriptions with spelling variations and unique character combinations.
- **Configuration:** Analyzes character sequences from single characters to five-character combinations.

### Model Architecture: Multi-Layer Perceptron (MLP)

- **Input Layer:** Accepts character-level embeddings.
- **Hidden Layers:** Three layers with 512, 256, and 128 neurons using ReLU activation.
- **Dropout Layers:** Added after each hidden layer to prevent overfitting.
- **Output Layer:** Uses a sigmoid activation function for multi-label classification.

---

## Command to Run the Code

To train the model and generate predictions, run the following command:

```bash
python run.py data/hw1_train.csv data/hw1_test.csv data/submission.csv
```

Ensure that your input files are correctly formatted and placed in the appropriate directories.

---
