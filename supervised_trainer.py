import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import joblib

# gets the relative file path to data.csv
try:
    base_dir = os.path.dirname(__file__)  # works if running as a script
except NameError:
    base_dir = os.getcwd()  # fallback for Spyder or notebooks
csv_path = os.path.join(base_dir, "data", "manga_dataset.csv")

# load csv data
df = pd.read_csv(csv_path)


# --- Prepare text input ---
# Combine title_en and description into one text field
df["text"] = (df["title_ja"].fillna("") + " " + df["title_en"].fillna("") + " " + df["description"].fillna("")).str.strip()

# --- Optional: split CamelCase in Romaji titles ---
def split_camel_case(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
df["text"] = df["text"].apply(split_camel_case)

# Drop rows with missing tags
df = df.dropna(subset=["tags"])

# Split the tag string into lists
df["tags"] = df["tags"].apply(lambda x: x.split("|"))

# --- Encode labels ---
#convert a list of labels per sample into a binary matrix.
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["tags"]) # to be kept seperate since it will be used for training

# Convert to a DataFrame with readable column headers (genre names) - this is here to make it easier to understand the Y labels
Y_df = pd.DataFrame(Y, columns=mlb.classes_)

# Optional: merge it back with your original df for inspection
#df_with_genres = pd.concat([df, Y_df], axis=1)


# --- TF-IDF vectorization for text features ---
#This converts raw text (your combined title_ja, title_en, and description) into a numeric matrix that a model can use.
#TF-IDF stands for Term Frequency–Inverse Document Frequency, a way to measure how important a word is in a document relative to all documents.
# this also automatically considers case sensitive words by making everything lowercase
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X = tfidf.fit_transform(df["text"])


# --- 10-fold cross-validation using iterative stratification ---
n_splits = 10

#IterativeStratification is from skmultilearn, specifically for multi-label datasets.
#Standard KFold or StratifiedKFold in scikit-learn is not suitable for multi-label data, because each sample can belong to multiple labels at once.
#Iterative stratification ensures that each fold roughly preserves the label distribution for all genres.
stratifier = IterativeStratification(n_splits=n_splits, order=1)
fold = 1

#This is an empty list that will store the classification reports for each fold.
#After running all folds, you can:
#Inspect each fold’s performance individually.
#Optionally aggregate metrics (precision, recall, F1) across folds.
all_reports = []

#stratifier.split(X, Y) generates indices for training and testing for each fold.
#train_idx → array of row indices to use for training.
#test_idx → array of row indices to use for testing.
for train_idx, test_idx in stratifier.split(X, Y):
    print(f"--- Fold {fold} ---")
    
    #Spliting data
    #X → the TF-IDF features of your manga text.
    #Y → the binary matrix of genres (output of MultiLabelBinarizer).
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]

    #OneVsRestClassifier wraps LogisticRegression to handle multi-label classification.
    #Each genre is treated as a separate binary classification task.
    #The classifier learns 0/1 predictions for each genre independently.
    #max_iter=200 → allows the logistic regression solver to converge (default 100 might be too low for some datasets).
    #max_iter=500 → allows enough iterations to converge.
    #solver="saga" → efficient for sparse, high-dimensional matrices like your TF-IDF. TF-IDF converts text into a matrix of numbers (features × samples). Most entries in this matrix are 0 because most words don’t appear in every document. A matrix with mostly zeros is called sparse.
    clf = OneVsRestClassifier(LogisticRegression(max_iter=500, solver="saga"))
    
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    #summarize how well your model is performing across all labels
    #It computes precision, recall, F1-score, and support for each class
    #Precision: how accurate your model positive predict are
    #Recall: how well your model finds all manga with a genre.
    #F1-score: Balances the trade-off between precision and recall.
    #Support: The number of true instances for that genre in the test set.
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)
    
    
    print(report)
    all_reports.append(report)
    fold += 1


# --- Save models---
os.makedirs("models", exist_ok=True)

#joblib is a Python library optimized for saving large objects efficiently, like machine learning models or large arrays.
#dump(obj, filename) saves the Python object obj to a file filename.
#You can later load it back with joblib.load(filename) to use the model or transformer.
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb, "models/genre_binarizer.pkl")
joblib.dump(clf, "models/genre_classifier.pkl")

