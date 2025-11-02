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
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["tags"]) # to be kept seperate since it will be used for training

# Convert to a DataFrame with readable column headers (genre names) - this is here to make it easier to understand the Y labels
Y_df = pd.DataFrame(Y, columns=mlb.classes_)

# Optional: merge it back with your original df for inspection
#df_with_genres = pd.concat([df, Y_df], axis=1)


# --- TF-IDF vectorization for text features ---
#This converts raw text (your combined title_ja, title_en, and description) into a numeric matrix that a model can use.
#TF-IDF stands for Term Frequency–Inverse Document Frequency, a way to measure how important a word is in a document relative to all documents.
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X = tfidf.fit_transform(df["text"])


# --- 10-fold cross-validation using iterative stratification ---
n_splits = 10
stratifier = IterativeStratification(n_splits=n_splits, order=1)
fold = 1
all_reports = []

for train_idx, test_idx in stratifier.split(X, Y):
    print(f"--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)
    print(report)
    all_reports.append(report)
    fold += 1


# --- Save models (optional) ---
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb, "models/genre_binarizer.pkl")
joblib.dump(clf, "models/genre_classifier.pkl")

