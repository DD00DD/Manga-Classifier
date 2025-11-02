# file: D:\NewUser\machine_learning_project_v2\train_genre_classifier.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Load dataset ---
df = pd.read_csv("data/manga_dataset.csv")

# --- Prepare text input ---
# Combine title_en and description into one text field
df["text"] = (df["title_en"].fillna("") + " " + df["description"].fillna("")).str.strip()

# Drop rows with missing tags
df = df.dropna(subset=["tags"])

# Split the tag string into lists
df["tags"] = df["tags"].apply(lambda x: x.split("|"))

# --- Encode labels ---
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["tags"])

# --- TF-IDF vectorization for text features ---
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X = tfidf.fit_transform(df["text"])

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Train classifier ---
clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# --- Save models (optional) ---
import joblib
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb, "models/genre_binarizer.pkl")
joblib.dump(clf, "models/genre_classifier.pkl")
