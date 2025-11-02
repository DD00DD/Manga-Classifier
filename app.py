from flask import Flask, render_template, request
import joblib
import requests
import re
import os

app = Flask(__name__)

# Load models
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
mlb = joblib.load("models/genre_binarizer.pkl")
clf = joblib.load("models/genre_classifier.pkl")

# Predict genres from text
def predict_genres_from_text(title, description):
    text = title + " " + description
    X = tfidf.transform([text])
    y_pred = clf.predict(X)
    genres = mlb.inverse_transform(y_pred)
    return genres[0]

# Accuracy calculation
def calculate_accuracy(predicted, actual):
    """
    Compute the accuracy of predicted genres against actual genres.

    Accuracy = fraction of actual genres correctly predicted (recall).

    Both predicted and actual can be lists or tuples.
    """

    # Ensure both are sets (removes duplicates and handles tuples/lists)
    predicted_set = set(predicted)
    actual_set = set(actual)

    if not actual_set:
        return 0  # Avoid division by zero

    # Count overlapping genres
    overlap = len(predicted_set & actual_set)

    # Accuracy as fraction of actual genres
    accuracy = overlap / len(actual_set)

    return accuracy


# Extract MangaDex ID from URL
def get_manga_id_from_url(url):
    match = re.search(r'/title/([a-f0-9-]+)', url)
    if match:
        return match.group(1)
    return None

# Fetch manga info and download cover locally
def fetch_manga_info(manga_id):
    manga_url = f"https://api.mangadex.org/manga/{manga_id}?includes[]=cover_art"
    response = requests.get(manga_url)
    
    if response.status_code != 200:
        return None, None, [], "/static/placeholder.png"

    data = response.json().get("data", {})
    attributes = data.get("attributes", {})

    # Title & description
    title = attributes.get("title", {}).get("en") or list(attributes.get("title", {}).values())[0]
    description = attributes.get("description", {}).get("en") or list(attributes.get("description", {}).values())[0]

    # Actual genres
    genres = []
    for tag in attributes.get("tags", []):
        tag_name = tag.get("attributes", {}).get("name", {}).get("en")
        if tag_name:
            genres.append(tag_name)

    # Default cover
    cover_url = "/static/placeholder.png"
    cover_file_name = None

    # Try relationships first
    for rel in data.get("relationships", []):
        if rel.get("type") == "cover_art":
            cover_file_name = rel.get("attributes", {}).get("fileName")
            break

    # Fallback to included array
    if not cover_file_name:
        included = response.json().get("included", [])
        for item in included:
            if item.get("type") == "cover_art":
                cover_file_name = item.get("attributes", {}).get("fileName")
                break

    # Download cover if available
    if cover_file_name:
        cover_url_api = f"https://uploads.mangadex.org/covers/{manga_id}/{cover_file_name}.256.jpg"
        os.makedirs("static/covers", exist_ok=True)
        cover_path = f"static/covers/{manga_id}.jpg"

        if not os.path.exists(cover_path):
            try:
                img_data = requests.get(cover_url_api).content
                with open(cover_path, "wb") as f:
                    f.write(img_data)
                print("Downloaded cover:", cover_path)
            except Exception as e:
                print("Error downloading cover:", e)
                cover_path = "/static/placeholder.png"

        cover_url = f"/static/covers/{manga_id}.jpg"

    return title, description, genres, cover_url

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_genres = []
    actual_genres = []
    accuracy = 0
    predicted_cover_url = "/static/placeholder.png"
    actual_cover_url = "/static/placeholder.png"
    error_message = None
    title = ""        # Initialize title
    description = ""  # Initialize description

    if request.method == 'POST':
        manga_url = request.form.get('manga_url')
        manga_id = get_manga_id_from_url(manga_url)

        if not manga_id:
            error_message = "Invalid MangaDex URL. Please enter a valid manga link."
        else:
            try:
                title, description, actual_genres, actual_cover_url = fetch_manga_info(manga_id)

                if not title or not description:
                    error_message = "Could not fetch manga info. Please try another link."
                else:
                    predicted_genres = predict_genres_from_text(title, description)
                    accuracy = calculate_accuracy(predicted_genres, actual_genres)
                    predicted_cover_url = actual_cover_url  # For simplicity, same cover

            except Exception as e:
                print("Error fetching MangaDex data:", e)
                error_message = "Failed to fetch data from MangaDex. Please try again later."

    # Pass title and description to template
    return render_template(
        'index.html',
        predicted_genres=predicted_genres,
        actual_genres=actual_genres,
        accuracy=accuracy,
        predicted_cover_url=predicted_cover_url,
        actual_cover_url=actual_cover_url,
        error_message=error_message,
        title=title,
        description=description
    )

if __name__ == '__main__':
    app.run(debug=True)
