from flask import Flask, render_template, request
#Flask: The main Flask class that creates your web app object (it’s the “engine” of the app).
#render_template: Lets you render HTML files (like index.html) stored in your templates/ folder, and dynamically insert variables into them
#request: Lets you access data sent from the user — like form inputs or URL parameters (POST or GET requests)

import joblib
import requests
import re
import os

#This line creates your Flask application object — the core of the app.
#The __name__ argument tells Flask where to find resources (templates, static files, etc.).
#If you’re running app.py directly, __name__ will be "__main__".
#If it’s imported, Flask still knows where to look for templates.
app = Flask(__name__)

# Load models
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
mlb = joblib.load("models/genre_binarizer.pkl")
clf = joblib.load("models/genre_classifier.pkl")

# Predict genres from text
def predict_genres_from_text(title, description):
    text = title + " " + description

    #This line converts the raw text into a numerical vector that your model understands
    X = tfidf.transform([text])

    #This uses your trained classifier model (clf) to predict genres.
    y_pred = clf.predict(X)

    # mlb is your MultiLabelBinarizer used during training. It knows how to map the binary output (1s and 0s) back to the genre names
    genres = mlb.inverse_transform(y_pred)

    #return the first element [0] to get just the tuple of genre names.
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

    #This line uses Python’s re module (regular expressions) to search for the manga ID pattern inside the URL.
    match = re.search(r'/title/([a-f0-9-]+)', url)

    # if match is found, return first captured group (the manga ID)
    if match:
        return match.group(1)
    return None

# Fetch manga info and download cover locally
def fetch_manga_info(manga_id):
    #URL retrieves data for a specific manga
    manga_url = f"https://api.mangadex.org/manga/{manga_id}?includes[]=cover_art"

    #sends a GET request to that URL.
    response = requests.get(manga_url)

    # If request failed, return None
    if response.status_code != 200:
        return None, None, [], "/static/placeholder.png"

    #converts the response body into a Python dictionary.
    data = response.json().get("data", {})

    #detailed info like title, description, tags, etc.
    attributes = data.get("attributes", {})

    # Title & description
    #MangaDex provides multiple languages. It first tries to get the English version (.get("en")).
    #If that doesn’t exist, it falls back to the first available translation
    title = attributes.get("title", {}).get("en") or list(attributes.get("title", {}).values())[0]
    description = attributes.get("description", {}).get("en") or list(attributes.get("description", {}).values())[0]

    # Actual genres
    # Each tag represents a genre (like “Action”, “Romance”, etc.).
    #This loop collects all English tag names into a list called genres
    genres = []
    for tag in attributes.get("tags", []):
        tag_name = tag.get("attributes", {}).get("name", {}).get("en")
        if tag_name:
            genres.append(tag_name)

    # Default cover
    cover_url = "/static/placeholder.png"
    cover_file_name = None

    # Try relationships  to get cover art file name
    for rel in data.get("relationships", []):
        if rel.get("type") == "cover_art":
            cover_file_name = rel.get("attributes", {}).get("fileName")
            break

    # Fallback to included array in case not found in relationships
    if not cover_file_name:
        #included is another key sometimes returned in MangaDex API responses.
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

        #check if the cover image already exists locally to avoid re-downloading it.
        if not os.path.exists(cover_path):
            try:
                #Sends an HTTP GET request to MangaDex to fetch the cover image.
                img_data = requests.get(cover_url_api).content
                
                #Opens a file at cover_path for writing binary data (wb = write binary)
                with open(cover_path, "wb") as f:
                    #Writes the downloaded bytes to your local file
                    f.write(img_data)
                print("Downloaded cover:", cover_path)

            except Exception as e:
                print("Error downloading cover:", e)
                cover_path = "/static/placeholder.png"

        cover_url = f"/static/covers/{manga_id}.jpg"

    # Return title, description, genres, cover URL of the manga
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

    #POST is used when a user submits a form (like entering a MangaDex URL). This ensures the code inside only runs after the user clicks “Predict”.
    if request.method == 'POST':
        #contains data submitted via the form
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

                    #use train model to predict genre from extracted title and description of specified manga
                    predicted_genres = predict_genres_from_text(title, description)

                    #calculate accuracy by comparing predicted genres to actual genres from MangaDex
                    accuracy = calculate_accuracy(predicted_genres, actual_genres)
                    predicted_cover_url = actual_cover_url  # For simplicity, same cover

            except Exception as e:
                print("Error fetching MangaDex data:", e)
                error_message = "Failed to fetch data from MangaDex. Please try again later."

    # Pass title and description to template
    # Loads the HTML template file (index.html) from your templates/ folder.
    # Injects variables into the template so that they can be used in Jinja2 syntax ({{ variable }} or {% ... %}).
    # Returns a fully rendered HTML page to the user’s browser.
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
