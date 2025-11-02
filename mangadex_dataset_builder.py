#testing this is DEV branch

# file: D:\NewUser\machine_learning_project_v2/mangadex_dataset_builder.py

#Imports Python’s built-in os module, which provides functions for interacting with the operating system, like creating directories, checking if files exist, and working with file paths
import os

# Imports the csv module, which allows you to read from and write to CSV (Comma-Separated Values) files. This will be used to save metadata about manga (like title, genres, description) in a structured format
import csv

# Imports the requests library, which is used to make HTTP requests in Python. Here it will be used to fetch manga data and images from the MangaDex API
import requests

# Imports tqdm, a library used to create progress bars. It’s helpful to visualize progress when downloading many images or iterating over a large dataset.
from tqdm import tqdm

# Imports the time module. This is often used for adding delays (time.sleep()) to avoid hitting rate limits of APIs or controlling execution timing
import time

# This is the base URL for the MangaDex API. All requests for manga data (like titles, genres, descriptions) will be made to endpoints under this URL.
API_URL = "https://api.mangadex.org"

#This is the base URL for cover images on MangaDex. Each manga cover image has an ID, and you append it to this URL to download the actual image.
IMG_URL = "https://uploads.mangadex.org/covers"

# This defines the local folder where manga cover images will be saved on your computer. The program will check if this directory exists and create it if needed, then save downloaded images here
DATA_DIR = "data/images"

#This is the path to the CSV file that stores metadata about each manga.
#Metadata includes things like: title, genres, cover filename, and possibly descriptions.
#The CSV allows your machine learning program to easily read and train on this structured data.
META_FILE = "data/manga_dataset.csv"

#This function creates a directory (or multiple nested directories) at the specified path.
os.makedirs(DATA_DIR, exist_ok=True)


#url: The API endpoint you want to request (e.g., https://api.mangadex.org/manga/).
#params: Optional dictionary of query parameters for the request.
#retries: How many times to retry the request if it fails (default is 3).
#delay: Seconds to wait between retries (default is 1.0 second).
#Function: Returns the response object if the request is successful.
def safe_get(url, params=None, retries=3, delay=1.0):
    """Make API requests with retry and rate-limit handling"""
    for _ in range(retries):

        #Makes a GET request to the URL using the requests library
        response = requests.get(url, params=params)

        #Checks if request was successful (status code 200)
        if response.status_code == 200:
            return response
        
        #Checks if the request was successful (status code 200).
        time.sleep(delay)

    #If all retries fail, this raises an HTTPError with details about why the request failed
    response.raise_for_status()

#limit: Maximum number of manga entries to fetch in one request (default is 100).
#offset: How many entries to skip before starting (default is 0). Useful for pagination
#Function: fetches a list of manga from the MangaDex API.
def get_manga_list(limit=100, offset=0):

    #builds a dictionary of query parameters to send with the API request.
    params = {
        "limit": limit,
        #"limit" → how many manga to fetch.

        "offset": offset,
        #"offset" → starting position in the full list of manga.

        "availableTranslatedLanguage[]": "en"
        #"availableTranslatedLanguage[]" → filter to only include manga that have English translations
    }

    #Sends a GET request to the MangaDex API endpoint for manga: https://api.mangadex.org/manga, Passes the params dictionary to filter and limit results.
    resp = safe_get(f"{API_URL}/manga", params)

    #Converts the API response from JSON (text) into a Python dictionary.
    return resp.json()

#relationships: The relationships field from a manga entry, which contains related entities like authors and cover art.
#Function: extracts the cover filename from the relationships field.

def get_cover_filename(manga_id): 
    #Sends a GET request to the MangaDex API endpoint for covers: https://api.mangadex.org/cover, with a query parameter specifying the manga cover ID. 
    resp = safe_get(f"{API_URL}/cover", params={"manga[]": manga_id}) 
    
    #Converts the response to a Python dictionary using .json(), If "data", which contains cover information, does not exist, it defaults to an empty list []. 
    data = resp.json().get("data", []) 
    
    #If no cover exists, the function returns None. 
    if not data: 
        return None
    
    # Accesses the first cover in the data list. Retrieves the "fileName" field from "attributes". 
    # This filename is what you append to the IMG_URL to download the actual cover image. 
    # #idk if this is correc***** 
    return data[0]["attributes"]["fileName"]

#manga_id: The unique identifier for the manga whose cover image you want to download.
#filename: The filename of the cover image to download.
#Function: downloads the cover image for a specific manga and saves it locally.
def download_cover(manga_id, filename):
    #Constructs the full URL of the cover image using an f-string
    url = f"{IMG_URL}/{manga_id}/{filename}"

    #Sends a GET request with cover image URL to download the image data.
    resp = requests.get(url)
    
    if resp.status_code == 200:

        #Defines the local path to save the image. Uses os.path.join to safely combine the directory (DATA_DIR) and filename and save image as .jpg file
        path = os.path.join(DATA_DIR, f"{manga_id}.jpg")

        #Using "wb" ensures that the bytes are written exactly as they came from the server
        #Writes the raw image data (resp.content) to the file.
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    
    return ""


#attr: The attributes dictionary from a manga entry.
#Function: extracts the description of a manga, prioritizing English and then Japanese.
def extract_description(attr):
    
    #Tries to get the "description" field from the attr dictionary.
    #If "description" doesn’t exist, it defaults to an empty dictionary {}.
    #On MangaDex, descriptions are usually returned as dictionaries with language codes as keys (idk if description is correct***)
    desc = attr.get("description", {})

    if isinstance(desc, dict):

        #Tries to get the English description first: desc.get("en"). If English doesn’t exist, falls back to Japanese: desc.get("ja"). If neither exists, returns an empty string ""
        return desc.get("en") or desc.get("ja") or ""
    return ""

#total: Total number of manga to collect.
#Function: builds the manga dataset by fetching manga data, downloading cover images, and saving metadata to a CSV file.
def build_dataset(total):

    # Opens the CSV file defined by META_FILE: This is a constant that holds the path to the CSV file where the dataset will be saved
    # for writing ("w") with UTF-8 encoding: UTF-8 encoding supports all Unicode characters, ensuring that your CSV file can correctly store Japanese titles, emoji, and other characters.
    # Using newline="" tells Python not to translate newlines, letting the csv module handle them correctly.
    with open(META_FILE, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        #Writes the header row with following columns headers
        writer.writerow(["title_ja", "title_en", "tags", "cover_image_path", "description"])


        #Pagination is a concept used when dealing with large amounts of data from APIs, databases, or web pages. Instead of fetching all data at once, you split it into smaller chunks (pages) and retrieve them one at a time.
        #Initializes offset to 0 for pagination.
        offset = 0

        #Creates a progress bar using tqdm to track how many manga have been processed.
        pbar = tqdm(total=total, desc="Collecting manga")


        while offset < total:
            # Loops until the total number of manga collected reaches total.
            # Calls get_manga_list to fetch up to 100 manga at a time from the API using the current offset.
            data = get_manga_list(limit=100, offset=offset)


            #Iterates over each manga in the API response and extracts relevant attributes.
            for manga in data.get("data", []):
                attr = manga["attributes"]

                #english title and japanese title
                title_en = attr["title"].get("en", "").replace(",", " ").strip()
                title_ja = attr["title"].get("ja", "").replace(",", " ").strip()

                # Skip entries with neither English nor Japanese titles
                if not title_en and not title_ja:
                    pbar.update(1)
                    time.sleep(0.25)
                    continue

                #genre tags
                tags = [t["attributes"]["name"]["en"] for t in attr["tags"]]

                #description
                desc = extract_description(attr).replace("\n", " ").replace(",", " ")

                #fetch and download cover image (will use in future for CNN model)
                #cover_filename = get_cover_filename(manga["id"])
                #img_path = download_cover(manga["id"], cover_filename) if cover_filename else ""
                img_path = "" # Skipping image download for now to speed up dataset creation

                #Writes a row to the CSV file with all the extracted data.
                writer.writerow([title_ja, title_en, "|".join(tags), img_path, desc])

                #update progress bar that tracks how many manga have been processed.
                pbar.update(1)
                time.sleep(0.25)  # stay under API rate limit

            # Writes a row to the CSV file with all the extracted data.
            offset += 100

        #Closes the progress bar after all manga have been processed.
        pbar.close()

if __name__ == "__main__":
    build_dataset(5000)  # Increase this to collect more manga
