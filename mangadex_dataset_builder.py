# file: D:\NewUser\machine_learning_project_v2/mangadex_dataset_builder.py
import os
import csv
import requests
from tqdm import tqdm
import time

API_URL = "https://api.mangadex.org"
IMG_URL = "https://uploads.mangadex.org/covers"
DATA_DIR = "data/images"
META_FILE = "data/manga_dataset.csv"

os.makedirs(DATA_DIR, exist_ok=True)

def safe_get(url, params=None, retries=3, delay=1.0):
    """Make API requests with retry and rate-limit handling"""
    for _ in range(retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response
        time.sleep(delay)
    response.raise_for_status()

def get_manga_list(limit=100, offset=0):
    params = {
        "limit": limit,
        "offset": offset,
        "availableTranslatedLanguage[]": "en"
    }
    resp = safe_get(f"{API_URL}/manga", params)
    return resp.json()

def get_cover_filename(manga_id):
    resp = safe_get(f"{API_URL}/cover", params={"manga[]": manga_id})
    data = resp.json().get("data", [])
    if not data:
        return None
    return data[0]["attributes"]["fileName"]

def download_cover(manga_id, filename):
    url = f"{IMG_URL}/{manga_id}/{filename}"
    resp = requests.get(url)
    if resp.status_code == 200:
        path = os.path.join(DATA_DIR, f"{manga_id}.jpg")
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    return ""

def extract_description(attr):
    desc = attr.get("description", {})
    if isinstance(desc, dict):
        return desc.get("en") or desc.get("ja") or ""
    return ""

def build_dataset(total=500):
    with open(META_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title_ja", "title_en", "tags", "cover_image_path", "description"])
        offset = 0
        pbar = tqdm(total=total, desc="Collecting manga")
        while offset < total:
            data = get_manga_list(limit=100, offset=offset)
            for manga in data.get("data", []):
                attr = manga["attributes"]
                title_en = attr["title"].get("en", "").replace(",", " ")
                title_ja = attr["title"].get("ja", "").replace(",", " ")
                tags = [t["attributes"]["name"]["en"] for t in attr["tags"]]
                desc = extract_description(attr).replace("\n", " ").replace(",", " ")
                cover_filename = get_cover_filename(manga["id"])
                img_path = download_cover(manga["id"], cover_filename) if cover_filename else ""
                writer.writerow([title_ja, title_en, "|".join(tags), img_path, desc])
                pbar.update(1)
                time.sleep(0.25)  # stay under API rate limit
            offset += 100
        pbar.close()

if __name__ == "__main__":
    build_dataset(total=500)  # Increase this to collect more manga
