import os
import zipfile
import pandas as pd
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from concurrent.futures import ThreadPoolExecutor

# --- Initialize API ---
api = KaggleApi()
api.authenticate()

# --- Settings ---
CSV_PATH = 'data/train_metadata.csv'
DEST_FOLDER = 'data/train'
COMPETITION = 'google-research-identify-contrails-reduce-global-warming'
THREADS = 4 # Keep this low to avoid 429 errors

df = pd.read_csv(CSV_PATH)

def download_and_extract(args):
    file_path, rid = args
    local_subdir = os.path.join(DEST_FOLDER, str(rid))
    file_name = os.path.basename(file_path) # e.g., band_11.npy
    final_path = os.path.join(local_subdir, file_name)

    # Skip if already exists
    if os.path.exists(final_path):
        return

    os.makedirs(local_subdir, exist_ok=True)

    try:
        api.competition_download_file(
            competition=COMPETITION,
            file_name=file_path,
            path=local_subdir,
            quiet=False
        )

        # 2. Handle the mandatory Zip
        zip_path = os.path.join(local_subdir, file_name + ".zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_subdir)
            os.remove(zip_path) # Clean up the zip

    except Exception as e:
        print(f"Error for {file_name}: {e}")

def main():
    # Create the task list
    tasks = []
    for rid in df['record_id'].astype(str):
        files = [
            f"train/{rid}/human_individual_masks.npy",
            f"train/{rid}/human_pixel_masks.npy",
            f"train/{rid}/band_11.npy",
            f"train/{rid}/band_14.npy",
            f"train/{rid}/band_15.npy"
        ]
        for f in files:
            tasks.append((f, rid))

    print(f"Downloading {len(tasks)} files...")

    # Basic Multithreading
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        list(tqdm(executor.map(download_and_extract, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()