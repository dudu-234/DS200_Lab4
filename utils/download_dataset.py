import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

dataset_name = "competitions/nyc-taxi-trip-duration"
download_dir = "./data"
zip_filename = "nyc-taxi-trip-duration.zip"
extract_subdir = "nyc-taxi-trip-duration"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)
    print(f"Created directory {download_dir}")

api = KaggleApi()
api.authenticate()

print("Downloading dataset...")
api.competition_download_files(
    competition="nyc-taxi-trip-duration",
    path=download_dir,
    quiet=False
)
print("Download complete.")

extract_dir = os.path.join(download_dir, extract_subdir)
os.makedirs(extract_dir, exist_ok=True)

zip_path = os.path.join(download_dir, zip_filename)
print(f"Unzipping {zip_path} into {extract_dir} ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Unzip complete.")

os.remove(zip_path)
print(f"Deleted zip file {zip_path}")

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            print(f"Unzipping nested {nested_zip_path} ...")
            with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                zip_ref.extractall(root)
            print(f"Unzipped {nested_zip_path}")
            os.remove(nested_zip_path)
            print(f"Deleted nested zip {nested_zip_path}")

print("All extraction complete.")
