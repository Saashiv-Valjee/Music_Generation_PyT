import os
import requests
import zipfile
from tqdm import tqdm

# downloads the dataset
def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        print(f"{filename} already exists. Skipping download.")
        return file_path

    # Stream download to handle large files efficiently
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    with open(file_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return file_path

def extract_zip(file_path, extract_to):
    print(f"Extracting {file_path} to {extract_to}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def main():
    maestro_url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
    data_folder = 'data'

    print("Starting download of MAESTRO dataset...")
    zip_file_path = download_file(maestro_url, data_folder)

    print("Download completed.")

    print("Starting extraction...")
    extract_zip(zip_file_path, data_folder)

    # Optional: Remove the ZIP file after extraction to save space
    os.remove(zip_file_path)
    print("ZIP file removed after extraction.")

if __name__ == '__main__':
    main()
