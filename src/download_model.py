import os
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# TODO: Add actual hash once we have it
EXPECTED_HASH = None  # We can add hash verification later

def verify_model(model_path: Path) -> bool:
    """Verify model file integrity"""
    if not model_path.exists():
        return False

    if EXPECTED_HASH:
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == EXPECTED_HASH

    # If no hash provided, just check file exists
    return True

def download_model(model_path: Path):
    """Download model with progress bar"""
    print(f"Creating model directory {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILENAME} to {model_path} from {MODEL_URL}")
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(model_path, 'wb') as file, tqdm(
        desc=f"Downloading {MODEL_FILENAME}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def ensure_model_downloaded():
    """Ensure model is downloaded and valid"""
    model_path = MODEL_DIR / MODEL_FILENAME

    if not verify_model(model_path):
        print(f"Model not found or invalid. Downloading {MODEL_FILENAME}...")
        download_model(model_path)

        if not verify_model(model_path):
            raise RuntimeError("Downloaded model failed verification")

        print("Model downloaded and verified successfully")
    else:
        print(f"Model {MODEL_FILENAME} verified successfully")

    return model_path
