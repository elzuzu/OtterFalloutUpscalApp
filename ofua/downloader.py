import urllib.request
import zipfile
from pathlib import Path
import shutil
import time
from .config import REALESRGAN_DOWNLOAD_URL


def download_with_retry(url: str, dest: Path, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"Échec téléchargement après {max_retries} tentatives: {e}")
    return False


def download_real_esrgan(dest_dir: Path, url: str = REALESRGAN_DOWNLOAD_URL) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "real-esrgan.zip"
    print(f"Téléchargement de Real-ESRGAN depuis {url}...")
    if not download_with_retry(url, zip_path):
        return dest_dir
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except Exception as e:
        print(f"Échec de l'extraction: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()
    print(f"Real-ESRGAN téléchargé dans {dest_dir}")
    return dest_dir


def find_realesrgan_exe(search_dir: Path) -> Path | None:
    for exe_name in ("realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"):
        for path in search_dir.rglob(exe_name):
            if path.is_file():
                return path
    return None
