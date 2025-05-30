import os
from pathlib import Path
FALLOUT_CE_REPO_URL = "https://github.com/elzuzu/fallout1-ce.git"
DEFAULT_WORKSPACE_DIR = Path("./fallout_upscaler_workspace")
DEFAULT_REALESRGAN_EXE = ""
DEFAULT_REALESRGAN_MODEL = ""
DEFAULT_UPSCALE_FACTOR = "4"
CPU_WORKERS = os.cpu_count() or 4
DEFAULT_GPU_ID = 0
REALESRGAN_DOWNLOAD_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.3.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
)
