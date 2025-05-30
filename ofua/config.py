import os
from pathlib import Path
import psutil
FALLOUT_CE_REPO_URL = "https://github.com/elzuzu/fallout1-ce.git"
DEFAULT_WORKSPACE_DIR = Path("./fallout_upscaler_workspace")
DEFAULT_REALESRGAN_EXE = ""
DEFAULT_REALESRGAN_MODEL = ""
DEFAULT_UPSCALE_FACTOR = "4"
def get_optimal_workers() -> int:
    """Return a sensible number of workers based on CPU and RAM."""
    cpu_count = os.cpu_count() or 4
    memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    max_workers_by_memory = int(memory_gb * 0.8 / 0.5)
    return min(cpu_count, max_workers_by_memory, 8)

CPU_WORKERS = get_optimal_workers()
DEFAULT_GPU_ID = 0
REALESRGAN_DOWNLOAD_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.3.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
)

def setup_logging(workspace_dir: Path):
    """Configure logging to file and console."""
    import logging

    log_file = workspace_dir / "upscaler.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
