"""Setup environment for the application, installing dependencies and tools."""
import shutil
import sys
import platform
import subprocess
from pathlib import Path

REQUIREMENTS = [
    "opencv-python>=4.5",
    "scikit-image",
    "Pillow",
    "numpy",
    "GitPython",
    "psutil",
    "requests"
]

def check_python_version(min_version=(3, 8)):
    if sys.version_info < min_version:
        raise EnvironmentError(
            f"Python {min_version[0]}.{min_version[1]} or higher is required"
        )


    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise EnvironmentError("pip is required")

def check_command(cmd):
    return shutil.which(cmd) is not None

def verify_dependencies():
    missing = []
    if not check_command("git"):
        missing.append("git")
    if platform.system() == "Windows":
        gpu_check = check_command("nvidia-smi")
    else:
        gpu_check = check_command("nvidia-smi")
    if not gpu_check:
        missing.append("GPU drivers")
    if missing:
        print("Warning: missing dependencies ->", ", ".join(missing))


def install_python_packages():
    for pkg in REQUIREMENTS:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def download_external_tools(target_dir: Path):
    # Placeholder for downloading Upscayl or ComfyUI
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pretend downloading tools to {target_dir}")


def test_installation():
    try:
        import cv2
        import skimage
        print("OpenCV version:", cv2.__version__)
        print("scikit-image version:", skimage.__version__)
    except Exception as exc:
        print("Installation test failed:", exc)


def create_default_config(config_path: Path):
    config = {
        "workspace": str(config_path),
        "upscale_factor": 4
    }
    import json
    with open(config_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def setup_environment():
    check_python_version()
    verify_dependencies()
    install_python_packages()
    tools_dir = Path("tools")
    download_external_tools(tools_dir)
    test_installation()
    create_default_config(Path("."))


if __name__ == "__main__":
    setup_environment()

