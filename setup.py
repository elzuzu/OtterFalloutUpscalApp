"""Dynamic environment setup for the Fallout Upscaler application.

This script installs required Python packages listed in requirements.txt,
checks for external dependencies like git, and optionally clones auxiliary
repositories (e.g. Real-ESRGAN) if they are missing. Run it once after
cloning the project to prepare a working environment.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


REQUIREMENTS_FILE = Path("requirements.txt")
TOOLS_DIR = Path("tools")
REALESRGAN_REPO = "https://github.com/xinntao/Real-ESRGAN.git"
REALESRGAN_DIR = TOOLS_DIR / "Real-ESRGAN"


def package_installed(pkg: str) -> bool:
    """Return True if the given package appears importable."""
    module_name = pkg.split("==")[0].replace("-", "_")
    return importlib.util.find_spec(module_name) is not None


def ensure_packages() -> None:
    """Install packages from requirements.txt if missing."""
    if not REQUIREMENTS_FILE.exists():
        print(f"{REQUIREMENTS_FILE} missing")
        return
    with open(REQUIREMENTS_FILE, "r", encoding="utf-8") as fh:
        packages = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

    to_install = [pkg for pkg in packages if not package_installed(pkg)]
    if to_install:
        print("Installing Python packages:", ", ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
    else:
        print("All required packages are already installed.")


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_git() -> None:
    if not have_cmd("git"):
        raise EnvironmentError("git command not found. Please install git and re-run this script.")


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Repository already present: {dest}")
        return
    print(f"Cloning {url} into {dest}…")
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def setup_external_tools() -> None:
    TOOLS_DIR.mkdir(exist_ok=True)
    clone_repo(REALESRGAN_REPO, REALESRGAN_DIR)


def create_default_config(cfg_path: Path) -> None:
    if cfg_path.exists():
        return
    cfg = {
        "workspace": str(Path("fallout_upscaler_workspace").resolve()),
        "upscale_factor": 4,
        "realesrgan_dir": str(REALESRGAN_DIR.resolve()),
    }
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Default config created at {cfg_path}")


def main() -> None:
    ensure_git()
    ensure_packages()
    setup_external_tools()
    create_default_config(Path("config.json"))
    print("Setup complete. You can now run the application with `python main.py`.\n")


if __name__ == "__main__":
    main()

