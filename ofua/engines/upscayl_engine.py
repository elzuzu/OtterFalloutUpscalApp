"""Upscayl engine integration for batch upscaling."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from ..workers import WorkerSignals
from ..downloader import download_with_retry


UPSCAYL_DOWNLOAD_URLS = {
    "Windows": "https://github.com/upscayl/upscayl/releases/latest/download/Upscayl-Setup.exe",
    "Darwin": "https://github.com/upscayl/upscayl/releases/latest/download/Upscayl.dmg",
    "Linux": "https://github.com/upscayl/upscayl/releases/latest/download/Upscayl.AppImage",
}


class UpscaylEngine:
    """Interface with Upscayl desktop application."""

    SUPPORTED_MODELS = {"realesrgan-x4plus", "anime", "cunet"}

    def __init__(self, upscayl_path: Optional[Path] = None, signals: Optional[WorkerSignals] = None):
        self._signals = signals
        self.upscayl_path = upscayl_path or self._detect_installation()

    # Logging helper
    def _log(self, message: str) -> None:
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def _detect_installation(self) -> Optional[Path]:
        """Try to locate Upscayl executable based on the OS."""
        system = platform.system()
        candidates = []
        if system == "Windows":
            candidates.extend([
                Path(os.getenv("ProgramFiles", "")) / "Upscayl" / "Upscayl.exe",
                Path(os.getenv("ProgramFiles(x86)", "")) / "Upscayl" / "Upscayl.exe",
                Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Upscayl" / "Upscayl.exe",
            ])
        elif system == "Darwin":
            candidates.append(Path("/Applications/Upscayl.app/Contents/MacOS/Upscayl"))
        else:  # Linux and others
            candidates.extend([
                Path("/usr/bin/upscayl"),
                Path("/app/bin/upscayl"),
                Path.home() / ".local" / "bin" / "upscayl",
            ])
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _download_upscayl(self, dest_dir: Path) -> Optional[Path]:
        """Download Upscayl to the given directory."""
        system = platform.system()
        url = UPSCAYL_DOWNLOAD_URLS.get(system)
        if not url:
            self._log(f"OS non supporté pour Upscayl: {system}")
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_name = url.split("/")[-1]
        dest_path = dest_dir / file_name
        self._log(f"Téléchargement d'Upscayl depuis {url}...")
        if download_with_retry(url, dest_path):
            dest_path.chmod(dest_path.stat().st_mode | 0o755)
            return dest_path
        self._log("Échec du téléchargement d'Upscayl")
        return None

    def _ensure_available(self) -> bool:
        if self.upscayl_path and self.upscayl_path.exists():
            return True
        self._log("Exécutable Upscayl introuvable, téléchargement automatique...")
        download_path = self._download_upscayl(Path.cwd() / "upscayl")
        if download_path and download_path.exists():
            self.upscayl_path = download_path
            return True
        if self._signals:
            self._signals.error.emit("Upscayl non trouvé.")
        return False

    def upscale_batch(self, input_dir: Path, output_dir: Path, model: str = "realesrgan-x4plus", scale: int = 2) -> bool:
        """Upscale a directory of images using Upscayl CLI."""
        if model not in self.SUPPORTED_MODELS:
            self._log(f"Modèle {model} non supporté")
            if self._signals:
                self._signals.error.emit(f"Modèle {model} non supporté")
            return False
        if not self._ensure_available():
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.upscayl_path),
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--model", model,
            "--scale", str(scale),
            "--no-gui",
        ]
        self._log("Lancement Upscayl: " + " ".join(command))
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    self._log(f"[Upscayl] {line.strip()}")
                    if self._signals:
                        from PyQt6.QtWidgets import QApplication
                        QApplication.processEvents()
            process.wait()
            if process.returncode != 0:
                self._log(f"Upscayl a échoué avec le code {process.returncode}")
                if self._signals:
                    self._signals.error.emit(f"Upscayl a échoué ({process.returncode})")
                return False
            self._log("Upscaling Upscayl terminé")
            return True
        except FileNotFoundError:
            self._log("Exécutable Upscayl non trouvé")
            if self._signals:
                self._signals.error.emit("Upscayl non trouvé")
            return False
        except Exception as exc:
            self._log(f"Erreur lors de l'exécution d'Upscayl: {exc}")
            if self._signals:
                self._signals.error.emit(str(exc))
            return False

