import subprocess
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication
from .workers import WorkerSignals
from .downloader import download_real_esrgan, find_realesrgan_exe
from .config import DEFAULT_GPU_ID


class AIScaler:
    """Gère l'upscaling des images via Real-ESRGAN."""

    def __init__(self, realesrgan_exe: Path, model_name: str, scale_factor: str, gpu_id: int = DEFAULT_GPU_ID, signals: Optional[WorkerSignals] = None):
        self.realesrgan_exe = Path(realesrgan_exe)
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.gpu_id = gpu_id
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def upscale_directory(self, input_dir_png: Path, output_dir_upscaled_png: Path) -> bool:
        if not self.realesrgan_exe or not self.realesrgan_exe.exists():
            self._log("Exécutable Real-ESRGAN introuvable, téléchargement automatique...")
            project_root = Path(__file__).parent
            download_real_esrgan(project_root)
            found = find_realesrgan_exe(project_root)
            if not found:
                self._log(f"Erreur: Exécutable Real-ESRGAN non trouvé dans {project_root}.")
                if self._signals:
                    self._signals.error.emit("Real-ESRGAN non trouvé.")
                return False
            self.realesrgan_exe = found
        model_param_path = self.realesrgan_exe.parent / f"{self.model_name}.param"
        model_bin_path = self.realesrgan_exe.parent / f"{self.model_name}.bin"
        alt_param = self.realesrgan_exe.parent / "models" / f"{self.model_name}.param"
        alt_bin = self.realesrgan_exe.parent / "models" / f"{self.model_name}.bin"
        if not ((model_param_path.exists() and model_bin_path.exists()) or (alt_param.exists() and alt_bin.exists())):
            self._log("Modèle Real-ESRGAN introuvable, téléchargement automatique...")
            download_real_esrgan(self.realesrgan_exe.parent)
            if not ((model_param_path.exists() and model_bin_path.exists()) or (alt_param.exists() and alt_bin.exists())):
                self._log(f"Erreur: Fichiers modèle Real-ESRGAN pour '{self.model_name}' non trouvés.")
                if self._signals:
                    self._signals.error.emit(f"Modèle '{self.model_name}' non trouvé.")
                return False
        output_dir_upscaled_png.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.realesrgan_exe),
            "-i", str(input_dir_png),
            "-o", str(output_dir_upscaled_png),
            "-n", self.model_name,
            "-s", self.scale_factor,
            "-f", "png",
            "-g", str(self.gpu_id)
        ]
        self._log("Lancement de Real-ESRGAN: " + " ".join(command))
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    self._log(f"[Real-ESRGAN] {line.strip()}")
                    if self._signals:
                        QApplication.processEvents()
            process.wait()
            if process.returncode != 0:
                self._log(f"Erreur Real-ESRGAN (code {process.returncode}).")
                if self._signals:
                    self._signals.error.emit(f"Real-ESRGAN a échoué (code {process.returncode}).")
                return False
            self._log("Upscaling Real-ESRGAN terminé.")
            return True
        except FileNotFoundError:
            self._log(f"Erreur: Exécutable Real-ESRGAN '{self.realesrgan_exe}' non trouvé.")
            if self._signals:
                self._signals.error.emit("Real-ESRGAN non trouvé ou non exécutable.")
            return False
        except Exception as e:
            self._log(f"Erreur lors de l'exécution de Real-ESRGAN: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur exécution Real-ESRGAN: {e}")
            return False
