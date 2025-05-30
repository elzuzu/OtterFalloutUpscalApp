import json
import shutil
import sys
import git
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QProgressBar, QLineEdit, QFileDialog, QLabel, QMessageBox, QGroupBox, QSplitter
)
from PyQt6.QtCore import QThreadPool, pyqtSlot

from .config import (
    FALLOUT_CE_REPO_URL, DEFAULT_WORKSPACE_DIR, DEFAULT_REALESRGAN_EXE,
    DEFAULT_REALESRGAN_MODEL, DEFAULT_UPSCALE_FACTOR, CPU_WORKERS, DEFAULT_GPU_ID
)
from .dat_tools import DatArchive, FRMConverter
from .ai_scaler import AIScaler
from .workers import TaskRunner, WorkerSignals
from .git_utils import GitProgress
from .validation import ValidationDialog
from .asset_utils import detect_asset_type


class FalloutUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fallout 1 CE Upscaler")
        self.resize(900, 700)
        self.workspace_dir = DEFAULT_WORKSPACE_DIR
        self.realesrgan_exe_path: Optional[Path] = None
        self.realesrgan_model_name: Optional[str] = None
        self.upscale_factor: str = DEFAULT_UPSCALE_FACTOR
        self.fallout_ce_cloned_path: Optional[Path] = None
        self.color_pal_path: Optional[Path] = None
        self.pil_palette_image_for_quantize: Optional[Path] = None
        if DEFAULT_REALESRGAN_EXE:
            self.realesrgan_exe_path = Path(DEFAULT_REALESRGAN_EXE)
        if DEFAULT_REALESRGAN_MODEL:
            self.realesrgan_model_name = DEFAULT_REALESRGAN_MODEL
        self.thread_pool = QThreadPool()
        self.log_messages = []
        self.validation_event = threading.Event()
        self.validation_result: Optional[bool] = None
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()

        ws_layout = QHBoxLayout()
        ws_layout.addWidget(QLabel("Workspace:"))
        self.workspace_edit = QLineEdit(str(self.workspace_dir))
        btn = QPushButton("Choisir")
        btn.clicked.connect(self._select_workspace_dir)
        ws_layout.addWidget(self.workspace_edit)
        ws_layout.addWidget(btn)
        config_layout.addLayout(ws_layout)

        exe_layout = QHBoxLayout()
        exe_layout.addWidget(QLabel("Real-ESRGAN:"))
        self.realesrgan_exe_edit = QLineEdit(str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "")
        btn_exe = QPushButton("Choisir")
        btn_exe.clicked.connect(self._select_realesrgan_exe)
        exe_layout.addWidget(self.realesrgan_exe_edit)
        exe_layout.addWidget(btn_exe)
        config_layout.addLayout(exe_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Modèle:"))
        self.realesrgan_model_edit = QLineEdit(self.realesrgan_model_name or "")
        model_layout.addWidget(self.realesrgan_model_edit)
        config_layout.addLayout(model_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Facteur:"))
        self.upscale_factor_edit = QLineEdit(self.upscale_factor)
        scale_layout.addWidget(self.upscale_factor_edit)
        config_layout.addLayout(scale_layout)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.start_button = QPushButton("Démarrer")
        self.start_button.clicked.connect(self._start_full_process)
        layout.addWidget(self.start_button)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        left_layout.addWidget(self.log_text_edit)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)
        splitter.addWidget(right)

        layout.addWidget(splitter)

        self.log_message("Application prête.")

    def _select_workspace_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Workspace", str(self.workspace_dir))
        if directory:
            self.workspace_dir = Path(directory)
            self.workspace_edit.setText(str(self.workspace_dir))
            self._save_settings()

    def _select_realesrgan_exe(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Executable Real-ESRGAN", "", "Executables (*.exe)")
        if filepath:
            self.realesrgan_exe_path = Path(filepath)
            self.realesrgan_exe_edit.setText(filepath)
            self._save_settings()

    def _settings_file(self) -> Path:
        return Path.home() / ".ofua_settings.json"

    def _save_settings(self):
        settings = {
            "workspace": str(self.workspace_dir),
            "realesrgan_exe": str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "",
            "realesrgan_model": self.realesrgan_model_name or "",
            "factor": self.upscale_factor,
        }
        self._settings_file().write_text(json.dumps(settings))

    def _load_settings(self):
        try:
            data = json.loads(self._settings_file().read_text())
            self.workspace_dir = Path(data.get("workspace", str(self.workspace_dir)))
            exe = data.get("realesrgan_exe")
            self.realesrgan_exe_path = Path(exe) if exe else None
            self.realesrgan_model_name = data.get("realesrgan_model") or None
            self.upscale_factor = data.get("factor", self.upscale_factor)
            self.workspace_edit.setText(str(self.workspace_dir))
            self.realesrgan_exe_edit.setText(str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "")
            self.realesrgan_model_edit.setText(self.realesrgan_model_name or "")
            self.upscale_factor_edit.setText(self.upscale_factor)
        except Exception:
            pass

    @pyqtSlot(str)
    def log_message(self, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{now}] {message}"
        self.log_messages.append(full_message)
        self.log_text_edit.append(full_message)
        self.log_text_edit.ensureCursorVisible()

    @pyqtSlot(int)
    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    @pyqtSlot(str)
    def handle_error(self, msg: str):
        QMessageBox.critical(self, "Erreur", msg)
        self.start_button.setEnabled(True)

    @pyqtSlot()
    def on_pipeline_finished(self):
        self.log_message("Processus terminé")
        self.start_button.setEnabled(True)

    def _start_full_process(self):
        self._update_config_from_ui()
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        worker = TaskRunner(self._run_full_pipeline_task)
        worker.signals.log.connect(self.log_message)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(self.on_pipeline_finished)
        worker.signals.request_validation.connect(self._on_request_validation)
        self.thread_pool.start(worker)

    def _update_config_from_ui(self):
        self.workspace_dir = Path(self.workspace_edit.text())
        self.realesrgan_exe_path = Path(self.realesrgan_exe_edit.text()) if self.realesrgan_exe_edit.text() else None
        self.realesrgan_model_name = self.realesrgan_model_edit.text() or None
        self.upscale_factor = self.upscale_factor_edit.text()
        self._save_settings()

    def _on_request_validation(self, orig_dir: str, upscaled_dir: str):
        origs = sorted(Path(orig_dir).rglob("*.png"))
        ups = sorted(Path(upscaled_dir).glob("*.png"))
        dialog = ValidationDialog(origs[:min(20, len(origs))], ups[:min(20, len(ups))])
        result = dialog.exec()
        self.validation_result = dialog.accepted and result == QDialog.Accepted
        self.validation_event.set()

    def _run_full_pipeline_task(self, signals: WorkerSignals):
        try:
            signals.log.emit("Clonage/Mise à jour du dépôt...")
            signals.progress.emit(5)
            self.fallout_ce_cloned_path = self.workspace_dir / "fallout1-ce"
            try:
                if self.fallout_ce_cloned_path.exists():
                    repo = git.Repo(self.fallout_ce_cloned_path)
                    repo.remotes.origin.pull()
                else:
                    git.Repo.clone_from(FALLOUT_CE_REPO_URL, self.fallout_ce_cloned_path, progress=GitProgress(signals))
                signals.log.emit("Dépôt prêt")
            except Exception as e:
                signals.error.emit(str(e))
                return
            signals.progress.emit(10)
            master_dat_path = self.fallout_ce_cloned_path / "master.dat"
            critter_dat_path = self.fallout_ce_cloned_path / "critter.dat"
            signals.log.emit("Extraction des DAT...")
            extracted_assets_dir = self.workspace_dir / "extracted"
            shutil.rmtree(extracted_assets_dir, ignore_errors=True)
            extracted_assets_dir.mkdir(parents=True, exist_ok=True)
            all_frms: List[Path] = []
            master = DatArchive(master_dat_path, signals)
            if not master.load_entries():
                return
            frms_master, pal_master = master.extract_all_frms_and_pal(extracted_assets_dir / "master")
            all_frms.extend(frms_master)
            if pal_master:
                self.color_pal_path = pal_master
            if critter_dat_path.exists():
                crit = DatArchive(critter_dat_path, signals)
                if crit.load_entries():
                    frms_crit, pal_crit = crit.extract_all_frms_and_pal(extracted_assets_dir / "critter")
                    all_frms.extend(frms_crit)
                    if pal_crit and not self.color_pal_path:
                        self.color_pal_path = pal_crit
            if not self.color_pal_path:
                signals.error.emit("COLOR.PAL manquant")
                return
            self.pil_palette_image_for_quantize = self.workspace_dir / "temp_palette.png"
            pal_bytes = self.color_pal_path.read_bytes()
            pal_list = []
            for i in range(0, 768, 3):
                r, g, b = pal_bytes[i:i+3]
                pal_list.extend([r * 4, g * 4, b * 4])
            while len(pal_list) < 768:
                pal_list.append(0)
            from PIL import Image
            img = Image.new('P', (16, 16))
            img.putpalette(pal_list)
            img.save(self.pil_palette_image_for_quantize)
            signals.log.emit("Conversion FRM -> PNG...")
            conv = FRMConverter(signals)
            conv.load_palette(self.color_pal_path)
            conv.set_pil_palette_image_path(self.pil_palette_image_for_quantize)
            png_dir = self.workspace_dir / "png_orig"
            png_dir_chars = png_dir / "characters"
            png_dir_textures = png_dir / "textures"
            shutil.rmtree(png_dir, ignore_errors=True)
            png_dir_chars.mkdir(parents=True, exist_ok=True)
            png_dir_textures.mkdir(parents=True, exist_ok=True)
            grouped: Dict[str, List[Path]] = {}
            orig_map: Dict[str, Path] = {}
            def conv_one(p: Path):
                asset_type = detect_asset_type(p)
                target_dir = png_dir_chars if asset_type == "character" else png_dir_textures
                return p, conv.frm_to_png(p, target_dir)
            with ThreadPoolExecutor(max_workers=CPU_WORKERS) as ex:
                for i, (frm_p, pngs) in enumerate(ex.map(conv_one, all_frms), 1):
                    base = frm_p.stem
                    grouped.setdefault(base, []).extend(pngs)
                    orig_map[base] = frm_p
                    signals.progress.emit(30 + int((i / len(all_frms)) * 20))
            signals.log.emit("Upscaling...")
            upscaled_dir = self.workspace_dir / "png_upscaled"
            shutil.rmtree(upscaled_dir, ignore_errors=True)
            upscaled_dir.mkdir(parents=True, exist_ok=True)
            scaler_chars = AIScaler(self.realesrgan_exe_path, "realesrgan-x4plus-anime", self.upscale_factor, DEFAULT_GPU_ID, signals)
            scaler_textures = AIScaler(self.realesrgan_exe_path, self.realesrgan_model_name or 'realesrgan-x4plus', self.upscale_factor, DEFAULT_GPU_ID, signals)
            if not scaler_chars.upscale_directory(png_dir_chars, upscaled_dir):
                signals.error.emit("Upscaling échoué (characters)")
                return
            if not scaler_textures.upscale_directory(png_dir_textures, upscaled_dir):
                signals.error.emit("Upscaling échoué (textures)")
                return
            signals.progress.emit(70)
            signals.request_validation.emit(str(png_dir), str(upscaled_dir))
            self.validation_event.clear()
            self.validation_event.wait()
            if not self.validation_result:
                shutil.rmtree(upscaled_dir, ignore_errors=True)
                signals.log.emit("Upscaling rejeté par l'utilisateur")
                return
            signals.log.emit("Conversion PNG -> FRM...")
            out_dir = self.workspace_dir / "frm_final"
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            up_grouped: Dict[str, List[Path]] = {}
            for up in upscaled_dir.glob("*.png"):
                base = up.stem.split('_d')[0]
                up_grouped.setdefault(base, []).append(up)
            def rebuild(item):
                base, orig = item
                if base in up_grouped:
                    rel = orig.relative_to(extracted_assets_dir)
                    sub = out_dir / rel.parent
                    sub.mkdir(parents=True, exist_ok=True)
                    conv.png_to_frm(up_grouped, sub, orig)
            with ThreadPoolExecutor(max_workers=CPU_WORKERS) as ex:
                for _ in ex.map(rebuild, orig_map.items()):
                    pass
            signals.progress.emit(100)
            signals.log.emit("Terminé. FRM dans " + str(out_dir))
        except Exception as e:
            signals.error.emit(str(e))
        finally:
            if self.pil_palette_image_for_quantize and self.pil_palette_image_for_quantize.exists():
                try:
                    self.pil_palette_image_for_quantize.unlink()
                except Exception:
                    pass

