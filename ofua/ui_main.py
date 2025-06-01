import json
import shutil
import sys
import git
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import time
from collections import defaultdict
from PIL import Image

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QLineEdit,
    QFileDialog,
    QLabel,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QTabWidget,
    QComboBox,
    QSlider,
    QCheckBox,
)
from PyQt6.QtCore import QThreadPool, pyqtSlot, QTimer, Qt

from .config import (
    FALLOUT_CE_REPO_URL, DEFAULT_WORKSPACE_DIR, DEFAULT_REALESRGAN_EXE,
    DEFAULT_REALESRGAN_MODEL, DEFAULT_UPSCALE_FACTOR, CPU_WORKERS, DEFAULT_GPU_ID,
    setup_logging,
)
from .dat_tools import DatArchive, FRMConverter, ProcessingStats
from .hybrid_upscaler import HybridUpscaler
from .post_processor import IntelligentPostProcessor
from .quality_metrics import QualityMetrics
from .direct3d_integration import Direct3DIntegration
from .workers import TaskRunner, WorkerSignals
from .git_utils import GitProgress
from .validation import ValidationDialog
from .asset_utils import detect_asset_type_enhanced


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
        self.enable_3d_generation: bool = False
        self.output_mode: int = 2
        if DEFAULT_REALESRGAN_EXE:
            self.realesrgan_exe_path = Path(DEFAULT_REALESRGAN_EXE)
        if DEFAULT_REALESRGAN_MODEL:
            self.realesrgan_model_name = DEFAULT_REALESRGAN_MODEL
        self.thread_pool = QThreadPool()
        self.log_messages = []
        self.validation_event = threading.Event()
        self.validation_result: Optional[bool] = None
        self.engine_settings: Dict[str, Dict[str, object]] = {}
        self.engine_stats: Dict[str, Dict[str, int]] = {
            "realesrgan": {"success": 0, "fail": 0},
            "upscayl": {"success": 0, "fail": 0},
            "comfyui": {"success": 0, "fail": 0},
        }
        self._init_ui()
        self._load_settings()
        setup_logging(self.workspace_dir)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_general_tab(), "Général")
        self.tabs.addTab(self._create_engines_tab(), "Moteurs")
        self.tabs.addTab(self._create_3d_generation_tab(), "3D")
        self.tabs.addTab(self._create_monitoring_tab(), "Monitoring")
        layout.addWidget(self.tabs)

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_monitoring)
        self.monitor_timer.start(1000)

        self.log_message("Application prête.")

    def _create_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

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
        self.realesrgan_exe_edit = QLineEdit(
            str(self.realesrgan_exe_path) if self.realesrgan_exe_path else ""
        )
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

        buttons_layout = QHBoxLayout()
        save_btn = QPushButton("Sauver Profil")
        load_btn = QPushButton("Charger Profil")
        export_btn = QPushButton("Exporter JSON")
        import_btn = QPushButton("Importer JSON")
        reset_btn = QPushButton("Reset")
        save_btn.clicked.connect(self.save_profile)
        load_btn.clicked.connect(self.load_profile)
        export_btn.clicked.connect(self.export_settings)
        import_btn.clicked.connect(self.import_settings)
        reset_btn.clicked.connect(self.reset_defaults)
        for b in [save_btn, load_btn, export_btn, import_btn, reset_btn]:
            buttons_layout.addWidget(b)
        layout.addLayout(buttons_layout)

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

        return widget

    def _create_engines_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.engine_widgets = {}
        engines = ["realesrgan", "upscayl", "comfyui"]
        for asset_type in ["character", "interface", "items", "scenery"]:
            grp = QGroupBox(asset_type.capitalize())
            grp_lay = QHBoxLayout()
            prim = QComboBox()
            prim.addItems(engines)
            fall = QComboBox()
            fall.addItems(["none"] + engines)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(70)
            chk = QCheckBox("Avancé")
            grp_lay.addWidget(QLabel("Moteur"))
            grp_lay.addWidget(prim)
            grp_lay.addWidget(QLabel("Fallback"))
            grp_lay.addWidget(fall)
            grp_lay.addWidget(QLabel("Seuil"))
            grp_lay.addWidget(slider)
            grp_lay.addWidget(chk)
            grp.setLayout(grp_lay)
            layout.addWidget(grp)
            self.engine_widgets[asset_type] = {
                "primary": prim,
                "fallback": fall,
                "threshold": slider,
                "advanced": chk,
            }
        layout.addStretch()
        return widget

    def _create_3d_generation_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.enable_3d_gen = QCheckBox("Générer modèles 3D")
        self.enable_3d_gen.setChecked(self.enable_3d_generation)
        layout.addWidget(self.enable_3d_gen)

        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Mode de sortie:"))
        self.output_format = QComboBox()
        self.output_format.addItems(["2D seulement", "3D seulement", "2D + 3D"])
        self.output_format.setCurrentIndex(self.output_mode)
        fmt_layout.addWidget(self.output_format)
        layout.addLayout(fmt_layout)

        layout.addStretch()
        return widget

    def _create_monitoring_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.monitor_text = QTextEdit()
        self.monitor_text.setReadOnly(True)
        layout.addWidget(self.monitor_text)
        return widget

    def _update_monitoring(self):
        lines = []
        for engine, stats in self.engine_stats.items():
            line = f"{engine}: {stats['success']} succès / {stats['fail']} échecs"
            lines.append(line)
        self.monitor_text.setPlainText("\n".join(lines))

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

    def _collect_settings(self) -> Dict[str, object]:
        engines_cfg = {}
        for asset, widgets in getattr(self, "engine_widgets", {}).items():
            engines_cfg[asset] = {
                "primary": widgets["primary"].currentText(),
                "fallback": widgets["fallback"].currentText(),
                "threshold": widgets["threshold"].value(),
                "advanced": widgets["advanced"].isChecked(),
            }
        return {
            "workspace": str(self.workspace_dir),
            "realesrgan_exe": str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "",
            "realesrgan_model": self.realesrgan_model_name or "",
            "factor": self.upscale_factor,
            "engines": engines_cfg,
            "enable_3d": getattr(self, "enable_3d_gen", None).isChecked() if hasattr(self, "enable_3d_gen") else False,
            "output_mode": getattr(self, "output_format", None).currentIndex() if hasattr(self, "output_format") else 0,
        }

    def _apply_settings(self, data: Dict[str, object]):
        self.workspace_dir = Path(data.get("workspace", str(self.workspace_dir)))
        exe = data.get("realesrgan_exe") or ""
        self.realesrgan_exe_path = Path(exe) if exe else None
        self.realesrgan_model_name = data.get("realesrgan_model") or None
        self.upscale_factor = data.get("factor", self.upscale_factor)
        engines_cfg = data.get("engines", {})
        for asset, widgets in getattr(self, "engine_widgets", {}).items():
            cfg = engines_cfg.get(asset, {})
            widgets["primary"].setCurrentText(cfg.get("primary", "realesrgan"))
            widgets["fallback"].setCurrentText(cfg.get("fallback", "none"))
            widgets["threshold"].setValue(int(cfg.get("threshold", 70)))
            widgets["advanced"].setChecked(bool(cfg.get("advanced", False)))
        self.workspace_edit.setText(str(self.workspace_dir))
        self.realesrgan_exe_edit.setText(str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "")
        self.realesrgan_model_edit.setText(self.realesrgan_model_name or "")
        self.upscale_factor_edit.setText(self.upscale_factor)
        if hasattr(self, "enable_3d_gen"):
            val = bool(data.get("enable_3d", False))
            self.enable_3d_gen.setChecked(val)
            self.enable_3d_generation = val
        if hasattr(self, "output_format"):
            idx = int(data.get("output_mode", 0))
            self.output_format.setCurrentIndex(idx)
            self.output_mode = idx

    def _save_settings(self):
        self._settings_file().write_text(json.dumps(self._collect_settings()))

    def _load_settings(self):
        try:
            data = json.loads(self._settings_file().read_text())
            self._apply_settings(data)
        except Exception:
            pass

    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer profil", str(self.workspace_dir), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._collect_settings(), indent=2))

    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger profil", str(self.workspace_dir), "JSON (*.json)")
        if path:
            try:
                data = json.loads(Path(path).read_text())
                self._apply_settings(data)
            except Exception as exc:
                QMessageBox.warning(self, "Erreur", str(exc))

    def export_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter paramètres", str(self.workspace_dir), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._collect_settings(), indent=2))

    def import_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importer paramètres", str(self.workspace_dir), "JSON (*.json)")
        if path:
            try:
                data = json.loads(Path(path).read_text())
                self._apply_settings(data)
            except Exception as exc:
                QMessageBox.warning(self, "Erreur", str(exc))

    def reset_defaults(self):
        defaults = {
            "workspace": str(DEFAULT_WORKSPACE_DIR),
            "realesrgan_exe": "",
            "realesrgan_model": "",
            "factor": DEFAULT_UPSCALE_FACTOR,
            "engines": {
                t: {"primary": "realesrgan", "fallback": "none", "threshold": 70, "advanced": False}
                for t in ["character", "interface", "items", "scenery"]
            },
            "enable_3d": False,
            "output_mode": 2,
        }
        self._apply_settings(defaults)

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
        if not self._validate_config():
            return
        self._check_previous_session()
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
        self.engine_settings = {}
        for asset, widgets in getattr(self, "engine_widgets", {}).items():
            self.engine_settings[asset] = {
                "primary": widgets["primary"].currentText(),
                "fallback": widgets["fallback"].currentText(),
                "threshold": widgets["threshold"].value(),
                "advanced": widgets["advanced"].isChecked(),
            }
        if hasattr(self, "enable_3d_gen"):
            self.enable_3d_generation = self.enable_3d_gen.isChecked()
        else:
            self.enable_3d_generation = False
        if hasattr(self, "output_format"):
            self.output_mode = self.output_format.currentIndex()
        else:
            self.output_mode = 0
        self._save_settings()

    def _create_backup(self) -> bool:
        backup_dir = self.workspace_dir / "backup"
        if self.fallout_ce_cloned_path and self.fallout_ce_cloned_path.exists():
            try:
                shutil.copytree(self.fallout_ce_cloned_path, backup_dir / "original", dirs_exist_ok=True)
                return True
            except Exception as e:
                self.log_message(f"Échec sauvegarde: {e}")
        return False

    def _create_preview_comparison(self, orig_dir: Path, upscaled_dir: Path) -> Path | None:
        samples = list(orig_dir.rglob("*.png"))[:9]
        if not samples:
            return None
        grid_size = 3
        img_size = Image.open(samples[0]).size
        comparison = Image.new('RGB', (img_size[0] * grid_size * 2, img_size[1] * grid_size))
        for idx, orig_path in enumerate(samples):
            up_path = upscaled_dir / orig_path.relative_to(orig_dir)
            if not up_path.exists():
                continue
            orig_img = Image.open(orig_path)
            up_img = Image.open(up_path)
            x = (idx % grid_size) * img_size[0] * 2
            y = (idx // grid_size) * img_size[1]
            comparison.paste(orig_img, (x, y))
            comparison.paste(up_img, (x + img_size[0], y))
            orig_img.close()
            up_img.close()
        output = self.workspace_dir / "preview_comparison.png"
        comparison.save(output)
        comparison.close()
        return output

    def _check_previous_session(self) -> bool:
        state_file = self.workspace_dir / ".session_state.json"
        if state_file.exists():
            reply = QMessageBox.question(
                self,
                "Session précédente",
                "Reprendre la session précédente ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def validate_engines_config(self) -> List[str]:
        """Valide la configuration des moteurs d'upscale."""
        errors = []
        uses_realesrgan = any(
            settings.get("primary") == "realesrgan" or settings.get("fallback") == "realesrgan"
            for settings in self.engine_settings.values()
        )
        if uses_realesrgan:
            if not self.realesrgan_exe_path or not self.realesrgan_exe_path.exists():
                errors.append("Real-ESRGAN non trouvé")
        return errors

    def _validate_config(self) -> bool:
        """Validate user configuration before starting."""
        if not self.workspace_dir or not self.workspace_dir.exists():
            QMessageBox.warning(self, "Erreur", "Workspace invalide")
            return False
        errors = self.validate_engines_config()
        if errors:
            QMessageBox.warning(self, "Erreur", " ; ".join(errors))
            return False
        return True

    def _on_request_validation(self, orig_dir: str, upscaled_dir: str):
        origs = sorted(Path(orig_dir).rglob("*.png"))
        ups = sorted(Path(upscaled_dir).glob("*.png"))
        dialog = ValidationDialog(origs[:min(20, len(origs))], ups[:min(20, len(ups))])
        result = dialog.exec()
        self.validation_result = dialog.accepted and result == QDialog.Accepted
        self.validation_event.set()

    def _run_full_pipeline_task(self, signals: WorkerSignals):
        try:
            self._create_backup()
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
            stats = ProcessingStats(total_frms=len(all_frms), start_time=time.time())
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
            img = Image.new('P', (16, 16))
            img.putpalette(pal_list)
            img.save(self.pil_palette_image_for_quantize)
            signals.log.emit("Conversion FRM -> PNG...")
            conv = FRMConverter(signals)
            conv.load_palette(self.color_pal_path)
            conv.set_pil_palette_image_path(self.pil_palette_image_for_quantize)
            png_dir = self.workspace_dir / "png_orig"
            shutil.rmtree(png_dir, ignore_errors=True)
            png_dir.mkdir(parents=True, exist_ok=True)

            grouped: Dict[str, List[Path]] = {}
            orig_map: Dict[str, Path] = {}
            assets_by_type: Dict[str, List[Path]] = defaultdict(list)
            for frm in all_frms:
                a_type, _ = detect_asset_type_enhanced(frm)
                assets_by_type[a_type].append(frm)

            processed = 0
            for a_type, frm_list in assets_by_type.items():
                target_dir = png_dir / a_type
                target_dir.mkdir(parents=True, exist_ok=True)
                def conv_one(p: Path):
                    return p, conv.frm_to_png(p, target_dir)

                with ThreadPoolExecutor(max_workers=CPU_WORKERS) as ex:
                    for frm_p, pngs in ex.map(conv_one, frm_list):
                        base = frm_p.stem
                        grouped.setdefault(base, []).extend(pngs)
                        orig_map[base] = frm_p
                        processed += 1
                        stats.processed_frms = processed
                        signals.progress.emit(30 + int((processed / len(all_frms)) * 20))

            signals.log.emit("Upscaling avec HybridUpscaler...")
            upscaled_dir = self.workspace_dir / "png_upscaled"
            shutil.rmtree(upscaled_dir, ignore_errors=True)
            upscaled_dir.mkdir(parents=True, exist_ok=True)

            hybrid_upscaler = HybridUpscaler(self.workspace_dir, signals)
            post_processor = IntelligentPostProcessor()

            for a_type in assets_by_type.keys():
                src = png_dir / a_type
                dst = upscaled_dir / a_type
                dst.mkdir(parents=True, exist_ok=True)
                if not hybrid_upscaler.smart_upscale(src, dst, a_type):
                    self.engine_stats["realesrgan"]["fail"] += 1
                    signals.error.emit(f"Upscaling échoué ({a_type})")
                    continue
                self.engine_stats["realesrgan"]["success"] += 1
                for img_path in dst.glob("*.png"):
                    post_processor.enhance_by_type(img_path, a_type)
                orig_sample = next(src.glob("*.png"), None)
                if orig_sample:
                    up_sample = dst / orig_sample.name
                    if up_sample.exists():
                        score = QualityMetrics.calculate_composite_score(orig_sample, up_sample, a_type)
                        signals.log.emit(f"Qualité {a_type}: {score:.3f}")

            if self.enable_3d_generation:
                signals.log.emit("Génération des modèles 3D...")
                try:
                    d3d = Direct3DIntegration(self.workspace_dir)
                    for a_type in assets_by_type.keys():
                        dst = upscaled_dir / a_type
                        out_3d = self.workspace_dir / "models_3d" / a_type
                        out_3d.mkdir(parents=True, exist_ok=True)
                        for img in dst.glob("*.png"):
                            try:
                                mesh = d3d.generate_3d_from_sprite(img, a_type)
                                mesh_path = out_3d / (img.stem + ".obj")
                                with open(mesh_path, "wb") as f:
                                    f.write(mesh)
                            except Exception as exc:
                                signals.log.emit(f"3D échoué pour {img.name}: {exc}")
                except Exception as exc:
                    signals.log.emit(f"Pipeline 3D indisponible: {exc}")

            signals.progress.emit(70)
            self._create_preview_comparison(png_dir, upscaled_dir)
            signals.log.emit("Conversion PNG -> FRM...")
            out_dir = self.workspace_dir / "frm_final"
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            up_grouped: Dict[str, List[Path]] = {}
            for up in upscaled_dir.rglob("*.png"):
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

