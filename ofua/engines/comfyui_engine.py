"""Engine ComfyUI for batch upscaling using predefined workflows."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from ..workers import WorkerSignals


class ComfyUIEngine:
    """Interact with a local ComfyUI instance using workflow JSON files."""

    def __init__(
        self,
        comfyui_path: Optional[Path] = None,
        host: str = "http://127.0.0.1:8188",
        signals: Optional[WorkerSignals] = None,
    ) -> None:
        self._signals = signals
        self.host = host.rstrip("/")
        self.comfyui_path = comfyui_path or self._detect_installation()
        self.workflows_dir = Path(__file__).parent / "workflows"

    def _log(self, message: str) -> None:
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def _detect_installation(self) -> Optional[Path]:
        """Try to locate a ComfyUI installation."""
        env_path = os.getenv("COMFYUI_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
        candidates = [
            Path.home() / "ComfyUI",
            Path.home() / "comfyui",
            Path("/opt/ComfyUI"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.host}{endpoint}"
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()

    def _get_json(self, endpoint: str) -> dict:
        url = f"{self.host}{endpoint}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    def _load_workflow(self, workflow_name: str, image: Path, output: Path) -> dict:
        workflow_path = self.workflows_dir / f"{workflow_name}.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_name} introuvable")
        text = workflow_path.read_text(encoding="utf-8")
        text = text.replace("__INPUT_IMAGE__", str(image))
        text = text.replace("__OUTPUT_PATH__", str(output))
        return json.loads(text)

    def execute_workflow(self, image_path: Path, workflow_name: str, output_dir: Path) -> bool:
        """Execute a workflow file via ComfyUI API."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / image_path.name
        try:
            payload = self._load_workflow(workflow_name, image_path, output_path)
        except Exception as exc:
            self._log(f"Erreur chargement workflow: {exc}")
            if self._signals:
                self._signals.error.emit(str(exc))
            return False
        try:
            self._log("Envoi du workflow à ComfyUI...")
            resp = self._post_json("/prompt", payload)
            prompt_id = resp.get("prompt_id")
            if not prompt_id:
                raise RuntimeError("prompt_id manquant")
            self._log(f"ID de la tâche: {prompt_id}")
            while True:
                status = self._get_json(f"/queue/{prompt_id}")
                if status.get("status") == "finished":
                    break
                time.sleep(1)
            self._log("Workflow terminé")
            return True
        except Exception as exc:
            self._log(f"Erreur ComfyUI: {exc}")
            if self._signals:
                self._signals.error.emit(str(exc))
            return False

    def upscale(self, image_path: Path, output_dir: Path, asset_type: str) -> bool:
        workflow_map = {
            "character": "character_upscale",
            "interface": "interface_upscale",
            "items": "items_upscale",
            "scenery": "scenery_upscale",
        }
        wf = workflow_map.get(asset_type)
        if not wf:
            self._log(f"Type d'asset inconnu: {asset_type}")
            return False
        return self.execute_workflow(image_path, wf, output_dir)
