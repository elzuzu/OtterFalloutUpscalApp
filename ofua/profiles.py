from __future__ import annotations

import json
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROFILE_DIR = Path.home() / ".ofua_profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "speed": {
        "description": "Privil\u00e9gie la vitesse (Upscayl seul, seuils bas)",
        "validation": {"strict": False},
        "engines": {
            t: {"primary": "upscayl", "fallback": "none", "threshold": 30, "advanced": False}
            for t in ["character", "interface", "items", "scenery"]
        },
    },
    "quality": {
        "description": "Privil\u00e9gie la qualit\u00e9 (ComfyUI + validation stricte)",
        "validation": {"strict": True},
        "engines": {
            t: {"primary": "comfyui", "fallback": "realesrgan", "threshold": 90, "advanced": True}
            for t in ["character", "interface", "items", "scenery"]
        },
    },
    "balanced": {
        "description": "\u00c9quilibre vitesse/qualit\u00e9 (hybride)",
        "validation": {"strict": False},
        "engines": {
            "character": {"primary": "comfyui", "fallback": "realesrgan", "threshold": 80, "advanced": True},
            "interface": {"primary": "upscayl", "fallback": "realesrgan", "threshold": 70, "advanced": False},
            "items": {"primary": "upscayl", "fallback": "realesrgan", "threshold": 70, "advanced": False},
            "scenery": {"primary": "comfyui", "fallback": "realesrgan", "threshold": 80, "advanced": False},
        },
    },
    "fallout_optimized": {
        "description": "Profil optimis\u00e9 pour Fallout 1",
        "validation": {"strict": True},
        "engines": {
            "character": {"primary": "comfyui", "fallback": "realesrgan", "threshold": 85, "advanced": True},
            "interface": {"primary": "realesrgan", "fallback": "none", "threshold": 80, "advanced": False},
            "items": {"primary": "upscayl", "fallback": "realesrgan", "threshold": 75, "advanced": False},
            "scenery": {"primary": "comfyui", "fallback": "realesrgan", "threshold": 85, "advanced": False},
        },
    },
}


class ProfileValidationError(Exception):
    pass


@dataclass
class Profile:
    name: str
    data: Dict[str, Any]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(json.dumps(self.data, indent=2))

    @staticmethod
    def load(path: Path) -> "Profile":
        data = json.loads(path.read_text())
        name = path.stem
        return Profile(name, data)


class ProfileManager:
    def __init__(self, user: Optional[str] = None):
        self.user = user or getpass.getuser()
        self.user_dir = PROFILE_DIR / self.user
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, profile_name: str) -> Path:
        return self.user_dir / f"{profile_name}.json"

    def _validate(self, data: Dict[str, Any]) -> None:
        engines = data.get("engines", {})
        if not isinstance(engines, dict):
            raise ProfileValidationError("'engines' doit \u00eatre un dictionnaire")
        for asset, cfg in engines.items():
            if not isinstance(cfg, dict):
                raise ProfileValidationError(f"configuration invalide pour {asset}")
            primary = cfg.get("primary")
            fallback = cfg.get("fallback")
            threshold = cfg.get("threshold")
            if primary not in {"realesrgan", "upscayl", "comfyui", "none"}:
                raise ProfileValidationError(f"Moteur primaire inconnu: {primary}")
            if fallback not in {"realesrgan", "upscayl", "comfyui", "none"}:
                raise ProfileValidationError(f"Moteur fallback inconnu: {fallback}")
            if not isinstance(threshold, int) or not 0 <= threshold <= 100:
                raise ProfileValidationError("'threshold' doit \u00eatre entre 0 et 100")

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Charger un profil depuis le disque ou les pr\u00e9d\u00e9finis."""
        path = self._profile_path(profile_name)
        if path.exists():
            data = json.loads(path.read_text())
        elif profile_name in DEFAULT_PROFILES:
            data = DEFAULT_PROFILES[profile_name]
        else:
            raise FileNotFoundError(f"Profil '{profile_name}' introuvable")
        self._validate(data)
        return data

    def save_profile(self, profile_name: str, data: Dict[str, Any]) -> None:
        """Valider et sauvegarder un profil pour l'utilisateur."""
        self._validate(data)
        Profile(profile_name, data).save(self.user_dir)

    def create_custom_profile(self, base_profile: str, overrides: Dict[str, Any]) -> str:
        """Cr\u00e9er un profil personnalis\u00e9 bas\u00e9 sur un autre."""
        base_data = self.load_profile(base_profile)
        new_data = json.loads(json.dumps(base_data))
        self._apply_overrides(new_data, overrides)
        self._validate(new_data)
        new_name = overrides.get("name") or f"{base_profile}_custom"
        i = 1
        final_name = new_name
        while self._profile_path(final_name).exists():
            i += 1
            final_name = f"{new_name}_{i}"
        self.save_profile(final_name, new_data)
        return final_name

    def _apply_overrides(self, data: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                self._apply_overrides(data[key], value)
            else:
                data[key] = value

    def list_profiles(self) -> Dict[str, Path]:
        """Retourne la liste des profils disponibles pour l'utilisateur."""
        profiles = {p.stem: p for p in self.user_dir.glob("*.json")}
        profiles.update({name: None for name in DEFAULT_PROFILES.keys()})
        return profiles

    def apply_cli_overrides(self, data: Dict[str, Any], cli_args: Dict[str, str]) -> None:
        """Applique des overrides venant de la ligne de commande."""
        overrides = {}
        for key, val in cli_args.items():
            parts = key.split('.')
            cur = overrides
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = self._parse_value(val)
        self._apply_overrides(data, overrides)
        self._validate(data)

    def _parse_value(self, val: str) -> Any:
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        try:
            return int(val)
        except ValueError:
            return val
