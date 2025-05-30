from pathlib import Path
import re
from typing import Tuple, Dict

from PIL import Image
import numpy as np

ASSET_TYPE_PATTERNS: Dict[str, list[str]] = {
    "critter": [
        r"^.+/art/critters/[^/]+\.frm$",
        r".*player.*\.frm$",
    ],
    "head": [
        r"^.+/art/heads/[^/]+\.frm$",
    ],
    "interface": [
        r"^.+/art/intrface/[^/]+\.frm$",
        r"^.+/art/misc/[^/]+\.frm$",
    ],
    "weapon": [
        r"^.+/art/items/.*weapon[^/]*\.frm$",
    ],
    "armor": [
        r"^.+/art/items/.*armor[^/]*\.frm$",
    ],
    "item": [
        r"^.+/art/items/[^/]+\.frm$",
        r"^.+/art/inven/[^/]+\.frm$",
    ],
    "scenery": [
        r"^.+/art/scenery/[^/]+\.frm$",
        r"^.+/art/walls/[^/]+\.frm$",
    ],
}

_TYPE_GROUPS = {
    "critter": "character",
    "head": "character",
    "weapon": "texture",
    "armor": "texture",
    "item": "texture",
    "interface": "texture",
    "scenery": "texture",
}


def detect_asset_type(frm_path: Path) -> str:
    label, _ = detect_asset_type_enhanced(frm_path, image_analysis=False)
    return _TYPE_GROUPS.get(label, "unknown")

def detect_asset_type_simple(frm_path: Path) -> str:
    """Backward compatible simple detection."""
    for pattern in ASSET_TYPE_PATTERNS["character"]:
        if re.match(pattern, str(frm_path).lower()):
            return "character"
    return "texture"


def _extract_image_features(image_path: Path) -> np.ndarray:
    """Extract simple numerical features from an image."""
    img = Image.open(image_path)
    img = img.convert("RGBA")
    arr = np.array(img)
    h, w = img.height, img.width
    aspect_ratio = w / h if h else 1.0
    flat = arr.reshape(-1, 4)
    colors = flat[:, :3]
    unique_colors = len(np.unique(colors, axis=0))
    color_std = float(np.std(colors))
    img.close()
    return np.array([aspect_ratio, unique_colors, color_std], dtype=float)


_FEATURE_PROTOTYPES: Dict[str, np.ndarray] = {
    "weapon": np.array([2.5, 40, 60], dtype=float),
    "armor": np.array([1.2, 50, 80], dtype=float),
    "critter": np.array([1.0, 60, 90], dtype=float),
    "head": np.array([1.0, 40, 50], dtype=float),
    "interface": np.array([1.0, 20, 30], dtype=float),
    "item": np.array([1.0, 35, 40], dtype=float),
    "scenery": np.array([1.3, 80, 110], dtype=float),
}


def _classify_features(features: np.ndarray) -> Tuple[str, float]:
    """Return label and confidence based on nearest prototype."""
    best_label = "unknown"
    best_dist = float("inf")
    for label, proto in _FEATURE_PROTOTYPES.items():
        dist = float(np.linalg.norm(features - proto))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    confidence = max(0.0, 1.0 - best_dist / 200.0)
    return best_label, confidence


def detect_asset_type_enhanced(frm_path: Path, image_analysis: bool = True) -> Tuple[str, float]:
    """Enhanced asset detection with regex patterns and optional image analysis."""
    path_str = str(frm_path).replace("\\", "/").lower()
    for asset_type, patterns in ASSET_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, path_str):
                return asset_type, 0.95

    if image_analysis:
        try:
            features = _extract_image_features(frm_path)
            return _classify_features(features)
        except Exception:
            pass

    return "unknown", 0.0
