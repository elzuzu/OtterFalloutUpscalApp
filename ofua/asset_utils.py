from pathlib import Path
import re

ASSET_TYPE_PATTERNS = {
    "character": [
        r"art/critters/.*\.frm$",
        r"art/heads/.*\.frm$",
        r".*player.*\.frm$",
    ],
    "interface": [
        r"art/intrface/.*\.frm$",
        r"art/misc/.*\.frm$",
    ],
    "items": [
        r"art/items/.*\.frm$",
        r"art/inven/.*\.frm$",
    ],
    "scenery": [
        r"art/scenery/.*\.frm$",
        r"art/walls/.*\.frm$",
    ],
}

def detect_asset_type(frm_path: Path) -> str:
    path_str = str(frm_path).lower()
    for asset_type, patterns in ASSET_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, path_str):
                return asset_type
    return "unknown"

def detect_asset_type_simple(frm_path: Path) -> str:
    """Backward compatible simple detection."""
    for pattern in ASSET_TYPE_PATTERNS["character"]:
        if re.match(pattern, str(frm_path).lower()):
            return "character"
    return "texture"
