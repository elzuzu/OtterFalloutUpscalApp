from pathlib import Path

CHARACTER_HINTS = [
    "art/critters",
    "art/heads",
    "critters",
    "heads",
    "npc",
]

def detect_asset_type(frm_path: Path) -> str:
    path_str = str(frm_path).lower()
    for hint in CHARACTER_HINTS:
        if hint in path_str:
            return "character"
    return "texture"
