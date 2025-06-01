from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from direct3d_s2.pipeline import Direct3DS2Pipeline
except Exception:  # pragma: no cover - library may not be installed
    Direct3DS2Pipeline = None


@dataclass
class Direct3DIntegration:
    """Wrapper around the Direct3D-S2 pipeline."""

    workspace_dir: Path
    device: str = "cuda:0"
    _pipeline: Optional[object] = None

    def __post_init__(self) -> None:
        if Direct3DS2Pipeline is None:
            raise ImportError("direct3d_s2 is not available")
        self._pipeline = Direct3DS2Pipeline.from_pretrained(
            "wushuang98/Direct3D-S2", subfolder="direct3d-s2-v-1-1"
        ).to(self.device)

    def generate_3d_from_sprite(self, sprite_path: Path, asset_type: str) -> object:
        """Generate a 3D mesh from a sprite."""
        resolution = 1024 if asset_type == "character" else 512
        return self._pipeline(
            str(sprite_path), sdf_resolution=resolution, remesh=True
        )
