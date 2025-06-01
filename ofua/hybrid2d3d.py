from pathlib import Path
from typing import Optional, Dict

from .direct3d_integration import Direct3DIntegration
from .hybrid_upscaler import HybridUpscaler


class Hybrid2D3DPipeline:
    """Simple wrapper combining 2D upscaling and 3D generation."""

    def __init__(self, upscaler: HybridUpscaler, d3d: Direct3DIntegration) -> None:
        self.upscaler = upscaler
        self.d3d = d3d

    def process_asset(self, frm_path: Path, asset_type: str, output_dir: Path, output_mode: str = "both") -> Dict[str, Optional[Path]]:
        """Process a single FRM file and optionally create a 3D model."""
        sprite_dir = output_dir / "2d"
        sprite_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir = output_dir / "3d"
        mesh_dir.mkdir(parents=True, exist_ok=True)

        if not self.upscaler.smart_upscale(frm_path.parent, sprite_dir, asset_type):
            return {"2d_sprite": None, "3d_model": None}
        upscaled_png = next(sprite_dir.glob("*.png"), None)

        mesh = None
        if upscaled_png and output_mode in {"3d", "both"}:
            mesh = self.d3d.generate_3d_from_sprite(upscaled_png, asset_type)

        return {"2d_sprite": upscaled_png if output_mode != "3d" else None, "3d_model": mesh}
