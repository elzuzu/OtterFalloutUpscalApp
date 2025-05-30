from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import time

from .workers import WorkerSignals
from .ai_scaler import AIScaler
from .config import DEFAULT_GPU_ID
from .engines.upscayl_engine import UpscaylEngine
from .engines.comfyui_engine import ComfyUIEngine
from .quality_metrics import QualityMetrics


class UpscalerEngine(Enum):
    UPSCAYL = "upscayl"
    COMFYUI = "comfyui"
    REALESRGAN = "realesrgan"
    HYBRID = "hybrid"


@dataclass
class AssetProfile:
    primary: UpscalerEngine
    fallback: UpscalerEngine


@dataclass
class UpscalingStats:
    engine_used: str
    processing_time: float
    quality_score: float
    asset_type: str
    original_size: Tuple[int, int]
    upscaled_size: Tuple[int, int]


ASSET_PROFILES: Dict[str, AssetProfile] = {
    "character": AssetProfile(UpscalerEngine.COMFYUI, UpscalerEngine.REALESRGAN),
    "interface": AssetProfile(UpscalerEngine.UPSCAYL, UpscalerEngine.REALESRGAN),
    "items": AssetProfile(UpscalerEngine.UPSCAYL, UpscalerEngine.REALESRGAN),
    "scenery": AssetProfile(UpscalerEngine.COMFYUI, UpscalerEngine.REALESRGAN),
}


class RealESRGANEngine:
    def __init__(
        self,
        realesrgan_exe: Optional[Path] = None,
        model_name: str = "",
        scale_factor: str = "4",
        gpu_id: int = DEFAULT_GPU_ID,
        signals: Optional[WorkerSignals] = None,
    ):
        self.scaler = AIScaler(realesrgan_exe or Path(), model_name, scale_factor, gpu_id, signals)

    def upscale(self, input_dir: Path, output_dir: Path) -> bool:
        return self.scaler.upscale_directory(input_dir, output_dir)


class HybridUpscaler:
    def __init__(self, workspace_dir: Path, signals: WorkerSignals):
        self.workspace_dir = workspace_dir
        self._signals = signals
        self.engines = {
            UpscalerEngine.UPSCAYL: UpscaylEngine(signals),
            UpscalerEngine.COMFYUI: ComfyUIEngine(signals),
            UpscalerEngine.REALESRGAN: RealESRGANEngine(signals=signals),
        }
        self.last_stats: Optional[UpscalingStats] = None
    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def _get_engine(self, engine: UpscalerEngine):
        return self.engines[engine]

    def _calculate_metrics(self, original: Path, upscaled: Path) -> Tuple[float, float]:
        img1 = np.array(Image.open(original).convert("RGB"))
        img2 = np.array(Image.open(upscaled).convert("RGB"))
        ssim_val = structural_similarity(img1, img2, channel_axis=-1)
        psnr_val = peak_signal_noise_ratio(img1, img2)
        return ssim_val, psnr_val

    def _collect_stats(
        self,
        engine: UpscalerEngine,
        asset_type: str,
        input_dir: Path,
        output_dir: Path,
        duration: float,
    ) -> UpscalingStats:
        originals = sorted(input_dir.glob("*.png"))
        orig = originals[0]
        up = output_dir / orig.name
        quality = 0.0
        orig_size = Image.open(orig).size
        up_size = orig_size
        if up.exists():
            up_size = Image.open(up).size
            quality = QualityMetrics.calculate_composite_score(orig, up, asset_type)
        stats = UpscalingStats(
            engine_used=engine.value,
            processing_time=duration,
            quality_score=quality,
            asset_type=asset_type,
            original_size=orig_size,
            upscaled_size=up_size,
        )
        self._log(f"Stats: {stats}")
        self.last_stats = stats
        return stats

    def _validate_quality(self, input_dir: Path, output_dir: Path) -> None:
        originals = sorted(input_dir.glob("*.png"))
        upscaled = sorted(output_dir.glob("*.png"))
        if not originals or not upscaled:
            return
        orig = originals[0]
        up = output_dir / orig.name
        if not up.exists():
            return
        ssim_val, psnr_val = self._calculate_metrics(orig, up)
        self._log(f"Qualité SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}dB")

    def smart_upscale(self, input_dir: Path, output_dir: Path, asset_type: str) -> bool:
        profile = ASSET_PROFILES.get(
            asset_type,
            AssetProfile(UpscalerEngine.REALESRGAN, UpscalerEngine.REALESRGAN),
        )
        self._log(f"Upscaling {asset_type} with {profile.primary.value}...")
        engine = self._get_engine(profile.primary)
        start = time.time()
        success = engine.upscale(input_dir, output_dir)
        duration = time.time() - start
        if success:
            self._validate_quality(input_dir, output_dir)
            self._collect_stats(profile.primary, asset_type, input_dir, output_dir, duration)
            return True
        if profile.fallback != profile.primary:
            self._log("Moteur principal échoué, tentative avec le moteur de secours...")
            fallback_engine = self._get_engine(profile.fallback)
            start = time.time()
            success = fallback_engine.upscale(input_dir, output_dir)
            duration = time.time() - start
            if success:
                self._validate_quality(input_dir, output_dir)
                self._collect_stats(profile.fallback, asset_type, input_dir, output_dir, duration)
            return success
        return False
