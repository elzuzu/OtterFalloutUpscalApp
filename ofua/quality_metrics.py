from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class QualityMetrics:
    """Calcul des métriques de qualité pour images upscalées."""

    WEIGHTS: Dict[str, Tuple[float, float, float]] = {
        "character": (0.5, 0.3, 0.2),
        "interface": (0.4, 0.4, 0.2),
        "items": (0.4, 0.3, 0.3),
        "scenery": (0.4, 0.3, 0.3),
        "default": (0.4, 0.3, 0.3),
    }

    THRESHOLDS: Dict[str, float] = {
        "character": 0.75,
        "interface": 0.85,
        "items": 0.7,
        "scenery": 0.7,
        "default": 0.7,
    }

    @staticmethod
    def _load_images(original: Path, upscaled: Path) -> Tuple[np.ndarray, np.ndarray]:
        img_orig = cv2.imread(str(original), cv2.IMREAD_COLOR)
        img_up = cv2.imread(str(upscaled), cv2.IMREAD_COLOR)
        if img_orig is None or img_up is None:
            raise FileNotFoundError("Impossible de charger les images pour les métriques")
        if img_orig.shape != img_up.shape:
            img_orig = cv2.resize(img_orig, (img_up.shape[1], img_up.shape[0]))
        return img_orig, img_up

    @staticmethod
    def _sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _artifact_penalty(orig_gray: np.ndarray, up_gray: np.ndarray) -> float:
        diff = cv2.absdiff(orig_gray, up_gray)
        diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
        diff_var = cv2.Laplacian(diff_blur, cv2.CV_64F).var()
        return 1.0 - min(diff_var / 1000.0, 1.0)

    @classmethod
    def calculate_metrics(cls, original: Path, upscaled: Path) -> Tuple[float, float, float, float]:
        img_orig, img_up = cls._load_images(original, upscaled)
        ssim_score = structural_similarity(img_orig, img_up, channel_axis=-1)
        psnr_val = peak_signal_noise_ratio(img_orig, img_up)
        sharp_orig = cls._sharpness(img_orig)
        sharp_up = cls._sharpness(img_up)
        sharp_ratio = sharp_up / (sharp_orig + 1e-8)
        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
        artifact_penalty = cls._artifact_penalty(gray_orig, gray_up)
        return ssim_score, psnr_val, sharp_ratio, artifact_penalty

    @classmethod
    def calculate_composite_score(cls, original: Path, upscaled: Path, asset_type: str) -> float:
        """Retourne un score composite entre 0 et 1."""
        ssim_score, psnr_val, sharp_ratio, artifact_penalty = cls.calculate_metrics(original, upscaled)
        weights = cls.WEIGHTS.get(asset_type, cls.WEIGHTS["default"])
        psnr_normalized = min(psnr_val / 50.0, 1.0)
        sharpness_normalized = min(sharp_ratio, 1.0)
        composite = (
            ssim_score * weights[0]
            + psnr_normalized * weights[1]
            + sharpness_normalized * weights[2]
        )
        composite *= artifact_penalty
        return max(0.0, min(composite, 1.0))

    @classmethod
    def is_quality_acceptable(cls, score: float, asset_type: str) -> bool:
        threshold = cls.THRESHOLDS.get(asset_type, cls.THRESHOLDS["default"])
        return score >= threshold
