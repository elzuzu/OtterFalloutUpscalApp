from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class IntelligentPostProcessor:
    """Apply advanced enhancements based on asset type."""

    def enhance_character(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def enhance_interface(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Sharpness(img).enhance(1.5)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv_img = cv2.GaussianBlur(cv_img, (0, 0), sigmaX=0.5)
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def enhance_items(self, img: Image.Image) -> Image.Image:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv_img = cv2.detailEnhance(cv_img, sigma_s=10, sigma_r=0.15)
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def enhance_scenery(self, img: Image.Image) -> Image.Image:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 3, 3, 7, 21)
        blurred = cv2.GaussianBlur(cv_img, (0, 0), 2)
        depth_img = cv2.addWeighted(cv_img, 1.5, blurred, -0.5, 0)
        return Image.fromarray(cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB))

    def enhance_by_type(self, image_path: Path, asset_type: str) -> bool:
        """Enhance image according to asset type and overwrite original."""
        try:
            img = Image.open(image_path).convert("RGB")
            asset_type = asset_type.lower()
            if asset_type == "character":
                img = self.enhance_character(img)
            elif asset_type == "interface":
                img = self.enhance_interface(img)
            elif asset_type == "items":
                img = self.enhance_items(img)
            elif asset_type == "scenery":
                img = self.enhance_scenery(img)
            else:
                return False
            img.save(image_path)
            return True
        except Exception:
            return False
