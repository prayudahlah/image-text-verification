"""
Image Validator Module
Validates article thumbnails using Pillow:
- Checks resolution (rejects icons, avatars, tracking pixels)
- Verifies image integrity and format (JPEG, PNG, WEBP)
- Rejects blacklisted URL patterns (logos, ads, trackers)
- Optionally saves validated images to disk
"""

import io
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Optional
import requests
from PIL import Image, UnidentifiedImageError

from config import (
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    MAX_IMAGE_FILE_SIZE,
    ALLOWED_IMAGE_FORMATS,
    DISALLOWED_IMAGE_PATTERNS,
    THUMBNAILS_DIR,
    REQUEST_TIMEOUT,
    DEFAULT_HEADERS
)

logger = logging.getLogger(__name__)


class ImageValidator:
    def __init__(self, download_dir: Optional[Path] = None, session: Optional[requests.Session] = None):
        self.download_dir = download_dir or THUMBNAILS_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()

    def is_url_blacklisted(self, url: str) -> Tuple[bool, Optional[str]]:
        """Quick check against blacklisted patterns in the URL."""
        if not url or not isinstance(url, str):
            return True, "Empty or invalid URL"
        
        lowered = url.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            return True, "Not a valid HTTP/HTTPS URL"
            
        for pattern in DISALLOWED_IMAGE_PATTERNS:
            if pattern in lowered:
                return True, f"URL matches disallowed pattern: '{pattern}'"
                
        return False, None

    def validate_and_save(
        self, 
        image_url: str, 
        save_locally: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates an image URL using Pillow.
        Returns:
            (is_valid: bool, reason: Optional[str], local_path: Optional[str])
        """
        blacklisted, reason = self.is_url_blacklisted(image_url)
        if blacklisted:
            return False, reason, None

        try:
            # Download image bytes with size control
            resp = self.session.get(
                image_url,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=True
            )
            if resp.status_code != 200:
                return False, f"HTTP status {resp.status_code} while downloading image", None

            # Check Content-Type header if present
            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type and not any(ct in content_type for ct in ["image/", "application/octet-stream"]):
                return False, f"Invalid Content-Type: {content_type}", None

            # Read content up to max size
            content = b""
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > MAX_IMAGE_FILE_SIZE:
                    return False, f"Image exceeds maximum allowed size of {MAX_IMAGE_FILE_SIZE} bytes", None

            if len(content) < 500:
                return False, f"Image payload too small ({len(content)} bytes), likely a placeholder or tracking pixel", None

            # Open with Pillow
            with Image.open(io.BytesIO(content)) as img:
                img_format = (img.format or "").upper()
                if img_format not in ALLOWED_IMAGE_FORMATS:
                    return False, f"Unsupported image format: {img_format}", None

                width, height = img.size
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    return False, f"Image dimensions ({width}x{height}) below minimum required ({MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT})", None

                # Check if it's a 1x1 or extreme ratio
                aspect_ratio = width / max(height, 1)
                if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                    return False, f"Unusual aspect ratio ({aspect_ratio:.2f}), likely a banner or decorative graphic", None

                local_path = None
                if save_locally:
                    url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
                    ext = img_format.lower()
                    if ext == "jpeg":
                        ext = "jpg"
                    file_name = f"thumb_{url_hash}.{ext}"
                    dest_file = self.download_dir / file_name
                    dest_file.write_bytes(content)
                    local_path = str(dest_file)

                return True, None, local_path

        except UnidentifiedImageError:
            return False, "Pillow could not identify image format (corrupted or unsupported)", None
        except requests.RequestException as e:
            return False, f"Network error fetching image: {str(e)}", None
        except Exception as e:
            logger.warning(f"Unexpected error validating image {image_url}: {e}")
            return False, f"Error validating image: {str(e)}", None

