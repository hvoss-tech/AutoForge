import os
import uuid
from pathlib import Path
from ..config import config


# Uploads are served back from /uploads with a type taken from the file
# extension. The client-chosen extension used to be kept verbatim, so a file
# that decodes as an image but is also valid HTML (polyglots are easy with
# GIF/BMP) and is named "x.html" was served as a same-origin HTML page.
_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".dib", ".webp", ".tif", ".tiff",
    ".gif", ".jp2", ".pbm", ".pgm", ".ppm", ".pnm", ".pxm", ".sr", ".ras",
    ".exr", ".hdr", ".pic", ".avif",
}


class ImageService:
    def save_upload(self, filename: str, data: bytes) -> str:
        os.makedirs(config.uploads_path, exist_ok=True)
        ext = Path(filename).suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            ext = ".png"
        image_id = str(uuid.uuid4()) + ext
        dest = os.path.join(config.uploads_path, image_id)
        with open(dest, "wb") as f:
            f.write(data)
        return image_id

    def get_path(self, image_id: str) -> str | None:
        resolved = os.path.realpath(os.path.join(config.uploads_path, image_id))
        uploads = os.path.realpath(config.uploads_path)
        if not resolved.startswith(uploads + os.sep) and resolved != uploads:
            return None
        return resolved if os.path.exists(resolved) else None

    def get_url(self, image_id: str) -> str | None:
        path = self.get_path(image_id)
        if path:
            return f"/uploads/{image_id}"
        return None


_service: ImageService | None = None


def get_image_service() -> ImageService:
    global _service
    if _service is None:
        _service = ImageService()
    return _service
