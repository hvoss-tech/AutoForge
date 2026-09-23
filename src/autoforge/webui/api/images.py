import asyncio

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.image_service import get_image_service

router = APIRouter()

# Generous but bounded: without a cap, an arbitrarily large upload is read
# fully into memory and handed to cv2.imdecode, which itself can allocate a
# multiple of the file size while decoding.
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _decode(data: bytes):
    # cv2.imdecode is a blocking C call; running it inline on the event loop
    # stalled every other request and WS message (including live job
    # progress) for the duration of a large upload's decode.
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED) if data else None


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    svc = get_image_service()
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    # Reject what OpenCV can't decode right away: a corrupt or non-image file
    # used to be accepted and only fail later, as a confusing auto-preview or
    # optimization error.
    decoded = await asyncio.to_thread(_decode, data)
    if decoded is None:
        raise HTTPException(400, "That file isn't an image this app can read (try PNG or JPEG).")
    filename = file.filename or "image.png"
    image_id = await asyncio.to_thread(svc.save_upload, filename, data)
    url = svc.get_url(image_id)
    return {"filename": image_id, "url": url}


@router.get("/{filename}")
async def get_image(filename: str):
    svc = get_image_service()
    path = svc.get_path(filename)
    if not path:
        raise HTTPException(404, "Image not found")
    from fastapi.responses import FileResponse
    return FileResponse(path)
