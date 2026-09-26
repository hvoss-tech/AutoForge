import os
from typing import Sequence

import cv2
import torch
from cv2.typing import MatLike
import numpy as np


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> MatLike:
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)


def imwrite(filename: str, img: MatLike, params: Sequence[int] = ()) -> None:
    success, encoded_img = cv2.imencode(os.path.splitext(filename)[1], img, params)
    if not success:
        raise OSError(f"cv2 could not write to path {filename}")
    encoded_img.tofile(filename)


def to_bgr_or_bgra_uint8(img) -> np.ndarray:
    """Bring any image OpenCV decodes into the 8-bit BGR/BGRA layout the
    pipeline is written for (used by both the CLI and the webui).

    ``cv2.imread(..., IMREAD_UNCHANGED)`` returns whatever the file holds, but
    the pipeline indexed ``img.shape[2]`` and treated values as 0-255: a
    grayscale PNG/JPEG (2-D array) failed with "tuple index out of range",
    grayscale+alpha had two channels and failed in ``cvtColor``, and a
    16-bit PNG ran on values up to 65535 — a nonsense target the optimizer
    could never match."""
    if img is None:
        raise ValueError("The input image could not be read.")
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).round().astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    channels = img.shape[2]
    if channels == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 2:
        gray, alpha = img[:, :, 0], img[:, :, 1:2]
        return np.concatenate([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), alpha], axis=2)
    return img


def resize_image(img, max_size):
    h_img, w_img, _ = img.shape

    # Compute the scaling factor based on the larger dimension
    if w_img >= h_img:
        scale = max_size / w_img
    else:
        scale = max_size / h_img

    # Compute new dimensions and round them
    new_w = int(round(w_img * scale))
    new_h = int(round(h_img * scale))

    # Resize the image with an appropriate interpolation method
    img_out = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
    # INTER_CUBIC is actually not a good choice for shrinking images. Except in our case.
    # As we shrink the image, we lose information and the image becomes blurry with INTER_AREA. This is detrimental for the solver. If you want to try it out be my guest.
    # INTER_AREA actually destroys a lot of the colors of our solver.
    return img_out


def resize_image_exact(img, new_w, new_h):
    img_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # INTER_CUBIC is actually not a good choice for shrinking images. Except in our case.
    # As we shrink the image, we lose information and the image becomes blurry with INTER_AREA. This is detrimental for the solver. If you want to try it out be my guest.
    # INTER_AREA actually destroys a lot of the colors of our solver.
    return img_out


# sRGB -> Lab conversion constants, cached per (device, dtype) so we don't
# rebuild + host->device transfer these small tensors on every call (this
# function runs at least twice per optimization step).
_RGB_TO_XYZ_CACHE: dict = {}
_WHITE_POINT_CACHE: dict = {}


def _get_rgb_to_xyz(dtype, device):
    key = (dtype, device)
    t = _RGB_TO_XYZ_CACHE.get(key)
    if t is None:
        t = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=dtype,
            device=device,
        )
        _RGB_TO_XYZ_CACHE[key] = t
    return t


def _get_white_point(dtype, device):
    key = (dtype, device)
    t = _WHITE_POINT_CACHE.get(key)
    if t is None:
        t = torch.tensor([0.95047, 1.0, 1.08883], dtype=dtype, device=device)
        _WHITE_POINT_CACHE[key] = t
    return t


def srgb_to_lab(srgb, eps=1e-6):
    """
    Converts an sRGB image (values in [0, 255]) to the CIELAB color space.
    The input tensor should have a shape ending in 3 (e.g., [H, W, 3] or [N, 3, H, W]).
    The function is fully differentiable.

    Args:
        srgb (torch.Tensor): The sRGB image tensor in the range [0, 255].
        eps (float): A small epsilon value to stabilize power operations.

    Returns:
        torch.Tensor: The image in CIELAB color space.
    """
    # Convert to float, normalize to [0, 1] and clamp to avoid negatives.
    srgb = torch.clamp(srgb / 255.0, 0.0, 1.0)

    # Inverse gamma correction to get linear RGB.
    threshold = 0.04045
    srgb_linear = torch.where(
        srgb <= threshold,
        srgb / 12.92,
        torch.pow(torch.clamp((srgb + 0.055) / 1.055, min=eps), 2.4),
    )

    # sRGB to XYZ conversion matrix (D65 illuminant)
    rgb_to_xyz = _get_rgb_to_xyz(srgb.dtype, srgb.device)

    # Reshape for matrix multiplication if needed.
    orig_shape = srgb_linear.shape
    srgb_linear = srgb_linear.view(-1, 3)
    xyz = torch.matmul(srgb_linear, rgb_to_xyz.t())
    xyz = xyz.view(orig_shape)

    # Normalize XYZ by the D65 white point.
    white_point = _get_white_point(srgb.dtype, srgb.device)
    xyz_scaled = xyz / white_point

    # Define the piecewise function f(t) used in the Lab conversion.
    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3  # 29^3/3^3

    def f(t):
        return torch.where(
            t > epsilon,
            torch.pow(torch.clamp(t, min=eps), 1 / 3),
            (kappa * t + 16) / 116,
        )

    f_xyz = f(xyz_scaled)

    # Separate channels.
    fX, fY, fZ = f_xyz[..., 0], f_xyz[..., 1], f_xyz[..., 2]

    # Compute L, a, and b.
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    # Stack channels back to form the Lab image.
    lab = torch.stack([L, a, b], dim=-1)
    return lab


def increase_saturation(image: torch.Tensor, percentage: float) -> torch.Tensor:
    """
    Increase the saturation of an image by a given percentage in a differentiable way.

    Supports both channel-first and channel-last image formats.

    Args:
        image (torch.Tensor): Input image tensor. Can be of shape:
                              - Channel-first: (C, H, W) or (B, C, H, W)
                              - Channel-last:  (H, W, C) or (B, H, W, C)
                              where C must be 3 (RGB) and pixel values are assumed to be normalized.
        percentage (float): Percentage increase in saturation. For example, 0.2 corresponds to a 20% increase.

    Returns:
        torch.Tensor: The image tensor with increased saturation.

    The function works by:
      1. Computing a grayscale image using the luminance formula:
           Y = 0.2989*R + 0.5870*G + 0.1140*B.
      2. Interpolating between the grayscale image and the original image:
           new_image = gray + (1 + percentage) * (image - gray).
      This operation is fully differentiable.
    """
    # Calculate the saturation factor (e.g., 1.2 for a 20% increase)
    factor = 1.0 + percentage

    # Define luminance weights for the RGB channels
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype
    )

    # Handle different input formats (channel-first vs channel-last)
    if image.ndim == 3:
        # Single image: could be (C, H, W) or (H, W, C)
        if image.shape[0] == 3 and image.shape[-1] != 3:
            # Channel-first format: (C, H, W)
            gray = (image * weights[:, None, None]).sum(dim=0, keepdim=True)
            gray = gray.expand_as(image)
        elif image.shape[-1] == 3:
            # Channel-last format: (H, W, C)
            gray = (image * weights).sum(dim=-1, keepdim=True)
            gray = gray.expand_as(image)
        else:
            raise ValueError(
                "Input image tensor must have 3 channels in either the first or last dimension"
            )
    elif image.ndim == 4:
        # Batched image: could be (B, C, H, W) or (B, H, W, C)
        if image.shape[1] == 3 and image.shape[-1] != 3:
            # Channel-first: (B, C, H, W)
            gray = (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            gray = gray.expand_as(image)
        elif image.shape[-1] == 3:
            # Channel-last: (B, H, W, C)
            gray = (image * weights.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True)
            gray = gray.expand_as(image)
        else:
            raise ValueError(
                "Input image tensor must have 3 channels in either the second or last dimension"
            )
    else:
        raise ValueError("Image tensor must be 3D or 4D")

    # Increase saturation by interpolating between grayscale and the original image
    new_image = gray + factor * (image - gray)
    return new_image
