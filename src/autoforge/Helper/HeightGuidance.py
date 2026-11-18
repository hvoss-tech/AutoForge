"""Height guidance: soft suggestions where colours sit in the print.

A heightfield print shows, at every pixel, the colour of its layer stack cut
off at that pixel's height. Two things decide which colour ends up how high,
and the guidance has a term for each:

* the stack (which filament is printed at which layer): filament *pins*
  ("keep a band of this filament near here") and an *order* (dark to light
  or light to dark from the base up);
* the pixel heights: a *target height map* built from the photo's depth
  estimate, painted raise/lower regions and picked image colours.

Both are costs added to the colour loss, never constraints: the optimizer
keeps whatever fits the picture much better. Positions are fractions of the
print's current top layer (``top``), the same axis the webui's colour column
shows (layer 0 = the base, ``top`` = the highest printed layer).

Units: the stack cost is in loss units (compute_loss: mean Lab squared error
/ 3); the pixel cost of one pixel is in Lab squared error, the unit of the
height assignment and the pixel refine, so a mean over the pixels divided by
3 is again in loss units.

A pin is "at least one band of this filament near here": a filament often
appears at several heights for colour mixing, and only what is near the pin
matters. With ``k(p)`` the pin's kernel over stack positions (1 within the
tolerance, falling off with the distance beyond it), the layers of the
filament near the pin count ``n = sum_p P(layer p is the filament) * k(p)``
(the base counts as a full band), and the pin costs ``(1 - n / N)^2`` until
there are ``N`` of them - enough to show (PIN_THICKNESS_MM); a single thin
layer of a translucent filament doesn't. The same formula for a discrete
stack (probabilities 0/1) and for the softmax of the training logits, so
training and every discrete search pursue the same thing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab

ORDERS = ("auto", "dark_to_light", "light_to_dark")

# Loss units a fully missed pin costs at strength 1.
STACK_SCALE = 10.0
# The order's cost relative to a pin's: a reversed stack is the extreme the
# 0..1 order cost only nears; most stacks sit around 0.1-0.25 of it.
ORDER_SCALE = 5.0
# Lab squared error of a pixel a whole print height away from its target
# height (beyond the tolerance) at strength 1.
PIXEL_SCALE = 120.0
# A pin is met anywhere within this fraction of the print height of its
# position; beyond it the kernel falls off over PIN_WIDTH.
PIN_TOLERANCE = 0.06
PIN_WIDTH = 0.2
# A pinned filament needs this much of itself near the pin to count as there.
PIN_THICKNESS_MM = 0.12
# Pixels within this fraction of the print height of their target cost nothing.
PIXEL_TOLERANCE = 0.1
# Painted regions and picked colours outweigh the depth estimate where both apply.
REGION_WEIGHT = 2.0
DEFAULT_COLOR_TOLERANCE = 15.0  # Delta E


@dataclass
class GuidanceSpec:
    """What the user asked for (see ``parse_spec``)."""

    strength: float = 1.0
    order: str = "auto"
    # (material index, position as a fraction of the print height)
    pins: list = field(default_factory=list)
    depth: bool = False
    depth_flip: bool = False
    # (rgb 0-255 tuple, tolerance in Delta E, +1 raise / -1 lower)
    colors: list = field(default_factory=list)

    @property
    def has_stack(self) -> bool:
        return self.strength > 0 and (bool(self.pins) or self.order != "auto")


def _hex_to_rgb(color: str) -> tuple:
    c = str(color).strip().lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Not a #RRGGBB colour: {color!r}")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def parse_spec(data, material_lookup: Optional[Callable[[dict], Optional[int]]] = None) -> Optional[GuidanceSpec]:
    """A ``GuidanceSpec`` from its JSON form (a dict, a JSON string or a path
    to a .json file); None when there is nothing to guide.

    ``{"strength": 1.0, "order": "auto"|"dark_to_light"|"light_to_dark",
    "pins": [{"material": 2, "position": 0.8}], "depth": false,
    "depth_flip": false, "colors": [{"color": "#aabbcc", "tolerance": 15,
    "direction": "raise"|"lower"}]}``

    A pin names its filament by ``material`` (index into the materials);
    ``material_lookup(pin) -> index | None`` resolves other keys (a webui
    uuid, a CLI filament name). Pins whose filament isn't there are dropped.
    """
    if data is None or data == "":
        return None
    if isinstance(data, str):
        text = data
        if os.path.isfile(data):
            with open(data, encoding="utf-8") as fh:
                text = fh.read()
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Height guidance must be a JSON object")
    strength = float(data.get("strength", 1.0) or 0.0)
    order = str(data.get("order") or "auto")
    if order not in ORDERS:
        raise ValueError(f"Unknown height order {order!r} (use one of {', '.join(ORDERS)})")
    pins = []
    for pin in data.get("pins") or []:
        m = pin.get("material")
        if m is None and material_lookup is not None:
            m = material_lookup(pin)
        if m is None:
            continue
        pos = float(pin.get("position", 0.5))
        pins.append((int(m), min(1.0, max(0.0, pos))))
    colors = []
    for c in data.get("colors") or []:
        direction = str(c.get("direction", "raise"))
        if direction not in ("raise", "lower"):
            raise ValueError(f"Unknown colour direction {direction!r} (raise or lower)")
        colors.append((
            _hex_to_rgb(c["color"]),
            max(1.0, float(c.get("tolerance", DEFAULT_COLOR_TOLERANCE))),
            1 if direction == "raise" else -1,
        ))
    spec = GuidanceSpec(
        strength=max(0.0, strength),
        order=order,
        pins=pins,
        depth=bool(data.get("depth", False)),
        depth_flip=bool(data.get("depth_flip", False)),
        colors=colors,
    )
    return spec


# --- target height map --------------------------------------------------

_depth_lock = threading.Lock()
_depth_pipe = None
_depth_cache: dict = {}
_DEPTH_CACHE_SIZE = 4


def _run_depth_model(image_rgb: np.ndarray) -> np.ndarray:
    """Raw Depth Anything V2 output (larger = nearer), image size."""
    global _depth_pipe
    from PIL import Image

    from autoforge.Helper.DeviceUtils import accelerator_device

    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:  # pragma: no cover - dependency missing
        raise ImportError("Depth guidance needs 'transformers' installed.") from e
    if _depth_pipe is None:
        accel = accelerator_device()
        try:
            _depth_pipe = pipeline(
                task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=accel
            )
        except Exception:
            if accel is None:
                raise
            _depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    out = _depth_pipe(Image.fromarray(image_rgb.astype(np.uint8)))
    depth = np.asarray(out["depth"], dtype=np.float32)
    if depth.shape[:2] != image_rgb.shape[:2]:
        import cv2

        depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def preload_depth_model() -> bool:
    """Load the depth model now (the webui does at startup, on a thread);
    False when it can't be (no 'transformers', no download)."""
    import importlib.util

    global _depth_pipe
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        with _depth_lock:
            if _depth_pipe is None:
                _run_depth_model(np.zeros((32, 32, 3), np.uint8))
        return True
    except Exception:
        import logging

        logging.getLogger(__name__).warning("Could not preload the depth model", exc_info=True)
        return False


def estimate_depth(image_rgb: np.ndarray) -> np.ndarray:
    """[H,W] float32 in 0..1, 1 = nearest: the photo's depth estimate,
    normalized between its 1st and 99th percentile. Cached per image."""
    key = (hashlib.sha1(np.ascontiguousarray(image_rgb).tobytes()).hexdigest(), image_rgb.shape)
    with _depth_lock:
        hit = _depth_cache.get(key)
        if hit is not None:
            return hit
        raw = _run_depth_model(image_rgb)
        lo, hi = np.percentile(raw, [1.0, 99.0])
        depth = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        if len(_depth_cache) >= _DEPTH_CACHE_SIZE:
            _depth_cache.pop(next(iter(_depth_cache)))
        _depth_cache[key] = depth
        return depth


def _resize(arr: np.ndarray, shape) -> np.ndarray:
    import cv2

    if arr.shape[:2] == tuple(shape):
        return arr
    interp = cv2.INTER_AREA if arr.shape[0] > shape[0] else cv2.INTER_LINEAR
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=interp)


def color_pick_weights(image_rgb: np.ndarray, rgb, tolerance: float) -> np.ndarray:
    """[H,W] 0..1: how closely each pixel matches a picked colour (1 at the
    colour, 0 from ``tolerance`` Delta E on)."""
    img = torch.as_tensor(image_rgb, dtype=torch.float32)
    lab = srgb_to_lab(img)
    ref = srgb_to_lab(torch.tensor(rgb, dtype=torch.float32).view(1, 1, 3))[0, 0]
    de = (lab - ref).norm(dim=-1)
    return (1.0 - de / float(tolerance)).clamp(0.0, 1.0).numpy().astype(np.float32)


def build_target_maps(
    spec: Optional[GuidanceSpec],
    image_rgb: np.ndarray,
    regions_rgba: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    depth_fn: Callable[[np.ndarray], np.ndarray] = None,
):
    """The pixel guidance of ``spec`` at the image's size: ``(target,
    weight)``, both [H,W] float32 - the height each pixel is pulled toward
    (0 = the base, 1 = the print's top) and how strongly (0..1) - or
    ``(None, None)`` when no pixel is guided.

    ``regions_rgba`` [h,w,4] uint8: painted strokes, red = raise, blue =
    lower, alpha = how fully painted (any size; resized to the image).
    """
    H, W = image_rgb.shape[:2]
    num = np.zeros((H, W), dtype=np.float32)
    den = np.zeros((H, W), dtype=np.float32)
    if spec is not None and spec.depth:
        depth = (depth_fn or estimate_depth)(image_rgb)
        depth = _resize(np.asarray(depth, dtype=np.float32), (H, W))
        if spec.depth_flip:
            depth = 1.0 - depth
        num += depth
        den += 1.0
    if regions_rgba is not None:
        r = np.asarray(regions_rgba)
        if r.ndim == 3 and r.shape[2] >= 4:
            a = r[..., 3].astype(np.float32) / 255.0
            up = _resize(r[..., 0].astype(np.float32) / 255.0 * a, (H, W))
            down = _resize(r[..., 2].astype(np.float32) / 255.0 * a, (H, W))
            num += REGION_WEIGHT * up
            den += REGION_WEIGHT * (up + down)
    if spec is not None:
        for rgb, tol, direction in spec.colors:
            w = REGION_WEIGHT * color_pick_weights(image_rgb, rgb, tol)
            if direction > 0:
                num += w
            den += w
    if not np.any(den > 0):
        return None, None
    target = np.where(den > 0, num / np.maximum(den, 1e-8), 0.0).astype(np.float32)
    weight = np.clip(den, 0.0, 1.0).astype(np.float32)
    if alpha is not None:
        a = np.asarray(alpha)
        if a.ndim == 3:
            a = a[..., 0]
        weight = np.where(_resize(a.astype(np.float32), (H, W)) < 128, 0.0, weight).astype(np.float32)
    return target, weight


# --- the costs ---------------------------------------------------------------


class HeightGuidance:
    """The guidance costs for one run (see module docstring).

    ``material_colors`` [M,3] 0..1, ``max_layers`` the stack length, the
    maps (full resolution, numpy or tensors; None for no pixel guidance)
    are resized to whatever resolution a cost is asked at.
    """

    def __init__(
        self,
        spec: Optional[GuidanceSpec],
        material_colors: torch.Tensor,
        max_layers: int,
        target_map=None,
        weight_map=None,
        device=None,
        layer_height: float = 0.04,
    ):
        self.spec = spec or GuidanceSpec()
        # The webui's default guidance, which runs without being shown.
        self.hidden = False
        device = device if device is not None else material_colors.device
        self.device = torch.device(device)
        self.max_layers = int(max_layers)
        self.pin_layers = max(1.0, float(np.ceil(PIN_THICKNESS_MM / max(float(layer_height), 1e-6) - 1e-6)))
        M = int(material_colors.shape[0])
        self.pins = [(m, p) for m, p in self.spec.pins if 0 <= m < M]
        s = max(0.0, float(self.spec.strength))
        self.stack_scale = s * STACK_SCALE
        self.pixel_scale = s * PIXEL_SCALE
        self.order = 0
        if self.spec.order == "dark_to_light":
            self.order = 1
        elif self.spec.order == "light_to_dark":
            self.order = -1
        self.has_stack = s > 0 and (bool(self.pins) or self.order != 0)
        self._pin_mats = torch.tensor([m for m, _ in self.pins], dtype=torch.long, device=self.device)
        self._pin_pos = torch.tensor([p for _, p in self.pins], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lab = srgb_to_lab(material_colors.detach().float().to(self.device).view(1, -1, 3) * 255.0)[0]
        self._lum = (lab[:, 0] / 100.0).contiguous()  # [M] L* / 100
        self._lum_range = float(self._lum.max() - self._lum.min()) if M > 1 else 1.0
        # The print's top layer, a device scalar updated in place (a captured
        # training graph reads it).
        self._top = torch.full((), float(self.max_layers), dtype=torch.float32, device=self.device)
        self._top_int = self.max_layers
        self._maps_full = None
        self._maps_cache = {}
        if target_map is not None and weight_map is not None and s > 0:
            t = torch.as_tensor(np.asarray(target_map, dtype=np.float32), device=self.device)
            w = torch.as_tensor(np.asarray(weight_map, dtype=np.float32), device=self.device)
            if bool((w > 0).any()):
                self._maps_full = (t, w)
        self.has_pixel = self._maps_full is not None

    @property
    def active(self) -> bool:
        return self.has_stack or self.has_pixel

    # -- the top ---------------------------------------------------------

    @property
    def top(self) -> int:
        return self._top_int

    def set_top(self, top: int) -> None:
        top = int(max(1, min(self.max_layers, int(top))))
        self._top_int = top
        self._top.fill_(float(top))

    @torch.no_grad()
    def sync_top(self, z: torch.Tensor) -> None:
        """Take the top from a height map (one host sync)."""
        self.set_top(int(z.max()))

    def _top_of(self, top, batch: int) -> torch.Tensor:
        """[B] float tops: ``top`` (int, tensor or None for the current)."""
        if top is None:
            return self._top.expand(batch)
        t = torch.as_tensor(top, dtype=torch.float32, device=self.device)
        return t.expand(batch) if t.dim() == 0 else t

    # -- stack ------------------------------------------------------------

    def _pin_kernel(self, top: torch.Tensor, n_pos: int) -> torch.Tensor:
        """[B,P,n_pos] kernel of every pin over stack positions 0..n_pos-1
        (0 = the base) for tops ``top`` [B]."""
        pos = torch.arange(n_pos, device=self.device, dtype=torch.float32).view(1, 1, -1)
        T = top.view(-1, 1, 1)
        centre = self._pin_pos.view(1, -1, 1) * T
        d = torch.clamp((pos - centre).abs() - PIN_TOLERANCE * T, min=0.0) / (PIN_WIDTH * T)
        k = torch.exp(-d * d)
        return torch.where(pos <= T, k, torch.zeros_like(k))

    def _pin_miss(self, n: torch.Tensor) -> torch.Tensor:
        """0..1 cost of a pin with ``n`` (kernel-weighted) layers near it."""
        return torch.relu(1.0 - n / self.pin_layers).pow(2)

    def _order_cost(self, lum_seq: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        """[B] 0..1 cost of the luminance sequence ``lum_seq`` [B,n]
        (position 0 = the base) going the wrong way: every pair of visible
        positions in the wrong order, weighted by how far apart they are, so
        a local alternation (colour mixing) costs little and a reversed
        stack everything."""
        n = lum_seq.shape[1]
        pos = torch.arange(n, device=self.device, dtype=torch.float32)
        gap = (pos.view(1, 1, -1) - pos.view(1, -1, 1)).clamp(min=0.0)  # [1,n,n] j - i for i < j
        vis = (pos.view(1, 1, -1) <= top.view(-1, 1, 1)).float()  # j visible
        w = gap * vis
        w = w / w.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        diff = lum_seq.unsqueeze(2) - lum_seq.unsqueeze(1)  # s_i - s_j
        if self.order < 0:
            diff = -diff
        return (torch.relu(diff) * w).sum(dim=(1, 2)) / max(self._lum_range, 1e-3)

    def _background_lum(self, background: torch.Tensor) -> torch.Tensor:
        bg = background.detach().float().to(self.device).view(1, 1, 3) * 255.0
        return srgb_to_lab(bg)[0, 0, 0] / 100.0

    def stack_cost(self, dg: torch.Tensor, base_material=None, background=None, top=None) -> torch.Tensor:
        """Stack cost (loss units) of discrete stacks ``dg`` [B,L] (or [L]:
        a 0-dim result). ``base_material`` the base's filament index (None
        for a custom colour), ``background`` its colour [3] 0..1."""
        single = dg.dim() == 1
        d = dg.view(1, -1) if single else dg
        d = d.to(device=self.device, dtype=torch.long)
        B, L = d.shape
        out = torch.zeros(B, dtype=torch.float32, device=self.device)
        if not self.has_stack:
            return out[0] if single else out
        T = self._top_of(top, B)
        if self.pins:
            k = self._pin_kernel(T, L + 1)  # [B,P,L+1]
            hit = (d.unsqueeze(1) == self._pin_mats.view(1, -1, 1)).float()  # [B,P,L]
            n = (hit * k[:, :, 1:]).sum(dim=2)
            if base_material is not None:
                on_base = (self._pin_mats == int(base_material)).float().view(1, -1)
                n = n + on_base * k[:, :, 0] * self.pin_layers
            out = out + self._pin_miss(n).sum(dim=1)
        if self.order != 0:
            if background is not None:
                b = self._background_lum(background).expand(B, 1)
            elif base_material is not None:
                b = self._lum[int(base_material)].expand(B, 1)
            else:
                b = None
            seq = self._lum[d]
            seq = torch.cat([b, seq], dim=1) if b is not None else torch.cat([seq[:, :1], seq], dim=1)
            out = out + ORDER_SCALE * self._order_cost(seq, T)
        out = out * self.stack_scale
        return out[0] if single else out

    def stack_cost_soft(self, global_logits: torch.Tensor, bg_probs=None, base_material=None, background=None) -> torch.Tensor:
        """Differentiable stack cost of the softmax of the training logits
        [L,M] (the expected stack of the snapshots). No host sync: safe in a
        captured training graph (the top is read from a device scalar)."""
        p = torch.softmax(global_logits.float(), dim=1)  # [L,M]
        L = p.shape[0]
        T = self._top.view(1)
        total = torch.zeros((), dtype=torch.float32, device=p.device)
        if self.pins:
            k = self._pin_kernel(T, L + 1)[0]  # [P,L+1]
            pm = p[:, self._pin_mats].t()  # [P,L]
            n = (pm * k[:, 1:]).sum(dim=1)
            if bg_probs is not None:
                n = n + bg_probs.float()[self._pin_mats] * k[:, 0] * self.pin_layers
            elif base_material is not None:
                on_base = (self._pin_mats == int(base_material)).float()
                n = n + on_base * k[:, 0] * self.pin_layers
            total = total + self._pin_miss(n).sum()
        if self.order != 0:
            seq = p @ self._lum  # [L]
            if bg_probs is not None:
                b = (bg_probs.float() @ self._lum).view(1)
            elif background is not None:
                b = self._background_lum(background).view(1)
            else:
                b = seq[:1]
            total = total + ORDER_SCALE * self._order_cost(torch.cat([b, seq]).view(1, -1), T)[0]
        return total * self.stack_scale

    # -- pixels ------------------------------------------------------------

    def maps(self, shape):
        """(target, weight) [H,W] at ``shape`` (cached)."""
        shape = tuple(int(s) for s in shape)
        hit = self._maps_cache.get(shape)
        if hit is None:
            t, w = self._maps_full
            if tuple(t.shape) != shape:
                mode = "area" if t.shape[0] * t.shape[1] > shape[0] * shape[1] else "bilinear"
                kw = {} if mode == "area" else {"align_corners": False}
                dev = t.device
                # "area" is adaptive_avg_pool2d, which MPS only supports for
                # divisible sizes; the result is cached, so resize on the CPU.
                if mode == "area" and dev.type == "mps":
                    t, w = t.cpu(), w.cpu()
                t = F.interpolate(t[None, None], size=shape, mode=mode, **kw)[0, 0].to(dev)
                w = F.interpolate(w[None, None], size=shape, mode=mode, **kw)[0, 0].to(dev)
            hit = (t.contiguous(), w.contiguous())
            if len(self._maps_cache) > 8:
                self._maps_cache.clear()
            self._maps_cache[shape] = hit
        return hit

    def pixel_cost_map(self, z: torch.Tensor, top=None) -> torch.Tensor:
        """Per-pixel cost (Lab squared error units) of heights ``z`` [H,W]
        (or [B,H,W] with ``top`` [B])."""
        if not self.has_pixel:
            return torch.zeros(z.shape, dtype=torch.float32, device=z.device)
        tgt, w = self.maps(z.shape[-2:])
        if z.dim() == 3:
            T = self._top_of(top, z.shape[0]).view(-1, 1, 1)
        else:
            T = self._top_of(top, 1).view(1, 1)
        dev = torch.clamp((z.to(torch.float32) / T - tgt).abs() - PIXEL_TOLERANCE, min=0.0)
        return self.pixel_scale * w * dev * dev

    def pixel_cost_table(self, shape, K: int) -> Optional[torch.Tensor]:
        """[H,W,K] cost of every pixel at every height 0..K-1, or None."""
        if not self.has_pixel:
            return None
        tgt, w = self.maps(shape)
        ks = torch.arange(K, device=self.device, dtype=torch.float32).view(1, 1, -1)
        # One [H,W,K] tensor, the rest in place (at the output resolution
        # it is as large as the assignment's own cost table).
        cost = ks / self._top - tgt.unsqueeze(-1)
        cost.abs_().sub_(PIXEL_TOLERANCE).clamp_(min=0.0)
        cost.mul_(cost)
        return cost.mul_(w.unsqueeze(-1) * self.pixel_scale)

    def pixel_loss(self, z: torch.Tensor, top=None) -> torch.Tensor:
        """Pixel cost in loss units: mean over the pixels / 3 (0-dim, or [B])."""
        if not self.has_pixel:
            n = z.shape[0] if z.dim() == 3 else None
            return torch.zeros(n if n is not None else (), dtype=torch.float32, device=z.device)
        return self.pixel_cost_map(z, top).mean(dim=(-2, -1)) / 3.0

    def total(self, dg: torch.Tensor, z: Optional[torch.Tensor], base_material=None, background=None, top=None) -> torch.Tensor:
        """Stack plus pixel cost of a solution (0-dim for one stack, [B] for
        a batch sharing the heights ``z``)."""
        out = self.stack_cost(dg, base_material, background, top)
        if z is not None and self.has_pixel:
            out = out + self.pixel_loss(z, top)
        return out

    # -- reporting ---------------------------------------------------------

    @torch.no_grad()
    def report(self, dg: torch.Tensor, z: torch.Tensor, base_material=None, background=None) -> dict:
        """How well a solution follows the guidance, for the user: every
        pin's nearest band (as a fraction of the print height) and whether
        it is within reach, the order's agreement, the pixels' mean
        distance from their targets."""
        top = max(1, int(z.max()))
        d = dg.to(device=self.device, dtype=torch.long).view(-1)
        out: dict = {"top": top, "pins": []}
        T = torch.tensor([float(top)], device=self.device)
        cost = None
        if self.pins:
            k = self._pin_kernel(T, d.numel() + 1)[0]  # [P,L+1]
            hit = (d.view(1, -1) == self._pin_mats.view(-1, 1)).float()
            n = (hit * k[:, 1:]).sum(dim=1)
            if base_material is not None:
                n = n + (self._pin_mats == int(base_material)).float() * k[:, 0] * self.pin_layers
            cost = self._pin_miss(n).tolist()
        for i, (m, p) in enumerate(self.pins):
            positions = [j + 1 for j, v in enumerate(d.tolist()) if v == m and j + 1 <= top]
            if base_material is not None and int(base_material) == m:
                positions.append(0)
            nearest = min(positions, key=lambda q: abs(q - p * top)) / top if positions else None
            out["pins"].append({
                "material": m, "position": p, "nearest": nearest,
                # Within reach: at most a quarter of the pin's cost left.
                "met": bool(cost[i] <= 0.25),
            })
        if self.order != 0:
            if background is not None:
                b = self._background_lum(background).view(1, 1)
            else:
                b = self._lum[d[:1]].view(1, 1)
            seq = torch.cat([b, self._lum[d].view(1, -1)], dim=1)
            out["order_agreement"] = float(1.0 - self._order_cost(seq, T)[0].clamp(0, 1))
        if self.has_pixel:
            tgt, w = self.maps(z.shape)
            wsum = float(w.sum())
            if wsum > 0:
                dist = ((z.float() / top - tgt).abs() * w).sum() / wsum
                out["pixel_mean_distance"] = float(dist)
        return out


def material_index_lookup(uuids: Sequence[str], names: Sequence[str] = (), colors: Sequence[str] = ()):
    """``material_lookup`` for ``parse_spec``: a pin's ``uuid`` (webui),
    ``filament`` (a name) or ``color`` (#RRGGBB) to a material index."""
    uuids = [str(u) for u in uuids]
    names = [str(n).strip().lower() for n in names]
    colors = [str(c).strip().lower() for c in colors]

    def lookup(pin: dict) -> Optional[int]:
        u = pin.get("uuid")
        if u and str(u) in uuids:
            return uuids.index(str(u))
        n = pin.get("filament")
        if n is not None:
            if isinstance(n, int) or (isinstance(n, str) and n.isdigit()):
                return int(n)
            key = str(n).strip().lower()
            if key in names:
                return names.index(key)
            # "Yellow" for "Brand - Yellow" (the CLI's names carry the brand).
            short = [i for i, full in enumerate(names) if full.split(" - ")[-1] == key]
            if len(short) == 1:
                return short[0]
        c = pin.get("color")
        if c and str(c).strip().lower() in colors:
            return colors.index(str(c).strip().lower())
        return None

    return lookup
