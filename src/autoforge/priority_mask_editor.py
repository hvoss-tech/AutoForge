#!/usr/bin/env python3
"""
Simple interactive priority mask editor.

Usage:
    python priority_mask_editor.py --input input_image.png [--output priority_mask.png]

Instructions:
    - Left mouse drag paints (increase priority) with a chosen brush size.
    - Right mouse drag erases (decrease priority).
    - Keys:
        q / ESC : Quit and save
        s       : Save without quitting
        c       : Clear mask
        +/-     : Increase / decrease brush radius
        g       : Toggle gradient mode (soft brush edges)
        r       : Reset mask to uniform 0

Output:
    A 8-bit grayscale PNG (0..255). During optimization this will be normalized to 0..1.
    White (255) -> full priority, Black (0) -> low priority.

If --output is omitted, the mask will be saved next to the input as <input>_mask.png.

You can create soft gradients by enabling gradient mode (g), which applies a
cosine falloff from the center of the brush.
"""

import argparse
import os
import cv2
import numpy as np

# The ImageHelper wrappers: OpenCV's own file I/O can't open non-ASCII
# paths on Windows.
from autoforge.Helper.ImageHelper import imread, imwrite


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to the input reference image")
    p.add_argument(
        "--output",
        default="",
        help="Path to save the priority mask (optional). If omitted, saves as <input>_mask.png",
    )
    p.add_argument(
        "--initial", default="", help="Optional existing mask to load and edit"
    )
    p.add_argument(
        "--brush", type=int, default=40, help="Initial brush radius in pixels"
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Paint strength (0..1)")
    return p.parse_args()


class PriorityMaskEditor:
    def __init__(self, ref_img, init_mask=None, brush_radius=40, alpha=1.0):
        self.ref_img = ref_img
        self.h, self.w = ref_img.shape[:2]
        if init_mask is None:
            self.mask = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            self.mask = init_mask.astype(np.float32) / 255.0
            if self.mask.shape != (self.h, self.w):
                self.mask = cv2.resize(
                    self.mask, (self.w, self.h), interpolation=cv2.INTER_LINEAR
                )
        self.brush_radius = brush_radius
        self.alpha = alpha
        self.gradient_mode = True
        self.drawing = False
        self.erasing = False
        self.last_point = None

    def _apply_brush(self, x, y, erase=False):
        rr = self.brush_radius
        # A drag leaving the window reports coordinates outside the image:
        # keep the window valid (x1 >= x0), or a negative end index sliced a
        # different region than the brush grid and the callback raised.
        x0 = min(max(0, x - rr), self.w)
        x1 = max(x0, min(self.w, x + rr + 1))
        y0 = min(max(0, y - rr), self.h)
        y1 = max(y0, min(self.h, y + rr + 1))
        if x1 == x0 or y1 == y0:
            return
        patch = self.mask[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        brush = dist <= rr
        if self.gradient_mode:
            # cosine falloff: 1 at center -> 0 at edge
            falloff = 0.5 * (np.cos(np.clip(dist / rr, 0, 1) * np.pi) + 1.0)
            falloff[dist > rr] = 0.0
        else:
            falloff = brush.astype(np.float32)
        if erase:
            patch[brush] = np.clip(patch[brush] - falloff[brush] * self.alpha, 0.0, 1.0)
        else:
            patch[brush] = np.clip(patch[brush] + falloff[brush] * self.alpha, 0.0, 1.0)
        self.mask[y0:y1, x0:x1] = patch

    def handle_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erasing = False
            self._apply_brush(x, y, erase=False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erasing = True
            self._apply_brush(x, y, erase=True)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._apply_brush(x, y, erase=self.erasing)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False

    def render_overlay(self):
        # Overlay mask as heatmap on reference image for visualization
        colored = cv2.applyColorMap(
            (self.mask * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(self.ref_img, 0.6, colored, 0.4, 0)
        cv2.putText(
            overlay,
            f"Brush: {self.brush_radius} Gradient: {'ON' if self.gradient_mode else 'OFF'}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Left drag: paint | Right drag: erase | g: toggle gradient | +/-: brush size | s: save | q: quit",
            (10, self.h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def get_mask_uint8(self):
        return (np.clip(self.mask, 0, 1) * 255).astype(np.uint8)


def _default_output_path(input_path: str) -> str:
    # Always PNG: a mask next to a .jpg input was saved as a lossy JPEG,
    # whose compression noise then weighted the loss.
    root, _ext = os.path.splitext(input_path)
    return f"{root}_mask.png"


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image '{args.input}' not found")
    ref = imread(args.input, cv2.IMREAD_COLOR)
    if ref is None:
        raise RuntimeError("Failed to load input image.")
    init_mask = None
    if args.initial and os.path.exists(args.initial):
        init_mask = imread(args.initial, cv2.IMREAD_GRAYSCALE)

    editor = PriorityMaskEditor(
        ref, init_mask, brush_radius=args.brush, alpha=args.alpha
    )
    win_name = "Priority Mask Editor"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, editor.handle_event)

    # Resolve output path
    out_path = args.output if args.output else _default_output_path(args.input)
    if not args.output:
        print(f"--output not provided. Will save to: {out_path}")

    while True:
        overlay = editor.render_overlay()
        cv2.imshow(win_name, overlay)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            imwrite(out_path, editor.get_mask_uint8())
            print(f"Saved mask to {out_path}")
            break
        elif key == ord("s"):
            imwrite(out_path, editor.get_mask_uint8())
            print(f"Saved mask to {out_path}")
        elif key == ord("+") or key == ord("="):
            editor.brush_radius = min(editor.brush_radius + 5, 1000)
        elif key == ord("-"):
            editor.brush_radius = max(editor.brush_radius - 5, 1)
        elif key == ord("g"):
            editor.gradient_mode = not editor.gradient_mode
        elif key == ord("c") or key == ord("r"):
            editor.mask[:] = 0.0
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
