import os
import sys
import time
from typing import Any

import numpy as np
import torch

from autoforge.Helper.DeviceUtils import (
    describe_device,
    mps_is_available,
    resolve_device,
)


def set_seed(args) -> Any:
    random_seed = args.random_seed
    if random_seed == 0:
        random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return random_seed


def perform_basic_check(args):
    # Basic checks. With a tolerance: in floating point 0.28 / 0.04 is
    # 7.000000000000001, so an exact is_integer() rejected valid settings.
    ratio = args.background_height / args.layer_height
    if abs(ratio - round(ratio)) > 1e-6:
        print(
            "Error: Background height must be a multiple of layer height.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.csv_file != "" and not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.json_file != "" and not os.path.exists(args.json_file):
        print(f"Error: Json file '{args.json_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.priority_mask != "" and not os.path.exists(args.priority_mask):
        print(
            f"Error: priority mask file '{args.priority_mask}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)
    if getattr(args, "priority_mask_strength", 10.0) < 1:
        print("Error: --priority_mask_strength must be at least 1.", file=sys.stderr)
        sys.exit(1)


def get_device(args=None) -> torch.device:
    """Select the torch device for a run.

    Auto-detects CUDA, ROCm (which PyTorch reports as CUDA) and Apple Metal,
    falling back to CPU. ``--device``/``AUTOFORGE_DEVICE`` override the choice;
    the legacy ``--mps`` flag is no longer needed - Metal is picked up on its
    own - but is still accepted so existing invocations keep working.
    """
    device = resolve_device(args=args)
    if (
        args is not None
        and getattr(args, "mps", False)
        and device.type != "mps"
        and mps_is_available()
    ):
        # Explicit --mps on a machine that also has a CUDA GPU: honor it.
        device = torch.device("mps")
    print("Using device:", describe_device(device))
    return device
