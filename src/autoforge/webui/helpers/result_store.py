"""A result's solution on disk, and the live result rebuilt from it.

Pruning, re-coloring and saving a project all work on a live pipeline result
(the ``FilamentOptimizer`` and everything around it). The server only keeps
one of those — the newest — and none survive a restart, so going back in
History, reloading after a restart or loading a project from another machine
left nothing to prune or re-color.

What a result *is*, though, fits in a few arrays: how many layers each pixel
has, which filament each layer is printed in, the filaments themselves and
the settings it was made with. Each finished run and prune saves exactly
that next to its files (``solution.npz``), and any of them can be turned
back into a full pipeline result whenever it is needed again — no training,
and no heightmap initialization either, since the heights are known.

A saved project file carries the same heightmap; its layer stack comes from
the color layers instead (see ``solution_from_sliders``).
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import cv2
import numpy as np
import torch

SOLUTION_FILE = "solution.npz"
SOLUTION_VERSION = 1


class ResultUnavailable(Exception):
    """A result that can't be brought back; the message is for the user."""


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def _args_dict(args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, np.generic):
            value = value.item()
        if _json_safe(value):
            out[key] = list(value) if isinstance(value, tuple) else value
    return out


def _materials_of(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The filaments the result's material indices refer to, in order."""
    active = result.get("active_filaments")
    if active:
        return [
            {
                "uuid": str(f.get("uuid", "")),
                "color": str(f["color"]),
                "td": float(f["td"]),
                "name": str(f.get("name", "")),
                "brand": str(f.get("brand", "")),
                "short_name": str(f.get("short_name", "")),
                "filament_type": str(f.get("filament_type") or f.get("filamentType") or "PLA"),
                "owned": bool(f.get("owned", False)),
            }
            for f in active
        ]
    colors = result.get("colors_list") or []
    tds = result.get("material_TDs_np")
    names = result.get("material_names") or []
    uuids = result.get("material_uuids") or []
    return [
        {
            "uuid": str(uuids[i]) if i < len(uuids) else "",
            "color": str(colors[i]),
            "td": float(tds[i]) if tds is not None else 1.0,
            "name": str(names[i]) if i < len(names) else "",
            "brand": str(names[i]).split(" - ", 1)[0].strip()
            if i < len(names) and " - " in str(names[i])
            else "",
            "short_name": str(names[i]).split(" - ", 1)[1].strip()
            if i < len(names) and " - " in str(names[i])
            else (str(names[i]) if i < len(names) else ""),
            "filament_type": "PLA",
            "owned": False,
        }
        for i in range(len(colors))
    ]


def solution_from_result(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The solution a live pipeline result holds, or None without one."""
    optimizer = result["optimizer"]
    stack, heights = optimizer.get_discretized_solution(best=True)
    if stack is None or heights is None:
        return None
    args = result["args"]
    heights_np = np.clip(heights.detach().cpu().numpy(), 0, 65535).astype(np.uint16)
    labels = getattr(optimizer, "pixel_height_labels", None)
    labels_np = None
    if isinstance(labels, torch.Tensor) and tuple(labels.shape) == heights_np.shape:
        labels_np = labels.detach().cpu().numpy().astype(np.int32)
    alpha = result.get("alpha")
    alpha_np = None
    if alpha is not None:
        alpha_np = np.asarray(alpha)
        if alpha_np.ndim == 3:
            alpha_np = alpha_np[:, :, 0]
        alpha_np = alpha_np.astype(np.uint8)
    index = getattr(args, "background_material_index", None)
    return {
        "heights": heights_np,
        "stack": stack.detach().cpu().numpy().astype(np.int32),
        "labels": labels_np,
        "alpha": alpha_np,
        "meta": {
            "version": SOLUTION_VERSION,
            "materials": _materials_of(result),
            "args": _args_dict(args),
            "background_color": str(getattr(args, "background_color", "#000000")),
            "background_material_index": int(index) if isinstance(index, (int, np.integer)) else None,
            "background_material_name": getattr(args, "background_material_name", None),
            "auto_background_color": bool(getattr(args, "auto_background_color", False)),
            "prune_runs": int(getattr(optimizer, "_prune_runs", 0) or 0),
        },
    }


def save_solution(job_dir: str, result: dict[str, Any]) -> bool:
    """Write ``result``'s solution into ``job_dir``. False when it has none."""
    solution = solution_from_result(result)
    if solution is None:
        return False
    os.makedirs(job_dir, exist_ok=True)
    arrays = {
        "heights": solution["heights"],
        "stack": solution["stack"],
        "meta": np.frombuffer(json.dumps(solution["meta"]).encode("utf-8"), dtype=np.uint8),
    }
    if solution["labels"] is not None:
        arrays["labels"] = solution["labels"]
    if solution["alpha"] is not None:
        arrays["alpha"] = solution["alpha"]
    path = os.path.join(job_dir, SOLUTION_FILE)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return True


def load_solution(job_dir: str) -> Optional[dict[str, Any]]:
    path = os.path.join(job_dir, SOLUTION_FILE)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            return {
                "heights": data["heights"].astype(np.int64),
                "stack": data["stack"].astype(np.int64),
                "labels": data["labels"] if "labels" in data.files else None,
                "alpha": data["alpha"] if "alpha" in data.files else None,
                "meta": meta,
            }
    except (OSError, ValueError, KeyError):
        return None


def solution_from_sliders(
    heights: np.ndarray,
    sliders: list[dict[str, Any]],
    active_filaments: list[dict[str, Any]],
    settings: dict[str, Any],
    background_color: Optional[str] = None,
    base: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """A solution made of a heightmap and the color layers painted onto it —
    a loaded project, or an auto-preview colored by hand."""
    from .slider_render import _dedupe_overlapping_layers, _layer_material_indices

    enabled = sorted(_dedupe_overlapping_layers(sliders), key=lambda s: int(s["layer"]))
    if not enabled:
        raise ResultUnavailable("Assign filaments to the color layers first — there is nothing to prune yet.")
    materials: list[dict[str, Any]] = []
    index_of: dict[str, int] = {}
    for f in active_filaments:
        uuid_ = str(f.get("uuid", ""))
        if uuid_ and uuid_ not in index_of and f.get("color"):
            index_of[uuid_] = len(materials)
            materials.append({
                "uuid": uuid_,
                "color": str(f["color"]),
                "td": float(f.get("td") or 1.0),
                "name": str(f.get("name", "")),
                "brand": str(f.get("brand", "")),
                "short_name": str(f.get("short_name", "")),
                "filament_type": str(f.get("filament_type") or f.get("filamentType") or "PLA"),
                "owned": bool(f.get("owned", False)),
            })
    slot_material = []
    for s in enabled:
        uuid_ = str(s.get("filament_uuid", ""))
        if uuid_ not in index_of:
            if not s.get("color"):
                raise ResultUnavailable("A color layer uses a filament that isn't active anymore; add it again first.")
            index_of[uuid_] = len(materials)
            materials.append({
                "uuid": uuid_,
                "color": str(s["color"]),
                "td": float(s.get("td") or 1.0),
                "name": "",
                "brand": "",
                "short_name": "",
                "filament_type": "PLA",
                "owned": False,
            })
        slot_material.append(index_of[uuid_])

    max_layer = int(heights.max(initial=0))
    stack_len = max(int(settings.get("max_layers") or 0), max_layer, int(enabled[-1]["layer"]), 1)
    slots = _layer_material_indices(enabled, stack_len)
    stack = np.asarray(slot_material, dtype=np.int64)[slots]

    color = background_color or str(settings.get("background_color") or "#000000")
    base_uuid = str((base or {}).get("filament_uuid") or "")
    base_index = index_of.get(base_uuid) if base_uuid else None
    if base_index is None:
        target = color.strip().lower()
        base_index = next((i for i, m in enumerate(materials) if m["color"].strip().lower() == target), None)
    args = {k: v for k, v in settings.items() if _json_safe(v)}
    return {
        "heights": heights.astype(np.int64),
        "stack": stack,
        "labels": None,
        "alpha": None,
        "meta": {
            "version": SOLUTION_VERSION,
            "materials": materials,
            "args": args,
            "background_color": color,
            "background_material_index": base_index,
            "background_material_name": materials[base_index]["name"] if base_index is not None else None,
            "auto_background_color": bool((base or {}).get("auto", settings.get("auto_background_color", False))),
            "prune_runs": 0,
        },
    }


def rebuild_pipeline_result(
    solution: dict[str, Any],
    input_image_path: str,
    output_dir: str,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """A live pipeline result holding exactly ``solution``, built like a run
    would build it (same image, sizes, focus map, materials) but with the
    saved heights in place of a heightmap initialization."""
    from autoforge.Helper.HeightAssign import heights_to_logits
    from autoforge.Helper.PruningHelper import disc_to_logits
    from .pipeline_runner import build_pipeline_state

    meta = solution["meta"]
    materials = meta["materials"]
    if not materials:
        raise ResultUnavailable("This result has no filaments to rebuild it with.")
    stack = np.asarray(solution["stack"], dtype=np.int64)
    if stack.size == 0 or stack.min() < 0 or stack.max() >= len(materials):
        raise ResultUnavailable("This result's saved layer stack is damaged.")

    settings = dict(meta.get("args") or {})
    for key in ("input_image", "output_folder", "csv_file", "json_file"):
        settings.pop(key, None)
    for key in ("priority_mask", "height_regions"):
        mask = settings.get(key) or ""
        if mask and not os.path.isfile(mask):
            settings[key] = ""
    # The base is what the result was built on: never picked again.
    settings["auto_background_color"] = False
    settings["optimize_background"] = False
    settings["background_color"] = meta.get("background_color") or "#000000"
    settings["tensorboard"] = False
    stack_len = int(stack.shape[0])
    settings["max_layers"] = max(int(settings.get("max_layers") or 0), stack_len)

    heights_saved = np.asarray(solution["heights"], dtype=np.int64)
    labels_saved = solution.get("labels")
    sizes: dict[str, Any] = {}

    def _fit(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if array.shape[:2] == shape:
            return array
        return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    def _heightmap_init(output_img_np: np.ndarray):
        shape = tuple(output_img_np.shape[:2])
        heights = np.clip(np.rint(_fit(heights_saved, shape)), 0, stack_len).astype(np.int64)
        labels = heights if labels_saved is None else np.rint(_fit(np.asarray(labels_saved), shape)).astype(np.int64)
        sizes["heights"] = heights
        sizes["labels"] = labels
        logits = heights_to_logits(torch.from_numpy(heights), stack_len).numpy().astype(np.float32)
        return logits, None, labels.astype(np.int32)

    result = build_pipeline_state(
        input_image_path, materials, output_dir, settings, device=device, heightmap_init=_heightmap_init,
    )
    optimizer = result["optimizer"]
    args = result["args"]
    device = result["device"]
    heights_t = torch.as_tensor(sizes["heights"], device=device)
    labels_t = torch.as_tensor(sizes["labels"], dtype=torch.int32, device=device)
    full_logits = heights_to_logits(heights_t, stack_len)
    stack_t = torch.as_tensor(stack, device=device)
    with torch.no_grad():
        optimizer.pixel_height_logits = full_logits.clone()
        optimizer.best_params = {
            "pixel_height_logits": full_logits,
            "global_logits": disc_to_logits(stack_t, len(materials)),
            "height_offsets": torch.zeros(int(labels_t.max().item()) + 1, 1, device=device),
        }
    optimizer.max_layers = stack_len
    optimizer.best_seed = 0
    optimizer.pixel_delta = None
    optimizer.target = result["output_target"]
    optimizer.pixel_height_labels = labels_t
    if result.get("focus_map_full") is not None and result.get("focus_map_proc") is not None:
        optimizer.focus_map = result["focus_map_full"]
    # Already at the output resolution: an export must not swap in the
    # (here meaningless) initial heights.
    optimizer._full_res_height_restored = True
    optimizer._prune_runs = int(meta.get("prune_runs") or 0)

    # The settings the result was made with, the base as it was resolved.
    args.max_layers = int((meta.get("args") or {}).get("max_layers") or stack_len)
    args.auto_background_color = bool(meta.get("auto_background_color", False))
    if meta.get("background_material_index") is not None:
        args.background_material_index = int(meta["background_material_index"])
    if meta.get("background_material_name"):
        args.background_material_name = meta["background_material_name"]
    alpha = solution.get("alpha")
    if alpha is not None and result.get("alpha") is None:
        result["alpha"] = _fit(np.asarray(alpha), tuple(sizes["heights"].shape))[..., None].astype(np.uint8)
    return result


# One rebuild at a time: two requests for the same result (a re-color and a
# base lookup right after a reload) must not build it twice on the GPU.
_rebuild_lock = threading.Lock()


class _UseLock:
    """Readers use live results; a writer replaces them. Registering a
    result releases every other one (OptimizationService.set_pipeline_result
    empties their dicts), so it waits until no request is still reading
    one: going back in History while the last color edit was still
    rendering otherwise pulled the result out from under that render."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writing = False

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writing:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            while self._writing or self._readers:
                self._cond.wait()
            self._writing = True
        try:
            yield
        finally:
            with self._cond:
                self._writing = False
                self._cond.notify_all()


_use = _UseLock()


def register_pipeline_result(job_id: str, result: dict[str, Any]) -> None:
    """``set_pipeline_result`` once no request is reading a result."""
    from ..services.optimization_service import get_optimization_service

    with _use.write():
        get_optimization_service().set_pipeline_result(job_id, result)


@contextmanager
def using_result(job_id: str) -> Iterator[dict[str, Any]]:
    """``resolve_pipeline_result``, kept live for the duration of the block."""
    from ..services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    for _attempt in range(3):
        result = resolve_pipeline_result(job_id)
        with _use.read():
            # Another result may have been registered in between.
            if result and svc.get_pipeline_result(job_id) is result:
                yield result
                return
    raise ResultUnavailable("The server is busy switching between results; try again.")


def resolve_pipeline_result(job_id: str, allow_while_job: Optional[str] = None) -> dict[str, Any]:
    """The live pipeline result of a completed job — rebuilt from its saved
    solution when the server no longer holds it (it was pruned since,
    another result replaced it, or the server restarted).

    ``allow_while_job`` names the one active job allowed to be running (the
    prune that asked); any other active job refuses a rebuild, which would
    compete with it for the GPU and evict the result it works on.

    Raises ResultUnavailable with a message for the user."""
    from ..config import config
    from ..services.image_service import get_image_service
    from ..services.optimization_service import get_optimization_service

    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if job is None or job.status != "completed":
        raise ResultUnavailable("Run an optimization first — there's no completed result to work on yet.")
    result = svc.get_pipeline_result(job_id)
    if result:
        return result
    with _rebuild_lock:
        result = svc.get_pipeline_result(job_id)
        if result:
            return result
        busy = svc.get_any_active_job()
        if busy is not None and busy.job_id != allow_while_job:
            raise ResultUnavailable(
                f"An optimization or pruning run is {busy.status}. Let it finish, then try again."
            )
        job_dir = os.path.join(config.checkpoints_path, job_id)
        solution = load_solution(job_dir)
        if solution is None:
            raise ResultUnavailable(
                "This result can't be brought back: the server no longer holds it (it was pruned since, "
                "or the server was restarted), and it was made before results were saved to disk. "
                "Go to the newest result, or run optimization first."
            )
        image_name = job.input_image or os.path.basename(str((solution["meta"].get("args") or {}).get("input_image") or ""))
        image_path = get_image_service().get_path(image_name) if image_name else None
        if image_path is None:
            raise ResultUnavailable("This result's image isn't on the server anymore; upload it again and rerun.")
        result = rebuild_pipeline_result(solution, image_path, job_dir)
        register_pipeline_result(job_id, result)
        return result
