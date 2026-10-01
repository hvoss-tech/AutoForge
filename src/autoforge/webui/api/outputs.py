import asyncio
import json
import logging
import os
import shutil
import re
import tempfile
import zipfile
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from ..config import config
from ..helpers.mesh_persist import wait_for_pending_mesh
from ..services.optimization_service import get_optimization_service

router = APIRouter()

def _job_path(job_id: str, filename: str) -> str | None:
    """Resolve a job's output file, refusing to escape checkpoints_path.

    Mirrors the traversal guard in ``services/image_service.py``. ``job_id``
    is attacker-controlled (a raw URL path segment), and unlike that
    service this module previously joined it straight into the filesystem
    path with no check.
    """
    checkpoints = os.path.realpath(config.checkpoints_path)
    resolved = os.path.realpath(os.path.join(checkpoints, job_id, filename))
    if not resolved.startswith(checkpoints + os.sep):
        return None
    return resolved


def _resolve_or_404(job_id: str, filename: str, not_found_msg: str) -> str:
    path = _job_path(job_id, filename)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, not_found_msg)
    return path


@router.get("/stl/{job_id}")
async def download_stl(job_id: str):
    path = _resolve_or_404(job_id, "final_model.stl", "STL not found")
    return FileResponse(path, filename=f"{job_id}.stl")


@router.get("/preview/{job_id}")
async def download_preview(job_id: str):
    path = _resolve_or_404(job_id, "final_model.png", "Preview not found")
    return FileResponse(path, filename=f"{job_id}_preview.png")


@router.get("/instructions/{job_id}")
async def download_instructions(job_id: str):
    path = _resolve_or_404(job_id, "swap_instructions.txt", "Instructions not found")
    return FileResponse(path, filename=f"{job_id}_instructions.txt")


@router.get("/project/{job_id}")
async def download_project(job_id: str):
    path = _resolve_or_404(job_id, "project_file.hfp", "Project file not found")
    return FileResponse(path, filename=f"{job_id}_project.hfp")


# Written by slider edits (api/preview.py) next to the optimizer's own
# outputs instead of over them: overwriting final_model_colored.ply /
# final_model.png meant an undo that briefly applied another state's sliders
# permanently replaced the real result, and the export zip mixed edited
# PNG/PLY with the unedited STL, swap instructions and project file.
EDITED_PLY = "edited_model_colored.ply"
EDITED_PNG = "edited_model.png"


# The mesh of a running prune's solution after its latest step (see
# api/pruning.py's _on_prune_step), written in the background under
# live_mesh_target(job_id).
LIVE_PLY = "live_model_colored.ply"


def live_mesh_target(job_id: str) -> str:
    return f"live:{job_id}"


@router.get("/live-ply/{job_id}")
async def download_live_ply(job_id: str):
    await asyncio.to_thread(wait_for_pending_mesh, live_mesh_target(job_id))
    path = _resolve_or_404(job_id, LIVE_PLY, "No live pruning mesh yet")
    return FileResponse(path, filename=f"{job_id}_live.ply")


def discard_slider_edits(job_id: str) -> None:
    """Called when a job's real outputs are regenerated (pruning)."""
    for name in (EDITED_PLY, EDITED_PNG):
        path = _job_path(job_id, name)
        if path and os.path.exists(path):
            os.remove(path)


@router.get("/current-preview/{job_id}")
async def current_preview(job_id: str):
    """The result image as it currently looks: slider-edited if edited,
    otherwise the optimizer's own. (/preview/{job_id} stays the optimizer's
    file, for downloads.)"""
    edited = _job_path(job_id, EDITED_PNG)
    if edited and os.path.exists(edited):
        return FileResponse(edited, media_type="image/png")
    path = _resolve_or_404(job_id, "final_model.png", "Preview not found")
    return FileResponse(path, media_type="image/png")


@router.get("/colored-ply/{job_id}")
async def download_colored_ply(job_id: str):
    """The mesh the 3D view shows: the user's slider-edited version when one
    exists, otherwise the optimizer's own."""
    # A live edit writes its mesh in the background; don't serve the one
    # from before it.
    await asyncio.to_thread(wait_for_pending_mesh, job_id)
    edited = _job_path(job_id, EDITED_PLY)
    if edited and os.path.exists(edited):
        return FileResponse(edited, filename=f"{job_id}_colored.ply")
    path = _resolve_or_404(job_id, "final_model_colored.ply", "Colored PLY not found")
    return FileResponse(path, filename=f"{job_id}_colored.ply")


# Files the export bundle may contain. The STL entry is special-cased below:
# a traditional run writes a single `final_model.stl`, while a FlatForge run
# writes one STL per material (`<material>_<hex>.stl`) and no final_model.stl
# at all — so the bundle has to take whatever STLs the job actually produced
# instead of assuming the traditional name.
_EXPORT_FILES = [
    "final_model_colored.ply",
    "final_model.png",
    "swap_instructions.txt",
    "final_loss.txt",
    "project_file.hfp",
]


def _export_files_for(job_dir: str) -> list[str]:
    """The file names a job's export bundle should contain.

    FlatForge runs write per-material STLs (``<material>_<hex>.stl``) and no
    ``final_model.stl``, so the single-STL assumption in ``_EXPORT_FILES``
    would have produced a bundle with *no* STL at all for them — the user
    could not download the print files from the UI. Listing the directory
    handles both shapes: the traditional single STL when it exists, and every
    per-material STL otherwise."""
    names = []
    final_stl = os.path.join(job_dir, "final_model.stl")
    if os.path.exists(final_stl):
        names.append("final_model.stl")
    else:
        for entry in sorted(os.listdir(job_dir)):
            if entry.lower().endswith(".stl") and os.path.isfile(os.path.join(job_dir, entry)):
                names.append(entry)
    return names + _EXPORT_FILES


@router.get("/stl-list/{job_id}")
async def list_stl_files(job_id: str):
    """Which STL file(s) a completed job actually produced: a single
    `final_model.stl` for a traditional run, or one per material for a
    FlatForge run (see `_export_files_for`). Lets the frontend offer each
    material's STL as its own download instead of only the bundled zip —
    the single `/stl/{job_id}` endpoint has nothing to serve for FlatForge."""
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job or job.status != "completed":
        raise HTTPException(400, "No completed optimization result")
    checkpoints = os.path.realpath(config.checkpoints_path)
    job_dir = os.path.realpath(os.path.join(checkpoints, job_id))
    if not job_dir.startswith(checkpoints + os.sep) or not os.path.isdir(job_dir):
        raise HTTPException(404, "Job output folder not found")
    return {"files": [f for f in _export_files_for(job_dir) if f.lower().endswith(".stl")]}


@router.get("/file/{job_id}/{filename}")
async def download_named_output(job_id: str, filename: str):
    """Download one of a job's own STL files by name.

    Only a name this job's own `/stl-list` would list is served — anything
    else (including an attempt to reach outside this job's directory, or a
    non-STL internal file) 404s rather than resolving arbitrary paths."""
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job or job.status != "completed":
        # Mirrors /stl-list/{job_id}'s guard: without it, a job that is
        # still running, failed, or an orphaned checkpoints/ directory could
        # have its .stl files served.
        raise HTTPException(400, "No completed optimization result")
    checkpoints = os.path.realpath(config.checkpoints_path)
    job_dir = os.path.realpath(os.path.join(checkpoints, job_id))
    if not job_dir.startswith(checkpoints + os.sep) or not os.path.isdir(job_dir):
        raise HTTPException(404, "Job output folder not found")
    if not filename.lower().endswith(".stl") or filename not in _export_files_for(job_dir):
        raise HTTPException(404, "File not found")
    path = _job_path(job_id, filename)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)


class ExportRequest(BaseModel):
    name: str | None = None
    # The webui project file (what File › Save project writes), bundled as
    # `<name>_project.json` so the zip holds everything needed to reopen it.
    project: dict[str, Any] | None = None


@router.post("/export/{job_id}")
async def export_project_with_file(job_id: str, body: ExportRequest):
    return await _export_zip_response(job_id, body.name, body.project)


@router.get("/export/{job_id}")
async def export_project(job_id: str, name: str | None = None):
    return await _export_zip_response(job_id, name, None)


async def _export_zip_response(job_id: str, name: str | None, project: dict[str, Any] | None):
    """Zip up a completed job's output files — STL (or the per-material STLs
    a FlatForge run produced), colored PLY, preview PNG, swap instructions,
    project file — as one downloadable bundle, mirroring the CLI's
    `--output-folder` contents after a run. ``name`` (the project name)
    names the zip and the files in it: `car_project.zip` with `car_model.stl`."""
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job or job.status != "completed":
        raise HTTPException(400, "No completed optimization result to export")

    checkpoints = os.path.realpath(config.checkpoints_path)
    job_dir = os.path.realpath(os.path.join(checkpoints, job_id))
    if not job_dir.startswith(checkpoints + os.sep) or not os.path.isdir(job_dir):
        raise HTTPException(404, "Job output folder not found")

    prefix = _export_prefix(name)
    fd, zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    edit_dir = tempfile.mkdtemp(prefix="autoforge_edit_")
    try:
        await asyncio.to_thread(wait_for_pending_mesh, job_id)
        replace = await asyncio.to_thread(_edited_replacements, job_id, job_dir, project, edit_dir)
        added = await asyncio.to_thread(_write_export_zip, zip_path, job_dir, prefix, project, replace)
    except BaseException:
        os.remove(zip_path)
        raise
    finally:
        shutil.rmtree(edit_dir, ignore_errors=True)
    if added == 0:
        os.remove(zip_path)
        raise HTTPException(404, "No output files found for this job")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{prefix}_project.zip" if prefix else f"{job_id}_export.zip",
        background=BackgroundTask(os.remove, zip_path),
    )


def _export_prefix(name: str | None) -> str:
    """The project name as a file-name prefix (same slug as the frontend's
    lib/project.ts), or "" when there is none."""
    return re.sub(r"[\W_]+", "-", (name or "").strip()).strip("-")[:80]


def _export_arcname(filename: str, prefix: str) -> str:
    """Name of a job file inside a project-named bundle: `car_model.stl`
    rather than `final_model.stl`. Per-material FlatForge STLs keep their
    material name after the prefix."""
    if not prefix:
        return filename
    renamed = {
        "final_model.stl": "model.stl",
        "final_model_colored.ply": "model_colored.ply",
        "final_model.png": "result.png",
        "final_loss.txt": "loss.txt",
        "project_file.hfp": "hueforge.hfp",
    }.get(filename, filename)
    return f"{prefix}_{renamed}"


def _edited_replacements(job_id: str, job_dir: str, project: dict[str, Any] | None, edit_dir: str) -> dict[str, str | None]:
    """Bundle entries to swap for a slider-edited result: {file name: path
    to use instead, or None to leave it out}.

    Slider edits write EDITED_PNG/EDITED_PLY next to the optimizer's files.
    When there are any, the bundle carries the edited image and mesh, and
    swap instructions and a .hfp regenerated from the edited stack (the
    project's colorSliders) - not the optimizer's, which name the colours
    the user replaced. If the result is no longer live (pruned since, or a
    server restart), those two can't be regenerated and are left out rather
    than shipped stale."""
    edited_png = os.path.join(job_dir, EDITED_PNG)
    if not os.path.exists(edited_png):
        return {}
    replace: dict[str, str | None] = {"final_model.png": edited_png}
    edited_ply = os.path.join(job_dir, EDITED_PLY)
    if os.path.exists(edited_ply):
        replace["final_model_colored.ply"] = edited_ply
    replace["swap_instructions.txt"] = None
    replace["project_file.hfp"] = None
    sliders = (project or {}).get("colorSliders")
    result = get_optimization_service().get_pipeline_result(job_id)
    if sliders and result:
        from ..helpers.edited_export import write_edited_outputs
        from ..services.filament_service import get_filament_service

        filaments = {f.uuid: f.model_dump() for f in get_filament_service().list()}
        for f in (project or {}).get("activeFilaments") or []:
            if isinstance(f, dict) and f.get("uuid"):
                filaments.setdefault(str(f["uuid"]), f)
        try:
            replace.update(write_edited_outputs(result, sliders, filaments, edit_dir))
        except Exception:
            logging.getLogger(__name__).exception("Could not regenerate the edited instructions for %s", job_id)
    return replace


def _write_export_zip(
    zip_path: str,
    job_dir: str,
    prefix: str,
    project: dict[str, Any] | None = None,
    replace: dict[str, str | None] | None = None,
) -> int:
    # Runs in a worker thread: the STL alone can be a few hundred MB, and
    # zipping it on the event loop froze every other request (and the
    # websocket) for the whole time. Level 1 is ~4x faster than the default
    # for ~2% larger output on these meshes. Written to a temp file rather
    # than a BytesIO — StreamingResponse iterated that line by line, which
    # took longer than the compression itself.
    added = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        replace = replace or {}
        for filename in _export_files_for(job_dir):
            path = replace.get(filename, os.path.join(job_dir, filename))
            if path is not None and os.path.exists(path):
                zf.write(path, arcname=_export_arcname(filename, prefix))
                added += 1
        if added and project is not None:
            project_name = f"{prefix}_project.json" if prefix else "autoforge-project.json"
            zf.writestr(project_name, json.dumps(project, indent=2))
    return added
