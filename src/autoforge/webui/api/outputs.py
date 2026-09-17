import io
import os
import zipfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from ..config import config
from ..services.optimization_service import get_optimization_service

router = APIRouter()

# Mirrors what the CLI leaves behind in its `--output-folder` after a run.
_EXPORT_FILES = [
    "final_model.stl",
    "final_model_colored.ply",
    "final_model.png",
    "swap_instructions.txt",
    "final_loss.txt",
    "project_file.hfp",
]


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


def discard_slider_edits(job_id: str) -> None:
    """Called when a job's real outputs are regenerated (pruning)."""
    for name in (EDITED_PLY, EDITED_PNG):
        path = _job_path(job_id, name)
        if path and os.path.exists(path):
            os.remove(path)


@router.get("/colored-ply/{job_id}")
async def download_colored_ply(job_id: str):
    """The mesh the 3D view shows: the user's slider-edited version when one
    exists, otherwise the optimizer's own."""
    edited = _job_path(job_id, EDITED_PLY)
    if edited and os.path.exists(edited):
        return FileResponse(edited, filename=f"{job_id}_colored.ply")
    path = _resolve_or_404(job_id, "final_model_colored.ply", "Colored PLY not found")
    return FileResponse(path, filename=f"{job_id}_colored.ply")


@router.get("/export/{job_id}")
async def export_project(job_id: str):
    """Zip up a completed job's output files — STL, colored PLY, preview
    PNG, swap instructions, project file — as one downloadable bundle,
    mirroring the CLI's `--output-folder` contents after a run."""
    svc = get_optimization_service()
    job = svc.get_job(job_id)
    if not job or job.status != "completed":
        raise HTTPException(400, "No completed optimization result to export")

    checkpoints = os.path.realpath(config.checkpoints_path)
    job_dir = os.path.realpath(os.path.join(checkpoints, job_id))
    if not job_dir.startswith(checkpoints + os.sep) or not os.path.isdir(job_dir):
        raise HTTPException(404, "Job output folder not found")

    buffer = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in _EXPORT_FILES:
            path = os.path.join(job_dir, filename)
            if os.path.exists(path):
                zf.write(path, arcname=filename)
                added += 1
    if added == 0:
        raise HTTPException(404, "No output files found for this job")

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_export.zip"'},
    )
