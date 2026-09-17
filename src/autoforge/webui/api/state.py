from typing import Optional

from fastapi import APIRouter, HTTPException
from ..models import StateSnapshot
from ..services.project_service import get_project_service

router = APIRouter()


@router.post("/snapshot")
async def save_snapshot(snapshot: StateSnapshot, discard_after: Optional[float] = None):
    """``discard_after``: the timestamp of the snapshot the client was on when
    it made this new edit. Everything newer is the redo branch the edit just
    abandoned — without deleting it here it came back (interleaved by time
    with the new branch) the next time the page hydrated its undo stack."""
    svc = get_project_service()
    if discard_after is not None:
        svc.delete_snapshots_after(discard_after)
    sid = svc.save_snapshot(snapshot)
    return {"snapshot_id": sid}


@router.post("/restore")
async def restore_snapshot(request: dict):
    timestamp = request.get("timestamp")
    snapshot_id = request.get("snapshot_id")
    svc = get_project_service()
    snapshot = svc.get_snapshot(str(snapshot_id)) if snapshot_id else None
    if snapshot is None and timestamp is not None:
        snapshot = svc.get_snapshot_by_timestamp(timestamp)
    if not snapshot:
        raise HTTPException(404, "Snapshot not found")

    return snapshot_to_client(snapshot)


def snapshot_to_client(snapshot: StateSnapshot) -> dict:
    """Top-level keys in camelCase (the frontend's ``Snapshot`` type), but
    nested filaments/sliders/settings in snake_case, matching the frontend's
    ``Filament``/``ColorSliderConfig``/``OptimizationSettings`` types.

    ``model_dump(by_alias=True)`` camelCases recursively, so after a page
    reload (which hydrates the undo stack from this endpoint) an undo applied
    e.g. ``settings.layerHeight``/``filamentUuid`` — the frontend then saw
    ``settings.input_image``/``layer_height`` and every slider's
    ``filament_uuid`` as undefined.
    """
    plain = snapshot.model_dump()
    return {StateSnapshot.model_fields[k].alias or k: v for k, v in plain.items()}


@router.delete("/history")
async def clear_snapshots():
    """Forget the undo/redo history (it otherwise survives reloads and
    restarts). The current project state itself is untouched."""
    svc = get_project_service()
    return {"deleted": svc.delete_snapshots_after(float("-inf"))}


@router.get("/history")
async def list_snapshots():
    svc = get_project_service()
    return [snapshot_to_client(s) for s in svc.list_snapshots()]
