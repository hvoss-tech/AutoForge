from typing import Optional
from fastapi import APIRouter, HTTPException
from ..models import Filament
from ..services.filament_service import get_filament_service

router = APIRouter()


@router.get("")
def list_filaments(
    filament_type: Optional[str] = None,
    brand: Optional[str] = None,
    query: Optional[str] = None,
):
    svc = get_filament_service()
    filaments = svc.list()
    if filament_type:
        filaments = [f for f in filaments if f.filament_type == filament_type]
    if brand:
        filaments = [f for f in filaments if f.brand == brand]
    if query:
        q = query.lower()
        filaments = [
            f for f in filaments if q in f.brand.lower() or q in f.name.lower()
        ]
    return [f.model_dump() for f in filaments]


@router.post("")
def create_filament(filament: Filament):
    svc = get_filament_service()
    result = svc.create(filament)
    return result.model_dump()


@router.put("/active")
def replace_active_filaments(filaments: list[Filament]):
    """Set the active list to exactly these filaments. Undo/redo and project
    loading use this so the backend (which is what /api/optimize/start reads)
    can't drift from what the UI shows.

    Registered before ``PUT /{uuid}``, which would otherwise match first."""
    svc = get_filament_service()
    return [f.model_dump() for f in svc.replace_active(filaments)]


@router.put("/{uuid}")
def update_filament(uuid: str, filament: Filament):
    svc = get_filament_service()
    result = svc.update(uuid, filament)
    if not result:
        raise HTTPException(404, "Filament not found")
    return result.model_dump()


@router.delete("/{uuid}")
def delete_filament(uuid: str):
    svc = get_filament_service()
    if not svc.delete(uuid):
        raise HTTPException(404, "Filament not found")
    return {"ok": True}


@router.get("/types")
def list_filament_types():
    svc = get_filament_service()
    return svc.get_types()


@router.get("/brands")
def list_filament_brands(filament_type: Optional[str] = None):
    # The frontend sends the selected type tab's filter; without it the
    # tabbed brand lists showed every type's brands.
    svc = get_filament_service()
    return svc.get_brands(filament_type)


@router.post("/import-csv")
async def import_csv(body: dict | None = None, contents: str | None = None, mode: str = "merge"):
    """Accepts JSON body with a `contents` string key (and optional `mode`:
    "merge" (default, overwrites same brand+name entries in place) or
    "replace" (wipes the library first)), or `contents`/`mode` query params.
    """
    svc = get_filament_service()
    if contents is not None:
        pass
    elif body is not None and isinstance(body, dict):
        contents = body.get("contents", "")
        mode = body.get("mode", mode)
    else:
        raise HTTPException(400, "Missing 'contents' (send as JSON body or query param)")
    if not contents:
        raise HTTPException(400, "Empty contents")
    if mode not in ("merge", "replace"):
        raise HTTPException(400, "mode must be 'merge' or 'replace'")
    result = svc.import_csv(contents, mode=mode)
    return {
        "status": "ok",
        "message": f"Imported {len(result)} filaments" + (" (library replaced)" if mode == "replace" else ""),
        "count": len(result),
    }


@router.post("/import-json")
async def import_json(data: list[dict], mode: str = "merge"):
    svc = get_filament_service()
    if mode not in ("merge", "replace"):
        raise HTTPException(400, "mode must be 'merge' or 'replace'")
    for i, item in enumerate(data):
        try:
            Filament(**item)
        except ValueError as e:
            # Validate everything before touching the library, so a bad
            # entry can't leave a "replace" half-applied (or crash with 500).
            first = str(e).splitlines()[1:3]
            raise HTTPException(400, f"Entry {i + 1} is not a valid filament: {' '.join(s.strip() for s in first)}")
    result = svc.import_json(data, mode=mode)
    return {
        "status": "ok",
        "message": f"Imported {len(result)} filaments" + (" (library replaced)" if mode == "replace" else ""),
        "count": len(result),
    }


@router.get("/has-custom-library")
def has_custom_library():
    svc = get_filament_service()
    return {"exists": svc.has_custom_library()}


@router.get("/active")
def get_active_filaments():
    svc = get_filament_service()
    return [f.model_dump() for f in svc.get_active()]


@router.post("/active")
def add_active_filament(filament: Filament):
    svc = get_filament_service()
    result = svc.set_active(filament)
    return result.model_dump()


@router.delete("/active/{uuid}")
def remove_active_filament(uuid: str):
    svc = get_filament_service()
    if not svc.remove_active(uuid):
        raise HTTPException(404, "Active filament not found")
    return {"ok": True}
