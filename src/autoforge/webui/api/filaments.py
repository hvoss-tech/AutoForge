import json
import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..config import config
from ..models import Filament
from ..services.catalog_service import get_catalog_service
from ..services.filament_service import (
    get_filament_service,
    looks_like_filament,
    normalize_filament_record,
)

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
    try:
        result = svc.import_csv(contents, mode=mode)
    except ValueError as e:
        # A row the Filament model rejects (e.g. a color that isn't hex) —
        # parsing happens before the library is touched, so nothing changed.
        raise HTTPException(400, f"The CSV has an invalid row: {_first_error_line(e)}")
    return {
        "status": "ok",
        "message": f"Imported {len(result)} filaments" + (" (library replaced)" if mode == "replace" else ""),
        "count": len(result),
    }


def _first_error_line(e: ValueError) -> str:
    lines = str(e).splitlines()
    return " ".join(s.strip() for s in (lines[1:3] or lines[:1]))


def _import_filament_records(data, mode: str) -> dict:
    """Validate every record, then import them all (or nothing)."""
    svc = get_filament_service()
    if mode not in ("merge", "replace"):
        raise HTTPException(400, "mode must be 'merge' or 'replace'")
    # HueForge's personal_library.json is {"Filaments": [...]}, which is what
    # the CLI's --json_file reads.
    if isinstance(data, dict):
        data = data.get("Filaments", data.get("filaments", [data]))
    if not isinstance(data, list) or not data:
        raise HTTPException(400, "No filaments found in that file.")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not looks_like_filament(item):
            # Filament(**item) accepts any dict (unknown keys are ignored),
            # so a wrapper object or an unrelated JSON file used to import as
            # a nameless white filament and report success.
            raise HTTPException(400, f"Entry {i + 1} is not a filament (no brand, name, color or TD).")
        try:
            Filament(**normalize_filament_record(item))
        except ValueError as e:
            # Validate everything before touching the library, so a bad
            # entry can't leave a "replace" half-applied (or crash with 500).
            raise HTTPException(400, f"Entry {i + 1} is not a valid filament: {_first_error_line(e)}")
    result = svc.import_json(data, mode=mode)
    return {
        "status": "ok",
        "message": f"Imported {len(result)} filaments" + (" (library replaced)" if mode == "replace" else ""),
        "count": len(result),
    }


@router.post("/import-json")
async def import_json(data: list[dict] | dict, mode: str = "merge"):
    return _import_filament_records(data, mode)


def _read_hueforge_library() -> tuple[str | None, list | None, str | None]:
    """(path, filaments, error). ``filaments`` is None when the file is
    missing or unreadable; ``error`` says why for a file that exists."""
    path = config.hueforge_library_path
    if not path or not os.path.isfile(path):
        return path, None, None
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return path, None, f"HueForge's library file could not be read: {e}"
    records = data.get("Filaments") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return path, None, "HueForge's library file has no filament list."
    return path, records, None


@router.get("/hueforge-library")
def hueforge_library():
    """Whether HueForge's personal filament library exists on this machine
    (the server's — the webui normally runs where HueForge is installed),
    and whether the one-time startup offer to import it was made yet."""
    path, records, error = _read_hueforge_library()
    return {
        "found": records is not None,
        "path": path,
        "count": len(records) if records is not None else 0,
        "error": error,
        "offered": get_filament_service().hueforge_offered(),
    }


@router.post("/hueforge-library/offered")
def mark_hueforge_library_offered():
    get_filament_service().mark_hueforge_offered()
    return {"ok": True}


@router.post("/import-hueforge")
def import_hueforge(mode: str = "merge"):
    path, records, error = _read_hueforge_library()
    if error:
        raise HTTPException(400, error)
    if records is None:
        raise HTTPException(404, "HueForge's personal filament library was not found on this computer.")
    result = _import_filament_records(records, mode)
    get_filament_service().mark_hueforge_offered()
    return result


class CatalogAddRequest(BaseModel):
    ids: list[int]
    owned: bool = True
    activate: bool = False


@router.get("/catalog")
def filament_catalog():
    """Every filamentcolors.xyz filament with a measured TD (the bundled
    snapshot plus anything the background update found since). The whole
    list is small enough to send at once and search in the browser.
    ``library_uuids`` maps swatch id -> library uuid for the ones already in
    the user's library."""
    catalog_svc = get_catalog_service()
    catalog = catalog_svc.catalog()
    matches = get_filament_service().catalog_matches(catalog["filaments"])
    library_uuids = {str(i): f.uuid for i, f in matches.items()}
    return {
        **{k: v for k, v in catalog.items() if k != "filaments"},
        **catalog_svc.status(),
        "filaments": catalog["filaments"],
        "library_uuids": library_uuids,
    }


@router.post("/catalog/add")
def add_from_catalog(body: CatalogAddRequest):
    """Add catalog filaments to the library (and optionally make them
    active). Filaments already in the library are reported, not duplicated."""
    if not body.ids:
        raise HTTPException(400, "No filaments selected.")
    by_id = get_catalog_service().entries_by_id()
    missing = [i for i in body.ids if i not in by_id]
    if missing:
        raise HTTPException(404, f"Not in the catalog: {', '.join(map(str, missing))}")
    svc = get_filament_service()
    added, existing = svc.add_catalog_entries([by_id[i] for i in dict.fromkeys(body.ids)], owned=body.owned)
    if body.activate:
        for f in added + existing:
            svc.set_active(f)
    return {
        "added": [f.model_dump() for f in added],
        "existing": [f.model_dump() for f in existing],
        "active": [f.model_dump() for f in svc.get_active()],
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
