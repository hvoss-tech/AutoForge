from __future__ import annotations
import json
import csv
import os
import threading
import tempfile
import shutil
import uuid
import logging
from typing import Any, Optional
from ..models import Filament

logger = logging.getLogger(__name__)

# Spellings of each Filament field in the files people import: our own
# export (snake/camelCase), HueForge's CSV header and HueForge's
# personal_library.json (PascalCase, "Transmissivity"). Only the first
# spelling present is used.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "brand": ("brand", "Brand"),
    "name": ("name", "Name"),
    "color": ("color", "Color"),
    "td": ("td", "TD", "Transmissivity", "transmissivity"),
    "owned": ("owned", "Owned"),
    "uuid": ("uuid", "UUID", "Uuid"),
    "filament_type": ("filament_type", "filamentType", "Type", "type"),
    "source": ("source",),
}


def _parse_owned(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


def normalize_filament_record(item: dict) -> dict:
    """Map an imported record's keys onto ``Filament``'s field names.

    ``Filament(**item)`` only knows snake_case/camelCase keys and silently
    ignores anything else, so a HueForge ``personal_library.json`` entry
    (``Brand``/``Name``/``Color``/``Transmissivity``) was imported as a blank
    white filament with no name and TD 0 — while reporting success."""
    out: dict[str, Any] = {}
    for field, aliases in _FIELD_ALIASES.items():
        for key in aliases:
            value = item.get(key)
            if value is None or (isinstance(value, str) and not value.strip() and field != "uuid"):
                continue
            out[field] = _parse_owned(value) if field == "owned" else value
            break
    return out


def looks_like_filament(item: dict) -> bool:
    """At least one of the fields that identify a filament is present."""
    return any(k in item for f in ("brand", "name", "color", "td") for k in _FIELD_ALIASES[f])


CATALOG_SOURCE = "filamentcolors"


def catalog_uuid(swatch_id: int) -> str:
    """Stable uuid for a filament added from the filamentcolors.xyz catalog,
    so adding the same swatch twice finds the first copy."""
    return f"filamentcolors-{int(swatch_id)}"


def _atomic_write_json(path: str, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", dir=parent or None, delete=False, suffix=".tmp")
    try:
        json.dump(data, tmp, indent=2)
        tmp.close()
        shutil.move(tmp.name, path)
    except BaseException:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


class FilamentService:
    def __init__(self, library_path: str = "filament_library"):
        self._lock = threading.RLock()
        self._filaments: dict[str, Filament] = {}
        self._active: dict[str, Filament] = {}
        self._library_path = library_path
        self._load_library()

    def _library_file(self) -> str:
        return os.path.join(self._library_path, "library.json")

    def _active_file(self) -> str:
        return os.path.join(self._library_path, "active.json")

    def _imported_marker_file(self) -> str:
        return os.path.join(self._library_path, "user_imported.marker")

    def _load_library(self):
        with self._lock:
            path = self._library_file()
            self._filaments = self._read_entries(path)
            self._active = self._read_entries(self._active_file())

    @staticmethod
    def _read_entries(path: str) -> dict[str, Filament]:
        """One invalid entry (a hand-edited file, an older version's field
        types, a color the model now rejects) used to raise out of
        ``FilamentService.__init__`` — and with it every filament route, the
        server's startup library seeding, and so the whole app. It is
        skipped instead, and the rest of the library still loads."""
        entries: dict[str, Filament] = {}
        if not os.path.exists(path):
            return entries
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError, UnicodeDecodeError):
            return entries
        if not isinstance(data, list):
            return entries
        for item in data:
            try:
                f = Filament(**item)
            except (ValueError, TypeError) as e:
                logger.warning("Skipping invalid filament entry in %s: %s", path, e)
                continue
            entries[f.uuid] = f
        return entries

    def _save_library(self):
        os.makedirs(self._library_path, exist_ok=True)
        data = [f.model_dump(by_alias=True) for f in self._filaments.values()]
        _atomic_write_json(self._library_file(), data)

        active_data = [f.model_dump(by_alias=True) for f in self._active.values()]
        _atomic_write_json(self._active_file(), active_data)

    def list(self) -> list[Filament]:
        with self._lock:
            return list(self._filaments.values())

    def create(self, filament: Filament) -> Filament:
        with self._lock:
            if not filament.uuid:
                filament.uuid = str(uuid.uuid4())
            self._filaments[filament.uuid] = filament
            self._save_library()
            return filament

    def update(self, uuid_: str, filament: Filament) -> Optional[Filament]:
        with self._lock:
            if uuid_ not in self._filaments:
                return None
            filament.uuid = uuid_
            self._filaments[uuid_] = filament
            # Active entries are independent copies, so a library edit
            # (e.g. a new TD) otherwise keeps silently driving runs and
            # renders with the old value until the user re-picks it.
            if uuid_ in self._active:
                self._active[uuid_] = filament
            self._save_library()
            return filament

    def delete(self, uuid_: str) -> bool:
        with self._lock:
            if uuid_ not in self._filaments:
                return False
            del self._filaments[uuid_]
            # Deleting from the library must retire it everywhere: an
            # active copy left behind keeps being used by the next run.
            self._active.pop(uuid_, None)
            self._save_library()
            return True

    def get_types(self) -> list[str]:
        with self._lock:
            types = set()
            for f in self._filaments.values():
                if f.filament_type:
                    types.add(f.filament_type)
            return sorted(types)

    def get_brands(self, filament_type: Optional[str] = None) -> list[str]:
        with self._lock:
            brands = set()
            for f in self._filaments.values():
                if f.brand and (not filament_type or f.filament_type == filament_type):
                    brands.add(f.brand)
            return sorted(brands)

    def _normalize_csv_row(self, reader: csv.DictReader) -> list[dict]:
        original_fieldnames = reader.fieldnames or []
        stripped = [h.strip() for h in original_fieldnames]
        rows = []
        for row in reader:
            values = list(row.values())
            normalized = {}
            for i, h in enumerate(stripped):
                normalized[h] = values[i] if i < len(values) else None
            rows.append(normalized)
        return rows

    def _parse_csv(self, content: str) -> list[Filament]:
        reader = csv.DictReader(content.splitlines())
        rows = self._normalize_csv_row(reader)
        parsed = []
        for row in rows:
            record = normalize_filament_record(row)
            td_str = record.get("td", "")
            try:
                record["td"] = float(td_str) if td_str not in ("", None) else 0.0
            except (ValueError, TypeError):
                record["td"] = 0.0
            # Leave uuid blank when the source doesn't have one —
            # merge_import()/replace_library() are responsible for
            # assigning one, since merge_import needs to know
            # "no uuid was given" to try matching by name instead.
            record["uuid"] = record.get("uuid") or ""
            parsed.append(Filament(**record))
        return parsed

    def _match_key(self, f: Filament) -> tuple[str, str]:
        return (f.brand.strip().lower(), f.name.strip().lower())

    def merge_import(self, filaments: list[Filament]) -> list[Filament]:
        """Add filaments into the existing library. An incoming filament
        whose (brand, name) matches an existing one overwrites it in place
        (keeping its uuid, so active-filament references and slider
        assignments referencing that uuid keep working) instead of adding a
        duplicate entry — this is what re-importing the same CSV/updated
        pricing sheet/etc. is expected to do."""
        with self._lock:
            existing_by_key = {self._match_key(f): f.uuid for f in self._filaments.values()}
            imported = []
            for f in filaments:
                key = self._match_key(f)
                if key in existing_by_key:
                    f.uuid = existing_by_key[key]
                elif not f.uuid:
                    f.uuid = str(uuid.uuid4())
                self._filaments[f.uuid] = f
                imported.append(f)
            self._refresh_active_locked()
            self._save_library()
            return imported

    def _refresh_active_locked(self) -> None:
        """Active entries are copies; point every one whose uuid is (still)
        in the library at the library's current version. update() did this
        for a single edit, but re-importing a CSV with corrected TDs/colors
        left the active copies — which is what runs and renders read — on
        the old values until each filament was re-picked by hand."""
        for u in list(self._active):
            if u in self._filaments:
                self._active[u] = self._filaments[u]

    def replace_library(self, filaments: list[Filament]) -> list[Filament]:
        """Replace the entire filament library with exactly these
        filaments — everything not in this list is removed."""
        with self._lock:
            self._filaments = {}
            imported = []
            for f in filaments:
                if not f.uuid:
                    f.uuid = str(uuid.uuid4())
                self._filaments[f.uuid] = f
                imported.append(f)
            # Same reason delete()/update() prune _active: an active entry
            # whose uuid no longer exists in the library is a ghost that
            # get_active() keeps returning — a "replace library" import
            # (fresh uuids for anything the source didn't carry one for)
            # otherwise orphaned every previously-active filament, and a run
            # right after would silently use their stale colors/TD.
            self._active = {u: f for u, f in self._active.items() if u in self._filaments}
            self._refresh_active_locked()
            self._save_library()
            return imported

    def import_csv(self, content: str, mode: str = "merge", _mark_user_import: bool = True) -> list[Filament]:
        with self._lock:
            parsed = self._parse_csv(content)
            imported = self.replace_library(parsed) if mode == "replace" else self.merge_import(parsed)
            if _mark_user_import:
                self._mark_user_imported()
            return imported

    def import_json(self, data: list[dict], mode: str = "merge") -> list[Filament]:
        with self._lock:
            parsed = [Filament(**normalize_filament_record(item)) for item in data]
            imported = self.replace_library(parsed) if mode == "replace" else self.merge_import(parsed)
            self._mark_user_imported()
            return imported

    @staticmethod
    def _catalog_key(brand: str, name: str, filament_type: str) -> tuple[str, str, str]:
        return (brand.strip().lower(), name.strip().lower(), filament_type.strip().lower())

    def catalog_matches(self, entries: list[dict]) -> dict[int, Filament]:
        """Swatch id -> the library filament that already is that catalog
        entry: the one added from it before (its uuid), or one of the user's
        own with the same brand, name and type (e.g. imported from HueForge)."""
        with self._lock:
            by_key = {self._catalog_key(f.brand, f.name, f.filament_type): f for f in self._filaments.values()}
            matches: dict[int, Filament] = {}
            for entry in entries:
                match = self._filaments.get(catalog_uuid(entry["id"])) or by_key.get(
                    self._catalog_key(entry["brand"], entry["name"], entry["filament_type"])
                )
                if match:
                    matches[entry["id"]] = match
            return matches

    def add_catalog_entries(self, entries: list[dict], owned: bool = True) -> tuple[list[Filament], list[Filament]]:
        """Add filamentcolors.xyz catalog entries to the library.
        Returns ``(added, already_in_library)``; entries already there are
        left untouched, so the user's own edits (a re-measured TD) survive."""
        with self._lock:
            added: list[Filament] = []
            existing: list[Filament] = []
            matches = self.catalog_matches(entries)
            for entry in entries:
                match = matches.get(entry["id"])
                if match:
                    existing.append(match)
                    continue
                f = Filament(
                    brand=entry["brand"],
                    name=entry["name"],
                    color=entry["color"],
                    td=entry["td"],
                    owned=owned,
                    uuid=catalog_uuid(entry["id"]),
                    filament_type=entry["filament_type"],
                    source=CATALOG_SOURCE,
                )
                self._filaments[f.uuid] = f
                added.append(f)
            if added:
                self._save_library()
                # The library panel hides non-"user" filaments until a
                # library of the user's own was loaded.
                self._mark_user_imported()
            return added, existing

    def _mark_user_imported(self):
        path = self._imported_marker_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("1")

    def _hueforge_offered_marker_file(self) -> str:
        return os.path.join(self._library_path, "hueforge_offered.marker")

    def hueforge_offered(self) -> bool:
        """Whether the startup offer to import HueForge's library was shown
        already — it is made once per install, not on every page load."""
        with self._lock:
            return os.path.exists(self._hueforge_offered_marker_file())

    def mark_hueforge_offered(self) -> None:
        with self._lock:
            path = self._hueforge_offered_marker_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("1")

    def has_custom_library(self) -> bool:
        with self._lock:
            return os.path.exists(self._imported_marker_file())

    def get_active(self) -> list[Filament]:
        with self._lock:
            return list(self._active.values())

    def set_active(self, filament: Filament) -> Filament:
        with self._lock:
            self._active[filament.uuid] = filament
            self._save_library()
            return filament

    def replace_active(self, filaments: list[Filament]) -> list[Filament]:
        with self._lock:
            self._active = {f.uuid: f for f in filaments}
            self._save_library()
            return list(self._active.values())

    def remove_active(self, uuid_: str) -> bool:
        with self._lock:
            if uuid_ not in self._active:
                return False
            del self._active[uuid_]
            self._save_library()
            return True


_service: Optional[FilamentService] = None
_init_lock = threading.Lock()


def reset_service():
    global _service
    _service = None


def get_filament_service() -> FilamentService:
    global _service
    if _service is None:
        with _init_lock:
            if _service is None:
                from ..config import config
                _service = FilamentService(library_path=config.library_dir)
    return _service
