from __future__ import annotations
import json
import csv
import os
import threading
import tempfile
import shutil
import uuid
from typing import Optional
from ..models import Filament


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
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    for item in data:
                        f = Filament(**item)
                        self._filaments[f.uuid] = f
                except (json.JSONDecodeError, IOError):
                    pass

            active_path = self._active_file()
            if os.path.exists(active_path):
                try:
                    with open(active_path) as f:
                        data = json.load(f)
                    for item in data:
                        f = Filament(**item)
                        self._active[f.uuid] = f
                except (json.JSONDecodeError, IOError):
                    pass

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
            self._save_library()
            return filament

    def delete(self, uuid_: str) -> bool:
        with self._lock:
            if uuid_ not in self._filaments:
                return False
            del self._filaments[uuid_]
            self._save_library()
            return True

    def get_types(self) -> list[str]:
        with self._lock:
            types = set()
            for f in self._filaments.values():
                if f.filament_type:
                    types.add(f.filament_type)
            return sorted(types)

    def get_brands(self) -> list[str]:
        with self._lock:
            brands = set()
            for f in self._filaments.values():
                if f.brand:
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
            td_str = row.get("TD") or row.get("td") or row.get("Transmissivity", "")
            try:
                td_val = float(td_str) if td_str else 0.0
            except (ValueError, TypeError):
                td_val = 0.0
            parsed.append(
                Filament(
                    brand=row.get("Brand", row.get("brand", "")),
                    name=row.get("Name", row.get("name", "")),
                    color=row.get("Color", row.get("color", "#ffffff")),
                    td=td_val,
                    # Leave uuid blank when the source doesn't have one —
                    # merge_import()/replace_library() are responsible for
                    # assigning one, since merge_import needs to know
                    # "no uuid was given" to try matching by name instead.
                    uuid=row.get("UUID", row.get("uuid", "")) or "",
                    filament_type=row.get("Type", row.get("filament_type", "")),
                )
            )
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
            self._save_library()
            return imported

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
            parsed = [Filament(**item) for item in data]
            imported = self.replace_library(parsed) if mode == "replace" else self.merge_import(parsed)
            self._mark_user_imported()
            return imported

    def _mark_user_imported(self):
        path = self._imported_marker_file()
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
