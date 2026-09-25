"""The filamentcolors.xyz catalog as the webui serves it.

The bundled snapshot (``autoforge/data/filamentcolors_catalog.json``) is
read-only and ships with each release. Swatches published after it are
fetched in the background on startup and kept in
``<library_dir>/filamentcolors_updates.json``, so they survive restarts and
are only ever downloaded once per install."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

from ...Helper import filamentcolors_library as fc

logger = logging.getLogger(__name__)


class CatalogService:
    def __init__(
        self,
        library_path: str,
        bundled_path: str = fc.BUNDLED_CATALOG_PATH,
        min_check_interval: float = 3600.0,
    ):
        self._lock = threading.RLock()
        self._library_path = library_path
        self._bundled_path = bundled_path
        self._min_check_interval = min_check_interval
        self._merged: Optional[dict] = None
        self._updating = False
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def _updates_file(self) -> str:
        return os.path.join(self._library_path, "filamentcolors_updates.json")

    def _load(self) -> dict:
        bundled = fc.load_catalog(self._bundled_path)
        updates = fc.load_catalog(self._updates_file())
        entries = fc.merge_entries(bundled["filaments"], updates["filaments"])
        return {
            "db_last_modified": updates.get("db_last_modified") or bundled.get("db_last_modified"),
            "newest_published": max(bundled.get("newest_published") or "", updates.get("newest_published") or ""),
            "generated_at": bundled.get("generated_at"),
            "last_checked": updates.get("checked_at"),
            "added_since_release": len(updates["filaments"]),
            "count": len(entries),
            "filaments": entries,
        }

    def catalog(self) -> dict:
        with self._lock:
            if self._merged is None:
                self._merged = self._load()
            return self._merged

    def entries_by_id(self) -> dict[int, dict]:
        return {e["id"]: e for e in self.catalog()["filaments"]}

    def status(self) -> dict:
        with self._lock:
            return {"updating": self._updating, "error": self._last_error}

    def check_for_updates(
        self,
        fetch: fc.FetchJson = fc.fetch_json,
        now: Callable[[], float] = time.time,
        force: bool = False,
        delay: float = 1.0,
    ) -> int:
        """Fetch swatches published since the newest one we know. Returns how
        many new filaments (with a TD) were added. Never raises for network
        trouble — the catalog simply stays as it is."""
        with self._lock:
            if self._updating:
                return 0
            self._updating = True
            self._last_error = None
        try:
            return self._check(fetch, now, force, delay)
        except Exception as e:  # network, bad JSON, site changes — all non-fatal
            logger.info("filamentcolors.xyz update check failed: %s", e)
            with self._lock:
                self._last_error = str(e)
            return 0
        finally:
            with self._lock:
                self._updating = False

    def _check(self, fetch, now, force, delay) -> int:
        updates = fc.load_catalog(self._updates_file())
        checked_at = updates.get("checked_at") or 0
        if not force and now() - float(checked_at) < self._min_check_interval:
            return 0

        current = self.catalog()
        version = fetch(fc.API_VERSION_URL)
        remote_modified = version.get("db_last_modified")
        added: list[dict] = []
        newest = current["newest_published"]
        if remote_modified is None or remote_modified != current["db_last_modified"]:
            known = {e["id"] for e in current["filaments"]}
            added, newest = fc.fetch_new_entries(known, current["newest_published"], fetch=fetch, delay=delay)

        entries = fc.merge_entries(updates["filaments"], added)
        catalog = fc.make_catalog(entries, remote_modified, newest)
        catalog["checked_at"] = now()
        fc.save_catalog(self._updates_file(), catalog)
        with self._lock:
            self._merged = None
        if added:
            logger.info("Added %d new filaments from filamentcolors.xyz", len(added))
        return len(added)

    def start_background_update(self, **kwargs) -> threading.Thread:
        """Runs ``check_for_updates`` on a daemon thread (startup must not
        wait on the network, and a hung request must not block shutdown)."""
        thread = threading.Thread(
            target=self.check_for_updates, kwargs=kwargs, name="filamentcolors-update", daemon=True
        )
        self._thread = thread
        thread.start()
        return thread


_service: Optional[CatalogService] = None
_init_lock = threading.Lock()


def reset_catalog_service():
    global _service
    _service = None


def get_catalog_service() -> CatalogService:
    global _service
    if _service is None:
        with _init_lock:
            if _service is None:
                from ..config import config

                _service = CatalogService(
                    library_path=config.library_dir,
                    min_check_interval=config.filamentcolors_check_interval_hours * 3600.0,
                )
    return _service
