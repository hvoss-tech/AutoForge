"""The filamentcolors.xyz filament catalog.

AutoForge ships a snapshot of every filamentcolors.xyz swatch that has a
measured transmission distance (TD) in ``autoforge/data/filamentcolors_catalog.json``,
so searching it never touches the site. The snapshot is regenerated for each
release with ``generate_filamentcolors_library.sh`` (a full, rate-limited crawl
of the API). Between releases the webui asks the site, at most once a day,
whether anything changed and then fetches only the swatches published since
the snapshot (see ``fetch_new_entries``) — a handful of requests instead of a
full crawl.

Catalog entries are compact dicts:
``{"id", "brand", "name", "color_name", "filament_type", "type_detail",
"color", "td", "color_family", "available", "published", "url"}``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

SITE_URL = "https://filamentcolors.xyz"
API_VERSION_URL = f"{SITE_URL}/api/version/"
# The API's default order is newest-published first, which is what lets the
# incremental update stop as soon as it reaches swatches it already knows.
SWATCH_API_URL = f"{SITE_URL}/api/swatch/"
PAGE_SIZE = 100
TIMEOUT = 30
# Hard floor between any two requests to filamentcolors.xyz from this
# process — the version check, update walks, the release crawl, and
# concurrent threads alike — so we never put load on their API.
MIN_REQUEST_INTERVAL = 1.0
USER_AGENT = "AutoForge filament catalog (+https://github.com/hvoss-techfak/AutoForge)"

BUNDLED_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "filamentcolors_catalog.json",
)

# filamentcolors.xyz's color_parent codes.
COLOR_FAMILIES = {
    "RED": "Red",
    "RNG": "Orange",
    "YLW": "Yellow",
    "GRN": "Green",
    "BLU": "Blue",
    "PPL": "Purple",
    "PNK": "Pink",
    "BRN": "Brown",
    "BLK": "Black",
    "GRY": "Gray",
    "WHT": "White",
    "TRN": "Transparent",
}

FetchJson = Callable[[str], Any]


class RequestThrottle:
    """Spaces calls at least ``interval`` seconds apart (thread-safe)."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last: Optional[float] = None

    def wait(self) -> None:
        # Held while sleeping, so concurrent callers queue up one by one.
        with self._lock:
            if self._last is not None:
                remaining = self.interval - (self._clock() - self._last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last = self._clock()


_throttle = RequestThrottle(MIN_REQUEST_INTERVAL)


def fetch_json(url: str) -> Any:
    _throttle.wait()
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _nested_name(value: Any) -> str:
    return str(value.get("name") or "").strip() if isinstance(value, dict) else ""


def swatch_to_entry(swatch: dict) -> Optional[dict]:
    """A raw API swatch as a compact catalog entry, or None when it can't be
    used (no measured TD, unpublished, or no valid color)."""
    td = swatch.get("td")
    try:
        td = float(td) if td is not None else None
    except (TypeError, ValueError):
        td = None
    if not td or td <= 0 or swatch.get("published") is False:
        return None
    hex_color = str(swatch.get("hex_color") or "").strip().lstrip("#").lower()
    if len(hex_color) != 6 or any(c not in "0123456789abcdef" for c in hex_color):
        return None
    try:
        swatch_id = int(swatch["id"])
    except (KeyError, TypeError, ValueError):
        return None

    brand = _nested_name(swatch.get("manufacturer")) or "Unknown"
    color_name = str(swatch.get("color_name") or "").strip() or f"#{hex_color}"
    ftype = swatch.get("filament_type") or {}
    type_detail = _nested_name(ftype)
    # "PLA", "PETG", "ABS / ASA", ... — the product line (e.g. "Silk PLA",
    # "Panchroma Starlight") goes into the name so the library's type tabs
    # stay a short list.
    parent = _nested_name(ftype.get("parent_type")) if isinstance(ftype, dict) else ""
    filament_type = parent or type_detail or "PLA"
    name = color_name if not type_detail or type_detail == filament_type else f"{color_name} ({type_detail})"

    published = str(swatch.get("date_published") or swatch.get("date_added") or "")[:10]
    return {
        "id": swatch_id,
        "brand": brand,
        "name": name,
        "color_name": color_name,
        "filament_type": filament_type,
        "type_detail": type_detail or filament_type,
        "color": f"#{hex_color}",
        "td": round(td, 2),
        "color_family": COLOR_FAMILIES.get(str(swatch.get("color_parent") or ""), ""),
        "available": bool(swatch.get("is_available", True)),
        "published": published,
        "url": f"{SITE_URL}/swatch/{swatch_id}/",
    }


def entries_from_swatches(swatches: Iterable[dict]) -> list[dict]:
    """Usable entries, one per swatch id, in a stable order."""
    by_id: dict[int, dict] = {}
    for swatch in swatches:
        entry = swatch_to_entry(swatch)
        if entry:
            by_id[entry["id"]] = entry
    return sort_entries(by_id.values())


def sort_entries(entries: Iterable[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: (e["brand"].lower(), e["name"].lower(), e["id"]))


def newest_published(swatches: Iterable[dict]) -> str:
    """The newest publish date among *all* swatches (with or without a TD) —
    the point the incremental update walks back to."""
    return max(
        (str(s.get("date_published") or "")[:10] for s in swatches),
        default="",
    )


def make_catalog(entries: list[dict], db_last_modified: Any, newest: str) -> dict:
    return {
        "source": SITE_URL,
        "attribution": "Swatch data from filamentcolors.xyz, measured by its community.",
        "db_last_modified": db_last_modified,
        "newest_published": newest,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "count": len(entries),
        "filaments": entries,
    }


def empty_catalog() -> dict:
    return {"db_last_modified": None, "newest_published": "", "count": 0, "filaments": []}


def load_catalog(path: str) -> dict:
    """A catalog file, or an empty catalog when it's missing or unreadable —
    a broken update file must never take the webui down."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return empty_catalog()
    if not isinstance(data, dict) or not isinstance(data.get("filaments"), list):
        return empty_catalog()
    return data


def save_catalog(path: str, catalog: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=parent, delete=False, suffix=".tmp", encoding="utf-8")
    try:
        json.dump(catalog, tmp, indent=1, ensure_ascii=False)
        tmp.write("\n")
        tmp.close()
        # NamedTemporaryFile is 0600; the bundled catalog must be readable
        # by whoever runs an installed copy.
        os.chmod(tmp.name, 0o644)
        shutil.move(tmp.name, path)
    except BaseException:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def merge_entries(*lists: Iterable[dict]) -> list[dict]:
    """Later lists win for the same swatch id."""
    by_id: dict[int, dict] = {}
    for entries in lists:
        for entry in entries:
            by_id[entry["id"]] = entry
    return sort_entries(by_id.values())


def _page_url(page: int) -> str:
    return f"{SWATCH_API_URL}?page_size={PAGE_SIZE}&page={page}"


def download_all_swatches(
    fetch: FetchJson = fetch_json,
    delay: float = 2.0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Every swatch on the site (raw API records), page by page, pausing
    ``delay`` seconds between requests to stay polite."""
    swatches: list[dict] = []
    url: Optional[str] = _page_url(1)
    page = 0
    while url:
        if page:
            time.sleep(delay)
        data = fetch(url)
        page += 1
        swatches.extend(data.get("results") or [])
        url = data.get("next")
        if progress:
            progress(len(swatches), int(data.get("count") or 0))
    return swatches


def fetch_new_entries(
    known_ids: set[int],
    since: str,
    fetch: FetchJson = fetch_json,
    delay: float = 1.0,
    max_pages: int = 10,
) -> tuple[list[dict], str]:
    """Swatches published since ``since`` (YYYY-MM-DD) that aren't known yet.

    Walks the newest-first listing and stops at the first page that reaches
    swatches older than ``since`` (or that holds nothing new at all), so a
    normal check costs one or two requests. Returns ``(new entries with a TD,
    newest publish date seen)``."""
    found: list[dict] = []
    newest = since
    for page in range(1, max_pages + 1):
        if page > 1:
            time.sleep(delay)
        data = fetch(_page_url(page))
        results = data.get("results") or []
        newest = max([newest, newest_published(results)])
        reached_old = False
        any_unknown = False
        for swatch in results:
            published = str(swatch.get("date_published") or "")[:10]
            if since and published and published < since:
                reached_old = True
                continue
            try:
                sid = int(swatch.get("id"))
            except (TypeError, ValueError):
                continue
            if sid in known_ids:
                continue
            any_unknown = True
            entry = swatch_to_entry(swatch)
            if entry:
                found.append(entry)
        if reached_old or not any_unknown or not data.get("next"):
            break
    return found, newest


def generate_catalog(
    output: str = BUNDLED_CATALOG_PATH,
    fetch: Optional[FetchJson] = None,
    delay: float = 2.0,
    quiet: bool = False,
) -> dict:
    """Crawl the whole site and write a fresh catalog snapshot."""
    fetch = fetch or fetch_json
    version = fetch(API_VERSION_URL)

    def report(done: int, total: int) -> None:
        if not quiet:
            print(f"\r  downloaded {done}/{total} swatches", end="", flush=True)

    swatches = download_all_swatches(fetch, delay=delay, progress=report)
    if not quiet:
        print()
    entries = entries_from_swatches(swatches)
    if not entries:
        raise RuntimeError("filamentcolors.xyz returned no swatches with a TD — not overwriting the catalog")
    catalog = make_catalog(entries, version.get("db_last_modified"), newest_published(swatches))
    save_catalog(output, catalog)
    if not quiet:
        print(f"Wrote {len(entries)} filaments with a TD (of {len(swatches)} swatches) to {output}")
    return catalog


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate AutoForge's bundled filamentcolors.xyz catalog.")
    parser.add_argument("--output", default=BUNDLED_CATALOG_PATH, help="Where to write the catalog JSON.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between API pages (requests are never less than 1s apart, whatever this is).")
    args = parser.parse_args(argv)
    try:
        generate_catalog(args.output, delay=args.delay)
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as e:
        print(f"Catalog generation failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
