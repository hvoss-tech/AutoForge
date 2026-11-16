"""OrcaSlicer printer, process and filament profiles.

A slicer project (.3mf) has to carry the printer's complete configuration:
when OrcaSlicer opens one, every key the project leaves out falls back to
Orca's built-in defaults, not to the printer's system preset, so a project with
only a handful of keys would print with an empty start G-code.

The profiles are OrcaSlicer's own (AGPL-3.0). AutoForge does not ship them.
It ships only ``autoforge/data/orca_printers.json``, a small index of which
printers exist (generated from a profile checkout with ``python -m
autoforge.Helper.OrcaProfiles``). When a project is exported for a printer, the
few profile files its presets are built from are fetched from OrcaSlicer's
GitHub repository at the pinned ``ORCA_TAG`` and cached on disk, so each
printer needs the internet once.

``AUTOFORGE_ORCA_PROFILES`` points at a local ``resources/profiles`` directory
instead (offline use, tests). ``AUTOFORGE_ORCA_CACHE`` moves the download cache.

Presets resolve the way Orca resolves them (PresetBundle::resolve_vendor_preset):
the parent chain named by ``inherits``, then the keys of every ``include`` in
order, then the preset's own keys on top.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

ORCA_TAG = "v2.5.0"
RAW_URL = "https://raw.githubusercontent.com/OrcaSlicer/OrcaSlicer/{tag}/resources/profiles/{path}"
USER_AGENT = "AutoForge printer profiles (+https://github.com/hvoss-techfak/AutoForge)"
TIMEOUT = 30
# Orca's base bundle: filaments of every vendor may inherit from it.
FILAMENT_LIBRARY = "OrcaFilamentLibrary"

INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orca_printers.json"
)

# Keys that describe a profile file rather than a setting.
_META_KEYS = frozenset({
    "name", "inherits", "include", "from", "instantiation", "type",
    "description", "setting_id", "renamed_from", "filament_id",
})
_KIND_LISTS = {
    "machine_model": "machine_model_list",
    "machine": "machine_list",
    "process": "process_list",
    "filament": "filament_list",
}
# Filament types AutoForge picks a matching preset for; anything else prints
# with the printer's default (PLA) filament preset.
FILAMENT_TYPES = ("PLA", "PETG", "ABS", "ASA", "TPU", "PLA-CF", "PETG-CF")


class ProfileError(Exception):
    """A preset doesn't exist or can't be resolved."""


class ProfileDownloadError(ProfileError):
    """A profile file couldn't be downloaded (no internet, GitHub down)."""


def _load_json_bytes(data: bytes, what: str) -> Any:
    try:
        # A few vendor files start with a UTF-8 BOM.
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"{what} is not valid JSON: {exc}") from exc


class LocalProfileSource:
    """Profiles from a ``resources/profiles`` directory on disk."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def read(self, path: str) -> Any:
        full = os.path.join(self.root, *path.split("/"))
        try:
            with open(full, "rb") as f:
                return _load_json_bytes(f.read(), path)
        except FileNotFoundError as exc:
            raise ProfileError(f"Profile file {path} not found in {self.root}") from exc

    def vendors(self) -> list[str]:
        return sorted(
            name[:-5] for name in os.listdir(self.root)
            if name.endswith(".json") and name != "blacklist.json"
            and os.path.isdir(os.path.join(self.root, name[:-5]))
        )


class GitHubProfileSource:
    """Profiles fetched from OrcaSlicer's repository at ``tag``, cached on disk.

    Different files download in parallel; two threads asking for the same
    file wait for one download."""

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, cache_dir: str, tag: str = ORCA_TAG):
        self.cache_dir = os.path.join(cache_dir, tag)
        self.tag = tag
        self.downloads = 0  # files fetched from GitHub (not the cache)

    @classmethod
    def _lock_for(cls, path: str) -> threading.Lock:
        with cls._locks_guard:
            return cls._locks.setdefault(path, threading.Lock())

    def _download(self, path: str) -> bytes:
        url = RAW_URL.format(tag=self.tag, path=urllib.parse.quote(path))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ProfileError(f"OrcaSlicer {self.tag} has no profile file {path}") from exc
            raise ProfileDownloadError(
                f"Couldn't download the OrcaSlicer profile {path} (HTTP {exc.code})"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ProfileDownloadError(
                f"Couldn't download the OrcaSlicer profile {path}. The first export for a "
                f"printer needs internet access to github.com ({exc})"
            ) from exc

    def read(self, path: str) -> Any:
        cached = os.path.join(self.cache_dir, *path.split("/"))
        if not os.path.exists(cached):
            with self._lock_for(cached):
                if not os.path.exists(cached):
                    data = self._download(path)
                    self.downloads += 1
                    _load_json_bytes(data, path)  # don't cache an error page
                    os.makedirs(os.path.dirname(cached), exist_ok=True)
                    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cached), suffix=".part")
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
                    os.replace(tmp, cached)
        with open(cached, "rb") as f:
            return _load_json_bytes(f.read(), path)


def default_cache_dir() -> str:
    env = os.environ.get("AUTOFORGE_ORCA_CACHE")
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "autoforge", "orca-profiles")


_shared: dict[tuple[str, str], "OrcaProfiles"] = {}
_shared_lock = threading.Lock()


def shared_profiles() -> "OrcaProfiles":
    """One OrcaProfiles per profile source, kept for the process, so resolved
    presets (and a prefetch) carry over from one export to the next."""
    key = (os.environ.get("AUTOFORGE_ORCA_PROFILES") or "", default_cache_dir())
    with _shared_lock:
        if key not in _shared:
            _shared[key] = OrcaProfiles(default_source())
        return _shared[key]


def default_source():
    local = os.environ.get("AUTOFORGE_ORCA_PROFILES")
    if local:
        return LocalProfileSource(local)
    return GitHubProfileSource(default_cache_dir())


class OrcaProfiles:
    """Looks presets up by name and resolves them to flat configs."""

    def __init__(self, source=None):
        self.source = source if source is not None else default_source()
        self._indexes: dict[str, dict[str, dict[str, str]]] = {}
        self._resolved: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._raw: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()

    def index(self, vendor: str) -> dict[str, dict[str, str]]:
        """{kind: {preset name: sub_path}} from the vendor's ``<vendor>.json``."""
        with self._lock:
            if vendor in self._indexes:
                return self._indexes[vendor]
        # Read outside the lock: other presets keep loading meanwhile.
        data = self.source.read(f"{vendor}.json")
        index = {
            kind: {e["name"]: e["sub_path"] for e in data.get(key) or [] if "name" in e and "sub_path" in e}
            for kind, key in _KIND_LISTS.items()
        }
        with self._lock:
            return self._indexes.setdefault(vendor, index)

    def locate(self, vendor: str, kind: str, name: str) -> tuple[str, str]:
        """(vendor, sub_path) of a preset; filaments fall back to Orca's library."""
        vendors = [vendor]
        if kind == "filament" and vendor != FILAMENT_LIBRARY:
            vendors.append(FILAMENT_LIBRARY)
        for v in vendors:
            sub_path = self.index(v)[kind].get(name)
            if sub_path:
                return v, sub_path
        raise ProfileError(f"OrcaSlicer has no {kind} preset named {name!r} for {vendor}")

    def raw(self, vendor: str, kind: str, name: str) -> tuple[str, dict[str, Any]]:
        where, sub_path = self.locate(vendor, kind, name)
        key = (where, sub_path)
        with self._lock:
            if key in self._raw:
                return where, self._raw[key]
        data = self.source.read(f"{where}/{sub_path}")
        with self._lock:
            return where, self._raw.setdefault(key, data)

    def resolve(self, vendor: str, kind: str, name: str, _seen: tuple = ()) -> dict[str, Any]:
        """The preset's flat config (settings only, plus ``filament_id``)."""
        key = (vendor, kind, name)
        with self._lock:
            if key in self._resolved:
                return dict(self._resolved[key])
        if key in _seen:
            raise ProfileError(f"Preset {name!r} inherits from itself")
        where, data = self.raw(vendor, kind, name)
        seen = _seen + (key,)
        config: dict[str, Any] = {}
        parent = data.get("inherits")
        if parent:
            config.update(self.resolve(where, kind, parent, seen))
        includes = data.get("include") or []
        if isinstance(includes, str):
            includes = [includes]
        for inc in includes:
            config.update({k: v for k, v in self.resolve(where, kind, inc, seen).items() if k != "filament_id"})
        config.update({k: v for k, v in data.items() if k not in _META_KEYS})
        # Orca takes filament_id from the preset, else from its parent.
        if data.get("filament_id"):
            config["filament_id"] = data["filament_id"]
        with self._lock:
            self._resolved[key] = config
        return dict(config)

    def prefetch(self, printer: dict[str, Any]) -> None:
        """Load everything a project for ``printer`` (an index entry) needs:
        its machine, process and filament presets, in parallel - so the
        export itself doesn't wait for GitHub."""
        from concurrent.futures import ThreadPoolExecutor

        vendor = printer["vendor"]
        self.index(vendor)
        if vendor != FILAMENT_LIBRARY:
            self.index(FILAMENT_LIBRARY)
        jobs = [("machine", printer["name"]), ("process", printer["process"])]
        jobs += [("filament", n) for n in dict.fromkeys(printer["filaments"].values())]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(self.resolve, vendor, kind, name) for kind, name in jobs]:
                future.result()

    def is_instance(self, vendor: str, kind: str, name: str) -> bool:
        return str(self.raw(vendor, kind, name)[1].get("instantiation", "")).lower() == "true"


# ---------------------------------------------------------------- printer index

def as_list(value) -> list:
    """A vector option as a list; profiles also write them comma-joined."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [v.strip() for v in str(value).split(",")]


def nozzle_list(value) -> list[str]:
    """``nozzle_diameter`` as one entry per extruder; some profiles write
    ``["0.4,0.4"]``."""
    return [part.strip() for item in as_list(value) for part in str(item).split(",") if part.strip()]


# A picture of 0.04-0.1 mm colour layers needs a fine nozzle: pellet and
# large-format extruders (5 mm and up) can't print one at all.
MAX_NOZZLE_MM = 1.2


def _bed_rect(printable_area) -> list[float]:
    xs, ys = [], []
    for point in as_list(printable_area):
        x, _, y = str(point).partition("x")
        try:
            px, py = float(x), float(y)
        except ValueError:
            continue
        xs.append(px)
        ys.append(py)
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _first(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _diameter(cfg: dict[str, Any]) -> float | None:
    try:
        return round(float(_first(as_list(cfg.get("filament_diameter")), "")), 2)
    except (TypeError, ValueError):
        return None


def _filament_choices(profiles: OrcaProfiles, vendor: str, printer: str, default_filament: str) -> dict[str, str]:
    """{filament type: preset name} for the presets compatible with ``printer``.

    Only explicit ``compatible_printers`` lists count; Orca's library presets
    ("Generic PLA @System") have none and fit every printer, so they are the
    fallback. The printer's default filament wins for its own type."""
    found: dict[str, list[str]] = {}
    # 2.85 mm printers (Lulzbot, Ultimaker...) must not get 1.75 mm presets.
    diameter = None
    try:
        diameter = _diameter(profiles.resolve(vendor, "filament", default_filament))
    except ProfileError:
        pass

    def consider(v: str, names):
        for name in names:
            try:
                if not profiles.is_instance(v, "filament", name):
                    continue
                cfg = profiles.resolve(v, "filament", name)
            except ProfileError:
                continue
            compat = cfg.get("compatible_printers") or []
            if v != FILAMENT_LIBRARY and printer not in compat:
                continue
            if v == FILAMENT_LIBRARY and (compat or cfg.get("compatible_printers_condition")):
                continue
            if diameter is not None and _diameter(cfg) not in (None, diameter):
                continue
            ftype = str(_first(cfg.get("filament_type"), "")).upper()
            if ftype in FILAMENT_TYPES:
                found.setdefault(ftype, []).append(name)

    consider(vendor, profiles.index(vendor)["filament"])
    if vendor != FILAMENT_LIBRARY:
        consider(FILAMENT_LIBRARY, [n for n in profiles.index(FILAMENT_LIBRARY)["filament"] if n.endswith("@System")])

    library = set(profiles.index(FILAMENT_LIBRARY)["filament"]) if vendor != FILAMENT_LIBRARY else set()

    def rank(name: str):
        # The printer's default, then the vendor's own presets for the
        # printer, then Orca's printer-independent library presets.
        lower = name.lower()
        return (name != default_filament, name in library, "generic" not in lower and "basic" not in lower, len(name), name)

    return {t: sorted(names, key=rank)[0] for t, names in found.items()}


def _default_process(profiles: OrcaProfiles, vendor: str, printer: str, cfg: dict[str, Any]) -> str | None:
    """The printer's default process, or else its closest compatible one.

    ``default_print_profile`` is sometimes a ``;`` list, sometimes names a
    preset that doesn't exist, and sometimes one that doesn't list the printer
    as compatible (OrcaSlicer then refuses the project); then the 0.20 mm-ish
    compatible preset wins."""

    def compatible(name: str) -> bool:
        try:
            if not profiles.is_instance(vendor, "process", name):
                return False
            process = profiles.resolve(vendor, "process", name)
        except ProfileError:
            return False
        listed = as_list(process.get("compatible_printers"))
        # A process with only a condition can't be checked here; trust it.
        return printer in listed if listed else True

    for name in str(cfg.get("default_print_profile") or "").split(";"):
        name = name.strip()
        if name and compatible(name):
            return name
    candidates = []
    for name in profiles.index(vendor)["process"]:
        try:
            if not profiles.is_instance(vendor, "process", name):
                continue
            if printer not in as_list(profiles.resolve(vendor, "process", name).get("compatible_printers")):
                continue
        except ProfileError:
            continue
        candidates.append(name)
    if not candidates:
        return None
    return sorted(candidates, key=lambda n: ("standard" not in n.lower(), "0.20mm" not in n, n))[0]


_ADD_RE = re.compile(r'this->add(?:_nullable)?\("([a-z0-9_]+)"')
_OVERRIDE_LIST_RE = re.compile(r"filament_extruder_override_keys\s*=\s*\{(.*?)\};", re.S)


_TYPE_RE = re.compile(r'this->add(?:_nullable)?\("([a-z0-9_]+)",\s*(co[A-Za-z]+)')


def extract_int_options(print_config_cpp: str) -> list[str]:
    """Integer options. Their "nil" is stored as the largest int, which
    OrcaSlicer's range check refuses even where the option is nullable."""
    with open(print_config_cpp, encoding="utf-8") as f:
        text = f.read()
    types = dict(_TYPE_RE.findall(text))
    ints = {k for k, t in types.items() if t in ("coInt", "coInts")}
    m = _OVERRIDE_LIST_RE.search(text)
    if m:  # filament overrides take their printer option's type
        for key in re.findall(r'"([a-z0-9_]+)"', m.group(1)):
            if types.get(key[len("filament_"):]) in ("coInt", "coInts"):
                ints.add(key)
    return sorted(ints)


def extract_nullable_options(print_config_cpp: str) -> list[str]:
    """Options that may hold "nil" (OrcaSlicer refuses it anywhere else):
    the filament overrides of printer options (filament_extruder_override_keys,
    declared with add_nullable in a loop) and those set ``def->nullable = true``."""
    with open(print_config_cpp, encoding="utf-8") as f:
        text = f.read()
    nullable = set()
    m = _OVERRIDE_LIST_RE.search(text)
    if m:
        nullable.update(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    current = None
    for line in text.splitlines():
        added = _ADD_RE.search(line)
        if added:
            current = added.group(1)
        elif current and re.search(r"def->nullable\s*=\s*true", line):
            nullable.add(current)
    return sorted(nullable)
_LIMIT_RE = re.compile(r"def->(min|max)\s*=\s*([^;]+);")


def extract_option_ranges(print_config_cpp: str) -> dict[str, list[float | None]]:
    """{option: [min, max]} from OrcaSlicer's PrintConfig.cpp - the ranges
    it refuses a project for ("not in range") when a value lies outside."""
    constants = {"max_temp": 1500.0}
    ranges: dict[str, list[float | None]] = {}
    current = None
    with open(print_config_cpp, encoding="utf-8") as f:
        for line in f:
            m = _ADD_RE.search(line)
            if m:
                current = m.group(1)
                continue
            m = _LIMIT_RE.search(line)
            if not m or current is None:
                continue
            raw = m.group(2).strip().rstrip("f")
            sign = -1.0 if raw.startswith("-") else 1.0
            name = raw.lstrip("-").strip()
            try:
                value = sign * (constants[name] if name in constants else float(name))
            except ValueError:
                continue
            entry = ranges.setdefault(current, [None, None])
            entry[0 if m.group(1) == "min" else 1] = value
    return ranges


_LIST_RE = re.compile(r"static std::vector<std::string> s_Preset_(print|filament|machine_limits|printer)_options\s*\{(.*?)\};", re.S)


def extract_preset_sections(preset_cpp: str) -> dict[str, list[str]]:
    """{"process"|"filament"|"machine": [option, ...]} - which preset each
    option belongs to, from OrcaSlicer's Preset.cpp."""
    with open(preset_cpp, encoding="utf-8") as f:
        text = re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    sections: dict[str, list[str]] = {"process": [], "filament": [], "machine": []}
    for kind, body in _LIST_RE.findall(text):
        target = {"print": "process", "filament": "filament"}.get(kind, "machine")
        sections[target].extend(re.findall(r'"([a-z0-9_]+)"', body))
    return sections


# Printers whose OrcaSlicer projects give the defaults: single-extruder ones
# without variants first, so per-extruder values come as one entry.
_DEFAULT_SOURCES = [
    "FLSun Q5 0.4 nozzle", "Voron 2.4 350 0.4 nozzle", "Prusa MK4 0.4 nozzle",
    "Elegoo Neptune 4 Max 0.4 nozzle", "Bambu Lab X1 Carbon 0.4 nozzle",
]


def extract_orca_defaults(profiles: "OrcaProfiles", index: dict[str, Any], sections: dict[str, list[str]],
                          orca_command: list[str], work_dir: str) -> dict[str, dict[str, Any]]:
    """Orca's default value of every option, per preset section.

    A project only holding what the presets set is not enough: OrcaSlicer
    takes a missing option from whatever it has loaded (another printer's
    bed exclusion area made every delta printer refuse to slice). OrcaSlicer's
    own project export has every option, with the built-in default where the
    presets set nothing - so each option's default is read from the export of
    a printer whose presets don't set it. Needs the OrcaSlicer CLI."""
    import subprocess
    import zipfile

    import numpy as np

    os.makedirs(work_dir, exist_ok=True)
    stl = os.path.join(work_dir, "box.stl")
    # A closed 10 mm cube (OrcaSlicer refuses a lone triangle).
    c = [[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)]
    faces = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
             (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    with open(stl, "wb") as f:
        f.write(bytes(80) + np.uint32(len(faces)).tobytes())
        for face in faces:
            f.write(np.zeros(3, np.float32).tobytes() + np.array([c[i] for i in face], np.float32).tobytes() + b"\0\0")
    by_name = {p["name"]: p for p in index["printers"]}
    sources = []
    for name in _DEFAULT_SOURCES:
        printer = by_name[name]
        files, own = [], {}
        for kind, preset in (("machine", printer["name"]), ("process", printer["process"]),
                             ("filament", printer["filaments"]["PLA"])):
            cfg = profiles.resolve(printer["vendor"], kind, preset)
            own[kind] = set(cfg)
            cfg.update({"type": kind, "name": preset, "from": "system", "instantiation": "true"})
            path = os.path.join(work_dir, f"{len(sources)}_{kind}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            files.append(path)
        out = os.path.join(work_dir, f"{len(sources)}.3mf")
        subprocess.run(orca_command + ["--load-settings", f"{files[0]};{files[1]}", "--load-filaments", files[2],
                                       "--export-3mf", os.path.basename(out), "--outputdir", work_dir, stl],
                       cwd=work_dir, capture_output=True, timeout=600, check=False)
        with zipfile.ZipFile(out) as z:
            exported = json.loads(z.read("Metadata/project_settings.config"))
        sources.append((own, exported))
    defaults: dict[str, dict[str, Any]] = {"process": {}, "filament": {}, "machine": {}}
    for kind, keys in sections.items():
        for key in keys:
            for own, exported in sources:
                if key in exported and key not in own[kind]:
                    defaults[kind][key] = exported[key]
                    break
    return defaults


def build_printer_index(profiles_dir: str, print_config_cpp: str | None = None) -> dict[str, Any]:
    """Index every printer preset Orca offers, from a ``resources/profiles`` checkout."""
    source = LocalProfileSource(profiles_dir)
    profiles = OrcaProfiles(source)
    printers = []
    for vendor in source.vendors():
        if vendor == FILAMENT_LIBRARY:
            continue
        try:
            idx = profiles.index(vendor)
        except ProfileError as exc:
            logger.warning("Skipping vendor %s: %s", vendor, exc)
            continue
        models = set(idx["machine_model"])
        for name in idx["machine"]:
            try:
                if not profiles.is_instance(vendor, "machine", name):
                    continue
                cfg = profiles.resolve(vendor, "machine", name)
            except ProfileError as exc:
                logger.warning("Skipping printer %s/%s: %s", vendor, name, exc)
                continue
            model = cfg.get("printer_model") or ""
            if model not in models:
                continue  # Orca ignores printers whose model the vendor doesn't list
            if str(cfg.get("belt_printer", "0")) == "1":
                continue  # a heightmap's layers can't be printed on a tilted belt
            process = _default_process(profiles, vendor, name, cfg)
            if process is None:
                logger.warning("Skipping printer %s: no process preset", name)
                continue
            nozzles = nozzle_list(cfg.get("nozzle_diameter"))
            nozzle = nozzles[0] if nozzles else str(cfg.get("printer_variant") or "")
            try:
                if min(float(n) for n in nozzles) > MAX_NOZZLE_MM:
                    logger.info("Skipping printer %s: %s mm nozzle is too coarse for colour layers", name, nozzle)
                    continue
            except ValueError:
                pass
            bed = _bed_rect(cfg.get("printable_area"))
            if bed[2] - bed[0] <= 0 or bed[3] - bed[1] <= 0:
                logger.warning("Skipping printer %s: no usable printable_area", name)
                continue
            default_filament = _first(cfg.get("default_filament_profile"), "")
            filaments = _filament_choices(profiles, vendor, name, default_filament)
            if "PLA" not in filaments:
                logger.warning("Skipping printer %s: no PLA filament preset", name)
                continue
            printers.append({
                "name": name,
                "vendor": vendor,
                "model": model,
                "nozzle": nozzle,
                "extruders": max(1, len(nozzles)),
                "bed": bed,
                "height": float(cfg.get("printable_height") or 0),
                "process": process,
                "filaments": filaments,
            })
    printers.sort(key=lambda p: (p["vendor"].lower(), p["model"].lower(), p["name"].lower()))
    if print_config_cpp is None:
        guess = os.path.join(profiles_dir, "..", "..", "src", "libslic3r", "PrintConfig.cpp")
        print_config_cpp = guess if os.path.exists(guess) else None
    ranges = extract_option_ranges(print_config_cpp) if print_config_cpp else {}
    nullable = extract_nullable_options(print_config_cpp) if print_config_cpp else []
    ints = extract_int_options(print_config_cpp) if print_config_cpp else []
    return {"orca_tag": ORCA_TAG, "printers": printers, "option_ranges": ranges, "nullable": nullable,
            "int_options": ints}


_index_cache: dict[str, Any] | None = None


def load_printer_index() -> dict[str, Any]:
    global _index_cache
    if _index_cache is None:
        with open(INDEX_PATH, encoding="utf-8") as f:
            _index_cache = json.load(f)
    return _index_cache


def find_printer(name: str) -> dict[str, Any]:
    for printer in load_printer_index()["printers"]:
        if printer["name"] == name:
            return printer
    raise ProfileError(f"Unknown printer {name!r}")


def _nozzle_mm(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_nozzle(a: Any, b: Any) -> bool:
    x, y = _nozzle_mm(a), _nozzle_mm(b)
    return x is not None and y is not None and abs(x - y) < 1e-3


def printer_for_nozzle(name: str, nozzle: float | None) -> tuple[dict[str, Any], str | None, str | None]:
    """The preset of printer ``name`` for a ``nozzle`` (mm): ``(printer,
    note, warning)``.

    A picked preset with another nozzle is swapped for the same printer's
    preset with this nozzle (``note`` says so). When the printer has no such
    preset the picked one stays and ``warning`` tells the user to set the
    nozzle by hand in the slicer. ``nozzle=None`` keeps the preset as is."""
    printer = find_printer(name)
    if nozzle is None or nozzle <= 0 or _same_nozzle(printer["nozzle"], nozzle):
        return printer, None, None
    siblings = [
        p for p in load_printer_index()["printers"]
        if p["vendor"] == printer["vendor"] and p["model"] == printer["model"]
    ]
    matches = [p for p in siblings if _same_nozzle(p["nozzle"], nozzle)]
    if matches:
        # Same naming as the picked preset ("... 0.4 nozzle" -> "... 0.6 nozzle"),
        # else the plainest one (no "HF" etc. variant).
        styled = printer["name"].replace(printer["nozzle"], "{n}", 1)
        same_style = [p for p in matches if styled.replace("{n}", p["nozzle"], 1) == p["name"]]
        chosen = (same_style or sorted(matches, key=lambda p: (len(p["name"]), p["name"])))[0]
        return chosen, f"Uses the {chosen['name']} preset to match the {nozzle:g} mm nozzle of this result.", None
    available = sorted({_nozzle_mm(p["nozzle"]) for p in siblings if _nozzle_mm(p["nozzle"]) is not None})
    have = ", ".join(f"{n:g}" for n in available)
    warning = (
        f"OrcaSlicer has no {nozzle:g} mm nozzle preset for the {printer['model']} (only {have} mm). "
        f"The project uses the {printer['nozzle']} mm preset: set the nozzle diameter to {nozzle:g} mm in "
        "OrcaSlicer's printer settings before slicing, or pick another printer."
    )
    return printer, None, warning


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate autoforge/data/orca_printers.json from an OrcaSlicer checkout "
        f"(resources/profiles at tag {ORCA_TAG})."
    )
    parser.add_argument("profiles_dir", help="Path to OrcaSlicer's resources/profiles")
    parser.add_argument("--out", default=INDEX_PATH)
    parser.add_argument(
        "--orca",
        default="flatpak run --filesystem={work} --command=orca-slicer com.orcaslicer.OrcaSlicer",
        help="Command that runs the OrcaSlicer CLI ({work} is the scratch directory), for the option defaults",
    )
    parser.add_argument(
        "--print-config",
        help="OrcaSlicer's src/libslic3r/PrintConfig.cpp, for the option ranges "
        "(default: next to the profiles in a checkout)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = build_printer_index(args.profiles_dir, args.print_config)
    if not index["option_ranges"]:
        parser.error("PrintConfig.cpp not found: pass --print-config (the option ranges are required)")
    preset_cpp = os.path.join(os.path.dirname(args.print_config or os.path.join(
        args.profiles_dir, "..", "..", "src", "libslic3r", "PrintConfig.cpp")), "Preset.cpp")
    sections = extract_preset_sections(preset_cpp)
    with tempfile.TemporaryDirectory(prefix="orca-defaults-") as work:
        command = [part.replace("{work}", work) for part in args.orca.split()]
        index["defaults"] = extract_orca_defaults(
            OrcaProfiles(LocalProfileSource(args.profiles_dir)), index, sections, command, work
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(index['printers'])} printers to {args.out}")


if __name__ == "__main__":
    main()
