"""The filamentcolors.xyz catalog helper: swatch conversion, the bundled
snapshot, the incremental update walk, the release crawl and the request
throttle. No test here touches the network — fetchers are fakes."""

import json
import os
import re
import subprocess
import sys

import pytest

from autoforge.Helper import filamentcolors_library as fc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def swatch(sid, td=2.5, published="2026-07-01", hex_color="aabbcc", brand="Acme", name="Red",
           ftype="PLA", parent="PLA", color_parent="RED", **extra):
    return {
        "id": sid,
        "manufacturer": {"id": 1, "name": brand},
        "color_name": name,
        "filament_type": {"id": 1, "name": ftype, "parent_type": {"name": parent}},
        "color_parent": color_parent,
        "hex_color": hex_color,
        "td": td,
        "date_published": f"{published}T10:00:00Z",
        "is_available": True,
        "published": True,
        **extra,
    }


class FakeApi:
    """Serves swatches newest-first in pages, like the real API."""

    def __init__(self, swatches, db_last_modified=100, page_size=fc.PAGE_SIZE):
        self.swatches = sorted(swatches, key=lambda s: (s["date_published"], s["id"]), reverse=True)
        self.db_last_modified = db_last_modified
        self.page_size = page_size
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url == fc.API_VERSION_URL:
            return {"db_version": 1, "db_last_modified": self.db_last_modified}
        page = int(re.search(r"page=(\d+)", url).group(1))
        start = (page - 1) * self.page_size
        chunk = self.swatches[start:start + self.page_size]
        more = start + self.page_size < len(self.swatches)
        return {
            "count": len(self.swatches),
            "next": fc._page_url(page + 1) if more else None,
            "results": chunk,
        }


# ---------------------------------------------------------------- conversion

def test_swatch_to_entry_maps_fields():
    e = fc.swatch_to_entry(swatch(7, td="3.25", hex_color="#E36F22", color_parent="RNG"))
    assert e == {
        "id": 7, "brand": "Acme", "name": "Red", "color_name": "Red", "filament_type": "PLA",
        "type_detail": "PLA", "color": "#e36f22", "td": 3.25, "color_family": "Orange",
        "available": True, "published": "2026-07-01", "url": "https://filamentcolors.xyz/swatch/7/",
    }


def test_product_line_goes_into_name_and_type_stays_the_parent():
    e = fc.swatch_to_entry(swatch(1, name="Galaxy Blue", ftype="Panchroma Starlight", parent="PLA"))
    assert e["filament_type"] == "PLA"
    assert e["type_detail"] == "Panchroma Starlight"
    assert e["name"] == "Galaxy Blue (Panchroma Starlight)"
    assert e["color_name"] == "Galaxy Blue"


@pytest.mark.parametrize("bad", [
    {"td": None}, {"td": 0}, {"td": -1}, {"td": "n/a"},
    {"hex_color": "xyz"}, {"hex_color": ""}, {"published": False}, {"id": None},
])
def test_unusable_swatches_are_skipped(bad):
    s = swatch(1)
    s.update(bad)
    assert fc.swatch_to_entry(s) is None


def test_missing_manufacturer_and_type_get_fallbacks():
    s = swatch(1)
    s["manufacturer"] = None
    s["filament_type"] = None
    s["color_parent"] = "???"
    e = fc.swatch_to_entry(s)
    assert (e["brand"], e["filament_type"], e["color_family"]) == ("Unknown", "PLA", "")


def test_entries_are_deduplicated_and_sorted():
    entries = fc.entries_from_swatches([
        swatch(3, brand="Zed"), swatch(1, brand="acme", name="b"), swatch(2, brand="Acme", name="a"),
        swatch(1, brand="acme", name="b"), swatch(4, td=None),
    ])
    assert [e["id"] for e in entries] == [2, 1, 3]


def test_merge_later_list_wins():
    a = fc.swatch_to_entry(swatch(1, td=1.0))
    b = fc.swatch_to_entry(swatch(1, td=2.0))
    c = fc.swatch_to_entry(swatch(2))
    merged = fc.merge_entries([a, c], [b])
    assert [(e["id"], e["td"]) for e in merged] == [(1, 2.0), (2, 2.5)]


# ----------------------------------------------------------- catalog files

def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "cat.json"
    cat = fc.make_catalog([fc.swatch_to_entry(swatch(1))], 5, "2026-07-01")
    fc.save_catalog(str(path), cat)
    assert fc.load_catalog(str(path)) == json.loads(json.dumps(cat))
    assert oct(os.stat(path).st_mode & 0o777) == "0o644"


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"filaments": 3}'])
def test_load_catalog_tolerates_missing_or_broken_files(tmp_path, content):
    path = tmp_path / "cat.json"
    if content is not None:
        path.write_text(content)
    assert fc.load_catalog(str(path)) == fc.empty_catalog()


def test_bundled_catalog_is_valid():
    """The shipped snapshot: only filaments with a TD, well-formed, unique."""
    assert os.path.isfile(fc.BUNDLED_CATALOG_PATH)
    cat = fc.load_catalog(fc.BUNDLED_CATALOG_PATH)
    entries = cat["filaments"]
    assert len(entries) > 300
    assert cat["count"] == len(entries)
    assert cat["db_last_modified"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cat["newest_published"])
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        assert e["td"] > 0
        assert re.fullmatch(r"#[0-9a-f]{6}", e["color"])
        assert e["brand"] and e["name"] and e["filament_type"]
        assert e["url"] == f"https://filamentcolors.xyz/swatch/{e['id']}/"
        assert e["color_family"] in set(fc.COLOR_FAMILIES.values()) | {""}


def test_bundled_catalog_is_declared_as_package_data():
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
        assert 'autoforge = ["data/*.json"]' in f.read()


# ------------------------------------------------------------- crawling

def test_download_all_swatches_follows_every_page(monkeypatch):
    api = FakeApi([swatch(i) for i in range(1, 251)])
    slept = []
    monkeypatch.setattr(fc.time, "sleep", slept.append)
    progress = []
    result = fc.download_all_swatches(api, delay=2.0, progress=lambda d, t: progress.append((d, t)))
    assert len(result) == 250
    assert len(api.calls) == 3
    assert all(f"page_size={fc.PAGE_SIZE}" in u for u in api.calls)
    assert slept == [2.0, 2.0]  # between pages, not before the first
    assert progress[-1] == (250, 250)


def test_generate_catalog_writes_only_td_filaments(tmp_path, monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    api = FakeApi([swatch(1), swatch(2, td=None, published="2026-08-01"), swatch(3, td=4)], db_last_modified=42)
    out = tmp_path / "catalog.json"
    cat = fc.generate_catalog(str(out), fetch=api, quiet=True)
    saved = json.loads(out.read_text())
    assert saved == cat
    assert [e["id"] for e in saved["filaments"]] == [1, 3]
    assert saved["db_last_modified"] == 42
    # newest over all swatches, including ones without a TD
    assert saved["newest_published"] == "2026-08-01"


def test_generate_catalog_refuses_to_write_an_empty_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    out = tmp_path / "catalog.json"
    out.write_text("keep me")
    with pytest.raises(RuntimeError):
        fc.generate_catalog(str(out), fetch=FakeApi([swatch(1, td=None)]), quiet=True)
    assert out.read_text() == "keep me"


def test_main_reports_network_failure(monkeypatch, tmp_path, capsys):
    import urllib.error

    def boom(url):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fc, "fetch_json", boom)
    assert fc.main(["--output", str(tmp_path / "x.json")]) == 1
    assert "failed" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


# ------------------------------------------------------- incremental walk

def test_fetch_new_entries_stops_at_known_history(monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    old = [swatch(i, published="2026-06-01") for i in range(1, 300)]
    new = [swatch(1000, published="2026-07-10"), swatch(1001, published="2026-07-11", td=None)]
    api = FakeApi(old + new)
    found, newest = fc.fetch_new_entries({i for i in range(1, 300)}, "2026-07-01", fetch=api)
    assert [e["id"] for e in found] == [1000]  # the TD-less one is not an entry
    assert newest == "2026-07-11"
    assert len(api.calls) == 1  # page 1 already reached older swatches


def test_fetch_new_entries_walks_further_when_a_page_is_all_new(monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    new = [swatch(1000 + i, published="2026-07-20") for i in range(150)]
    api = FakeApi(new + [swatch(1, published="2026-06-01")])
    found, _ = fc.fetch_new_entries({1}, "2026-07-01", fetch=api)
    assert len(found) == 150
    assert len(api.calls) == 2


def test_fetch_new_entries_stops_when_nothing_is_unknown(monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    # Same publish date as the snapshot: the known-id check has to stop it.
    same_day = [swatch(i, published="2026-07-01") for i in range(1, 400)]
    api = FakeApi(same_day)
    found, _ = fc.fetch_new_entries({i for i in range(1, 400)}, "2026-07-01", fetch=api)
    assert found == [] and len(api.calls) == 1


def test_fetch_new_entries_respects_max_pages(monkeypatch):
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    api = FakeApi([swatch(i, published="2026-07-20") for i in range(1, 1000)])
    found, _ = fc.fetch_new_entries(set(), "2026-07-01", fetch=api, max_pages=3)
    assert len(api.calls) == 3 and len(found) == 300


# --------------------------------------------------------------- throttle

class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


def test_throttle_spaces_requests_at_least_one_second_apart():
    c = FakeClock()
    t = fc.RequestThrottle(1.0, clock=c.clock, sleep=c.sleep)
    times = []
    for _ in range(4):
        t.wait()
        times.append(c.now)
    assert c.sleeps == [1.0, 1.0, 1.0]
    assert all(b - a >= 1.0 for a, b in zip(times, times[1:]))


def test_throttle_does_not_wait_when_enough_time_passed():
    c = FakeClock()
    t = fc.RequestThrottle(1.0, clock=c.clock, sleep=c.sleep)
    t.wait()
    c.now += 0.4
    t.wait()  # only the remaining 0.6s
    c.now += 5
    t.wait()  # nothing
    assert c.sleeps == [0.6]


def test_throttle_is_thread_safe():
    import threading
    import time

    t = fc.RequestThrottle(0.05)
    stamps = []
    lock = threading.Lock()

    def worker():
        t.wait()
        with lock:
            stamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    stamps.sort()
    assert all(b - a >= 0.049 for a, b in zip(stamps, stamps[1:]))


def test_every_real_request_goes_through_the_one_second_throttle(monkeypatch):
    assert fc.MIN_REQUEST_INTERVAL >= 1.0
    assert fc._throttle.interval == fc.MIN_REQUEST_INTERVAL
    waits = []
    monkeypatch.setattr(fc._throttle, "wait", lambda: waits.append(1))

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": 1}'

    seen = []

    def fake_urlopen(req, timeout):
        seen.append((req.full_url, req.get_header("User-agent"), timeout))
        return Resp()

    monkeypatch.setattr(fc.urllib.request, "urlopen", fake_urlopen)
    assert fc.fetch_json("https://filamentcolors.xyz/api/version/") == {"ok": 1}
    assert waits == [1]
    assert "AutoForge" in seen[0][1]


# ------------------------------------------------------------ shell script

def test_generation_script_exists_and_targets_the_bundled_file():
    path = os.path.join(REPO_ROOT, "generate_filamentcolors_library.sh")
    assert os.access(path, os.X_OK)
    text = open(path).read()
    assert "autoforge.Helper.filamentcolors_library" in text
    assert "src/autoforge/data/filamentcolors_catalog.json" in text


def test_cli_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "autoforge.Helper.filamentcolors_library", "--help"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "src")},
    )
    assert out.returncode == 0
    assert "--delay" in out.stdout and "1s apart" in out.stdout
