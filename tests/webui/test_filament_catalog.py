"""The filamentcolors.xyz catalog in the webui: CatalogService (bundled
snapshot + per-install updates, the startup update check) and the
/api/filaments/catalog endpoints. Fetchers are fakes — no network."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from autoforge.Helper import filamentcolors_library as fc
from autoforge.webui.config import config
from autoforge.webui.server import create_app
from autoforge.webui.services import catalog_service
from autoforge.webui.services.catalog_service import CatalogService, get_catalog_service
from autoforge.webui.services.filament_service import get_filament_service


def entry(sid, brand="Acme", name="Red", ftype="PLA", td=2.5, color="#aa0000", published="2026-07-01"):
    return {
        "id": sid, "brand": brand, "name": name, "color_name": name, "filament_type": ftype,
        "type_detail": ftype, "color": color, "td": td, "color_family": "Red", "available": True,
        "published": published, "url": f"https://filamentcolors.xyz/swatch/{sid}/",
    }


def raw_swatch(sid, published, td=3.0, name="New"):
    return {
        "id": sid, "manufacturer": {"name": "Acme"}, "color_name": name,
        "filament_type": {"name": "PLA", "parent_type": {"name": "PLA"}}, "color_parent": "BLU",
        "hex_color": "0000ff", "td": td, "date_published": f"{published}T00:00:00Z",
        "is_available": True, "published": True,
    }


@pytest.fixture
def bundled(tmp_path):
    path = tmp_path / "bundled.json"
    fc.save_catalog(str(path), fc.make_catalog([entry(1), entry(2, name="Blue", color="#0000aa")], 100, "2026-07-01"))
    return str(path)


@pytest.fixture(autouse=True)
def _reset_catalog_singleton():
    catalog_service.reset_catalog_service()
    yield
    catalog_service.reset_catalog_service()


@pytest.fixture
def svc(tmp_path, bundled):
    return CatalogService(str(tmp_path / "lib"), bundled_path=bundled, min_check_interval=3600)


class Api:
    def __init__(self, swatches, db_last_modified=200):
        self.swatches = swatches
        self.db_last_modified = db_last_modified
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url == fc.API_VERSION_URL:
            return {"db_last_modified": self.db_last_modified}
        return {"count": len(self.swatches), "next": None, "results": self.swatches}


# ---------------------------------------------------------------- service

def test_catalog_serves_the_bundled_snapshot(svc):
    cat = svc.catalog()
    assert cat["count"] == 2
    assert [e["id"] for e in cat["filaments"]] == [2, 1]  # sorted by brand, name
    assert cat["db_last_modified"] == 100
    assert cat["added_since_release"] == 0
    assert cat["last_checked"] is None


def test_real_service_uses_the_real_bundled_catalog():
    cat = get_catalog_service().catalog()
    assert cat["count"] > 300


def test_update_adds_new_filaments_and_persists_them(svc, tmp_path, bundled):
    api = Api([raw_swatch(10, "2026-08-01"), raw_swatch(11, "2026-08-02", td=None), raw_swatch(1, "2026-07-01")])
    assert svc.check_for_updates(fetch=api, now=lambda: 10_000.0, delay=0) == 1
    assert [e["id"] for e in svc.catalog()["filaments"]] == [2, 10, 1]  # Blue, New, Red
    assert svc.catalog()["added_since_release"] == 1
    assert svc.catalog()["db_last_modified"] == 200
    assert svc.catalog()["newest_published"] == "2026-08-02"
    saved = json.loads((tmp_path / "lib" / "filamentcolors_updates.json").read_text())
    assert saved["checked_at"] == 10_000.0
    assert [e["id"] for e in saved["filaments"]] == [10]

    # A new process (restart) still has it, without asking the site again.
    again = CatalogService(str(tmp_path / "lib"), bundled_path=bundled, min_check_interval=3600)
    assert 10 in again.entries_by_id()


def test_unchanged_site_costs_one_request(svc):
    api = Api([raw_swatch(10, "2026-08-01")], db_last_modified=100)  # same as the snapshot
    assert svc.check_for_updates(fetch=api, now=lambda: 10_000.0) == 0
    assert api.calls == [fc.API_VERSION_URL]


def test_checks_are_rate_limited_between_restarts(svc):
    api = Api([raw_swatch(10, "2026-08-01")])
    svc.check_for_updates(fetch=api, now=lambda: 10_000.0, delay=0)
    n = len(api.calls)
    assert svc.check_for_updates(fetch=api, now=lambda: 10_000.0 + 60) == 0
    assert len(api.calls) == n  # too soon: no request at all
    svc.check_for_updates(fetch=api, now=lambda: 10_000.0 + 3601, delay=0)
    assert len(api.calls) > n
    # force bypasses the interval
    m = len(api.calls)
    svc.check_for_updates(fetch=api, now=lambda: 10_000.0 + 3602, force=True, delay=0)
    assert len(api.calls) > m


def test_network_errors_are_not_fatal(svc):
    def offline(url):
        raise OSError("no route to host")

    assert svc.check_for_updates(fetch=offline, now=lambda: 10_000.0) == 0
    assert svc.status() == {"updating": False, "error": "no route to host"}
    assert svc.catalog()["count"] == 2


def test_corrupt_updates_file_is_ignored(svc, tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "filamentcolors_updates.json").write_text("{broken")
    assert svc.catalog()["count"] == 2


def test_only_one_check_runs_at_a_time(svc):
    started, release = threading.Event(), threading.Event()

    def slow(url):
        started.set()
        release.wait(5)
        return {"db_last_modified": 100}

    t = svc.start_background_update(fetch=slow, now=lambda: 10_000.0)
    assert started.wait(5)
    assert t.daemon
    assert svc.status()["updating"] is True
    assert svc.check_for_updates(fetch=slow, now=lambda: 10_000.0) == 0  # skipped, not queued
    release.set()
    t.join(5)
    assert svc.status()["updating"] is False


# ------------------------------------------------------------ startup

def test_startup_runs_the_update_check_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(CatalogService, "start_background_update", lambda self, **kw: calls.append(1))
    monkeypatch.setattr(config, "filamentcolors_auto_update", False)
    with TestClient(create_app()):
        pass
    assert calls == []
    monkeypatch.setattr(config, "filamentcolors_auto_update", True)
    with TestClient(create_app()):
        pass
    assert calls == [1]


def test_config_defaults():
    from autoforge.webui.config import WebUIConfig

    c = WebUIConfig()
    assert c.filamentcolors_auto_update is True
    assert c.filamentcolors_check_interval_hours == 1.0


# ------------------------------------------------------------------ API

@pytest.fixture
def client(monkeypatch, bundled):
    catalog_service._service = CatalogService(config.library_dir, bundled_path=bundled)
    with TestClient(create_app()) as c:
        yield c


def test_get_catalog(client):
    data = client.get("/api/filaments/catalog").json()
    assert data["count"] == 2
    assert {e["id"] for e in data["filaments"]} == {1, 2}
    assert data["library_uuids"] == {}
    assert data["updating"] is False and data["error"] is None


def test_add_creates_a_library_filament(client):
    res = client.post("/api/filaments/catalog/add", json={"ids": [1]})
    assert res.status_code == 200
    body = res.json()
    assert body["existing"] == []
    f = body["added"][0]
    assert f == {"brand": "Acme", "name": "Red", "color": "#aa0000", "td": 2.5, "owned": True,
                 "uuid": "filamentcolors-1", "filament_type": "PLA", "source": "filamentcolors"}
    library = client.get("/api/filaments").json()
    assert any(x["uuid"] == "filamentcolors-1" for x in library)
    assert client.get("/api/filaments/catalog").json()["library_uuids"] == {"1": "filamentcolors-1"}
    # The library panel only shows non-"user" filaments once a custom library exists.
    assert client.get("/api/filaments/has-custom-library").json()["exists"] is True
    assert client.get("/api/filaments/active").json() == []


def test_add_respects_owned_and_activate(client):
    body = client.post("/api/filaments/catalog/add", json={"ids": [1, 2], "owned": False, "activate": True}).json()
    assert [f["owned"] for f in body["added"]] == [False, False]
    assert {f["uuid"] for f in body["active"]} == {"filamentcolors-1", "filamentcolors-2"}


def test_adding_twice_does_not_duplicate_or_overwrite_edits(client):
    client.post("/api/filaments/catalog/add", json={"ids": [1]})
    edited = {"brand": "Acme", "name": "Red", "color": "#aa0000", "td": 9.9, "owned": True,
              "uuid": "filamentcolors-1", "filament_type": "PLA", "source": "filamentcolors"}
    client.put("/api/filaments/filamentcolors-1", json=edited)
    body = client.post("/api/filaments/catalog/add", json={"ids": [1, 1]}).json()
    assert body["added"] == []
    assert [f["td"] for f in body["existing"]] == [9.9]
    assert sum(1 for f in client.get("/api/filaments").json() if f["name"] == "Red" and f["brand"] == "Acme") == 1


def test_users_own_matching_filament_counts_as_in_library(client):
    own = client.post("/api/filaments", json={"brand": "ACME", "name": "red ", "color": "#ff0000", "td": 1, "filament_type": "pla"}).json()
    assert client.get("/api/filaments/catalog").json()["library_uuids"] == {"1": own["uuid"]}
    body = client.post("/api/filaments/catalog/add", json={"ids": [1], "activate": True}).json()
    assert body["added"] == [] and body["existing"][0]["uuid"] == own["uuid"]
    assert [f["uuid"] for f in body["active"]] == [own["uuid"]]


def test_same_name_other_type_is_not_a_match(client):
    client.post("/api/filaments", json={"brand": "Acme", "name": "Red", "color": "#ff0000", "td": 1, "filament_type": "PETG"})
    assert client.get("/api/filaments/catalog").json()["library_uuids"] == {}


def test_deleting_the_library_copy_makes_it_addable_again(client):
    client.post("/api/filaments/catalog/add", json={"ids": [1]})
    client.delete("/api/filaments/filamentcolors-1")
    assert client.get("/api/filaments/catalog").json()["library_uuids"] == {}
    assert len(client.post("/api/filaments/catalog/add", json={"ids": [1]}).json()["added"]) == 1


def test_add_rejects_unknown_and_empty(client):
    res = client.post("/api/filaments/catalog/add", json={"ids": [1, 999]})
    assert res.status_code == 404 and "999" in res.json()["detail"]
    assert client.get("/api/filaments/catalog").json()["library_uuids"] == {}  # nothing half-added
    assert client.post("/api/filaments/catalog/add", json={"ids": []}).status_code == 400
    assert client.post("/api/filaments/catalog/add", json={"ids": ["x"]}).status_code == 422


def test_added_filaments_show_up_in_type_and_brand_lists(client):
    client.post("/api/filaments/catalog/add", json={"ids": [2]})
    assert "PLA" in client.get("/api/filaments/types").json()
    assert "Acme" in client.get("/api/filaments/brands", params={"filament_type": "PLA"}).json()
