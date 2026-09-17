"""Isolation for the webui test suite.

Every test under tests/webui/ (and tests/test_webui_backend.py, which
mirrors this) creates its own FastAPI app via ``create_app()``, but every
service behind that app is a process-wide singleton backed by files at
CWD-relative paths (``WebUIConfig.checkpoints_dir``/``uploads_dir``/
``library_dir`` all default to relative paths, with nothing overriding
them for tests). Run pytest from the repo root — the normal way to run it —
and every filament these tests create, every project-state snapshot they
save, and every upload they make land in the exact same
``filament_library/``, ``checkpoints/`` and ``uploads/`` directories a real,
interactively-run webui session reads from. Confirmed in practice:
``filament_library/library.json`` and ``active.json`` had accumulated dozens
of "Test"/"OptGuardFilament"/"VerifyTest" entries from previous test runs,
which then showed up as real Active Filaments in a real session.

These fixtures redirect all webui storage to a throwaway temp directory for
the whole test session, and reset the in-process singletons between tests
so state doesn't leak from one test into the next either.
"""

import os
import shutil
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_webui_storage():
    from autoforge.webui.config import config

    tmp_dir = tempfile.mkdtemp(prefix="autoforge_webui_test_")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "checkpoints_dir", os.path.join(tmp_dir, "checkpoints"))
    mp.setattr(config, "uploads_dir", os.path.join(tmp_dir, "uploads"))
    mp.setattr(config, "library_dir", os.path.join(tmp_dir, "filament_library"))
    try:
        yield
    finally:
        mp.undo()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_webui_singletons():
    from autoforge.webui.api import project as project_api
    from autoforge.webui.api import settings as settings_api
    from autoforge.webui.services import filament_service, optimization_service, project_service

    def _reset():
        filament_service.reset_service()
        optimization_service.reset_service()
        project_service.reset_project_service()
        settings_api._settings = settings_api.OptimizationSettings()
        project_api._state = project_api.ProjectState()

    _reset()
    yield
    _reset()
