import os

import pytest
from fastapi.testclient import TestClient
from autoforge.webui import create_app
from autoforge.webui.server import FRONTEND_DIST


def test_server_health():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/docs")
    assert resp.status_code in (200, 307)


@pytest.mark.skipif(
    not os.path.isdir(FRONTEND_DIST),
    reason="Frontend not built (run `npm run build` in webui/frontend, or ./run_webui.sh)",
)
def test_frontend_served():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<div id=\"root\">" in resp.text
