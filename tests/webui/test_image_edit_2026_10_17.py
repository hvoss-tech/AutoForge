"""The image editor's layers (settings.image_edit) are kept with the
project: the settings model used to drop any field it didn't know, so an
edited picture lost the edit it was made with on the next reload and could
no longer be edited again."""

from fastapi.testclient import TestClient

from autoforge.webui.models import OptimizationSettings
from autoforge.webui.server import create_app

EDIT = {
    "version": 1,
    "source": "photo.png",
    "layers": [
        {
            "id": "layer-1", "name": "Sky", "visible": True, "selection": "mask", "mask": "selection.png", "feather": 10,
            "adjust": {"exposure": -0.5, "contrast": 20, "curves": {"master": [[0, 0], [128, 150], [255, 255]]},
                       "reduce": {"enabled": True, "colors": 12}},
        }
    ],
}


def test_the_image_edit_survives_the_project_state():
    client = TestClient(create_app())
    state = client.get("/api/project/state").json()
    state["settings"]["image_edit"] = EDIT
    assert client.post("/api/project/state", json=state).status_code == 200
    assert client.get("/api/project/state").json()["settings"]["image_edit"] == EDIT

    # And a picture used as uploaded has none.
    state["settings"]["image_edit"] = None
    assert client.post("/api/project/state", json=state).status_code == 200
    assert client.get("/api/project/state").json()["settings"]["image_edit"] is None


def test_runs_accept_settings_with_an_image_edit():
    settings = OptimizationSettings(input_image="edited.png", image_edit=EDIT)
    assert settings.model_dump()["image_edit"] == EDIT
