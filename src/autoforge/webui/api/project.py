import json
import os
from fastapi import APIRouter, HTTPException
from ..models import ProjectState
from ..config import config

router = APIRouter()

_state = ProjectState()


def _state_file() -> str:
    return os.path.join(config.checkpoints_path, "project_state.json")


@router.get("/state")
async def get_project_state():
    global _state
    path = _state_file()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            _state = ProjectState(**data)
        except (json.JSONDecodeError, IOError, ValueError):
            # ValueError: pydantic validation of an outdated/invalid file.
            pass
    return _state.model_dump()


@router.post("/state")
async def save_project_state(state: ProjectState):
    global _state
    _state = state
    path = _state_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state.model_dump(), f, indent=2)
    return {"status": "ok"}
