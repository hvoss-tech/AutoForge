import json
import os
from fastapi import APIRouter, HTTPException
from ..models import ProjectState
from ..config import config
from ..services.project_service import atomic_write_json

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
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                # A non-object top level (e.g. an old-format file that was a
                # bare list) — nothing to restore from.
                raise TypeError("project_state.json is not an object")
            _state = ProjectState(**data)
        except (json.JSONDecodeError, IOError, ValueError, TypeError):
            # ValueError: pydantic validation of an outdated/invalid file.
            pass
    return _state.model_dump()


@router.post("/state")
async def save_project_state(state: ProjectState):
    global _state
    _state = state
    path = _state_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_json(path, state.model_dump())
    return {"status": "ok"}
