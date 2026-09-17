import json
import os
import uuid
from typing import Optional
from ..models import StateSnapshot
from ..config import config

MAX_SNAPSHOTS = 50


class ProjectService:
    def __init__(self, snapshot_dir: str = ""):
        self._snapshots: dict[str, StateSnapshot] = {}
        self._snapshot_dir = snapshot_dir or os.path.join(config.checkpoints_path, "snapshots")
        self._load_snapshots()

    def _snapshot_path(self, snapshot_id: str) -> str:
        safe = os.path.basename(snapshot_id.replace("..", ""))
        if not safe or safe.startswith("."):
            safe = f"snapshot_{abs(hash(snapshot_id)) % (10**16)}"
        return os.path.join(self._snapshot_dir, f"{safe}.json")

    def _load_snapshots(self):
        if not os.path.exists(self._snapshot_dir):
            return
        for fname in sorted(os.listdir(self._snapshot_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._snapshot_dir, fname)) as f:
                        data = json.load(f)
                    snapshot = StateSnapshot(**data)
                    sid = data.get("_snapshot_id", os.path.splitext(fname)[0])
                    self._snapshots[sid] = snapshot
                except (json.JSONDecodeError, IOError, KeyError, ValueError):
                    # ValueError covers pydantic's ValidationError: one bad
                    # or outdated snapshot file must not take down the rest.
                    pass

    def save_snapshot(self, snapshot: StateSnapshot) -> str:
        sid = str(uuid.uuid4())
        os.makedirs(self._snapshot_dir, exist_ok=True)
        path = self._snapshot_path(sid)
        data = snapshot.model_dump(by_alias=True)
        data["_snapshot_id"] = sid
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._snapshots[sid] = snapshot
        # The client only ever keeps the newest MAX_SNAPSHOTS; without a cap
        # here the directory grew by one file per edit forever.
        if len(self._snapshots) > MAX_SNAPSHOTS:
            oldest = sorted(self._snapshots.items(), key=lambda kv: kv[1].timestamp)
            for old_sid, _ in oldest[: len(self._snapshots) - MAX_SNAPSHOTS]:
                self._delete(old_sid)
        return sid

    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        if not snapshot_id:
            return None
        if snapshot_id in self._snapshots:
            return self._snapshots[snapshot_id]
        for v in self._snapshots.values():
            if str(v.timestamp) == snapshot_id:
                return v
        return None

    def get_snapshot_by_timestamp(self, timestamp) -> Optional[StateSnapshot]:
        """Match numerically, not by string: the frontend's timestamps are
        ``Date.now() / 1000``, and when that lands on a whole second JSON
        sends ``1726570000`` (an int in Python) while the stored float
        stringifies as ``1726570000.0`` — a string comparison missed it and
        undo/redo to that snapshot silently did nothing."""
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return None
        for v in self._snapshots.values():
            if abs(v.timestamp - ts) < 1e-6:
                return v
        return None

    def delete_snapshots_after(self, timestamp: float) -> int:
        doomed = [sid for sid, v in self._snapshots.items() if v.timestamp > timestamp + 1e-6]
        for sid in doomed:
            self._delete(sid)
        return len(doomed)

    def _delete(self, sid: str):
        self._snapshots.pop(sid, None)
        try:
            os.remove(self._snapshot_path(sid))
        except OSError:
            pass

    def list_snapshots(self) -> list[StateSnapshot]:
        return sorted(self._snapshots.values(), key=lambda s: s.timestamp, reverse=True)



_service: Optional[ProjectService] = None


def get_project_service() -> ProjectService:
    global _service
    if _service is None:
        _service = ProjectService()
    return _service


def reset_project_service():
    global _service
    _service = None
