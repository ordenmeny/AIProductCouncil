from __future__ import annotations

import json
from pathlib import Path

from backend.app.models import MeetingState


class MeetingStorage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, meeting_id: str) -> Path:
        return self.storage_dir / f"{meeting_id}.json"

    def save(self, meeting: MeetingState) -> MeetingState:
        meeting.touch()
        self._path(meeting.id).write_text(
            meeting.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return meeting

    def get(self, meeting_id: str) -> MeetingState:
        path = self._path(meeting_id)
        if not path.exists():
            raise FileNotFoundError(meeting_id)
        return MeetingState.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        return sorted(path.stem for path in self.storage_dir.glob("*.json"))

    def raw_json(self, meeting_id: str) -> dict:
        return json.loads(self._path(meeting_id).read_text(encoding="utf-8"))
