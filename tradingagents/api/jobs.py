"""In-memory job registry for the HTTP API. Single-user/local deployment —
no Redis/Celery needed; job state doesn't need to survive a process restart.
"""
import uuid
from typing import Any


class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "queued",
            "reports_completed": 0,
            "reports_total": 0,
            "reports": {},
            "final_decision": None,
            "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **fields: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(fields)
