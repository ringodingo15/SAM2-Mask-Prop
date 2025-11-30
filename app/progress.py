from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class JobState:
    job_id: str
    status: str = "created"  # created|queued|running|completed|failed
    progress: int = 0
    message: str = ""
    meta: Dict[str, Any] = None

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, JobState] = {}

    def create(self, job_id: str):
        self._jobs[job_id] = JobState(job_id=job_id)

    def get(self, job_id: str):
        j = self._jobs.get(job_id)
        return asdict(j) if j else None

    def update(self, job_id: str, **kwargs):
        j = self._jobs.get(job_id)
        if not j:
            return
        for k, v in kwargs.items():
            setattr(j, k, v)