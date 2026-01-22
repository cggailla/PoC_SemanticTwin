"""
Job manager for handling asynchronous audit tasks.

This module manages background job execution, status tracking, and result storage.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobResult:
    """Container for job results and metadata."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class JobManager:
    """Manages asynchronous job execution and tracking."""

    def __init__(self, output_base_dir: Path = Path("./output/jobs")):
        """Initialize job manager.

        Args:
            output_base_dir: Base directory for job outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, JobResult] = {}
        self._lock = threading.RLock()

        logger.info(f"JobManager initialized with output dir: {self.output_base_dir}")

    def create_job(self) -> str:
        """Create and register a new job.

        Returns:
            Job ID as UUID string
        """
        job_id = str(uuid.uuid4())

        with self._lock:
            self._jobs[job_id] = JobResult(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
            )

        logger.info(f"Created job: {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status and metadata.

        Args:
            job_id: Job identifier

        Returns:
            JobResult object or None if job not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: JobStatus) -> bool:
        """Update job status.

        Args:
            job_id: Job identifier
            status: New job status

        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Job not found: {job_id}")
                return False

            job = self._jobs[job_id]
            job.status = status

            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.now()
            elif status == JobStatus.COMPLETED and job.completed_at is None:
                job.completed_at = datetime.now()

            logger.info(f"Job {job_id} status updated to: {status.value}")
            return True

    def set_job_output(self, job_id: str, output_path: str) -> bool:
        """Set job output path (when audit completes successfully).

        Args:
            job_id: Job identifier
            output_path: Path to output file

        Returns:
            True if set, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Job not found: {job_id}")
                return False

            self._jobs[job_id].output_path = output_path
            logger.info(f"Job {job_id} output set to: {output_path}")
            return True

    def set_job_error(self, job_id: str, error_message: str) -> bool:
        """Set job error message.

        Args:
            job_id: Job identifier
            error_message: Error description

        Returns:
            True if set, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Job not found: {job_id}")
                return False

            self._jobs[job_id].error_message = error_message
            logger.error(f"Job {job_id} error: {error_message}")
            return True

    def set_job_metadata(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        """Set job metadata.

        Args:
            job_id: Job identifier
            metadata: Metadata dictionary

        Returns:
            True if set, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Job not found: {job_id}")
                return False

            self._jobs[job_id].metadata.update(metadata)
            return True

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get output directory for a job.

        Args:
            job_id: Job identifier

        Returns:
            Path to job output directory
        """
        job_dir = self.output_base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def execute_job(self, job_id: str, task_func, *args, **kwargs) -> threading.Thread:
        """Execute a job in background thread.

        Args:
            job_id: Job identifier
            task_func: Callable to execute
            *args: Positional arguments for task_func
            **kwargs: Keyword arguments for task_func

        Returns:
            Thread object for the background job
        """

        def wrapped_task():
            """Wraps task execution with status tracking."""
            self.update_job_status(job_id, JobStatus.RUNNING)
            try:
                logger.info(f"Starting job execution: {job_id}")
                result = task_func(*args, **kwargs)
                self.update_job_status(job_id, JobStatus.COMPLETED)
                logger.info(f"Job completed successfully: {job_id}")
                return result
            except Exception as e:
                self.update_job_status(job_id, JobStatus.FAILED)
                self.set_job_error(job_id, str(e))
                logger.exception(f"Job failed: {job_id}")
                raise

        thread = threading.Thread(target=wrapped_task, daemon=True)
        thread.start()
        return thread

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """Clean up old job data (not implemented in basic version).

        Args:
            days: Keep jobs from last N days

        Returns:
            Number of jobs cleaned up
        """
        # TODO: Implement cleanup logic
        return 0
