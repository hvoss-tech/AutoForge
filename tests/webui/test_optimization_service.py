import pytest
import tempfile
from autoforge.webui.services.optimization_service import OptimizationService, reset_service


@pytest.fixture
def svc():
    reset_service()
    return OptimizationService(checkpoints_dir=tempfile.mkdtemp())


def test_create_and_list_jobs(svc):
    job = svc.create_job({"iterations": 100}, job_id="test-001")
    assert job.job_id == "test-001"
    assert job.status == "pending"

    # Pending jobs don't appear in history — only terminal states
    history = svc.get_history()
    assert len(history) == 0

    svc.update_status("test-001", "completed")
    history = svc.get_history()
    assert len(history) == 1
    assert history[0].job_id == "test-001"


def test_job_state_machine(svc):
    job = svc.create_job({}, job_id="sm-001")
    assert job.status == "pending"

    svc.update_status("sm-001", "running")
    assert svc.get_job("sm-001").status == "running"

    svc.update_status("sm-001", "paused")
    assert svc.get_job("sm-001").status == "paused"

    svc.update_status("sm-001", "completed")
    result = svc.get_result("sm-001")
    assert result is not None
    assert result.status == "completed"


def test_pause_resume(svc):
    svc.create_job({}, job_id="pr-001")
    svc.update_status("pr-001", "running")

    assert svc.pause("pr-001") is True
    assert svc.get_job("pr-001").status == "paused"

    assert svc.resume("pr-001") is True
    assert svc.get_job("pr-001").status == "running"

    assert svc.cancel("pr-001") is True
    assert svc.get_job("pr-001").status == "cancelled"


def test_progress_callback_does_not_unpause(svc):
    svc.create_job({}, job_id="pg-001")
    svc.update_status("pg-001", "running")
    svc.pause("pg-001")
    assert svc.get_job("pg-001").status == "paused"

    # A progress callback firing while paused must NOT flip the job back to
    # "running" — that is what made the UI toggle back to the Pause button.
    # Its payload is dropped too: the report describes work that finished
    # before the pause, and applying it made a paused job look like it was
    # still advancing (see test_a_paused_job_stops_advancing).
    svc.update_status("pg-001", "running", progress=42.0)
    job = svc.get_job("pg-001")
    assert job.status == "paused"
    assert job.progress == 0.0

    svc.resume("pg-001")
    assert svc.get_job("pg-001").status == "running"
    # Reports land again once it is actually running.
    svc.update_status("pg-001", "running", progress=42.0)
    assert svc.get_job("pg-001").progress == 42.0


def test_cancel_event(svc):
    svc.create_job({}, job_id="ce-001")
    ev = svc.cancel_event("ce-001")
    assert ev is not None
    assert not ev.is_set()

    svc.cancel("ce-001")
    assert ev.is_set()
