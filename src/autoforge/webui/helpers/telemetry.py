"""Server-side error reporting to PostHog, mirroring the frontend's
posthog-js capture_exceptions/captureException (see
webui/frontend/src/lib/telemetry.ts). Disabled whenever the frontend's
/api/system/telemetry endpoint would report disabled — same config, same
--no-telemetry / AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false switch.

No image data, filament data or other user information is sent - only the
exception type, message, traceback, and a small context dict (e.g. job
phase) callers pass in explicitly. The message and traceback can contain
file paths; the user's home folder in them is replaced by ``~`` (see
scrub_paths). The request path of a failed API call is sent as context.
"""
import logging
import traceback
import uuid
from pathlib import Path

from ..config import config

logger = logging.getLogger(__name__)

_client = None
_client_built = False
_distinct_id: str | None = None

# Truncate long tracebacks/messages before sending — PostHog events have a
# size limit, and a huge tensor repr accidentally caught in an exception
# message shouldn't blow past it.
_MAX_FIELD_LEN = 4000


def _anon_distinct_id() -> str:
    """A random id, stable across runs and projects for this machine/user,
    persisted outside any project checkout so reinstalling or updating
    AutoForge doesn't generate a new one each time.

    Also handed to the frontend (see /api/system/telemetry ->
    lib/telemetry.ts's posthog.identify() call) so backend exception events
    and frontend pings/errors from the same install land under the same
    PostHog person instead of two unrelated anonymous ids."""
    global _distinct_id
    if _distinct_id is not None:
        return _distinct_id

    id_path = Path.home() / ".autoforge" / "telemetry_id"
    try:
        if id_path.exists():
            _distinct_id = id_path.read_text().strip()
        if not _distinct_id:
            _distinct_id = str(uuid.uuid4())
            id_path.parent.mkdir(parents=True, exist_ok=True)
            id_path.write_text(_distinct_id)
    except OSError:
        # Read-only home dir, permissions issue, etc. — fall back to a
        # per-process id rather than failing telemetry entirely.
        _distinct_id = str(uuid.uuid4())
    return _distinct_id


def anon_distinct_id() -> str:
    """Public accessor for the frontend's /api/system/telemetry response."""
    return _anon_distinct_id()


def scrub_paths(text: str) -> str:
    """``text`` with the user's home folder replaced by ``~``: messages and
    tracebacks carry absolute paths (the install location, input files),
    and with them the user's account name."""
    home = str(Path.home())
    if not home or home in ("/", "\\"):
        return text
    for variant in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
        text = text.replace(variant, "~")
    return text


def _get_client():
    global _client, _client_built
    if _client_built:
        return _client
    _client_built = True

    if not config.telemetry_enabled or not config.posthog_key:
        return None
    try:
        from posthog import Posthog
        _client = Posthog(project_api_key=config.posthog_key, host=config.posthog_host)
    except Exception:
        logger.debug("Could not initialize PostHog client", exc_info=True)
        _client = None
    return _client


def capture_exception(exc: BaseException, context: dict | None = None) -> None:
    """Best-effort — never raises, never masks the original exception. Call
    this alongside (not instead of) normal logging/traceback.print_exc()."""
    try:
        client = _get_client()
        if client is None:
            return
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        properties = {
            "$exception_type": type(exc).__name__,
            "$exception_message": scrub_paths(str(exc))[:_MAX_FIELD_LEN],
            "$exception_stack_trace_raw": scrub_paths(tb)[-_MAX_FIELD_LEN:],
            "source": "backend",
        }
        if context:
            properties.update({k: scrub_paths(v) if isinstance(v, str) else v for k, v in context.items()})
        client.capture(
            distinct_id=_anon_distinct_id(),
            event="$exception",
            properties=properties,
        )
    except Exception:
        logger.debug("Failed to report exception to PostHog", exc_info=True)


def flush() -> None:
    """Called from the server's shutdown lifespan so a crash right before
    exit doesn't drop its own report — posthog-python batches events and
    flushes on an interval otherwise."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
