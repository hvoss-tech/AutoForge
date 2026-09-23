import os
from pathlib import Path
from pydantic_settings import BaseSettings

class WebUIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    checkpoints_dir: str = "checkpoints"
    uploads_dir: str = "uploads"
    library_dir: str = "filament_library"

    # PostHog telemetry — anonymous usage analytics, on by default for every
    # install so the project has aggregate usage data. This is the project's
    # own PostHog key, safe to ship publicly (PostHog project keys are
    # write-only, client-side keys by design). Opt out per-run with
    # --no-telemetry (run_webui.sh/.bat/run_dev.sh) or permanently with
    # AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false.
    telemetry_enabled: bool = True
    posthog_key: str = "phc_zT5ANJNoUKnUTQqRnxF5EfkWzG5LNjfKWMvhgBwBMHTA"
    posthog_host: str = "https://eu.i.posthog.com"

    model_config = {"env_prefix": "AUTOFORGE_WEBUI_"}

    @property
    def checkpoints_path(self) -> str:
        return os.path.abspath(self.checkpoints_dir)

    @property
    def uploads_path(self) -> str:
        return os.path.abspath(self.uploads_dir)

    @property
    def library_path(self) -> str:
        return os.path.abspath(self.library_dir)


config = WebUIConfig()
