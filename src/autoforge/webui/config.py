import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings

class WebUIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    checkpoints_dir: str = "checkpoints"
    uploads_dir: str = "uploads"
    library_dir: str = "filament_library"
    # Where HueForge keeps the user's personal filament library. Empty means
    # "look in HueForge's per-user data folder" (see hueforge_library_candidates).
    hueforge_library: str = ""
    # Look for filaments published on filamentcolors.xyz since the bundled
    # catalog snapshot, in the background on startup. Checks are skipped when
    # the last one was less than this many hours ago (restart loops, dev
    # reloads). AUTOFORGE_WEBUI_FILAMENTCOLORS_AUTO_UPDATE=false disables it.
    filamentcolors_auto_update: bool = True
    filamentcolors_check_interval_hours: float = 1.0

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

    def hueforge_library_candidates(self) -> list[str]:
        """Where HueForge's personal_library.json may live, most likely first.

        HueForge (a Qt app) keeps it in Qt's per-user app-data folder:
        %APPDATA%\\HueForge\\Filaments on Windows, ~/Library/Application
        Support/HueForge/Filaments on macOS and
        $XDG_DATA_HOME (~/.local/share)/HueForge/Filaments on Linux (confirmed
        against a real install). ~/.config is a Linux fallback, and an
        APPDATA variable is honored on every platform (e.g. under Wine)."""
        if self.hueforge_library:
            return [os.path.abspath(os.path.expanduser(self.hueforge_library))]
        tail = os.path.join("HueForge", "Filaments", "personal_library.json")
        home = Path.home()
        roots: list[str] = []
        if os.environ.get("APPDATA"):
            roots.append(os.environ["APPDATA"])
        if sys.platform == "darwin":
            roots.append(str(home / "Library" / "Application Support"))
        elif not sys.platform.startswith("win"):
            roots.append(os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share"))
            roots.append(os.environ.get("XDG_CONFIG_HOME") or str(home / ".config"))
        return [os.path.join(root, tail) for root in dict.fromkeys(roots)]

    @property
    def hueforge_library_path(self) -> str | None:
        """The first candidate that exists, else the most likely one (so the
        UI can say where it looked), else None."""
        candidates = self.hueforge_library_candidates()
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0] if candidates else None

    @property
    def library_path(self) -> str:
        return os.path.abspath(self.library_dir)


config = WebUIConfig()
