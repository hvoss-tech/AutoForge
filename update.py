#!/usr/bin/env python3
"""Check GitHub for a newer AutoForge release and update this checkout.

Usage:
    uv run python update.py            # check, and update if a newer release exists
    uv run python update.py --check    # only check, don't change anything

Only works when this directory is a git checkout of the AutoForge repo (the
normal case if you cloned it or used install.sh/install.bat). If you
installed AutoForge some other way (e.g. `pip install autoforge`), this will
tell you where to download the new release instead.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "hvoss-tech/AutoForge"
ROOT = Path(__file__).resolve().parent


def current_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        for name in ("autoforge", "AutoForge"):
            try:
                return version(name)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    # Editable/dev checkouts may not have package metadata registered —
    # fall back to reading it straight out of pyproject.toml.
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text())
        if match:
            return match.group(1)
    return "unknown"


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(re.sub(r"\D", "", part) or 0) for part in v.split("."))


def latest_release() -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "AutoForge-updater"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_git_checkout() -> bool:
    return (ROOT / ".git").exists()


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only check for an update, don't apply it")
    args = parser.parse_args()

    current = current_version()
    print(f"Current version: {current}")

    try:
        release = latest_release()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"Could not reach GitHub to check for updates: {e}")
        return 1

    latest_tag = str(release.get("tag_name", "")).lstrip("v")
    release_url = release.get("html_url", f"https://github.com/{REPO}/releases/latest")
    print(f"Latest release:  {latest_tag or 'unknown'} ({release_url})")

    if not latest_tag or current == "unknown":
        print("Could not compare versions.")
        return 1

    if version_tuple(latest_tag) <= version_tuple(current):
        print("You're already on the latest version.")
        return 0

    print(f"A newer version is available: {current} -> {latest_tag}")
    if args.check:
        print("Run without --check to update.")
        return 0

    if not is_git_checkout():
        print(
            "This isn't a git checkout, so it can't be updated automatically.\n"
            f"Download the latest release yourself from: {release_url}"
        )
        return 1

    print("Updating via git...")
    run(["git", "fetch", "--tags", "origin"])
    tag_ref = release.get("tag_name", latest_tag)
    try:
        run(["git", "checkout", tag_ref])
    except subprocess.CalledProcessError:
        print(f"Could not check out tag '{tag_ref}' directly — falling back to 'git pull' on the current branch.")
        run(["git", "pull"])

    if shutil.which("uv"):
        print("Reinstalling Python dependencies...")
        run(["uv", "sync"])
    else:
        print("WARNING: 'uv' not found — skipping dependency reinstall. Run install.sh/install.bat afterwards.")

    frontend_dir = ROOT / "webui" / "frontend"
    if frontend_dir.exists() and shutil.which("npm"):
        print("Rebuilding the web UI frontend...")
        run(["npm", "install"], cwd=frontend_dir)
        run(["npm", "run", "build"], cwd=frontend_dir)

    print("Update complete. Restart AutoForge to use the new version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
