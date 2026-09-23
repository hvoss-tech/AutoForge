@echo off
setlocal
rem One-shot setup for AutoForge on Windows: installs `uv` (a fast Python
rem package manager) if it's missing, installs all Python dependencies, and
rem builds the web UI frontend. After this finishes, run run_webui.bat to
rem start AutoForge.
cd /d "%~dp0"

echo === AutoForge installer ===

where uv >nul 2>nul
if errorlevel 1 (
    echo [install] "uv" not found - installing it now...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [install] ERROR: uv installed but isn't on PATH yet.
    echo   Close and reopen this terminal and re-run install.bat, or see https://docs.astral.sh/uv/ for manual install steps.
    exit /b 1
)

echo [install] Installing Python dependencies with uv...
call uv sync
if errorlevel 1 exit /b 1

where npm >nul 2>nul
if errorlevel 1 (
    echo [install] WARNING: npm not found - skipping the web UI frontend build.
    echo   Install Node.js from https://nodejs.org/ if you want to use the web UI, then re-run install.bat.
) else (
    echo [install] Building the web UI frontend ^(this can take a minute^)...
    pushd webui\frontend
    call npm install
    call npm run build
    popd
)

echo.
echo === Install complete ===
echo Start AutoForge with:   run_webui.bat
echo Check for updates with: update.bat
echo.
echo Note: the web UI sends anonymous usage telemetry (PostHog) by default.
echo   Disable it with run_webui.bat --no-telemetry, or permanently via
echo   AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false. See README.md for details.
