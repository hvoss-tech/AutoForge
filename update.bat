@echo off
cd /d "%~dp0"
uv run python update.py %*
