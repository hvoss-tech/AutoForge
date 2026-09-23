@echo off
setlocal enabledelayedexpansion
rem Starts the AutoForge web UI on Windows. Run install.bat first if you
rem haven't already. Mirrors run_webui.sh - see that file for the
rem Linux/macOS equivalent.
cd /d "%~dp0"

if "%WEBUI_HOST%"=="" set "WEBUI_HOST=0.0.0.0"
if "%WEBUI_PORT%"=="" set "WEBUI_PORT=8000"
if "%BUILD_FRONTEND%"=="" set "BUILD_FRONTEND=true"

rem --no-telemetry disables PostHog telemetry for this run (same effect as
rem AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false).
for %%A in (%*) do (
    if "%%A"=="--no-telemetry" set "AUTOFORGE_WEBUI_TELEMETRY_ENABLED=false"
)

set "SHOULD_BUILD=false"
if "%BUILD_FRONTEND%"=="true" set "SHOULD_BUILD=true"
if "%BUILD_FRONTEND%"=="auto" if not exist "webui\frontend\dist" set "SHOULD_BUILD=true"

if "%SHOULD_BUILD%"=="true" (
    echo [webui] Building frontend...
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [webui] ERROR: npm not found. Install Node.js or set BUILD_FRONTEND=false.
        exit /b 1
    )
    pushd webui\frontend
    call npm install
    call npm run build
    popd
    echo [webui] Frontend built.
)

echo [webui] Starting server on http://%WEBUI_HOST%:%WEBUI_PORT%
if not "%AUTOFORGE_WEBUI_TELEMETRY_ENABLED%"=="false" (
    echo [webui] Anonymous usage telemetry is on ^(PostHog^). Disable with --no-telemetry.
)

rem A flat 2-second wait guessed at server startup time and opened the
rem browser too early on a slower machine (bare "problem loading page"
rem instead of the app). Poll the health endpoint instead via a small
rem generated PowerShell script and open only once it actually answers,
rem falling back to the old fixed wait if PowerShell itself isn't available.
rem Kept out of an `if (...)` block on purpose: the PowerShell lines below
rem contain literal `)` characters, which would otherwise prematurely close
rem a parenthesized batch block.
if "%NO_BROWSER%"=="true" goto after_browser_wait
set "WAIT_PS1=%TEMP%\autoforge_webui_wait_%RANDOM%.ps1"
> "%WAIT_PS1%" echo $ok=$false
>> "%WAIT_PS1%" echo for ($i=0; $i -lt 150; $i++) {
>> "%WAIT_PS1%" echo   try {
>> "%WAIT_PS1%" echo     $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%WEBUI_PORT%/api/system/health' -TimeoutSec 1
>> "%WAIT_PS1%" echo     if ($r.StatusCode -eq 200) { $ok=$true; break }
>> "%WAIT_PS1%" echo   } catch {}
>> "%WAIT_PS1%" echo   Start-Sleep -Milliseconds 200
>> "%WAIT_PS1%" echo }
>> "%WAIT_PS1%" echo if (-not $ok) { Start-Sleep -Seconds 2 }
start "" cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File "%WAIT_PS1%" & del "%WAIT_PS1%" & start "" http://localhost:%WEBUI_PORT%"
:after_browser_wait

uv run uvicorn autoforge.webui.server:app --host %WEBUI_HOST% --port %WEBUI_PORT%
