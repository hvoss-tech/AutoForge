// Starts a throwaway AutoForge webui backend for the Playwright suite.
//
// The suite used to run against whatever was listening on :8000 — normally
// the user's real server — and its global setup reset that server's project
// state and deleted "test" filaments from the real library. This launches a
// separate server whose checkpoints/uploads/filament library live in a
// fresh temp directory, deleted again on exit.
//
//   AUTOFORGE_PYTHON  python executable with autoforge installed (default: python)
//   WEBUI_TEST_PORT   port to listen on (default: 8799)
//
// Server output is written to <tmpdir>/autoforge-webui-e2e-server.log.
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'autoforge-webui-e2e-'))
const port = process.env.WEBUI_TEST_PORT || '8799'
const python = process.env.AUTOFORGE_PYTHON || 'python'
// Server output (including optimizer progress bars) goes to a log file kept
// after the run, rather than flooding the test output.
const logPath = path.join(os.tmpdir(), 'autoforge-webui-e2e-server.log')
const logFd = fs.openSync(logPath, 'w')
// Never the developer's real HueForge library: its presence opens a
// first-start import offer over the whole app. Specs that want one write a
// file here (keep in sync with HUEFORGE_TEST_LIBRARY in tests/helpers.ts).
const hueforgeLibrary = path.join(os.tmpdir(), 'autoforge-webui-e2e-hueforge', 'personal_library.json')
fs.rmSync(path.dirname(hueforgeLibrary), { recursive: true, force: true })

const server = spawn(
  python,
  ['-m', 'uvicorn', 'autoforge.webui.server:app', '--host', '127.0.0.1', '--port', port, '--log-level', 'warning'],
  { cwd: dataDir, stdio: ['ignore', logFd, logFd], env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      AUTOFORGE_WEBUI_HUEFORGE_LIBRARY: hueforgeLibrary,
      // The catalog specs use the bundled snapshot; never call filamentcolors.xyz from tests.
      AUTOFORGE_WEBUI_FILAMENTCOLORS_AUTO_UPDATE: 'false',
    } },
)
console.log(`AutoForge test server log: ${logPath}`)

let cleaned = false
const cleanup = () => {
  if (cleaned) return
  cleaned = true
  fs.rmSync(dataDir, { recursive: true, force: true })
}
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    server.kill('SIGTERM')
  })
}
server.on('exit', (code) => {
  cleanup()
  process.exit(code ?? 0)
})
