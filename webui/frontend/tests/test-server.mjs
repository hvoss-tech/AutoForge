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

const server = spawn(
  python,
  ['-m', 'uvicorn', 'autoforge.webui.server:app', '--host', '127.0.0.1', '--port', port, '--log-level', 'warning'],
  { cwd: dataDir, stdio: ['ignore', logFd, logFd], env: { ...process.env, PYTHONUNBUFFERED: '1' } },
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
