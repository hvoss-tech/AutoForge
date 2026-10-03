// Test runner: node --import tsx/esm --test tests/bugreport-2026-10-03.test.mjs
//
// Regression tests for the frontend fixes from the 2026-10-03 bug report
// (BUG_REPORT.md: H1, M5, M6, M7, L27-L30). The store is imported for real,
// with the few browser globals it touches at module scope shimmed.

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const storage = {}
globalThis.localStorage = {
  getItem: (k) => storage[k] ?? null,
  setItem: (k, v) => { storage[k] = String(v) },
  removeItem: (k) => { delete storage[k] },
}
const windowListeners = {}
globalThis.window = {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  addEventListener(type, fn) { (windowListeners[type] ??= []).push(fn) },
  removeEventListener() {},
  location: { protocol: 'http:', host: 'localhost' },
}
globalThis.document = {
  documentElement: { classList: { add() {}, remove() {}, toggle() {} }, setAttribute() {}, style: {} },
  createElement: () => ({ click() {}, remove() {} }),
  body: { appendChild() {} },
}
URL.createObjectURL = () => 'blob:test'
URL.revokeObjectURL = () => {}

let routes = []
const requests = []
globalThis.fetch = async (url, init = {}) => {
  requests.push({ url: String(url), init })
  for (const [prefix, handler] of routes) {
    if (String(url).startsWith(prefix)) return handler(url, init)
  }
  return { ok: true, status: 200, json: async () => ({}), blob: async () => new Blob([]) }
}
const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

const { useAppStore, defaultSettings, initRetryDelay, INIT_BUSY_RETRY_MS } = await import('../src/store/appStore.ts')
const { adoptsPrunedResult } = await import('../src/lib/jobTracking.ts')

const job = (id, status, extra = {}) => ({
  job_id: id, status, progress: 0, iteration: 0, total_iterations: 10, loss: null, error: null,
  started_at: null, completed_at: null, preview_image: null, ...extra,
})
const reset = () => {
  routes = []
  requests.length = 0
  useAppStore.setState({ currentJob: null, toasts: [], pruningJob: null })
}
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms))
const src = (path) => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8')

test('H1: runs are not started with the matplotlib visualization', () => {
  assert.equal(defaultSettings.visualize, false)
})

test('M5: another build running makes the auto-preview wait before asking again', async () => {
  reset()
  routes = [['/api/init/run', () => json(400, { detail: 'Already initializing' })]]
  useAppStore.setState({ settings: { ...useAppStore.getState().settings, input_image: 'a.png' } })
  assert.equal(await useAppStore.getState().runInit(), 'busy')
  const wait = initRetryDelay()
  assert.ok(wait > 0 && wait <= INIT_BUSY_RETRY_MS, `waits (${wait} ms)`)
  assert.equal(initRetryDelay(Date.now() + INIT_BUSY_RETRY_MS + 1), 0)
  assert.match(src('hooks/useAutoPreviewInit.ts'), /initRetryDelay\(\)/)
})

test('M5: a build superseded by a reset is retried at once', async () => {
  reset()
  routes = [['/api/init/run', () => json(409, { detail: 'Auto-preview was reset while it was being prepared.' })]]
  await tick(INIT_BUSY_RETRY_MS + 20)
  assert.equal(await useAppStore.getState().runInit(), 'busy')
  assert.equal(initRetryDelay(), 0)
})

test('M6: a prune finishing after an undo does not take over the screen', async () => {
  reset()
  assert.equal(adoptsPrunedResult('opt-1', 'opt-1'), true)
  assert.equal(adoptsPrunedResult('opt-0', 'opt-1'), false)
  let pollStatus = 'running'
  routes = [
    ['/api/pruning/start', () => json(200, { job_id: 'prune-m6', status: 'running' })],
    ['/api/optimize/status/prune-m6', () => json(200, job('prune-m6', pollStatus))],
  ]
  useAppStore.setState({ currentJob: job('opt-1', 'completed') })
  await useAppStore.getState().startPruning()
  await tick(10)
  // The user steps back in History to another result meanwhile.
  useAppStore.setState({ currentJob: job('opt-0', 'completed') })
  pollStatus = 'completed'
  await tick(1100)
  assert.equal(useAppStore.getState().currentJob.job_id, 'opt-0')
  assert.equal(useAppStore.getState().pruningJob.status, 'completed')
})

test('M7: ordinary edits do not use keepalive; unloading does', async () => {
  reset()
  useAppStore.getState().captureSnapshot('edit')
  const posts = requests.filter((r) => r.url.startsWith('/api/state/snapshot') || r.url === '/api/project/state')
  assert.equal(posts.length, 2)
  assert.ok(posts.every((r) => r.init.keepalive === false))

  requests.length = 0
  useAppStore.getState().setSettings({ ...useAppStore.getState().settings, iterations: 123 })
  for (const fn of windowListeners.pagehide ?? []) fn()
  const unloading = requests.filter((r) => r.url.startsWith('/api/state/snapshot'))
  assert.equal(unloading.length, 1)
  assert.equal(unloading[0].init.keepalive, true)
  for (const fn of windowListeners.pageshow ?? []) fn()
})

test('L27/L28: stale mesh downloads are aborted and divider drags cleaned up', () => {
  const view = src('components/ThreeDView.tsx')
  assert.match(view, /fetch\(plyUrl, \{ signal: abort\.signal \}\)/)
  assert.match(view, /abort\.abort\(\)/)
  const panel = src('components/InputImagePanel.tsx')
  assert.match(panel, /addEventListener\('pointercancel', up\)/)
  assert.match(panel, /useEffect\(\(\) => \(\) => dividerDragCleanup\.current\?\.\(\), \[\]\)/)
})

test('L29: a loaded project drops uploads this server does not have', async () => {
  reset()
  routes = [
    ['/uploads/mask.png', () => json(404, {})],
    ['/uploads/photo.png', () => json(200, {})],
    ['/api/filaments', () => json(200, [])],
  ]
  useAppStore.setState({ sliderLayerRange: { min: 3, max: 9 } })
  await useAppStore.getState().loadProjectFromFile({
    settings: { ...defaultSettings, input_image: 'photo.png', priority_mask: 'mask.png', max_layers: 40 },
    inputImage: '/uploads/photo.png',
  })
  const s = useAppStore.getState()
  assert.equal(s.settings.priority_mask, '')
  assert.equal(s.settings.input_image, 'photo.png')
  assert.deepEqual(s.sliderLayerRange, { min: 0, max: 40 })

  routes = [['/uploads/', () => json(404, {})], ['/api/filaments', () => json(200, [])]]
  await useAppStore.getState().loadProjectFromFile({
    settings: { ...defaultSettings, input_image: 'gone.png' },
    inputImage: '/uploads/gone.png',
  })
  assert.equal(useAppStore.getState().settings.input_image, '')
  assert.equal(useAppStore.getState().inputImage, null)
})

test('L30: adding a filament that is already active changes nothing', async () => {
  await tick(600) // let the earlier tests' debounced snapshots go out
  reset()
  const fil = { uuid: 'f1', brand: 'B', name: 'Red', color: '#ff0000', td: 2, owned: true, filament_type: 'PLA', source: 'user' }
  useAppStore.setState({ activeFilaments: [fil] })
  await useAppStore.getState().addActiveFilament(fil)
  await tick(600)
  assert.deepEqual(requests.map((r) => r.url), [], 'no request, no undo snapshot')
  assert.equal(useAppStore.getState().activeFilaments.length, 1)
})
