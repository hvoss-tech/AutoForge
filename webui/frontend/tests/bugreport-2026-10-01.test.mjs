// Test runner: node --import tsx/esm --test tests/bugreport-2026-10-01.test.mjs
//
// Regression tests for the frontend fixes from the 2026-10-01 bug report
// (F-1 ... F-7). The store is imported for real, with the few browser
// globals it touches at module scope shimmed.

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const storage = {}
globalThis.localStorage = {
  getItem: (k) => storage[k] ?? null,
  setItem: (k, v) => { storage[k] = String(v) },
  removeItem: (k) => { delete storage[k] },
}
globalThis.window = {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  addEventListener() {},
  removeEventListener() {},
  location: { protocol: 'http:', host: 'localhost' },
}
const clicked = []
globalThis.document = {
  documentElement: { classList: { add() {}, remove() {}, toggle() {} }, setAttribute() {}, style: {} },
  createElement: () => ({ click() { clicked.push(this.download) }, remove() {} }),
  body: { appendChild() {} },
}
const revoked = []
URL.createObjectURL = () => 'blob:test'
URL.revokeObjectURL = (u) => revoked.push(u)

// Routed fetch: tests install handlers by URL prefix.
let routes = []
const requests = []
globalThis.fetch = async (url, init = {}) => {
  requests.push({ url, init })
  for (const [prefix, handler] of routes) {
    if (String(url).startsWith(prefix)) return handler(url, init)
  }
  return { ok: true, status: 200, json: async () => ({}), blob: async () => new Blob([]) }
}
const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

const { useAppStore } = await import('../src/store/appStore.ts')
const { acceptsStatusFor, finishedResultCounts, isFailureTransition, pruningPollStep, PRUNING_POLL_MAX_NETWORK_ERRORS } =
  await import('../src/lib/jobTracking.ts')
const { downloadBlob, releaseReplacedObjectUrl, DOWNLOAD_URL_LIFETIME_MS } = await import('../src/lib/download.ts')
const { addMissingFilaments } = await import('../src/lib/projectLoad.ts')

const job = (id, status, extra = {}) => ({
  job_id: id, status, progress: 0, iteration: 0, total_iterations: 10, loss: null, error: null,
  started_at: null, completed_at: null, preview_image: null, ...extra,
})
const reset = () => {
  routes = []
  requests.length = 0
  revoked.length = 0
  useAppStore.setState({ currentJob: null, toasts: [], pruningJob: null })
}
const toasts = () => useAppStore.getState().toasts.map((t) => t.message)
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms))

test('F-1: a late status for another job does not replace the job on screen', () => {
  reset()
  assert.equal(acceptsStatusFor(null, job('a', 'running')), true)
  assert.equal(acceptsStatusFor(job('b', 'running'), job('a', 'completed')), false)
  assert.equal(acceptsStatusFor(job('b', 'running'), null), true)

  useAppStore.setState({ currentJob: job('b', 'running') })
  useAppStore.getState().setCurrentJob(job('a', 'completed'))
  assert.equal(useAppStore.getState().currentJob.job_id, 'b')
  useAppStore.getState().setCurrentJob(job('b', 'running', { iteration: 5 }))
  assert.equal(useAppStore.getState().currentJob.iteration, 5)
})

test('F-1: the hook aborts in-flight polls and checks the job id', () => {
  const hook = readFileSync(new URL('../src/hooks/useJobWebSocket.ts', import.meta.url), 'utf8')
  assert.match(hook, /pollAbortRef\.current\?\.abort\(\)/)
  assert.match(hook, /data\.job_id !== polledJobId/)
})

test('F-2: the automatic prune keeps the server-reported counts of the result', async () => {
  reset()
  assert.deepEqual(finishedResultCounts({ result_colors: 5, result_swaps: 7, result_layers: 40 }, { colors: 1, swaps: 0, layers: 1 }),
    { colors: 5, swaps: 7, layers: 40 })
  assert.deepEqual(finishedResultCounts({}, { colors: 2, swaps: 3, layers: 4 }), { colors: 2, swaps: 3, layers: 4 })

  let pruneBody = null
  routes = [
    ['/api/pruning/start', (_u, init) => { pruneBody = JSON.parse(init.body); return json(200, { job_id: 'prune-1', status: 'running' }) }],
    ['/api/optimize/status/prune-1', () => json(200, job('prune-1', 'running'))],
  ]
  // Stale sliders from the last training preview: one band, one swap.
  useAppStore.setState({
    currentJob: job('opt-1', 'running'),
    colorSliders: [{ td: 2, layer: 10, depth_mm: 0.4, filament_uuid: 'x', enabled: true }],
    settings: { ...useAppStore.getState().settings, auto_initial_prune: true },
  })
  useAppStore.getState().setCurrentJob(job('opt-1', 'completed', { result_colors: 6, result_swaps: 9, result_layers: 42 }))
  await tick(10)
  assert.ok(pruneBody, 'the automatic prune started')
  assert.equal(pruneBody.pruning_max_colors, 6)
  assert.equal(pruneBody.pruning_max_swaps, 9)
  assert.equal(pruneBody.pruning_max_layer, 42)
  useAppStore.setState({ settings: { ...useAppStore.getState().settings, auto_initial_prune: false } })
})

test('F-3: the pruning poll stops when the server forgets the job', async () => {
  reset()
  assert.equal(pruningPollStep({ status: 404 }, 0), 'lost')
  assert.equal(pruningPollStep({ status: 200, jobStatus: 'running' }, 0), 'continue')
  assert.equal(pruningPollStep({ status: 200, jobStatus: 'completed' }, 0), 'done')
  assert.equal(pruningPollStep(null, PRUNING_POLL_MAX_NETWORK_ERRORS - 1), 'continue')
  assert.equal(pruningPollStep(null, PRUNING_POLL_MAX_NETWORK_ERRORS), 'lost')

  routes = [
    ['/api/pruning/start', () => json(200, { job_id: 'prune-2', status: 'running' })],
    ['/api/optimize/status/prune-2', () => json(404, { detail: 'Job not found' })],
  ]
  useAppStore.setState({ currentJob: job('opt-2', 'completed') })
  await useAppStore.getState().startPruning()
  await tick(10)
  assert.equal(useAppStore.getState().pruningJob.status, 'cancelled')
  const polls = requests.filter((r) => r.url === '/api/optimize/status/prune-2').length
  await tick(1100)
  assert.equal(requests.filter((r) => r.url === '/api/optimize/status/prune-2').length, polls, 'no more polling')
  assert.ok(toasts().some((m) => m.includes('Lost track of the pruning run')))
})

test('F-3: pausing a prune that already finished shows its real state', async () => {
  reset()
  routes = [
    ['/api/optimize/pause/prune-3', () => json(200, { status: 'completed' })],
    ['/api/optimize/status/prune-3', () => json(200, job('prune-3', 'completed'))],
  ]
  useAppStore.setState({ pruningJob: job('prune-3', 'running') })
  await useAppStore.getState().pausePruning('prune-3')
  assert.equal(useAppStore.getState().pruningJob.status, 'completed')
})

test('F-4: a failure raises one toast whichever channel reports it', () => {
  reset()
  assert.equal(isFailureTransition('running', job('a', 'failed')), true)
  assert.equal(isFailureTransition('failed', job('a', 'failed')), false)
  useAppStore.setState({ currentJob: job('f', 'running') })
  // e.g. the 1 s early poll saw it first ...
  useAppStore.getState().setCurrentJob(job('f', 'failed', { error: 'Input image not found\n\ntraceback' }))
  // ... and then the socket delivers the same status.
  useAppStore.getState().setCurrentJob(job('f', 'failed', { error: 'Input image not found\n\ntraceback' }))
  assert.deepEqual(toasts().filter((m) => m.startsWith('Optimization failed')), ['Optimization failed: Input image not found'])
})

test('F-5: replacing an uploaded image releases its local preview URL', async () => {
  reset()
  releaseReplacedObjectUrl('blob:old', 'blob:new')
  releaseReplacedObjectUrl('/uploads/a.png', 'blob:new')
  releaseReplacedObjectUrl('blob:same', 'blob:same')
  assert.deepEqual(revoked, ['blob:old'])

  revoked.length = 0
  useAppStore.setState({ inputImage: 'blob:first-upload' })
  await useAppStore.getState().applyUploadedImage('second.png', 'blob:second-upload')
  assert.deepEqual(revoked, ['blob:first-upload'])
})

test('F-6: downloads keep their object URL alive past the click', async () => {
  reset()
  const realSetTimeout = globalThis.setTimeout
  const scheduled = []
  globalThis.setTimeout = (fn, ms) => { scheduled.push(ms); return realSetTimeout(() => {}, 0) }
  try {
    downloadBlob(new Blob(['x']), 'a.txt')
  } finally {
    globalThis.setTimeout = realSetTimeout
  }
  assert.deepEqual(revoked, [], 'not revoked synchronously')
  assert.deepEqual(scheduled, [DOWNLOAD_URL_LIFETIME_MS])
  for (const file of ['components/ImportModal.tsx', 'components/FilamentLibrary.tsx', 'components/PrintPlanPanel.tsx', 'store/appStore.ts', 'components/FileMenu.tsx']) {
    const src = readFileSync(new URL(`../src/${file}`, import.meta.url), 'utf8')
    assert.ok(!src.includes('URL.revokeObjectURL(url)'), `${file} still revokes right after the click`)
  }
})

test('F-7: refused filaments in a loaded project are reported and left out', async () => {
  reset()
  const good = { uuid: 'g', brand: 'Acme', name: 'Good', color: '#112233', td: 2, owned: false }
  const bad = { uuid: 'b', brand: 'Acme', name: 'Bad', color: 'red', td: 2, owned: false }
  const result = await addMissingFilaments([good, bad], new Set(), async (f) =>
    f.uuid === 'b' ? { ok: false, status: 422, detail: 'not a hex color' } : { ok: true, status: 200 })
  assert.equal(result.libraryChanged, true)
  assert.deepEqual(result.failed.map((x) => [x.filament.uuid, x.detail]), [['b', 'not a hex color']])

  let activeBody = null
  routes = [
    ['/api/init/reset', () => json(200, {})],
    ['/api/filaments/active', (_u, init) => { activeBody = JSON.parse(init.body); return json(200, activeBody) }],
    ['/api/filaments', (_u, init) => {
      if (init.method === 'POST') {
        const f = JSON.parse(init.body)
        return f.uuid === 'b' ? json(422, { detail: "'red' is not a hex color like #RRGGBB" }) : json(200, f)
      }
      return json(200, [])
    }],
  ]
  await useAppStore.getState().loadProjectFromFile({ activeFilaments: [good, bad], colorSliders: [] }, 'p.json')
  assert.deepEqual(activeBody.map((f) => f.uuid), ['g'])
  assert.ok(toasts().some((m) => m.includes('Acme - Bad') && m.includes('hex color')))
})
