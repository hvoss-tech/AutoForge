import { test, expect } from '@playwright/test'
import zlib from 'zlib'
import * as fs from 'fs'

// Tiny 32x32 four-quadrant image (red/green/blue/yellow) so optimization runs fast.
const W = 32
const raw = Buffer.alloc(W * W * 3)
for (let y = 0; y < W; y++) {
  for (let x = 0; x < W; x++) {
    const q = (x < W / 2 ? 0 : 1) + (y < W / 2 ? 0 : 2)
    const o = (y * W + x) * 3
    raw[o] = q === 0 ? 255 : q === 1 ? 30 : q === 2 ? 40 : 255
    raw[o + 1] = q === 0 ? 30 : q === 1 ? 255 : q === 2 ? 40 : 220
    raw[o + 2] = q === 0 ? 30 : q === 1 ? 30 : q === 2 ? 255 : 30
  }
}
const stride = W * 3 + 1
const scanlines = Buffer.alloc(stride * W)
for (let y = 0; y < W; y++) {
  scanlines[y * stride] = 0
  raw.copy(scanlines, y * stride + 1, y * W * 3, (y + 1) * W * 3)
}
const chunk = (type: string, payload: Buffer) => {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(payload.length)
  const typeBuf = Buffer.from(type)
  const crcTable: number[] = []
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    crcTable[n] = c >>> 0
  }
  let c = 0xffffffff
  for (const b of Buffer.concat([typeBuf, payload])) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8)
  const crcBuf = Buffer.alloc(4)
  crcBuf.writeUInt32BE((c ^ 0xffffffff) >>> 0)
  return Buffer.concat([len, typeBuf, payload, crcBuf])
}
const ihdr = Buffer.alloc(13)
ihdr.writeUInt32BE(W, 0)
ihdr.writeUInt32BE(W, 4)
ihdr[8] = 8
ihdr[9] = 2
const PNG = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', zlib.deflateSync(scanlines)),
  chunk('IEND', Buffer.alloc(0)),
])

const waitForStatus = async (request: any, jobId: string) => {
  for (let i = 0; i < 240; i++) {
    await new Promise((r) => setTimeout(r, 500))
    const res = await request.get(`/api/optimize/status/${jobId}`)
    const body = await res.json()
    if (body.status === 'completed' || body.status === 'failed') return body
  }
  throw new Error(`job ${jobId} did not finish`)
}

test('sliders and color core stay populated after optimization and pruning (live WS)', async ({ page, request }) => {
  // Clear any active filaments left over from previous test runs so the
  // optimizer only sees the filaments created below.
  const active = await request.get('/api/filaments/active')
  for (const f of await active.json()) {
    await request.delete(`/api/filaments/active/${f.uuid}`)
  }

  const upload = await request.post('/api/images/upload', {
    multipart: { file: { name: 'quad.png', mimeType: 'image/png', buffer: PNG } },
  })
  expect(upload.ok()).toBeTruthy()
  const filename = (await upload.json()).filename

  for (const [name, color, td] of [
    ['Red', '#FF0000', 1.0],
    ['Green', '#00FF00', 2.0],
    ['Blue', '#0000FF', 3.0],
    ['Yellow', '#FFFF00', 4.0],
  ]) {
    const created = await request.post('/api/filaments', {
      data: { brand: 'E2E', name, color, td, filament_type: 'PLA' },
    })
    expect(created.ok()).toBeTruthy()
    const filament = await created.json()
    await request.post('/api/filaments/active', { data: filament })
  }

  // Open the UI BEFORE optimizing so it is subscribed to /ws/preview.
  await page.goto('/')

  // Run a short optimization. The backend fires preview_callback at the final
  // iteration and broadcasts the derived slider stack over the WebSocket.
  const start = await request.post('/api/optimize/start', {
    data: {
      input_image: filename,
      iterations: 4,
      max_layers: 12,
      layer_height: 0.04,
      background_height: 0.12,
      stl_output_size: 20,
      processing_reduction_factor: 1,
      random_seed: 42,
      num_init_rounds: 1,
      num_init_cluster_layers: 4,
      learning_rate: 0.01,
      init_tau: 1.0,
      final_tau: 0.5,
      early_stopping: 1000,
      visualize: false,
      perform_pruning: false,
      num_init_threads: 1,
      best_of: 1,
      discrete_check: 1,
      csv_file: '',
      json_file: '',
    },
  })
  expect(start.ok()).toBeTruthy()
  const optJobId = (await start.json()).job_id
  const optResult = await waitForStatus(request, optJobId)
  expect(optResult.status).toBe('completed')

  // The WS broadcast must have repopulated the sliders (bug: sliders:[] wiped them).
  // Other specs may run in parallel and broadcast their own stacks, so we assert
  // the invariant (sliders present with a real derived range) rather than exact
  // values: the default sliderLayerRange max is 75, so a non-"75" max proves a
  // solution-derived range was applied instead of leaving the store untouched.
  // The column count is NOT fixed at 15 — it matches however many distinct
  // layer bands the optimizer actually produced (can be more or fewer), so
  // just assert that some real columns showed up.
  await expect(page.locator('[data-testid^="slider-column-"]').first()).toBeVisible({ timeout: 30000 })
  expect(await page.locator('[data-testid^="slider-column-"]').count()).toBeGreaterThan(0)
  await expect(page.locator('[data-testid^="color-core-handle-"]').first()).toBeVisible({ timeout: 30000 })
  const optimizeMax = await page.locator('[data-testid="slider-0"]').getAttribute('max')
  expect(optimizeMax).not.toBe('75')
  expect(await page.locator('[data-testid^="color-core-handle-"]').count()).toBeGreaterThan(0)

  // Prune down to max_layer 6 and verify the UI follows the reduced stack.
  const pruneResp = await request.post('/api/pruning/start', {
    data: { pruning_max_colors: 4, pruning_max_swaps: 4, pruning_max_layer: 6, job_id: optJobId },
  })
  expect(pruneResp.ok()).toBeTruthy()
  const pruneResult = await waitForStatus(request, (await pruneResp.json()).job_id)
  expect(pruneResult.status).toBe('completed')

  // Sliders must still be present after pruning completes — again, however
  // many bands pruning left (pruning_max_colors=4 above bounds it, but not
  // to exactly any fixed UI column count).
  const columns = page.locator('[data-testid^="slider-column-"]')
  await expect(columns.first()).toBeVisible({ timeout: 30000 })
  expect(await columns.count()).toBeGreaterThan(0)
  const handles = page.locator('[data-testid^="color-core-handle-"]')
  expect(await handles.count()).toBeGreaterThan(0)

  // Enabled slider positions must be within their own range bound. Because the
  // suite may run other optimize jobs in parallel (shared backend + global WS
  // broadcasts), compare each slider against the range of its own input rather
  // than a hardcoded bound — this still fails if sliders are wiped or positions
  // exceed the printed layer range.
  const values = await page.evaluate(() =>
    Array.from(document.querySelectorAll('[data-testid^="slider-column-"]'))
      .map((col) => {
        const range = col.querySelector('[data-testid^="slider-"]') as HTMLInputElement
        const toggle = col.querySelector('[data-testid^="toggle-"]') as HTMLButtonElement
        return {
          layer: Number(range.value),
          min: Number(range.min),
          max: Number(range.max),
          enabled: !toggle.textContent?.includes('+'),
        }
      })
      .filter((s) => s.enabled && s.layer > 0),
  )
  expect(values.length).toBeGreaterThan(0)
  for (const v of values) {
    expect(v.layer).toBeGreaterThanOrEqual(v.min)
    expect(v.layer).toBeLessThanOrEqual(v.max)
  }
})

test('slider column count is not capped at 15 — it matches the real result exactly, with horizontal scroll', async ({ page, request }) => {
  // Regression test: the backend used to merge segments down to a fixed 15
  // and the frontend separately padded/truncated to 15 too, so a result
  // with a different number of real layer bands than 15 got silently
  // rewritten — and since render-with-sliders reconstructs the whole
  // composite/mesh from exactly this slider list, that didn't just mis-draw
  // the UI, it changed the actual 3D preview colors. Neither cap exists
  // anymore; the UI must show exactly what the backend derives.
  const active = await request.get('/api/filaments/active')
  for (const f of await active.json()) {
    await request.delete(`/api/filaments/active/${f.uuid}`)
  }

  const upload = await request.post('/api/images/upload', {
    multipart: { file: { name: 'quad2.png', mimeType: 'image/png', buffer: PNG } },
  })
  const filename = (await upload.json()).filename

  for (const [name, color, td] of [
    ['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0],
    ['Yellow', '#FFFF00', 4.0],
  ]) {
    const created = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
    const filament = await created.json()
    await request.post('/api/filaments/active', { data: filament })
  }

  await page.goto('/')

  const start = await request.post('/api/optimize/start', {
    data: {
      input_image: filename, iterations: 4, max_layers: 12, layer_height: 0.04,
      background_height: 0.12, stl_output_size: 20, processing_reduction_factor: 1,
      random_seed: 42, num_init_rounds: 1, num_init_cluster_layers: 4, learning_rate: 0.01,
      init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
      perform_pruning: false, best_of: 1, discrete_check: 1, csv_file: '', json_file: '',
    },
  })
  expect(start.ok()).toBeTruthy()
  const jobId = (await start.json()).job_id
  const result = await waitForStatus(request, jobId)
  expect(result.status).toBe('completed')

  // Ask the backend directly for the derived slider stack — the authority
  // on how many bands the real solution has — and compare against the UI.
  const derived = await (await request.get('/api/sliders/from-optimizer')).json()
  const expectedCount = derived.sliders.length
  expect(expectedCount).toBeGreaterThan(0)

  await expect(page.locator('[data-testid^="slider-column-"]').first()).toBeVisible({ timeout: 30000 })
  await expect(page.locator('[data-testid^="slider-column-"]')).toHaveCount(expectedCount, { timeout: 10000 })

  // The scroll container must actually allow horizontal scrolling (not
  // clip or wrap) so a count that overflows the panel width stays usable.
  const overflowX = await page.locator('[data-testid="slider-columns"]').evaluate(
    (el) => getComputedStyle(el).overflowX,
  )
  expect(overflowX).toBe('auto')
})
