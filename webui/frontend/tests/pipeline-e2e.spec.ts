import { test, expect } from '@playwright/test'

const TINY_PNG_BYTES = Buffer.from([
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
  0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04,
  0x08, 0x02, 0x00, 0x00, 0x00, 0x26, 0x93, 0x09, 0x29, 0x00, 0x00, 0x00,
  0x1D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xFC, 0xCF, 0x80, 0x00,
  0x2C, 0x0C, 0x0C, 0x0C, 0x8C, 0x30, 0x01, 0x46, 0x86, 0xFF, 0x08, 0x1E,
  0x23, 0x03, 0x03, 0x42, 0x21, 0x00, 0x8B, 0xD8, 0x05, 0x05, 0x57, 0x19,
  0xF0, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
  0x60, 0x82,
])

test.describe('Pipeline E2E Tests', () => {
  // Only one optimization may run at a time; don't let a test that stops
  // watching its job early (e.g. once progress shows up) block the next one.
  test.afterEach(async ({ request }) => {
    const latest = await request.get('/api/optimize/latest')
    if (!latest.ok()) return
    const job = await latest.json()
    if (!['pending', 'running', 'paused'].includes(job.status)) return
    await request.post(`/api/optimize/cancel/${job.job_id}`)
    for (let i = 0; i < 100; i++) {
      const s = await (await request.get(`/api/optimize/status/${job.job_id}`)).json()
      if (s.status !== 'running' && s.status !== 'paused' && s.status !== 'pending') break
      await new Promise((r) => setTimeout(r, 200))
    }
    // The worker thread notices the cancel between iterations; give it a moment.
    await new Promise((r) => setTimeout(r, 1000))
  })


  test('upload image and add active filaments via API', async ({ request }) => {
    // Upload a test image
    const uploadResp = await request.post('/api/images/upload', {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES }
      }
    })
    expect(uploadResp.ok()).toBeTruthy()
    const { filename } = await uploadResp.json()
    expect(filename).toBeTruthy()

    // Create and activate filaments
    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const fResp = await request.post('/api/filaments', {
        data: { brand: 'E2ETest', name, color, td, filament_type: 'PLA' }
      })
      expect(fResp.ok()).toBeTruthy()
      const filament = await fResp.json()
      const aResp = await request.post('/api/filaments/active', { data: filament })
      expect(aResp.ok()).toBeTruthy()
    }

    // Verify active filaments
    const activeResp = await request.get('/api/filaments/active')
    expect(activeResp.ok()).toBeTruthy()
    const active = await activeResp.json()
    expect(active.length).toBeGreaterThanOrEqual(3)
  })

  test('optimization runs and produces 2D preview', async ({ request }) => {
    // Setup
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename } = await uploadResp.json()

    // Create and activate filaments
    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await f.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    // Start optimization
    const startResp = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 2, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp.ok()).toBeTruthy()
    const { job_id } = await startResp.json()
    expect(job_id).toBeDefined()

    // Poll until completion
    const deadline = Date.now() + 120_000
    let finalStatus: any = null
    while (Date.now() < deadline) {
      const sr = await request.get(`/api/optimize/status/${job_id}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (['completed', 'failed', 'cancelled'].includes(s.status)) { finalStatus = s; break }
      }
      await new Promise(r => setTimeout(r, 500))
    }
    expect(finalStatus).not.toBeNull()
    expect(finalStatus.status).toBe('completed')
    expect(finalStatus.progress).toBe(100)

    // Verify preview image generated
    const previewResp = await request.get(`/api/outputs/preview/${job_id}`)
    expect(previewResp.ok()).toBeTruthy()
  })

  test('full pipeline: optimize → prune → verify STL and colored PLY', async ({ request }) => {
    // Upload image
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename } = await uploadResp.json()

    // Create and activate filaments
    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await f.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    // Run optimization
    const startResp = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 2, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp.ok()).toBeTruthy()
    const { job_id } = await startResp.json()

    // Wait for optimization
    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      const sr = await request.get(`/api/optimize/status/${job_id}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed') break
        if (s.status === 'failed') { expect.fail(`Optimization failed: ${s.error}`) }
      }
      await new Promise(r => setTimeout(r, 500))
    }

    // Upload image again (after optimization, settings may have changed)
    const uploadResp2 = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename: filename2 } = await uploadResp2.json()

    // Ensure active filaments still exist
    const activeResp = await request.get('/api/filaments/active')
    if ((await activeResp.json()).length === 0) {
      for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
        const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
        const filament = await f.json()
        await request.post('/api/filaments/active', { data: filament })
      }
    }

    // Start a 2nd optimization to get a pipeline result for pruning
    const startResp2 = await request.post('/api/optimize/start', {
      data: {
        input_image: filename2,
        iterations: 2, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp2.ok()).toBeTruthy()
    const { job_id: job_id2 } = await startResp2.json()

    // Wait for optimization 2
    const deadline2 = Date.now() + 120_000
    while (Date.now() < deadline2) {
      const sr = await request.get(`/api/optimize/status/${job_id2}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed') break
        if (s.status === 'failed') { expect.fail(`Optimization 2 failed: ${s.error}`) }
      }
      await new Promise(r => setTimeout(r, 500))
    }

    // Now run pruning
    const pruneResp = await request.post('/api/pruning/start', {
      data: { pruning_max_colors: 5, pruning_max_swaps: 5, pruning_max_layer: 5, job_id: job_id2 }
    })
    expect(pruneResp.ok()).toBeTruthy()
    const { job_id: pruneJobId } = await pruneResp.json()

    // Wait for pruning
    const deadline3 = Date.now() + 120_000
    while (Date.now() < deadline3) {
      const sr = await request.get(`/api/optimize/status/${pruneJobId}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed') break
        if (s.status === 'failed') { expect.fail(`Prune failed: ${s.error}`) }
      }
      await new Promise(r => setTimeout(r, 500))
    }

    // Verify output files exist via API
    // STL
    const stlResp = await request.get(`/api/outputs/stl/${job_id2}`)
    expect(stlResp.ok()).toBeTruthy()

    // Colored PLY
    const plyResp = await request.get(`/api/outputs/colored-ply/${job_id2}`)
    expect(plyResp.ok()).toBeTruthy()

    // Preview PNG
    const previewResp = await request.get(`/api/outputs/preview/${job_id2}`)
    expect(previewResp.ok()).toBeTruthy()
  }, 300_000) // 5 min timeout

  test('error when no active filaments', async ({ request }) => {
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    expect(uploadResp.ok()).toBeTruthy()
    const { filename } = await uploadResp.json()

    await request.put('/api/filaments/active', { data: [] })

    // Rejected up front — no job is created that would briefly look "running".
    const startResp = await request.post('/api/optimize/start', { data: { input_image: filename, iterations: 2 } })
    expect(startResp.status()).toBe(400)
    expect((await startResp.json()).detail).toContain('active filament')
  })

  test('pause, resume and cancel an optimization job from the UI', async ({ page }) => {
    await page.goto('/')

    // Upload the image through the real drop-zone file input, which is what
    // actually wires up store.inputImage + settings.input_image — uploading
    // via the raw API alone (as the other tests here do) leaves the
    // frontend unaware of the image, so the UI "Run" button would fail with
    // "No input image specified".
    await page.locator('[data-testid="image-file-input"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: TINY_PNG_BYTES,
    })
    await expect(page.locator('[data-testid="input-image"]')).toBeVisible()

    // Activate a real filament from the library via its "+" toggle.
    const toggle = page.locator('[data-testid^="toggle-filament-"]').first()
    await toggle.waitFor({ state: 'visible', timeout: 10000 })
    await toggle.click()
    await expect(page.locator('[data-testid="active-filaments-summary"]')).not.toHaveAttribute('data-count', '0')

    // Use a large-ish iteration count so there's a real window to pause
    // within before the job would naturally complete.
    await page.locator('[data-testid="settings-button"]').click()
    await page.locator('[data-testid="setting-iterations"]').fill('20000')
    await page.locator('[data-testid="setting-max_layers"]').fill('3')
    await page.locator('[data-testid="close-settings"]').click()

    await page.locator('[data-testid="top-start-btn"]').click()
    await expect(page.locator('[data-testid="top-pause-btn"]')).toBeVisible({ timeout: 15000 })

    // --- Pause ---
    await page.locator('[data-testid="top-pause-btn"]').click()
    await expect(page.locator('[data-testid="top-resume-btn"]')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('[data-testid="top-pause-btn"]')).not.toBeVisible()

    // The job must actually stop advancing while paused, not just show a
    // paused label in the UI.
    const iterationText = page.locator('[data-testid="job-iteration-count"]')
    const pausedIteration = await iterationText.textContent()
    await page.waitForTimeout(1500)
    const stillPausedIteration = await iterationText.textContent()
    expect(stillPausedIteration).toBe(pausedIteration)

    // --- Resume ---
    await page.locator('[data-testid="top-resume-btn"]').click()
    await expect(page.locator('[data-testid="top-pause-btn"]')).toBeVisible({ timeout: 10000 })

    // --- Cancel ---
    await page.locator('[data-testid="top-cancel-btn"]').click()
    await expect(page.locator('[data-testid="top-start-btn"]')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('[data-testid="top-pause-btn"]')).not.toBeVisible()
    await expect(page.locator('[data-testid="top-resume-btn"]')).not.toBeVisible()
  })

  test('progress updates appear before iteration 100', async ({ request }) => {
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename } = await uploadResp.json()

    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await f.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    const startResp = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 200, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp.ok()).toBeTruthy()
    const { job_id } = await startResp.json()

    // Poll. The very first status should already show iteration > 0.
    const deadline = Date.now() + 60_000
    let seenProgress = false
    while (Date.now() < deadline) {
      const sr = await request.get(`/api/optimize/status/${job_id}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed' || s.status === 'failed') {
          seenProgress = s.iteration > 0 || s.progress > 0
          break
        }
        if (s.iteration > 0 && s.iteration < 100) {
          seenProgress = true
          break
        }
      }
      await new Promise(r => setTimeout(r, 200))
    }
    expect(seenProgress).toBeTruthy()
  })

  test('optimization produces all output files (STL, colored PLY, preview PNG)', async ({ request }) => {
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename } = await uploadResp.json()

    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await f.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    const startResp = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 2, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp.ok()).toBeTruthy()
    const { job_id } = await startResp.json()

    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      const sr = await request.get(`/api/optimize/status/${job_id}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed') break
        if (s.status === 'failed') { expect.fail(`Optimization failed: ${s.error}`) }
      }
      await new Promise(r => setTimeout(r, 500))
    }

    // All three output files must return 200, not 404
    const stl = await request.get(`/api/outputs/stl/${job_id}`)
    expect(stl.ok()).toBeTruthy()

    const ply = await request.get(`/api/outputs/colored-ply/${job_id}`)
    expect(ply.ok()).toBeTruthy()

    const preview = await request.get(`/api/outputs/preview/${job_id}`)
    expect(preview.ok()).toBeTruthy()
  })

  test('colored PLY is parseable after optimization', async ({ request, page }) => {
    const uploadResp = await request.post('/api/images/upload', {
      multipart: { file: { name: 'test.png', mimeType: 'image/png', buffer: TINY_PNG_BYTES } }
    })
    const { filename } = await uploadResp.json()

    for (const [name, color, td] of [['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0]]) {
      const f = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await f.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    const startResp = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 2, max_layers: 3, layer_height: 0.04, background_height: 0.12,
        stl_output_size: 20, processing_reduction_factor: 1, random_seed: 42,
        num_init_rounds: 1, num_init_cluster_layers: 3, learning_rate: 0.01,
        init_tau: 1.0, final_tau: 0.5, early_stopping: 1000, visualize: false,
        perform_pruning: false, num_init_threads: 1, best_of: 1, discrete_check: 1,
        csv_file: '', json_file: '',
      }
    })
    expect(startResp.ok()).toBeTruthy()
    const { job_id } = await startResp.json()

    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      const sr = await request.get(`/api/optimize/status/${job_id}`)
      if (sr.ok()) {
        const s = await sr.json()
        if (s.status === 'completed') break
        if (s.status === 'failed') { expect.fail(`Optimization failed: ${s.error}`) }
      }
      await new Promise(r => setTimeout(r, 500))
    }

    const plyResp = await request.get(`/api/outputs/colored-ply/${job_id}`)
    expect(plyResp.ok()).toBeTruthy()
    const plyBuf = await plyResp.body()

    const decoder = new TextDecoder('ascii')
    const headerBytes = new Uint8Array(plyBuf.slice(0, Math.min(plyBuf.byteLength, 4096)))
    const headerText = decoder.decode(headerBytes)
    const end = headerText.indexOf('end_header')
    expect(end).toBeGreaterThan(0)
    const headerLines = headerText.substring(0, end).split('\n').filter(l => l.trim())

    let vertexCount = 0
    let faceCount = 0
    let vertexStride = 0
    let colorStride = 0
    let hasColors = false
    for (const line of headerLines) {
      const parts = line.trim().split(/\s+/)
      if (parts[0] === 'element') {
        if (parts[1] === 'vertex') vertexCount = parseInt(parts[2])
        if (parts[1] === 'face') faceCount = parseInt(parts[2])
      } else if (parts[0] === 'property') {
        const typeSizes: Record<string, number> = { float: 4, double: 8, uchar: 1, char: 1, ushort: 2, short: 2, int: 4, uint: 4 }
        const byteSize = typeSizes[parts[1]]
        if (byteSize === undefined) continue
        vertexStride += byteSize
        const name = parts[2].toLowerCase()
        if (name === 'red' || name === 'green' || name === 'blue' || name === 'r' || name === 'g' || name === 'b') {
          colorStride += byteSize
          hasColors = true
        }
      }
    }

    expect(hasColors).toBeTruthy()
    expect(colorStride).toBe(3)
    expect(vertexStride).toBe(16)

    const dv = new DataView(plyBuf.buffer, plyBuf.byteOffset, plyBuf.byteLength)
    const headerLen = end + 'end_header'.length + 1
    let offset = headerLen
    const positions = new Float32Array(3)
    positions[0] = dv.getFloat32(offset, true); offset += 4
    positions[1] = dv.getFloat32(offset, true); offset += 4
    positions[2] = dv.getFloat32(offset, true); offset += 4

    const r = dv.getUint8(offset); offset += 1
    const g = dv.getUint8(offset); offset += 1
    const b = dv.getUint8(offset); offset += 1
    offset += 1

    expect(Number.isFinite(positions[0])).toBeTruthy()
    expect(Number.isFinite(positions[1])).toBeTruthy()
    expect(Number.isFinite(positions[2])).toBeTruthy()
  }, 300_000)

  test('non-existent job returns not_found via status API', async ({ request }) => {
    const sr = await request.get('/api/optimize/status/does-not-exist')
    expect(await sr.json()).toHaveProperty('detail')
  })

  test('optimize start requires input_image', async ({ request }) => {
    await request.post('/api/filaments/active', { data: { uuid: 'e2e-needs-image', brand: 'E2E', name: 'Img', color: '#ff0000', td: 1 } })
    const sr = await request.post('/api/optimize/start', {
      data: { iterations: 1, max_layers: 1, visualize: false, csv_file: '', json_file: '', input_image: '' }
    })
    expect(sr.status()).toBe(400)
    expect((await sr.json()).detail).toContain('input image')
  })

  test('pruning without completed optimization returns error', async ({ request }) => {
    const sr = await request.post('/api/pruning/start', {
      data: { pruning_max_colors: 5, pruning_max_swaps: 5, pruning_max_layer: 5 }
    })
    // Pruning is always pinned to an explicit job_id (the result on screen),
    // so a request without one is rejected regardless of job history.
    expect(sr.status()).toBe(400)
    expect((await sr.json()).detail).toContain('Run an optimization first')
  })

  test('filament CRUD is idempotent', async ({ request }) => {
    // Create
    const c = await request.post('/api/filaments', {
      data: { brand: 'TestBrand', name: 'TestFilament', color: '#123456', td: 1.5, filament_type: 'PLA' }
    })
    expect(c.ok()).toBeTruthy()
    const f = await c.json()
    expect(f.uuid).toBeTruthy()

    // Read back
    const list = await request.get('/api/filaments')
    expect(list.ok()).toBeTruthy()
    const filaments = await list.json()
    expect(filaments.some((x: any) => x.uuid === f.uuid)).toBeTruthy()

    // Delete
    const d = await request.delete(`/api/filaments/${f.uuid}`)
    expect(d.ok()).toBeTruthy()

    // Verify deleted
    const list2 = await request.get('/api/filaments')
    const filaments2 = await list2.json()
    expect(filaments2.some((x: any) => x.uuid === f.uuid)).toBeFalsy()
  })

  test('active filaments are persisted after creation', async ({ request }) => {
    // Clear active
    for (const f of await (await request.get('/api/filaments/active')).json()) {
      await request.delete(`/api/filaments/active/${f.uuid}`)
    }

    // Create and activate a filament
    const c = await request.post('/api/filaments', {
      data: { brand: 'PersistTest', name: 'Red', color: '#FF0000', td: 1.0, filament_type: 'PLA' }
    })
    const f = await c.json()
    await request.post('/api/filaments/active', { data: f })

    // Verify active
    const active = await request.get('/api/filaments/active')
    const activeList = await active.json()
    expect(activeList.some((x: any) => x.uuid === f.uuid)).toBeTruthy()
  })
})
