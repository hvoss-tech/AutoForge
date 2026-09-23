import { test, expect, type Page } from '@playwright/test'
import { activePalette, byTestId, createFilament, openApp, resetBackend, setActive, uploadAndWaitForPreview } from '../helpers'

/**
 * The 2026-09-23 round of WebUI bug-report fixes, in the browser.
 *
 * Backend-only items (GPU-mutex, CUDA graph release, cancel-before-start,
 * concurrent init builds, job input_image, FlatForge export contents,
 * preview-image exposure, corrupted history files, update-check blocking,
 * brand filtering, delete/update propagation, base-color job pinning) are
 * covered by tests/webui/test_review_2026_09_23.py. This file covers the
 * frontend halves that need a real DOM: the ColorCore drag runaway, the
 * filament auto-save debounce, band-drop feedback, FlatForge downloads in
 * FileMenu, and the job WebSocket's not_found handling.
 */

const libraryItem = (page: Page, uuid: string) => page.locator(`[data-testid="filament-${uuid}"]`)

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

// --- 15. ColorCore: dragging the topmost handle must not run away ----------

test.describe('ColorCore drag', () => {
  test('dragging the topmost handle a small amount moves it a proportional amount, not to the layer max', async ({ page }) => {
    await openApp(page)
    // Default columns: index 3 is the topmost (layer 27 of a 75 max).
    const handle = byTestId(page, 'color-core-handle-3')
    const before = Number(await handle.getAttribute('data-layer'))
    const box = (await handle.boundingBox())!

    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down()
    // Many small steps upward — this is exactly the shape of movement the
    // bug turned into a runaway: `segmentHeight` was recomputed from the
    // handle's own (just-moved) layer on every one of these intermediate
    // mousemove events, so each step's denominator kept shrinking and the
    // next same-sized step mapped to a larger and larger layer jump.
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 - 40, { steps: 40 })
    await page.mouse.up()

    const after = Number(await byTestId(page, 'color-core-handle-3').getAttribute('data-layer'))
    expect(after).toBeGreaterThan(before)
    // The bug pinned this at (or within a couple of layers of) the max
    // within a handful of steps, regardless of how small the drag was.
    expect(after).toBeLessThan(70)
    await expect(byTestId(page, 'layer-input-3')).toHaveValue(String(after))
  })
})

// --- 16. Filament auto-save must re-arm on every edit, and flush on close --

test.describe('Filament library auto-save', () => {
  test('an edit made after the first debounced save fired is still auto-saved', async ({ page, request }) => {
    const f = await createFilament(request, { td: 2 })
    await openApp(page)
    await libraryItem(page, f.uuid).dblclick()

    // Slow the PUT down so the save the first edit's 400ms debounce fires
    // has not resolved (and so `baseline`/`dirty` haven't reset) by the
    // time a second edit lands — this is what a drag that runs long than
    // 400ms looks like: `dirty` stays continuously true across it, so a
    // timer that only re-arms on the false->true transition (rather than on
    // every field change) never fires again for the rest of the drag.
    let putCount = 0
    await page.route(`**/api/filaments/${f.uuid}`, async (route) => {
      if (route.request().method() !== 'PUT') return route.continue()
      putCount++
      await new Promise((r) => setTimeout(r, 1200))
      await route.continue()
    })

    await byTestId(page, 'edit-filament-td').fill('3.5')
    await page.waitForTimeout(600) // past the 400ms debounce: the first (slow) PUT is now in flight
    await byTestId(page, 'edit-filament-td').fill('7.25')
    await page.waitForTimeout(2200) // long enough for both slowed PUTs to land

    expect(putCount).toBeGreaterThanOrEqual(2)
    const library = await (await request.get('/api/filaments')).json()
    expect(library.find((x: { uuid: string }) => x.uuid === f.uuid)?.td).toBe(7.25)
    await byTestId(page, 'close-edit-filament-btn').click()
  })

  test('closing the dialog inside the debounce window still saves the edit', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Flush Before' })
    await openApp(page)
    await libraryItem(page, f.uuid).dblclick()

    await byTestId(page, 'edit-filament-name').fill('E2E Flush After')
    // Close immediately — well inside the 400ms debounce window. The old
    // behavior just clearTimeout'd the pending save away.
    await byTestId(page, 'close-edit-filament-btn').click()
    await expect(byTestId(page, 'edit-filament-modal')).toHaveCount(0)

    const library = await (await request.get('/api/filaments')).json()
    expect(library.find((x: { uuid: string }) => x.uuid === f.uuid)?.name).toBe('E2E Flush After')
  })
})

// --- Band drop onto an unplaced row: rejected with feedback -----------------

test.describe('Band reordering', () => {
  test('dropping a band onto an unplaced column is rejected with a toast, not a silent no-op', async ({ page }) => {
    await openApp(page)
    // Defaults: columns 0-3 are placed (layers 8/13/20/27); columns 4-14 are
    // unplaced (layer 0, disabled) and sit at the end of the list.
    //
    // Dispatched manually rather than via locator.dragTo(): Playwright's
    // composite drag action does not reliably deliver dragover/drop to rows
    // this far down the scrollable list in this environment (verified with
    // the same custom MIME type against a nearer row, which does work via
    // dragTo — this is an automation quirk, not app behavior). Manually
    // dispatched DragEvents exercise the exact same handlers the browser's
    // own drag-and-drop would call.
    await page.evaluate(() => {
      const source = document.querySelector('[data-testid="band-grip-1"]') as HTMLElement
      const target = document.querySelector('[data-testid="slider-column-4"]') as HTMLElement
      const dt = new DataTransfer()
      source.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: dt }))
      dt.setData('application/x-autoforge-band', '1')
      target.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: dt }))
      target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt }))
    })

    await expect(byTestId(page, 'toast-warning')).toContainText('has no layer yet')
    // Nothing was reordered, and no bogus "Moved band" history entry exists.
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('13')
    await byTestId(page, 'history-open-btn').click()
    await expect(page.locator('[data-testid^="history-entry-"]', { hasText: 'Moved band' })).toHaveCount(0)
  })
})

// --- 6. FlatForge: FileMenu offers each material's STL, not a dead link ----

test.describe('FlatForge export', () => {
  test('FileMenu lists each material STL for a FlatForge result', async ({ page }) => {
    const jobId = 'e2e-flatforge-job'
    await page.route('**/api/optimize/latest', (route) =>
      route.fulfill({
        json: {
          job_id: jobId,
          status: 'completed',
          progress: 100,
          iteration: 10,
          total_iterations: 10,
          loss: 0.1,
          error: null,
          started_at: null,
          completed_at: null,
          phase: null,
          flatforge: true,
        },
      }),
    )
    await page.route(`**/api/outputs/stl-list/${jobId}`, (route) =>
      route.fulfill({ json: { files: ['Red_#ff0000.stl', 'Blue_#0000ff.stl'] } }),
    )
    let requestedFile: string | null = null
    await page.route(`**/api/outputs/file/${jobId}/*`, (route) => {
      requestedFile = decodeURIComponent(route.request().url().split('/').pop()!)
      route.fulfill({ status: 200, contentType: 'application/octet-stream', body: 'solid test\nendsolid test\n' })
    })

    await openApp(page)
    await byTestId(page, 'file-menu-btn').click()

    // The single-STL entry (which 404s for FlatForge) is gone, replaced by
    // one entry per material.
    await expect(byTestId(page, 'download-stl')).toHaveCount(0)
    const materialButtons = byTestId(page, 'download-stl-flatforge')
    await expect(materialButtons).toHaveCount(2)
    await expect(materialButtons.first()).toContainText('Red_#ff0000.stl')

    const [download] = await Promise.all([page.waitForEvent('download'), materialButtons.first().click()])
    expect(requestedFile).toBe('Red_#ff0000.stl')
    expect(download.suggestedFilename()).toBe('Red_#ff0000.stl')
  })
})

// --- 9. A backend-restart "not_found" status must be handled, not stored ---

test.describe('Job WebSocket not_found', () => {
  test('a not_found status clears the job with a toast instead of being stored verbatim', async ({ page, request }) => {
    await activePalette(request)
    await page.route('**/api/optimize/start', (route) =>
      route.fulfill({ json: { job_id: 'e2e-mock-job', total_iterations: 10, status: 'running' } }),
    )
    await page.routeWebSocket(/\/ws\/optimize\//, (ws) => {
      ws.send(
        JSON.stringify({
          job_id: 'e2e-mock-job',
          status: 'running',
          progress: 10,
          iteration: 1,
          total_iterations: 10,
          loss: 1,
          error: null,
          started_at: null,
          completed_at: null,
          phase: null,
        }),
      )
      setTimeout(() => ws.send(JSON.stringify({ job_id: 'e2e-mock-job', status: 'not_found', error: 'Job not found' })), 200)
    })
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await byTestId(page, 'top-start-btn').click()
    await expect(byTestId(page, 'job-phase')).toBeVisible()

    await expect(byTestId(page, 'toast-warning')).toContainText('server may have restarted')
    // The job is gone from the UI — not stuck showing a status value
    // outside the JobStatus union.
    await expect(byTestId(page, 'job-phase')).toHaveCount(0)
    await expect(byTestId(page, 'top-cancel-btn')).toHaveCount(0)
  })
})

// --- A library TD edit must not mark the layers "edited by hand" -----------

test.describe('Filament TD propagation', () => {
  test('a library TD edit propagated to sliders does not trigger the "replace your layers" confirm', async ({ page, request }) => {
    test.setTimeout(120_000)
    const f = await createFilament(request, { td: 2 })
    await setActive(request, [f])
    // Assign it to a slider directly through project state — not through any
    // frontend "hand edit" action — so slidersEditedByHand starts false.
    const state = await (await request.get('/api/project/state')).json()
    state.color_sliders[0] = { ...state.color_sliders[0], filament_uuid: f.uuid, td: 2, enabled: true, layer: 10 }
    await request.post('/api/project/state', { data: state })

    await openApp(page)
    await expect(byTestId(page, 'td-input-0')).toHaveValue('2')

    await libraryItem(page, f.uuid).dblclick()
    await byTestId(page, 'edit-filament-td').fill('4.5')
    await byTestId(page, 'save-filament-btn').click()
    await expect(byTestId(page, 'edit-filament-modal')).toHaveCount(0)
    await expect(byTestId(page, 'td-input-0')).toHaveValue('4.5')

    await page.route('**/api/optimize/start', (route) => route.fulfill({ status: 409, json: { detail: 'mocked' } }))
    await uploadAndWaitForPreview(page)
    await byTestId(page, 'top-start-btn').click()
    // Regression: setSliders() (used to propagate the new TD onto the
    // slider) unconditionally marked slidersEditedByHand, so this TD-only
    // library edit alone triggered "Replace your color layers?" on the
    // next Run even though the user never touched a slider by hand.
    await expect(byTestId(page, 'confirm-dialog')).toHaveCount(0)
    await expect(byTestId(page, 'start-error')).toHaveText('mocked')
  })
})
