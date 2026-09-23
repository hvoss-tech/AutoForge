import { test, expect, type Page, type WebSocketRoute } from '@playwright/test'
import { activePalette, byTestId, createFilament, openApp, resetBackend, setActive, uploadAndWaitForPreview, uploadImage, waitForSnapshot } from '../helpers'

/**
 * The 2026-09-23 usability round: a top bar that no longer overflows during
 * a run, a visible "lost connection" state, slim color-column markers for
 * dense stacks, the quieter filament library, the base-color field, and
 * focus-area painting. The Differences view needs a real result and is
 * checked in jobs/optimization.spec.ts; the base-color field in
 * autoforge.spec.ts; the tutorial text in webui-fixes-2026-09-17.spec.ts.
 */

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

async function setSliders(page: Page, sliders: object[]) {
  const state = await (await page.request.get('/api/project/state')).json()
  await page.request.post('/api/project/state', { data: { ...state, color_sliders: sliders } })
}

const RUNNING = {
  job_id: 'e2e-mock-job',
  status: 'running',
  progress: 25,
  iteration: 250,
  total_iterations: 1000,
  loss: 1,
  error: null,
  started_at: null,
  completed_at: null,
  phase: null,
}

/** Mocks a run: /api/optimize/start answers with a job id, and `onSocket`
 * decides what each (re)connection of its socket does. Must be called
 * before the page loads — routeWebSocket only covers pages loaded after it. */
async function mockRun(page: Page, onSocket: (ws: WebSocketRoute, connection: number) => void) {
  let connections = 0
  await page.route('**/api/optimize/start', (route) => route.fulfill({ json: { job_id: 'e2e-mock-job', total_iterations: 1000, status: 'running' } }))
  await page.routeWebSocket(/\/ws\/optimize\//, (ws) => onSocket(ws, connections++))
}

async function clickRun(page: Page) {
  await byTestId(page, 'top-start-btn').click()
  await expect(byTestId(page, 'job-progress')).toBeVisible()
}

const right = async (page: Page, id: string) => {
  const box = (await byTestId(page, id).boundingBox())!
  return box.x + box.width
}
const left = async (page: Page, id: string) => (await byTestId(page, id).boundingBox())!.x

// --- 1. The top bar fits while a run is going --------------------------------

test.describe('Top bar during a run', () => {
  test('at 1280px nothing overlaps: project name, undo buttons, steps and progress', async ({ page, request }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await activePalette(request)
    await mockRun(page, (ws) => ws.send(JSON.stringify(RUNNING)))
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await clickRun(page)

    // The name box used to keep its width and paint over the undo buttons.
    expect(await right(page, 'project-name-input')).toBeLessThanOrEqual(await left(page, 'undo-btn'))
    // The steps used to slide underneath the progress readout.
    expect(await right(page, 'workflow-steps')).toBeLessThanOrEqual(await left(page, 'job-progress'))
    // Only the current step keeps its label while running.
    await expect(byTestId(page, 'workflow-step-run')).toHaveAttribute('data-state', 'current')
    await expect(byTestId(page, 'workflow-step-run')).toContainText('Run')
    await expect(byTestId(page, 'workflow-step-export').locator('.sr-only')).toHaveCount(1)
    // Pruning is unavailable during a run, so it gives its room to the progress.
    await expect(byTestId(page, 'top-pruning-btn')).toHaveCount(0)
  })
})

// --- 2. A server that goes away mid-run is reported, not a frozen bar --------

test.describe('Lost connection', () => {
  test('the run shows reconnecting, then a banner; Try again recovers once the server answers', async ({ page, request }) => {
    test.setTimeout(150_000)
    await activePalette(request)
    let serverUp = true
    await page.route('**/api/optimize/status/e2e-mock-job', (route) => (serverUp ? route.fulfill({ json: RUNNING }) : route.abort('connectionrefused')))
    await mockRun(page, (ws, connection) => {
      if (connection === 0) {
        ws.send(JSON.stringify(RUNNING))
        // The server dies a moment into the run.
        setTimeout(() => {
          serverUp = false
          ws.close()
        }, 300)
      } else if (serverUp) {
        ws.send(JSON.stringify(RUNNING))
      } else {
        ws.close()
      }
    })
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await clickRun(page)

    await expect(byTestId(page, 'job-progress')).toHaveAttribute('data-connection', 'reconnecting')
    await expect(byTestId(page, 'job-phase')).toHaveText('Reconnecting…')
    // Reconnects run out (~15s), then status polls fail too: say so plainly.
    await expect(byTestId(page, 'connection-lost-banner')).toBeVisible({ timeout: 40_000 })
    await expect(byTestId(page, 'job-progress')).toHaveAttribute('data-connection', 'lost')
    await expect(byTestId(page, 'job-phase')).toHaveText('Disconnected')
    await expect(byTestId(page, 'top-pause-btn')).toBeDisabled()

    await byTestId(page, 'connection-retry-btn').click()
    await expect(byTestId(page, 'connection-still-down')).toBeVisible()

    serverUp = true
    await byTestId(page, 'connection-retry-btn').click()
    await expect(byTestId(page, 'connection-lost-banner')).toHaveCount(0)
    await expect(byTestId(page, 'job-progress')).toHaveAttribute('data-connection', 'ok')
    await expect(byTestId(page, 'top-pause-btn')).toBeEnabled()
  })

  test('Cancel while the server is gone clears the run instead of failing', async ({ page, request }) => {
    test.setTimeout(150_000)
    await activePalette(request)
    await page.route('**/api/optimize/status/e2e-mock-job', (route) => route.abort('connectionrefused'))
    await page.route('**/api/optimize/cancel/e2e-mock-job', (route) => route.abort('connectionrefused'))
    await mockRun(page, (ws, connection) => {
      if (connection === 0) {
        ws.send(JSON.stringify(RUNNING))
        setTimeout(() => ws.close(), 300)
      } else {
        ws.close()
      }
    })
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await clickRun(page)
    await expect(byTestId(page, 'connection-lost-banner')).toBeVisible({ timeout: 40_000 })

    await byTestId(page, 'top-cancel-btn').click()
    await expect(byTestId(page, 'job-progress')).toHaveCount(0)
    await expect(byTestId(page, 'connection-lost-banner')).toHaveCount(0)
    await expect(byTestId(page, 'top-start-btn')).toBeVisible()
  })
})

// --- 3. Dense stacks in the color column --------------------------------------

test.describe('Color column with many bands', () => {
  test('tightly packed bands get slim markers, and the one pointed at grows back to a full handle', async ({ page, request }) => {
    const a = await createFilament(request, { name: 'E2E Dense A', color: '#223366', td: 2 })
    const b = await createFilament(request, { name: 'E2E Dense B', color: '#ddcc44', td: 2 })
    await setActive(request, [a, b])
    await setSliders(
      page,
      Array.from({ length: 50 }, (_, i) => ({ td: 2, layer: i + 1, depth_mm: 0, filament_uuid: i % 2 ? b.uuid : a.uuid, enabled: true })),
    )
    await openApp(page)

    const compact = page.locator('[data-testid^="color-core-handle-"][data-compact="true"]')
    await expect.poll(() => compact.count()).toBeGreaterThan(10)
    // Nothing overlaps any more: every marker is shorter than the gap to its neighbours.
    const boxes = await page.locator('[data-testid^="color-core-marker-"]').evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).map((r) => ({ top: r.top, bottom: r.bottom })))
    boxes.sort((x, y) => x.top - y.top)
    for (let i = 1; i < boxes.length; i++) expect(boxes[i].top).toBeGreaterThanOrEqual(boxes[i - 1].bottom - 0.5)

    const handle = compact.nth(5)
    const testId = (await handle.getAttribute('data-testid'))!
    const hb = (await handle.boundingBox())!
    await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2)
    await expect(byTestId(page, testId)).not.toHaveAttribute('data-compact', 'true')
    await expect(byTestId(page, 'color-core-tooltip')).toBeVisible()
    // The zoom-in button points the way to picking bands apart.
    await expect(byTestId(page, 'color-core-zoom-in')).toHaveAttribute('title', /zoom in/i)
  })
})

// --- 8. The filament library --------------------------------------------------

test.describe('Filament library rows', () => {
  test('TD is shown once per row, and active rows are marked without tinting the whole row', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Row', color: '#c0392b', td: 1.5 })
    await setActive(request, [f])
    await openApp(page)
    const row = page.locator(`[data-testid="filament-${f.uuid}"]`)
    await expect(row).toHaveAttribute('data-active', 'true')
    expect((await row.innerText()).match(/1\.5/g)).toHaveLength(1)
    await expect(row).not.toHaveClass(/bg-emerald/)
    await expect(byTestId(page, 'library-auto-save-state')).toHaveText('— edits are saved right away')
  })
})

// --- 14. Focus areas ----------------------------------------------------------

async function paintStroke(page: Page, fromX = 0.3, toX = 0.6) {
  const box = (await byTestId(page, 'focus-mask-canvas').boundingBox())!
  const y = box.y + box.height / 2
  await page.mouse.move(box.x + box.width * fromX, y)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * toX, y, { steps: 12 })
  await page.mouse.up()
}

async function savedMask(page: Page): Promise<string> {
  const state = await (await page.request.get('/api/project/state')).json()
  return state.settings.priority_mask ?? ''
}

test.describe('Focus areas', () => {
  test('painting saves a greyscale mask the optimizer can read; Done, Clear and undo behave', async ({ page }) => {
    await openApp(page)
    await uploadImage(page)

    await byTestId(page, 'focus-areas-btn').click()
    await expect(byTestId(page, 'focus-toolbar')).toBeVisible()
    // One strength for all painted areas: 10× by default, 2× to 100×, kept in the settings.
    await expect(byTestId(page, 'focus-strength-value')).toHaveText('10×')
    await byTestId(page, 'focus-strength').fill('100')
    await expect(byTestId(page, 'focus-strength-value')).toHaveText('100×')
    await byTestId(page, 'focus-strength').fill('0')
    await expect(byTestId(page, 'focus-strength-value')).toHaveText('2×')
    await waitForSnapshot(page)
    expect((await (await page.request.get('/api/project/state')).json()).settings.priority_mask_strength).toBe(2)
    await byTestId(page, 'focus-strength').fill(String(Math.round((Math.log(5) / Math.log(50)) * 100)))
    await expect(byTestId(page, 'focus-strength-value')).toHaveText('10×')
    await expect(byTestId(page, 'focus-share')).toHaveText('Nothing marked yet')

    await paintStroke(page)
    await expect(byTestId(page, 'focus-share')).toContainText('% marked')
    await expect.poll(() => savedMask(page), { timeout: 10_000 }).not.toBe('')
    const name = await savedMask(page)

    // Opaque greyscale: white where painted, black elsewhere (the backend
    // drops alpha, so transparency would read as "all focus").
    const pixels = await page.evaluate(async (url) => {
      const img = new Image()
      img.src = url
      await img.decode()
      const c = document.createElement('canvas')
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      const d = ctx.getImageData(0, 0, c.width, c.height).data
      let white = 0, black = 0, transparent = 0, colored = 0
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] < 255) transparent++
        if (d[i] !== d[i + 1] || d[i] !== d[i + 2]) colored++
        if (d[i] > 200) white++
        else if (d[i] < 50) black++
      }
      return { white, black, transparent, colored }
    }, `/uploads/${name}`)
    expect(pixels.white).toBeGreaterThan(0)
    expect(pixels.black).toBeGreaterThan(pixels.white)
    expect(pixels.transparent).toBe(0)
    expect(pixels.colored).toBe(0)

    // Erasing over the stroke takes paint away again.
    await byTestId(page, 'focus-tool-erase').click()
    await paintStroke(page, 0.25, 0.65)
    await expect(byTestId(page, 'focus-share')).toHaveText('Nothing marked yet')
    await expect.poll(() => savedMask(page), { timeout: 10_000 }).toBe('')

    await byTestId(page, 'focus-tool-paint').click()
    await paintStroke(page)
    await expect.poll(() => savedMask(page), { timeout: 10_000 }).not.toBe('')

    // Done (or Esc) returns to the normal view; the button shows focus areas are set.
    await byTestId(page, 'focus-done-btn').click()
    await expect(byTestId(page, 'focus-toolbar')).toHaveCount(0)
    await expect(byTestId(page, 'image-view-toggle')).toBeVisible()
    await expect(byTestId(page, 'focus-areas-dot')).toBeVisible()

    // Clear, then undo brings the same mask back.
    const painted = await savedMask(page)
    await byTestId(page, 'focus-areas-btn').click()
    await byTestId(page, 'focus-clear-btn').click()
    await expect(byTestId(page, 'focus-share')).toHaveText('Nothing marked yet')
    await waitForSnapshot(page)
    expect(await savedMask(page)).toBe('')
    await page.keyboard.press('Escape')
    await expect(byTestId(page, 'focus-areas-dot')).toHaveCount(0)

    await byTestId(page, 'undo-btn').click()
    await expect(byTestId(page, 'focus-areas-dot')).toBeVisible()
    await byTestId(page, 'focus-areas-btn').click()
    await expect(byTestId(page, 'focus-share')).toContainText('% marked')
    await waitForSnapshot(page)
    expect(await savedMask(page)).toBe(painted)
  })

  test('a new picture starts without focus areas', async ({ page }) => {
    await openApp(page)
    await uploadImage(page)
    await byTestId(page, 'focus-areas-btn').click()
    await paintStroke(page)
    await expect.poll(() => savedMask(page), { timeout: 10_000 }).not.toBe('')

    await uploadImage(page)
    await expect(byTestId(page, 'focus-toolbar')).toHaveCount(0)
    await expect(byTestId(page, 'focus-areas-dot')).toHaveCount(0)
    await expect.poll(() => savedMask(page), { timeout: 10_000 }).toBe('')
  })
})
