import { test, expect, type Page } from '@playwright/test'
import fs from 'node:fs'
import { FAST_SETTINGS, activePalette, byTestId, openApp, presetSettings, resetBackend, uploadAndWaitForPreview, waitForJob } from '../helpers'

// Borders (settings.border): picked in the 3D panel at any time, shown on
// the 3D model, merged into the downloaded STL — without a new run.
test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

/** x/y extent (mm) of a binary STL. */
function stlExtent(path: string): [number, number] {
  const buf = fs.readFileSync(path)
  const count = buf.readUInt32LE(80)
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (let t = 0; t < count; t++) {
    for (let v = 0; v < 3; v++) {
      const at = 84 + t * 50 + 12 + v * 12
      const x = buf.readFloatLE(at)
      const y = buf.readFloatLE(at + 4)
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
    }
  }
  return [maxX - minX, maxY - minY]
}

async function downloadStl(page: Page): Promise<string> {
  await byTestId(page, 'file-menu-btn').click()
  const download = page.waitForEvent('download')
  await byTestId(page, 'download-stl').click()
  return (await (await download).path())!
}

async function borderReady(page: Page) {
  await expect(byTestId(page, 'border-size')).toBeVisible({ timeout: 30000 })
  await expect(byTestId(page, 'border-btn').locator('.animate-spin')).toHaveCount(0, { timeout: 30000 })
}

test('a border shows on the preview and the result, and goes into the STL', async ({ page, request, baseURL }) => {
  const borderRequests: string[] = []
  page.on('request', (r) => { if (r.url().includes('/border-preview/')) borderRequests.push(r.url()) })
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)
  await uploadAndWaitForPreview(page)

  // Picked before any run: drawn around the auto-preview.
  await byTestId(page, 'border-btn').click()
  await byTestId(page, 'border-shape-rectangle').click()
  const width = byTestId(page, 'border-width-input')
  await width.fill('4')
  await width.press('Enter')
  await borderReady(page)
  expect(borderRequests.some((u) => u.endsWith('/__init__'))).toBeTruthy()
  await expect(byTestId(page, 'border-size')).toContainText('picture')
  await expect(byTestId(page, 'border-btn')).toContainText('Rectangle')
  await page.keyboard.press('Escape')
  await expect(byTestId(page, 'border-panel')).toHaveCount(0)

  await byTestId(page, 'top-start-btn').click()
  await expect(byTestId(page, 'job-progress').or(byTestId(page, 'three-d-view'))).toBeVisible({ timeout: 30000 })
  const jobId = (await (await request.get('/api/optimize/latest')).json()).job_id
  expect((await waitForJob(request, jobId)).status).toBe('completed')
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  await expect.poll(() => borderRequests.some((u) => u.endsWith(`/${jobId}`)), { timeout: 15000 }).toBeTruthy()

  // The STL: plain without a border, 2 x 4 mm larger with it.
  await byTestId(page, 'border-btn').click()
  await byTestId(page, 'border-shape-none').click()
  await page.keyboard.press('Escape')
  const [plainW, plainH] = stlExtent(await downloadStl(page))
  await byTestId(page, 'border-btn').click()
  await byTestId(page, 'border-shape-rectangle').click()
  await borderReady(page)
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
  await byTestId(page, 'three-d-view').screenshot({ path: test.info().outputPath('border-rectangle.png') })
  await byTestId(page, 'file-menu-btn').click()
  await expect(byTestId(page, 'file-menu-border-note')).toContainText('Rectangle 4 mm')
  await byTestId(page, 'file-menu-btn').click()
  const [w, h] = stlExtent(await downloadStl(page))
  expect(w).toBeCloseTo(plainW + 8, 3)
  expect(h).toBeCloseTo(plainH + 8, 3)

  // Picking a border never makes the result out of date.
  await expect(byTestId(page, 'job-stale-badge')).toHaveCount(0)

  // It is part of the project: still there after a reload.
  await page.reload()
  await expect(byTestId(page, 'border-btn')).toContainText('Rectangle', { timeout: 15000 })
  await byTestId(page, 'border-btn').click()
  await byTestId(page, 'border-shape-circle').click()
  await byTestId(page, 'border-hanger-top').click()
  await borderReady(page)
  // Nothing in the panel is cut off by the 3D panel's edges.
  const panel = await byTestId(page, 'border-panel').boundingBox()
  expect(panel!.y + panel!.height).toBeLessThanOrEqual((page.viewportSize()?.height ?? 720))
  await expect(byTestId(page, 'border-info')).toBeInViewport({ ratio: 1 })
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
  await byTestId(page, 'three-d-view').screenshot({ path: test.info().outputPath('border-circle-hole.png') })
  await expect(page.locator('[data-testid^="toast-"]')).toHaveCount(0)
})
