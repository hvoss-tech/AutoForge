import { test, expect, type Page, type APIRequestContext } from '@playwright/test'
import { FAST_SETTINGS, activePalette, byTestId, latestJob, openApp, presetSettings, resetBackend, uploadAndWaitForPreview, waitForJob } from '../helpers'

// Best of N: once every run is done, a gallery shows all of them; the user
// inspects (pan/zoom), compares two A/B and keeps one, which then finishes
// like a single run. Real (tiny) optimizations; tests/webui/test_best_of.py
// covers the server side.
test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

async function startSeries(page: Page, request: APIRequestContext, baseURL: string, autoPrune = false) {
  await resetBackend(baseURL)
  await activePalette(request)
  await presetSettings(request, { ...FAST_SETTINGS, best_of: 2, auto_initial_prune: autoPrune })
  await openApp(page)
  await uploadAndWaitForPreview(page)
  await byTestId(page, 'top-start-btn').click()
  await byTestId(page, 'confirm-ok').click()
  await expect(byTestId(page, 'best-of-gallery')).toBeVisible({ timeout: 240_000 })
  return (await latestJob(request)).job_id as string
}

async function imageLoaded(page: Page, testId: string) {
  await expect
    .poll(() => byTestId(page, testId).locator('img').first().evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0))
    .toBe(true)
}

test('every run in a gallery: inspect, compare A/B, keep one', async ({ page, request, baseURL }) => {
  // With the automatic pruning pass: every run had it before it is shown.
  const jobId = await startSeries(page, request, baseURL!, true)
  const job = await (await request.get(`/api/optimize/status/${jobId}`)).json()
  expect(job.best_of_choosing).toBe(true)
  const lowest: number = job.best_of_run
  const other = 1 - lowest

  // Both runs, side by side, with their losses and the lowest marked.
  const gallery = byTestId(page, 'best-of-gallery')
  await expect(byTestId(page, 'best-of-grid')).toHaveAttribute('data-columns', '2')
  await expect(gallery.locator('[data-testid^="best-of-card-"]')).toHaveCount(2)
  await imageLoaded(page, 'best-of-card-0')
  await imageLoaded(page, 'best-of-card-1')
  await expect(byTestId(page, `best-of-card-${lowest}`).getByTestId('best-of-lowest')).toBeVisible()
  await expect(byTestId(page, `best-of-card-${other}`).getByTestId('best-of-lowest')).toHaveCount(0)
  await expect(byTestId(page, `best-of-card-${lowest}`).getByTestId('best-of-loss')).toContainText(job.best_of_losses[lowest].toFixed(4))

  // Hidden, it comes back from the top bar; nothing to pause meanwhile.
  await byTestId(page, 'best-of-hide').click()
  await expect(gallery).toBeHidden()
  await expect(byTestId(page, 'top-pause-btn')).toHaveCount(0)
  await byTestId(page, 'best-of-open').click()
  await expect(gallery).toBeVisible()

  // Inspect a run: zoom in, flip to the other run at the same zoom, back out.
  await byTestId(page, 'best-of-card-0').click()
  await expect(gallery).toHaveAttribute('data-view', 'inspect')
  await expect(byTestId(page, 'best-of-inspect-run')).toContainText('Run 1')
  const viewport = byTestId(page, 'best-of-inspect-viewport')
  await viewport.hover()
  await page.mouse.wheel(0, -300)
  await expect(byTestId(page, 'best-of-zoom')).toBeVisible()
  const zoomed = await byTestId(page, 'best-of-zoom').textContent()
  await page.keyboard.press('ArrowRight')
  await expect(byTestId(page, 'best-of-inspect-run')).toContainText('Run 2')
  await expect(byTestId(page, 'best-of-zoom')).toHaveText(zoomed!)
  await page.keyboard.press('Escape')
  await expect(gallery).toHaveAttribute('data-view', 'grid')
  await expect(gallery).toBeVisible()

  // Deselecting a marked run, then A/B of both.
  await expect(byTestId(page, 'best-of-compare-btn')).toBeDisabled()
  await byTestId(page, 'best-of-mark-0').click()
  await byTestId(page, 'best-of-mark-0').click()
  await expect(byTestId(page, 'best-of-card-0')).toHaveAttribute('data-marked', 'false')
  await byTestId(page, `best-of-mark-${lowest}`).click()
  await byTestId(page, `best-of-mark-${other}`).click()
  await byTestId(page, 'best-of-compare-btn').click()
  await expect(gallery).toHaveAttribute('data-view', 'compare')
  await expect(byTestId(page, 'best-of-slider')).toBeVisible()
  await expect(byTestId(page, 'best-of-side-a')).toHaveValue(String(lowest))
  await expect(byTestId(page, 'best-of-side-b')).toHaveValue(String(other))
  await byTestId(page, 'best-of-swap').click()
  await expect(byTestId(page, 'best-of-side-a')).toHaveValue(String(other))
  await byTestId(page, 'best-of-mode-split').click()
  await expect(byTestId(page, 'best-of-split-a')).toBeVisible()
  await expect(byTestId(page, 'best-of-split-b')).toBeVisible()
  // The picture itself can be one side.
  await byTestId(page, 'best-of-side-b').selectOption('original')
  await expect(byTestId(page, `best-of-keep-${lowest}`)).toHaveCount(0)
  await byTestId(page, 'best-of-side-b').selectOption(String(lowest))

  // Keep the run that didn't have the lowest loss - after a while: the
  // wait for a choice isn't a stalled run.
  await page.waitForTimeout(32_000)
  await byTestId(page, `best-of-keep-${other}`).click()
  await expect(gallery).toBeHidden()
  await expect(byTestId(page, 'job-progress')).toBeVisible()
  // Read once, as it appears: the flag clears again as soon as the export moves.
  expect(await byTestId(page, 'job-stalled').count()).toBe(0)
  const done = await waitForJob(request, jobId)
  expect(done.status).toBe('completed')
  expect(done.best_of_run).toBe(other)
  expect(done.loss).toBeCloseTo(done.best_of_losses[other], 6)
  expect(done.best_of_pruned).toBe(true)
  // Pruned already: no second pruning pass starts after the pick.
  await page.waitForTimeout(3000)
  expect((await latestJob(request)).job_id).toBe(jobId)
  await expect(byTestId(page, 'pruning-indicator')).toHaveCount(0)
  await expect(byTestId(page, 'result-best-of')).toBeVisible({ timeout: 15_000 })
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15_000 })
  await expect(page.locator('[data-testid^="toast-"]').filter({ hasText: /failed|couldn't/i })).toHaveCount(0)
})

test('the gallery comes back after a reload, and discarding keeps nothing', async ({ page, request, baseURL }) => {
  const jobId = await startSeries(page, request, baseURL!)

  await page.reload()
  await expect(byTestId(page, 'best-of-gallery')).toBeVisible({ timeout: 30_000 })
  await byTestId(page, 'best-of-hide').click()

  await byTestId(page, 'top-cancel-btn').click()
  await expect(byTestId(page, 'confirm-dialog')).toContainText('Discard all 2 runs?')
  await byTestId(page, 'confirm-ok').click()
  expect((await waitForJob(request, jobId)).status).toBe('cancelled')
  await expect(byTestId(page, 'best-of-open')).toHaveCount(0)
  await expect(byTestId(page, 'best-of-gallery')).toHaveCount(0)
})
