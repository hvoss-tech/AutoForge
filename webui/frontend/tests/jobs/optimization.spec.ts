import crypto from 'node:crypto'
import fs from 'node:fs'
import { test, expect, type APIRequestContext } from '@playwright/test'
import {
  FAST_SETTINGS,
  activePalette,
  byTestId,
  historyLabels,
  openApp,
  presetSettings,
  resetBackend,
  typeInto,
  uploadAndWaitForPreview,
  uploadImage,
  waitForJob,
  waitForSnapshot,
} from '../helpers'

// Real optimizations on a tiny image. Serial: later tests build on the
// result of the first one, like a user continuing to work on it.
test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const md5 = (buf: Buffer) => crypto.createHash('md5').update(buf).digest('hex')
const fetchHash = async (request: APIRequestContext, url: string) => {
  const res = await request.get(url)
  expect(res.ok(), url).toBeTruthy()
  return md5(await res.body())
}

let jobId = ''

test('a run from the UI: progress, result view, and a single history step for it', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)
  await uploadAndWaitForPreview(page)
  await waitForSnapshot(page)
  const labelsBefore = await historyLabels(page)

  await byTestId(page, 'top-start-btn').click()
  // The iteration count itself is only shown on very wide screens (it's in
  // the progress details); the progress chip is always there.
  await expect(byTestId(page, 'job-progress').or(byTestId(page, 'three-d-view'))).toBeVisible({ timeout: 30000 })

  // One optimization at a time.
  const second = await request.post('/api/optimize/start', { data: { input_image: 'x.png' } })
  if (second.status() !== 400) {
    const latest = await (await request.get('/api/optimize/latest')).json()
    if (latest.status === 'running' || latest.status === 'paused') expect(second.status()).toBe(409)
  }

  jobId = (await (await request.get('/api/optimize/latest')).json()).job_id
  const done = await waitForJob(request, jobId)
  expect(done.status, done.error ?? '').toBe('completed')

  await expect(page.getByText('Done', { exact: true })).toBeVisible({ timeout: 15000 })
  await expect(byTestId(page, 'three-d-view')).toBeVisible()
  await expect(byTestId(page, 'top-pruning-btn')).toBeEnabled()
  expect(await page.locator('[data-testid^="slider-column-"]').count()).toBeGreaterThan(0)

  // The result is shown next to the original, and the 3D view has camera controls.
  await expect(byTestId(page, 'image-view-split')).toHaveAttribute('aria-pressed', 'true')
  await expect(byTestId(page, 'result-image')).toBeVisible()
  await expect.poll(() => byTestId(page, 'result-image').evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0)
  // The Differences view compares the result with the picture and says how close it is.
  await byTestId(page, 'image-view-difference').click()
  await expect(byTestId(page, 'difference-summary')).toBeVisible({ timeout: 15000 })
  expect(Number(await byTestId(page, 'difference-summary').getAttribute('data-mean'))).toBeGreaterThan(0)
  await expect.poll(() => byTestId(page, 'difference-image').evaluate((c: HTMLCanvasElement) => c.width)).toBeGreaterThan(0)
  await byTestId(page, 'image-view-split').click()
  for (const id of ['view-top-btn', 'view-angled-btn', 'view-reset-btn']) await byTestId(page, id).click()
  await expect(byTestId(page, 'workflow-step-export')).toHaveAttribute('data-state', 'current')
  // The print plan reflects the optimizer's stack.
  await byTestId(page, 'tab-print-plan').click()
  await expect(page.locator('[data-testid="plan-swap-row"]').first()).toBeVisible()
  await byTestId(page, 'tab-color-layers').click()

  await waitForSnapshot(page)
  // Live slider updates during the run used to be recorded as dozens of
  // "Color slider edit" steps.
  expect(await historyLabels(page)).toEqual([...labelsBefore.map((l) => l.replace(/^▶/, '')), '▶Optimization completed'])

  // Changing a setting the run used flags the result as out of date and makes
  // Run the primary action again; changing it back clears that.
  await expect(byTestId(page, 'top-start-btn')).toHaveAttribute('data-primary', 'false')
  await typeInto(page, 'global-layer-height', '0.08')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  await expect(byTestId(page, 'job-stale-badge')).toBeVisible()
  await expect(byTestId(page, 'top-start-btn')).toHaveAttribute('data-primary', 'true')
  await typeInto(page, 'global-layer-height', '0.04')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  await expect(byTestId(page, 'job-done-badge')).toBeVisible()
  await waitForSnapshot(page)
})

test('downloads and the export zip contain the result files', async ({ page, request }) => {
  await openApp(page)
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })

  // All downloads live in File › Export result (they used to be hidden in Settings).
  for (const [button, filename] of [['download-stl', 'final_model.stl'], ['download-preview', 'final_model.png'], ['download-instructions', 'swap_instructions.txt'], ['download-project', 'final_model.hfp']]) {
    await byTestId(page, 'file-menu-btn').click()
    const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, button).click()])
    expect(download.suggestedFilename()).toBe(filename)
    expect(fs.statSync(await download.path()).size).toBeGreaterThan(0)
  }

  await byTestId(page, 'file-menu-btn').click()
  const [zip] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'file-menu-export').click()])
  expect(zip.suggestedFilename()).toBe(`${jobId}_export.zip`)
  const bytes = fs.readFileSync(await zip.path())
  for (const name of ['final_model.stl', 'final_model_colored.ply', 'final_model.png', 'swap_instructions.txt']) {
    expect(bytes.includes(Buffer.from(name)), name).toBe(true)
  }
})

test('editing a slider re-renders the 3D view without touching the result files', async ({ page, request }) => {
  await openApp(page)
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  const pngBefore = await fetchHash(request, `/api/outputs/preview/${jobId}`)
  const plyBefore = await fetchHash(request, `/api/outputs/colored-ply/${jobId}`)

  const render = page.waitForResponse((r) => r.url().includes('/api/preview/render-with-sliders') && r.ok())
  await typeInto(page, 'td-input-0', '9')
  expect((await (await render).json()).status).toBe('ok')

  await expect.poll(() => fetchHash(request, `/api/outputs/colored-ply/${jobId}`)).not.toBe(plyBefore)
  // The optimizer's own outputs (what STL/instructions/export correspond to) stay.
  expect(await fetchHash(request, `/api/outputs/preview/${jobId}`)).toBe(pngBefore)
  await expect(page.locator('[data-testid^="toast-"]')).toHaveCount(0)
  await waitForSnapshot(page)
})

test('undoing past the run and redoing back keeps history, preview and result intact', async ({ page, request }) => {
  await openApp(page)
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  await waitForSnapshot(page)
  const labels = await historyLabels(page)
  const resultIndex = labels.findIndex((l) => l.replace(/^▶/, '') === 'Optimization completed')
  expect(resultIndex).toBeGreaterThan(0)
  const pngBefore = await fetchHash(request, `/api/outputs/preview/${jobId}`)
  const stepsBack = labels.length - 1 - (resultIndex - 1)
  const undo = byTestId(page, 'undo-btn')
  const redo = byTestId(page, 'redo-btn')

  // Back to before the result: the pre-run preview, no result.
  for (let i = 0; i < stepsBack; i++) {
    await undo.click()
    await page.waitForTimeout(700)
  }
  await expect(byTestId(page, 'init-three-d-view').or(byTestId(page, 'preview-placeholder')).or(byTestId(page, 'color-stack-preview'))).toBeVisible()
  await expect(byTestId(page, 'three-d-view')).toHaveCount(0)
  await expect(page.getByText('Done', { exact: true })).toHaveCount(0)
  await expect(byTestId(page, 'top-pruning-btn')).toBeDisabled()
  // Give the (formerly) polling pre-run preview time to misbehave.
  await page.waitForTimeout(3000)

  // Nothing was appended while undoing (it used to add an entry per second,
  // wiping redo and making undo walk *forward* in time).
  const during = await historyLabels(page)
  expect(during.map((l) => l.replace(/^▶/, ''))).toEqual(labels.map((l) => l.replace(/^▶/, '')))
  expect(during[resultIndex - 1]).toMatch(/^▶/)

  for (let i = 0; i < stepsBack; i++) {
    await redo.click()
    await page.waitForTimeout(700)
  }
  await expect(redo).toBeDisabled()
  await expect(byTestId(page, 'three-d-view')).toBeVisible()
  await expect(page.getByText('Done', { exact: true })).toBeVisible()
  await expect(byTestId(page, 'td-input-0')).toHaveValue('9')
  expect(await fetchHash(request, `/api/outputs/preview/${jobId}`)).toBe(pngBefore)
  await expect(page.locator('[data-testid="toast-error"]')).toHaveCount(0)
})

test('a reload shows the finished result again, image included', async ({ page }) => {
  await openApp(page)
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  // The image comes back with the result (it used to show an empty upload box
  // next to "Done").
  await expect(byTestId(page, 'input-image')).toBeVisible()
  await expect(byTestId(page, 'session-restored-banner')).toContainText('result')
  await byTestId(page, 'image-view-compare').click()
  await expect(byTestId(page, 'image-compare')).toBeVisible()
  const divider = byTestId(page, 'compare-divider')
  await divider.focus()
  await page.keyboard.press('ArrowLeft')
  await expect(divider).toHaveAttribute('aria-valuenow', '45')
  await expect(byTestId(page, 'top-pruning-btn')).toBeEnabled()
  await byTestId(page, 'file-menu-btn').click()
  await expect(byTestId(page, 'file-menu-export')).toBeEnabled()
})

test('pruning from the dialog reduces the stack and is an undo step', async ({ page, request }) => {
  await openApp(page)
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  const columnsBefore = await page.locator('[data-testid^="slider-column-"]').count()

  await byTestId(page, 'top-pruning-btn').click()
  // The dialog shows what there is to reduce and opens on exactly those
  // numbers — it no longer proposes a reduction the user didn't ask for, so
  // the cut below is the user's own choice.
  const currentSwaps = Number(await byTestId(page, 'pruning-current-swaps').getAttribute('data-value'))
  expect(currentSwaps).toBeGreaterThan(0)
  await expect(byTestId(page, 'pruning-max-swaps')).toHaveValue(String(currentSwaps))
  await typeInto(page, 'pruning-max-colors', '3')
  await typeInto(page, 'pruning-max-swaps', '4')
  await expect(byTestId(page, 'pruning-max-colors')).toHaveValue('3')
  await byTestId(page, 'pruning-start-btn').click()
  // Progress is shown in the dialog; it can be closed and keeps running.
  await expect(byTestId(page, 'pruning-progress-view')).toBeVisible()
  await expect(byTestId(page, 'pruning-modal-status')).toHaveText('Pruning finished — the color layers were updated.', { timeout: 200000 })
  await expect(byTestId(page, 'pruning-summary')).toContainText('→')
  await byTestId(page, 'pruning-close-btn').click()

  await expect(byTestId(page, 'pruning-done')).toBeVisible()
  await expect.poll(() => page.locator('[data-testid^="slider-column-"]').count(), { timeout: 15000 }).toBeLessThanOrEqual(Math.max(columnsBefore, 5))
  const materials = new Set(
    await page.locator('[data-testid^="filament-label-"]').evaluateAll((els) => els.map((e) => e.textContent)),
  )
  expect(materials.size).toBeLessThanOrEqual(3)
  await waitForSnapshot(page)
  expect((await historyLabels(page)).at(-1)).toBe('▶Pruning completed')
  const instructions = await request.get(`/api/outputs/instructions/${jobId}`)
  expect(instructions.ok()).toBeTruthy()
})

test('pause, resume and cancel from the UI', async ({ page, request }) => {
  await presetSettings(request, { ...FAST_SETTINGS, iterations: 20000, early_stopping: 100000 })
  await openApp(page)
  await uploadImage(page)

  await byTestId(page, 'top-start-btn').click()
  await expect(byTestId(page, 'top-pause-btn')).toBeVisible({ timeout: 30000 })
  await expect(byTestId(page, 'job-progress')).toBeVisible()
  await expect(byTestId(page, 'job-iteration-count')).toHaveText(/^\(\d+\/\d+\)$/)

  await byTestId(page, 'top-pause-btn').click()
  await expect(byTestId(page, 'top-resume-btn')).toBeVisible()
  const paused = await (await request.get('/api/optimize/latest')).json()
  expect(paused.status).toBe('paused')
  const iteration = paused.iteration
  await page.waitForTimeout(1500)
  expect((await (await request.get(`/api/optimize/status/${paused.job_id}`)).json()).iteration).toBe(iteration)

  // Settings can be opened while paused without offering a second run.
  await byTestId(page, 'settings-button').click()
  await expect(byTestId(page, 'settings-modal')).toContainText('not the one currently running')
  await byTestId(page, 'close-settings').click()
  await expect(byTestId(page, 'job-phase')).toHaveText('Paused')
  // Details (loss curve, timing) open from the progress pill at any width.
  await byTestId(page, 'job-progress').click()
  await expect(byTestId(page, 'job-progress-details')).toContainText('Iteration')
  await expect(byTestId(page, 'job-loss')).toBeVisible()
  await page.keyboard.press('Escape')
  await byTestId(page, 'top-resume-btn').click()
  await expect(byTestId(page, 'top-pause-btn')).toBeVisible()

  await byTestId(page, 'top-cancel-btn').click()
  await expect(byTestId(page, 'top-start-btn')).toBeVisible()
  const cancelled = await waitForJob(request, paused.job_id, 30000)
  expect(cancelled.status).toBe('cancelled')
  await expect(page.getByText('Done', { exact: true })).toHaveCount(0)
})
