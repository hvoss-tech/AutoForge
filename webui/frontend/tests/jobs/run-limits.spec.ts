import { test, expect, type APIRequestContext, type Page } from '@playwright/test'
import {
  FAST_SETTINGS,
  activePalette,
  byTestId,
  openApp,
  presetSettings,
  resetBackend,
  typeInto,
  uploadAndWaitForPreview,
  waitForJob,
} from '../helpers'

// Real optimizations with color / swap limits set in the UI before running:
// the optimizer has to keep to them itself (no pruning runs — auto cleanup
// is off in the test reset). Serial: the unlimited run first shows the
// limits are actually below what this picture would use.
test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

const MAX_COLORS = 3 // base + 2 colors, of the 4 active filaments
const MAX_SWAPS = 2

let unlimited = { colors: 0, swaps: 0 }

async function runFromUi(page: Page, request: APIRequestContext) {
  const started = page.waitForResponse((r) => r.url().endsWith('/api/optimize/start') && r.request().method() === 'POST')
  await byTestId(page, 'top-start-btn').click()
  const response = await started
  expect(response.ok(), await response.text()).toBeTruthy()
  const body = response.request().postDataJSON()
  const jobId = (await response.json()).job_id
  expect(jobId).toBeTruthy()
  const done = await waitForJob(request, jobId, 280_000)
  expect(done.status, done.error ?? '').toBe('completed')
  await expect(byTestId(page, 'result-counts')).toBeVisible({ timeout: 30000 })
  return { body, done }
}

const shownCounts = async (page: Page) => ({
  colors: Number(await byTestId(page, 'result-counts').getAttribute('data-colors')),
  swaps: Number(await byTestId(page, 'result-counts').getAttribute('data-swaps')),
})

test('without limits the result uses more colors/swaps than the limits below allow', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)
  await uploadAndWaitForPreview(page)
  await expect(byTestId(page, 'run-limits-summary')).toHaveText('No limits')

  const { body, done } = await runFromUi(page, request)
  expect(body.max_colors ?? null).toBeNull()
  expect(body.max_swaps ?? null).toBeNull()
  unlimited = { colors: done.result_colors, swaps: done.result_swaps }
  // The job's own counts and what the page shows agree.
  expect(await shownCounts(page)).toEqual(unlimited)
  // Otherwise the limited run below would prove nothing.
  expect(unlimited.colors > MAX_COLORS || unlimited.swaps > MAX_SWAPS).toBeTruthy()
})

test('limits set in the UI before running are kept by the optimizer itself', async ({ page, request }) => {
  await openApp(page)
  await uploadAndWaitForPreview(page)

  // Set the limits the way a user would: the button next to Run.
  await byTestId(page, 'run-limits-btn').click()
  await byTestId(page, 'run-limit-max_colors-limited').click()
  await typeInto(page, 'run-limit-max_colors-value', String(MAX_COLORS))
  await byTestId(page, 'run-limit-max_swaps-limited').click()
  await typeInto(page, 'run-limit-max_swaps-value', String(MAX_SWAPS))
  await byTestId(page, 'run-limit-max_swaps-value').blur()
  await page.keyboard.press('Escape')
  await expect(byTestId(page, 'run-limits-summary')).toHaveText(`≤ ${MAX_COLORS} colors · ${MAX_SWAPS} swaps`)

  const { body, done } = await runFromUi(page, request)
  // The run was started with the limits...
  expect(body.max_colors).toBe(MAX_COLORS)
  expect(body.max_swaps).toBe(MAX_SWAPS)
  // ...and the optimizer's result keeps to them (nothing pruned it: the
  // automatic cleanup is off in tests).
  expect(done.result_colors).toBeLessThanOrEqual(MAX_COLORS)
  expect(done.result_swaps).toBeLessThanOrEqual(MAX_SWAPS)
  expect(done.result_colors < unlimited.colors || done.result_swaps < unlimited.swaps).toBeTruthy()

  // The page shows the same, within the limits of this run.
  const shown = await shownCounts(page)
  expect(shown).toEqual({ colors: done.result_colors, swaps: done.result_swaps })
  await expect(byTestId(page, 'result-counts')).toHaveAttribute('data-within-limits', 'true')
  await expect(byTestId(page, 'job-done-badge')).toBeVisible()

  // The band list and print plan the user prints from agree with it.
  const bands = page.locator('[data-testid^="slider-column-"]')
  await expect(bands.first()).toBeVisible()
  await byTestId(page, 'tab-print-plan').click()
  const swapRows = await page.locator('[data-testid="plan-swap-row"]').count()
  expect(Math.max(0, swapRows - 1)).toBeLessThanOrEqual(MAX_SWAPS)
  await byTestId(page, 'tab-color-layers').click()

  // Loosening a limit afterwards marks the result out of date.
  await byTestId(page, 'run-limits-btn').click()
  await byTestId(page, 'run-limit-max_swaps-unlimited').click()
  await page.keyboard.press('Escape')
  await expect(byTestId(page, 'job-stale-badge')).toBeVisible()
})

// The user-reported case: a limited run, the automatic cleanup prune that
// follows it, then a manual prune at the same limits. Pruning's stack search
// used to accept stacks beyond its limits (8 colors / 20 swaps became 12/25
// after the automatic pass and 14/31 after pruning again at 8/20).
test('the automatic cleanup and a manual prune at the same limits keep to them', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, { ...FAST_SETTINGS, auto_initial_prune: true, max_colors: MAX_COLORS, max_swaps: MAX_SWAPS })
  await openApp(page)
  await uploadAndWaitForPreview(page)
  await expect(byTestId(page, 'run-limits-summary')).toHaveText(`≤ ${MAX_COLORS} colors · ${MAX_SWAPS} swaps`)

  const { done } = await runFromUi(page, request)
  expect(done.result_colors).toBeLessThanOrEqual(MAX_COLORS)
  expect(done.result_swaps).toBeLessThanOrEqual(MAX_SWAPS)

  // The page starts the cleanup prune by itself once the run is done.
  let cleanup: { job_id?: string; status?: string; result_colors?: number; result_swaps?: number } = {}
  const deadline = Date.now() + 200_000
  while (Date.now() < deadline) {
    cleanup = await (await request.get('/api/optimize/latest')).json()
    if (cleanup.job_id?.startsWith('prune-') && ['completed', 'failed', 'cancelled'].includes(cleanup.status ?? '')) break
    await page.waitForTimeout(500)
  }
  expect(cleanup.job_id).toMatch(/^prune-/)
  expect(cleanup.status).toBe('completed')
  expect(cleanup.result_colors).toBeLessThanOrEqual(MAX_COLORS)
  expect(cleanup.result_swaps).toBeLessThanOrEqual(MAX_SWAPS)
  await expect(byTestId(page, 'pruning-done')).toBeVisible({ timeout: 30000 })
  await expect(byTestId(page, 'result-counts')).toHaveAttribute('data-within-limits', 'true')

  // Prune again by hand, at the limits the run had.
  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()
  await typeInto(page, 'pruning-max-colors', String(MAX_COLORS))
  await typeInto(page, 'pruning-max-swaps', String(MAX_SWAPS))
  const started = page.waitForResponse((r) => r.url().endsWith('/api/pruning/start') && r.request().method() === 'POST')
  await byTestId(page, 'pruning-start-btn').click()
  const pruneJobId = (await (await started).json()).job_id
  const pruned = await waitForJob(request, pruneJobId, 280_000)
  expect(pruned.status, pruned.error ?? '').toBe('completed')
  expect(pruned.result_colors).toBeLessThanOrEqual(MAX_COLORS)
  expect(pruned.result_swaps).toBeLessThanOrEqual(MAX_SWAPS)
  await expect(byTestId(page, 'pruning-again-btn')).toBeVisible({ timeout: 30000 })
  expect(Number(await byTestId(page, 'pruning-current-colors').getAttribute('data-value'))).toBeLessThanOrEqual(MAX_COLORS)
  expect(Number(await byTestId(page, 'pruning-current-swaps').getAttribute('data-value'))).toBeLessThanOrEqual(MAX_SWAPS)
  await byTestId(page, 'pruning-close-btn').click()
  await expect(byTestId(page, 'result-counts')).toHaveAttribute('data-within-limits', 'true')
})
