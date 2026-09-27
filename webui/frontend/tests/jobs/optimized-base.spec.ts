import { test, expect, type Locator } from '@playwright/test'
import {
  FAST_SETTINGS,
  activePalette,
  byTestId,
  openApp,
  presetSettings,
  resetBackend,
  uploadAndWaitForPreview,
  waitForJob,
} from '../helpers'

// With the base color on automatic (the default) the optimizer chooses the
// base filament along with the layer colors. Every place that shows the base
// must show the one the run chose, not the starting pick.
test.setTimeout(300_000)

const rgb = (hex: string) => {
  const n = parseInt(hex.replace('#', ''), 16)
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`
}
const bg = (l: Locator) => l.evaluate((el) => getComputedStyle(el).backgroundColor)

test('after a run every base display shows the base the optimizer chose', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  const palette = await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)
  await uploadAndWaitForPreview(page)

  const started = page.waitForResponse((r) => r.url().endsWith('/api/optimize/start') && r.request().method() === 'POST')
  await byTestId(page, 'top-start-btn').click()
  const jobId = (await (await started).json()).job_id
  const done = await waitForJob(request, jobId, 280_000)
  expect(done.status, done.error ?? '').toBe('completed')

  // The backend's word on this result's base (see derive_base_from_result).
  const base = (await (await request.get(`/api/sliders/base?job_id=${jobId}`)).json()).base
  expect(base.auto).toBe(true)
  const chosen = palette.find((f) => f.uuid === base.filament_uuid)
  expect(chosen, 'the base is one of the active filaments').toBeTruthy()
  expect(base.color.toLowerCase()).toBe(chosen!.color.toLowerCase())

  // Color layers: the base row and the stack overview.
  await expect(byTestId(page, 'base-band-row')).toHaveAttribute('data-base-color', new RegExp(`^${base.color}$`, 'i'), { timeout: 15000 })
  await expect(byTestId(page, 'base-filament-label')).toHaveText(chosen!.name)
  expect(await bg(byTestId(page, 'stack-overview-base'))).toBe(rgb(base.color))

  // Print plan.
  await byTestId(page, 'tab-print-plan').click()
  expect(await bg(byTestId(page, 'plan-base-swatch'))).toBe(rgb(base.color))
  await byTestId(page, 'tab-color-layers').click()

  // Settings: the automatic base shows what was chosen for this picture.
  await byTestId(page, 'settings-button').click()
  const auto = byTestId(page, 'setting-background_color-auto')
  await expect(auto).toContainText(chosen!.name)
  await byTestId(page, 'close-settings').click()

  // The exported swap instructions start with the same base filament.
  const swaps = await request.get(`/api/outputs/instructions/${jobId}`)
  expect(swaps.ok()).toBeTruthy()
  expect(await swaps.text()).toContain(`background filament E2E Brand - ${chosen!.name}`)
})
