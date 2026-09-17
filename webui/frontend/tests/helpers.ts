import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, type APIRequestContext, type Page } from '@playwright/test'
import { cleanupTestFilaments, resetProjectState } from './reset-state'

export const FIXTURE_IMAGE = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'cat_128x72.png')

/** Small, fast optimization settings (seconds on a GPU, still a real run). */
export const FAST_SETTINGS = {
  iterations: 150,
  max_layers: 20,
  layer_height: 0.04,
  background_height: 0.24,
  stl_output_size: 30,
  processing_reduction_factor: 1,
  num_init_rounds: 1,
  early_stopping: 10000,
  discrete_check: 25,
  random_seed: 1,
}

export async function resetBackend(baseURL: string) {
  await resetProjectState(baseURL)
  await cleanupTestFilaments(baseURL)
}

export interface TestFilament {
  uuid: string
  brand: string
  name: string
  color: string
  td: number
  filament_type: string
  owned?: boolean
}

let counter = 0
/** Creates a library filament under a brand the cleanup helper recognises. */
export async function createFilament(request: APIRequestContext, overrides: Partial<TestFilament> = {}): Promise<TestFilament> {
  counter += 1
  const body = {
    brand: 'E2E Brand',
    name: `E2E Filament ${Date.now()}-${counter}`,
    color: '#cc3300',
    td: 2,
    filament_type: 'PLA',
    ...overrides,
  }
  const res = await request.post('/api/filaments', { data: body })
  expect(res.ok()).toBeTruthy()
  return res.json()
}

export async function setActive(request: APIRequestContext, filaments: TestFilament[]) {
  const res = await request.put('/api/filaments/active', { data: filaments })
  expect(res.ok()).toBeTruthy()
}

/** Four clearly distinct filaments, created and made active. */
export async function activePalette(request: APIRequestContext): Promise<TestFilament[]> {
  const palette = [
    await createFilament(request, { name: 'E2E Black', color: '#111111', td: 0.6 }),
    await createFilament(request, { name: 'E2E Red', color: '#c0392b', td: 1.5 }),
    await createFilament(request, { name: 'E2E Yellow', color: '#f1c40f', td: 3 }),
    await createFilament(request, { name: 'E2E White', color: '#f5f5f5', td: 5 }),
  ]
  await setActive(request, palette)
  return palette
}

/** Persist settings the page will load with (the store reads
 * /api/project/state on mount). */
export async function presetSettings(request: APIRequestContext, overrides: Record<string, unknown>) {
  const current = await (await request.get('/api/project/state')).json()
  const res = await request.post('/api/project/state', {
    data: { ...current, settings: { ...current.settings, ...overrides } },
  })
  expect(res.ok()).toBeTruthy()
}

export const byTestId = (page: Page, id: string) => page.locator(`[data-testid="${id}"]`)

/** Page loaded and the library populated. */
export async function openApp(page: Page) {
  await page.goto('/')
  await expect(byTestId(page, 'app')).toBeVisible()
  await expect(page.locator('[data-testid="filament-list"] [data-testid^="filament-"]').first()).toBeVisible({ timeout: 15000 })
}

export async function uploadImage(page: Page, file = FIXTURE_IMAGE) {
  await page.locator('[data-testid="image-file-input"]').setInputFiles(file)
  await expect(byTestId(page, 'input-image')).toBeVisible()
}

/** Upload + wait for the auto-preview (heightmap init) to finish. */
export async function uploadAndWaitForPreview(page: Page) {
  await uploadImage(page)
  await expect(byTestId(page, 'init-three-d-view')).toBeVisible({ timeout: 60000 })
}

export async function waitForJob(request: APIRequestContext, jobId: string, timeoutMs = 240000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const status = await (await request.get(`/api/optimize/status/${jobId}`)).json()
    if (['completed', 'failed', 'cancelled'].includes(status.status)) return status
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`job ${jobId} did not finish in ${timeoutMs}ms`)
}

export async function latestJob(request: APIRequestContext) {
  const res = await request.get('/api/optimize/latest')
  return res.ok() ? res.json() : null
}

export async function historyLabels(page: Page): Promise<string[]> {
  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  const labels: string[] = []
  for (let i = 0; i < (await entries.count()); i++) {
    const current = (await entries.nth(i).getAttribute('data-current')) === 'true'
    const text = (await entries.nth(i).locator('span').first().innerText()).replace(/^▶\s*/, '')
    labels.push(current ? `▶${text}` : text)
  }
  await byTestId(page, 'history-close-btn').click()
  return labels
}

/** Clears the field and types like a person, one key at a time. `fill()`
 * sets the whole value at once and would hide per-keystroke coercion bugs. */
export async function typeInto(page: Page, testId: string, text: string) {
  const input = byTestId(page, testId)
  await input.click()
  await input.press('ControlOrMeta+a')
  await input.press('Backspace')
  await input.pressSequentially(text, { delay: 30 })
}

export async function waitForSnapshot(page: Page) {
  // captureSnapshot is debounced by 500ms.
  await page.waitForTimeout(900)
}
