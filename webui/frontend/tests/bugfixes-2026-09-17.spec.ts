import { test, expect, type Page } from '@playwright/test'
import { resetProjectState, cleanupTestFilaments } from './reset-state'

// Coverage for the 2026-09-17 webui audit (see also
// tests/webui/test_bugfix_regressions_2026_09.py and
// tests/webui-bugfixes.test.mjs):
//  - Undo/redo restored the active-filament list only in the browser; the
//    backend list (what /api/optimize/start actually uses) kept the old set.
//  - After a reload, the undo stack was hydrated from /api/state/history with
//    nested fields camelCased, so undoing wiped settings (layer_height,
//    background_height, input_image, ...) and slider filament assignments.

const SNAPSHOT_DEBOUNCE_MS = 1200

test.beforeEach(async ({ page, baseURL }) => {
  await resetProjectState(baseURL!)
  await cleanupTestFilaments(baseURL!)
  await page.goto('/')
})

async function activateLibraryFilaments(page: Page, count: number) {
  const toggles = page.locator('[data-testid="filament-list"] [data-testid^="toggle-filament-"]')
  const active = page.locator('[data-testid="filament-list"] [data-testid^="filament-"][data-active="true"]')
  for (let i = 0; i < count; i++) {
    await toggles.nth(i).click()
    await expect(active).toHaveCount(i + 1)
    // Let the debounced snapshot for this step land, so each step is its
    // own undo entry.
    await page.waitForTimeout(SNAPSHOT_DEBOUNCE_MS)
  }
}

async function backendActiveCount(page: Page): Promise<number> {
  const res = await page.request.get('/api/filaments/active')
  return (await res.json()).length
}

test('undo restores the backend active-filament list, not just the UI', async ({ page }) => {
  await activateLibraryFilaments(page, 2)
  expect(await backendActiveCount(page)).toBe(2)

  await page.keyboard.press('Control+z')

  await expect(page.locator('[data-testid="filament-list"] [data-testid^="filament-"][data-active="true"]')).toHaveCount(1)
  await expect.poll(() => backendActiveCount(page)).toBe(1)
})

test('undo after a page reload keeps settings and filaments intact', async ({ page }) => {
  await activateLibraryFilaments(page, 2)

  await page.reload()
  const active = page.locator('[data-testid="filament-list"] [data-testid^="filament-"][data-active="true"]')
  await expect(active).toHaveCount(2)
  const meshLabel = page.locator('[data-testid="mesh-height-label"]')
  const labelBefore = await meshLabel.textContent()
  // Give loadProjectState() time to hydrate the undo stack from the backend.
  await page.waitForTimeout(1000)

  // Loading the page queues snapshots of its own, so step back until the
  // one-filament state; every hydrated snapshot passed through must keep the
  // settings usable. The label is computed from settings.background_height/
  // layer_height/max_layers, which camelCased settings made undefined.
  for (let i = 0; i < 6 && (await active.count()) !== 1; i++) {
    await page.keyboard.press('Control+z')
    await page.waitForTimeout(400)
    await expect(meshLabel).toHaveText(labelBefore!)
  }
  await expect(active).toHaveCount(1)
  await expect.poll(() => backendActiveCount(page)).toBe(1)
})
