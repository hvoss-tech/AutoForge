// HueForge's personal filament library: offered once on the first start,
// and always available as the highlighted option in the Import dialog.
// The isolated test server looks for it at HUEFORGE_TEST_LIBRARY, never at
// the developer's real HueForge install (see tests/test-server.mjs).
import fs from 'node:fs'
import path from 'node:path'
import { test, expect } from '@playwright/test'
import { byTestId, HUEFORGE_TEST_LIBRARY, openApp, resetBackend } from '../helpers'

const LIBRARY = {
  Filaments: [
    { Brand: 'E2E Brand', Name: 'E2E HF Grey', Color: '#595959', Owned: true, Tags: [], Transmissivity: 1, Type: 'PLA', uuid: '{e2e-hf-grey}' },
    { Brand: 'E2E Brand', Name: 'E2E HF White', Color: '#F2F2F2', Owned: false, Tags: [], Transmissivity: 4.5, Type: 'PLA', uuid: '{e2e-hf-white}' },
  ],
}

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
  fs.mkdirSync(path.dirname(HUEFORGE_TEST_LIBRARY), { recursive: true })
  fs.writeFileSync(HUEFORGE_TEST_LIBRARY, JSON.stringify(LIBRARY))
})

test.afterEach(() => {
  // Every other spec must start without a HueForge library.
  fs.rmSync(HUEFORGE_TEST_LIBRARY, { force: true })
})

test.describe.configure({ mode: 'serial' })

test('the first start offers to import it — once, not on every start', async ({ page, request }) => {
  // The offer is remembered server-side, so this must be the first time this
  // test server sees a HueForge library.
  expect((await (await request.get('/api/filaments/hueforge-library')).json()).offered).toBe(false)

  await page.goto('/')
  const offer = byTestId(page, 'hueforge-import-option')
  await expect(offer).toBeVisible({ timeout: 15000 })
  await expect(offer).toContainText('Found your HueForge filament library')
  await expect(offer).toContainText('2 filaments')
  await expect(byTestId(page, 'hueforge-library-path')).toHaveText(HUEFORGE_TEST_LIBRARY)

  // Declining (just closing) still counts as having been asked.
  await byTestId(page, 'close-import').click()
  await expect(byTestId(page, 'import-modal')).toHaveCount(0)

  await page.reload()
  await expect(byTestId(page, 'app')).toBeVisible()
  await page.waitForTimeout(1500)
  await expect(byTestId(page, 'import-modal')).toHaveCount(0)
  expect((await (await request.get('/api/filaments/hueforge-library')).json()).offered).toBe(true)
})

test('the Import dialog highlights the HueForge library and imports it', async ({ page, request }) => {
  await openApp(page)
  await expect(byTestId(page, 'import-modal')).toHaveCount(0)
  await byTestId(page, 'import-btn').click()
  const offer = byTestId(page, 'hueforge-import-option')
  await expect(offer).toBeVisible()
  await expect(offer).not.toContainText('Found your')

  await byTestId(page, 'hueforge-import-btn').click()
  await expect(byTestId(page, 'import-mode-choice')).toContainText('HueForge library ready')
  await byTestId(page, 'import-mode-merge').click()
  await expect(byTestId(page, 'import-modal')).toContainText('Imported 2 filaments')

  const all = await (await request.get('/api/filaments')).json()
  expect(all.find((f: any) => f.uuid === '{e2e-hf-grey}')).toMatchObject({ name: 'E2E HF Grey', td: 1, owned: true, color: '#595959' })
  expect(all.find((f: any) => f.uuid === '{e2e-hf-white}')).toMatchObject({ name: 'E2E HF White', td: 4.5, owned: false })
})

test('without a HueForge library the dialog shows no such option', async ({ page }) => {
  fs.rmSync(HUEFORGE_TEST_LIBRARY, { force: true })
  await openApp(page)
  await byTestId(page, 'import-btn').click()
  await expect(byTestId(page, 'import-formats')).toBeVisible()
  await expect(byTestId(page, 'hueforge-import-option')).toHaveCount(0)
})
