// DOM-level regressions for the 2026-09-24 WebUI fixes (no GPU needed).
// The pure logic is in tests/webui-fixes-2026-09-24.test.mjs, the backend
// side in tests/webui/test_bugfixes_2026_09_24.py.
import fs from 'node:fs'
import { test, expect } from '@playwright/test'
import { activePalette, byTestId, openApp, presetSettings, resetBackend, uploadImage } from '../helpers'

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

test("importing HueForge's personal_library.json adds its real filaments", async ({ page, request }, testInfo) => {
  // The dialog wrapped this {"Filaments": [...]} object in an array, so the
  // whole file went to the server as one entry — imported as a single
  // nameless white filament, with a success message.
  const file = testInfo.outputPath('personal_library.json')
  fs.writeFileSync(file, JSON.stringify({
    Filaments: [
      { Brand: 'E2E Brand', Name: 'E2E HueForge Teal', Color: '#008080', Transmissivity: 2.5, Type: 'PLA', Owned: true, Uuid: 'e2e-hf-teal' },
      { Brand: 'E2E Brand', Name: 'E2E HueForge Plum', Color: '#8E4585', Transmissivity: 1.25, Type: 'PLA', Owned: false, Uuid: 'e2e-hf-plum' },
    ],
  }))
  const countBefore = (await (await request.get('/api/filaments')).json()).length

  await openApp(page)
  await byTestId(page, 'import-btn').click()
  await byTestId(page, 'file-input').setInputFiles(file)
  await byTestId(page, 'import-mode-merge').click()
  await expect(byTestId(page, 'import-modal')).toContainText('Imported 2 filaments')

  const all = await (await request.get('/api/filaments')).json()
  expect(all.length).toBe(countBefore + 2)
  const teal = all.find((f: any) => f.uuid === 'e2e-hf-teal')
  const plum = all.find((f: any) => f.uuid === 'e2e-hf-plum')
  expect(teal).toMatchObject({ name: 'E2E HueForge Teal', color: '#008080', td: 2.5, owned: true })
  expect(plum).toMatchObject({ name: 'E2E HueForge Plum', color: '#8E4585', td: 1.25, owned: false })
  expect(all.some((f: any) => !f.name && !f.brand)).toBe(false)
  await expect(page.locator('[data-testid="filament-e2e-hf-teal"]')).toBeVisible()
})

test('a base height that is not a whole number of layers is refused with an explanation', async ({ page, request }) => {
  // The CLI refuses this; the webui used to start the run anyway.
  await activePalette(request)
  await presetSettings(request, { layer_height: 0.08, background_height: 0.2 })
  // Skip the (GPU) auto-preview build: only the Run request matters here.
  await page.route('**/api/init/run', (route) =>
    route.fulfill({ json: { status: 'ready', preview_image: null, min_layer: 0, max_layer: 20, base: null } }),
  )
  const jobsBefore = (await (await request.get('/api/optimize/history')).json()).length
  await openApp(page)
  await uploadImage(page)

  const run = byTestId(page, 'top-start-btn')
  await expect(run).toBeEnabled({ timeout: 15000 })
  await run.click()
  await expect(byTestId(page, 'start-error')).toContainText('multiple of the layer height')
  expect((await (await request.get('/api/optimize/history')).json()).length).toBe(jobsBefore)
})
