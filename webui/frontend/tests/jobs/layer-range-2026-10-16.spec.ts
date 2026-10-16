import fs from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import { FAST_SETTINGS, activePalette, byTestId, openApp, presetSettings, resetBackend, uploadAndWaitForPreview, waitForJob } from '../helpers'

// Bug (2026-10-16): after a reload or loading a project, the color core ran
// up to the max_layers setting (75) instead of the result's own top layer.
test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const coreMax = async (page: Page) => Number(await byTestId(page, 'color-core-handle-0').getAttribute('aria-valuemax'))

test('the color core keeps the result’s layer range after a reload and a project load', async ({ page, request, baseURL }, testInfo) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)
  await uploadAndWaitForPreview(page)
  await byTestId(page, 'top-start-btn').click()
  await expect(byTestId(page, 'job-progress').or(byTestId(page, 'three-d-view'))).toBeVisible({ timeout: 30000 })
  const jobId = (await (await request.get('/api/optimize/latest')).json()).job_id
  expect((await waitForJob(request, jobId)).status).toBe('completed')
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 15000 })
  const top = (await (await request.get(`/api/project/heightmap?job_id=${jobId}`)).json()).max_layer
  expect(top).toBeLessThanOrEqual(FAST_SETTINGS.max_layers)
  await expect.poll(() => coreMax(page), { timeout: 15000 }).toBe(top)
  // The setting now says 75 (the default); the result still has its own layers.
  const maxLayers = byTestId(page, 'global-max-layers')
  await maxLayers.fill('75')
  await maxLayers.press('Enter')
  await page.waitForTimeout(500)
  expect(await coreMax(page)).toBe(top)
  // Let the debounced snapshot land, as a user's session would.
  await page.waitForTimeout(1500)

  await page.reload()
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 30000 })
  await expect(byTestId(page, 'color-core-handle-0')).toBeVisible({ timeout: 15000 })
  await page.waitForTimeout(2000)
  expect(await coreMax(page)).toBe(top)

  await byTestId(page, 'file-menu-btn').click()
  const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'file-menu-save').click()])
  const file = testInfo.outputPath('saved.json')
  await download.saveAs(file)
  await byTestId(page, 'file-menu-load-input').setInputFiles(file)
  await expect(byTestId(page, 'init-three-d-view').or(byTestId(page, 'three-d-view'))).toBeVisible({ timeout: 30000 })
  await page.waitForTimeout(3000)
  expect(await coreMax(page)).toBe(top)

  // And the loaded project survives a reload too.
  await page.waitForTimeout(1000)
  await page.reload()
  await expect(byTestId(page, 'color-core-handle-0')).toBeVisible({ timeout: 30000 })
  await page.waitForTimeout(3000)
  expect(await coreMax(page)).toBe(top)
})
