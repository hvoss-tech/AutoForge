import fs from 'node:fs'
import { test, expect, type APIRequestContext } from '@playwright/test'
import { FAST_SETTINGS, FIXTURE_IMAGE, activePalette, byTestId, openApp, presetSettings, resetBackend, waitForJob } from '../helpers'

/**
 * File › Slicer project (.3mf) against a real (tiny) result: the first
 * export asks for the printer, later ones are one click; both show progress.
 * The printer's OrcaSlicer profiles are downloaded from GitHub unless the
 * test server has AUTOFORGE_ORCA_PROFILES set.
 */

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

const SHOTS = process.env.SLICER_SPEC_SCREENSHOTS

async function runOptimization(request: APIRequestContext, baseURL: string, extra: Record<string, unknown> = {}) {
  await resetBackend(baseURL)
  await activePalette(request)
  await presetSettings(request, { ...FAST_SETTINGS, orca_printer: '', orca_manual_swaps: null, ...extra })
  const upload = await request.post('/api/images/upload', {
    multipart: { file: { name: 'cat.png', mimeType: 'image/png', buffer: fs.readFileSync(FIXTURE_IMAGE) } },
  })
  const filename = (await upload.json()).filename
  await presetSettings(request, { input_image: filename })
  const start = await request.post('/api/optimize/start', { data: { ...FAST_SETTINGS, ...extra, input_image: filename } })
  expect(start.ok(), await start.text()).toBeTruthy()
  const done = await waitForJob(request, (await start.json()).job_id)
  expect(done.status, done.error ?? '').toBe('completed')
}

test('the first slicer project asks for the printer, the next is one click', async ({ page, request, baseURL }) => {
  await runOptimization(request, baseURL!)
  await openApp(page)

  await byTestId(page, 'file-menu-btn').click()
  await expect(byTestId(page, 'download-3mf-printer')).toHaveText('Choose your printer…')
  await expect(byTestId(page, 'download-3mf-change-printer')).toHaveCount(0)
  await byTestId(page, 'download-3mf').click()

  // No printer yet: the picker pops up, search focused, popular printers listed.
  const dialog = byTestId(page, 'slicer-project-dialog')
  await expect(dialog).toBeVisible()
  await expect(byTestId(page, 'slicer-printer-search')).toBeFocused()
  await expect(byTestId(page, 'slicer-printer-result').first()).toBeVisible()
  await expect(byTestId(page, 'slicer-export')).toBeDisabled()
  // How colour changes happen depends on the printer: asked once one is picked.
  await expect(byTestId(page, 'slicer-swaps')).toHaveCount(0)
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/1-picker.png` })

  await page.keyboard.type('x1 carbon')
  await expect(byTestId(page, 'slicer-printer-result').first()).toContainText('Bambu Lab X1 Carbon')
  await page.keyboard.press('Enter')
  await expect(byTestId(page, 'slicer-selected-printer')).toContainText('Bambu Lab X1 Carbon')
  // Enter takes the usual 0.4 mm nozzle, not the first preset in the list (0.2)
  await expect(byTestId(page, 'slicer-selected-printer')).toContainText('0.4 mm nozzle')
  await expect(byTestId(page, 'slicer-swaps-auto')).toHaveAttribute('aria-checked', 'true')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/2-chosen.png` })

  if (SHOTS) {
    // Slow the download down so the running card can be captured.
    await page.route('**/api/outputs/tasks/*/file', async (route) => {
      await new Promise((r) => setTimeout(r, 1500))
      await route.continue()
    })
  }
  const download = page.waitForEvent('download')
  await byTestId(page, 'slicer-export').click()
  await expect(dialog).toHaveCount(0)
  await expect(byTestId(page, 'export-progress')).toBeVisible()
  if (SHOTS) {
    await expect(byTestId(page, 'export-progress-cancel')).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/3-progress.png` })
  }
  expect((await download).suggestedFilename()).toMatch(/\.3mf$/)
  await expect(byTestId(page, 'export-progress-stage')).toHaveText('Downloaded')
  await expect(byTestId(page, 'export-progress')).toHaveCount(0, { timeout: 10_000 })

  // Remembered: the menu names it and exports without asking.
  await byTestId(page, 'file-menu-btn').click()
  await expect(byTestId(page, 'download-3mf-printer')).toHaveText('for Bambu Lab X1 Carbon')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/4-menu.png` })
  const second = page.waitForEvent('download')
  await byTestId(page, 'download-3mf').click()
  await expect(byTestId(page, 'export-progress')).toBeVisible()
  await expect(dialog).toHaveCount(0)
  await second

  // Another printer from the menu opens the picker on the current one.
  await byTestId(page, 'file-menu-btn').click()
  await byTestId(page, 'download-3mf-change-printer').click()
  await expect(byTestId(page, 'slicer-selected-printer')).toContainText('Bambu Lab X1 Carbon')
  // Change, look around, keep the old one after all.
  await byTestId(page, 'slicer-change-printer').click()
  await page.keyboard.type('neptune 4')
  await byTestId(page, 'slicer-keep-printer').click()
  await expect(byTestId(page, 'slicer-selected-printer')).toContainText('Bambu Lab X1 Carbon')
  await byTestId(page, 'slicer-cancel').click()
})

test('the printer lives in the settings, survives Reset and can be forgotten', async ({ page }) => {
  await openApp(page)
  await byTestId(page, 'settings-button').click()
  await expect(byTestId(page, 'settings-printer-name')).toHaveText('Bambu Lab X1 Carbon 0.4 nozzle')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/5-settings.png` })

  await byTestId(page, 'settings-reset-btn').click()
  await byTestId(page, 'confirm-ok').click()
  await expect(byTestId(page, 'settings-printer-name')).toHaveText('Bambu Lab X1 Carbon 0.4 nozzle')

  // Choosing from the settings saves without exporting.
  await byTestId(page, 'settings-printer-choose').click()
  await byTestId(page, 'slicer-change-printer').click()
  await page.keyboard.type('prusa mk4s')
  await expect(byTestId(page, 'slicer-printer-result').first()).toContainText('Prusa MK4S')
  await byTestId(page, 'slicer-printer-result').first().click()
  await expect(byTestId(page, 'slicer-swaps-manual')).toHaveAttribute('aria-checked', 'true')
  await byTestId(page, 'slicer-export').click()
  await expect(byTestId(page, 'settings-printer-name')).toHaveText(/^Prusa MK4S/)

  await byTestId(page, 'settings-printer-forget').click()
  await expect(byTestId(page, 'settings-printer-name')).toContainText('None yet')
})

test('a failed export says why and offers another printer', async ({ page, request }) => {
  // A printer remembered from an older printer list that no longer exists.
  await presetSettings(request, { orca_printer: 'Gone Printer 0.4 nozzle', orca_manual_swaps: null })
  await openApp(page)
  await byTestId(page, 'file-menu-btn').click()
  await byTestId(page, 'download-3mf').click()
  await expect(byTestId(page, 'export-progress-error')).toContainText('Gone Printer')
  await byTestId(page, 'export-progress-change-printer').click()
  await expect(byTestId(page, 'export-progress')).toHaveCount(0)
  await expect(byTestId(page, 'slicer-project-dialog')).toBeVisible()
  await expect(byTestId(page, 'slicer-printer-search')).toBeFocused()
})

test('a nozzle the printer has no preset for is flagged before and after the export', async ({ page, request, baseURL }) => {
  // OrcaSlicer's X1 Carbon presets: 0.2, 0.4, 0.6 and 0.8 mm - no 0.5.
  await runOptimization(request, baseURL!, { nozzle_diameter: 0.5 })
  await openApp(page)
  await byTestId(page, 'file-menu-btn').click()
  await byTestId(page, 'download-3mf').click()
  await expect(byTestId(page, 'slicer-printer-search')).toBeFocused()
  await page.keyboard.type('x1 carbon')
  await expect(byTestId(page, 'slicer-printer-result').first()).toContainText('Bambu Lab X1 Carbon')
  await page.keyboard.press('Enter')
  await expect(byTestId(page, 'slicer-nozzle-warning')).toContainText('no 0.5 mm nozzle preset for the Bambu Lab X1 Carbon')
  // the closest preset was picked, not the first in the list (0.2)
  await expect(byTestId(page, 'slicer-nozzle-warning')).toContainText('uses the 0.4 mm preset')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/6-nozzle-warning.png` })

  const download = page.waitForEvent('download')
  await byTestId(page, 'slicer-export').click()
  await download
  // The warning stays until it's read and closed.
  await expect(byTestId(page, 'export-progress-warning')).toContainText('set the nozzle diameter to 0.5 mm')
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/7-export-warning.png` })
  await page.waitForTimeout(3500)
  await expect(byTestId(page, 'export-progress')).toBeVisible()
  await byTestId(page, 'export-progress-close').click()
  await expect(byTestId(page, 'export-progress')).toHaveCount(0)
})
