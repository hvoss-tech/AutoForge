import fs from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import { byTestId, createFilament, historyLabels, openApp, resetBackend, typeInto, waitForSnapshot } from '../helpers'

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

/** Three distinct undo steps: TD of column 0 set to 1, 2, 3. */
async function makeSteps(page: Page, values = ['1', '2', '3']) {
  for (const v of values) {
    await typeInto(page, 'td-input-0', v)
    await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
    await waitForSnapshot(page)
  }
}

const undoBtn = (page: Page) => page.getByRole('button', { name: '↶ Undo' })
const redoBtn = (page: Page) => page.getByRole('button', { name: '↷ Redo' })
const td0 = (page: Page) => byTestId(page, 'td-input-0')

test('Undo/Redo buttons step backwards and forwards in order', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await expect(redoBtn(page)).toBeDisabled()

  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('2')
  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('1')
  await expect(undoBtn(page)).toBeDisabled()

  await redoBtn(page).click()
  await expect(td0(page)).toHaveValue('2')
  await redoBtn(page).click()
  await expect(td0(page)).toHaveValue('3')
  await expect(redoBtn(page)).toBeDisabled()
})

test('keyboard: Ctrl+Z undoes, Ctrl+Shift+Z and Ctrl+Y redo', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await page.locator('body').click({ position: { x: 600, y: 5 } })

  await page.keyboard.press('Control+z')
  await expect(td0(page)).toHaveValue('2')
  await page.keyboard.press('Control+z')
  await expect(td0(page)).toHaveValue('1')
  // Ctrl+Shift+Z used to do nothing: with Shift held the key is "Z".
  await page.keyboard.press('Control+Shift+Z')
  await expect(td0(page)).toHaveValue('2')
  await page.keyboard.press('Control+y')
  await expect(td0(page)).toHaveValue('3')
})

test('Ctrl+Z inside a text field is left to the field', async ({ page }) => {
  await openApp(page)
  await makeSteps(page, ['1', '2'])
  await byTestId(page, 'filter-input').click()
  await byTestId(page, 'filter-input').pressSequentially('abc')
  await page.keyboard.press('Control+z')
  await page.waitForTimeout(500)
  await expect(td0(page)).toHaveValue('2')
})

test('pressing undo twice quickly goes back two steps', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await page.locator('body').click({ position: { x: 600, y: 5 } })
  // Each undo awaits network calls; the second used to read the index
  // before the first had updated it and restored the same step again.
  await page.keyboard.press('Control+z')
  await page.keyboard.press('Control+z')
  await expect(td0(page)).toHaveValue('1')
})

test('an edit made right before Undo is kept as a step, not lost', async ({ page }) => {
  await openApp(page)
  await makeSteps(page, ['1', '2'])
  await typeInto(page, 'td-input-0', '7')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  // No wait: the debounced snapshot for "7" is still pending.
  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('2')
  await redoBtn(page).click()
  await expect(td0(page)).toHaveValue('7')
})

test('a new edit after undoing replaces the redo branch, also after a reload', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await undoBtn(page).click()
  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('1')
  await makeSteps(page, ['9'])
  await expect(redoBtn(page)).toBeDisabled()

  const labelsBefore = await historyLabels(page)
  await page.reload()
  await expect(td0(page)).toHaveValue('9')
  // The abandoned "2" and "3" steps came back after a reload, interleaved by
  // time, which made undo jump around.
  expect(await historyLabels(page)).toEqual(labelsBefore)
  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('1')
})

test('library browsing does not create undo steps', async ({ page, request }) => {
  await createFilament(request, { filament_type: 'PETG' })
  await openApp(page)
  await makeSteps(page, ['1'])
  const before = await historyLabels(page)
  await byTestId(page, 'tab-PETG').click()
  await byTestId(page, 'filter-input').fill('e2e')
  await byTestId(page, 'filter-input').fill('')
  await waitForSnapshot(page)
  // Every library refresh used to add a "Filament library updated" step.
  expect(await historyLabels(page)).toEqual(before)
})

test('undo restores the active filament list on the server as well', async ({ page, request }) => {
  const a = await createFilament(request, { name: 'E2E Undo A' })
  const b = await createFilament(request, { name: 'E2E Undo B' })
  await openApp(page)
  await byTestId(page, `toggle-filament-${a.uuid}`).click()
  await waitForSnapshot(page)
  await byTestId(page, `toggle-filament-${b.uuid}`).click()
  await waitForSnapshot(page)

  await undoBtn(page).click()
  await expect(byTestId(page, `active-filament-${b.uuid}`)).toHaveCount(0)
  await expect.poll(async () => (await (await request.get('/api/filaments/active')).json()).map((f: any) => f.uuid)).toEqual([a.uuid])
})

test('an edit made right before reloading is not lost', async ({ page }) => {
  await openApp(page)
  await typeInto(page, 'td-input-0', '6.5')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  // No wait for the 500ms debounce.
  await page.reload()
  await expect(td0(page)).toHaveValue('6.5')
  expect((await historyLabels(page)).at(-1)).toBe('▶Color slider edit')
})

test('undo is persisted: a reload shows the undone state', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await undoBtn(page).click()
  await expect(td0(page)).toHaveValue('2')
  await page.waitForTimeout(500)
  await page.reload()
  await expect(td0(page)).toHaveValue('2')
})

test('History panel: jump to a step, marker follows, Clear keeps the current state', async ({ page }) => {
  await openApp(page)
  await makeSteps(page)
  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  const count = await entries.count()
  await expect(entries.nth(count - 1)).toHaveAttribute('data-current', 'true')

  await entries.nth(count - 3).click()
  await expect(td0(page)).toHaveValue('1')
  await byTestId(page, 'history-open-btn').click()
  await expect(page.locator('[data-testid^="history-entry-"]').nth(count - 3)).toHaveAttribute('data-current', 'true')

  page.once('dialog', (d) => d.accept())
  await byTestId(page, 'history-clear-btn').click()
  await expect(page.locator('[data-testid^="history-entry-"]')).toHaveCount(1)
  await expect(td0(page)).toHaveValue('1')
  await expect(undoBtn(page)).toBeDisabled()
  expect((await (await page.request.get('/api/state/history')).json()).length).toBe(1)
})

test.describe('Project files', () => {
  test('Save Project then Load Project restores sliders, settings and active filaments', async ({ page, request }, testInfo) => {
    const a = await createFilament(request, { name: 'E2E Project A' })
    const b = await createFilament(request, { name: 'E2E Project B' })
    await openApp(page)
    await byTestId(page, `toggle-filament-${a.uuid}`).click()
    await typeInto(page, 'td-input-0', '4.2')
    await typeInto(page, 'global-layer-height', '0.12')
    await waitForSnapshot(page)

    await byTestId(page, 'file-menu-btn').click()
    const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'file-menu-save').click()])
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'))
    expect(saved).toMatchObject({ version: 1, settings: { layer_height: 0.12 } })
    expect(saved.colorSliders[0].td).toBe(4.2)
    expect(saved.activeFilaments.map((f: any) => f.uuid)).toEqual([a.uuid])

    // Change everything, then load the file back.
    await byTestId(page, `remove-filament-${a.uuid}`).click()
    await byTestId(page, `toggle-filament-${b.uuid}`).click()
    await typeInto(page, 'td-input-0', '1')
    await typeInto(page, 'global-layer-height', '0.04')
    const file = testInfo.outputPath('project.json')
    fs.writeFileSync(file, JSON.stringify(saved))
    await byTestId(page, 'file-menu-load-input').setInputFiles(file)

    await expect(td0(page)).toHaveValue('4.2')
    await expect(byTestId(page, 'global-layer-height')).toHaveValue('0.12')
    await expect(byTestId(page, `active-filament-${a.uuid}`)).toBeVisible()
    await expect(byTestId(page, `active-filament-${b.uuid}`)).toHaveCount(0)
    await expect.poll(async () => (await (await request.get('/api/filaments/active')).json()).map((f: any) => f.uuid)).toEqual([a.uuid])
  })

  test('loading a project whose filaments are missing from the library adds them', async ({ page, request }, testInfo) => {
    const file = testInfo.outputPath('foreign.json')
    const foreign = { uuid: 'e2e-foreign-uuid', brand: 'E2E Elsewhere', name: 'E2E Foreign', color: '#abcdef', td: 2.5, filament_type: 'PLA', owned: false }
    fs.writeFileSync(file, JSON.stringify({ version: 1, activeFilaments: [foreign] }))
    await openApp(page)
    await byTestId(page, 'file-menu-load-input').setInputFiles(file)
    await expect(byTestId(page, 'active-filament-e2e-foreign-uuid')).toBeVisible()
    const library = await (await request.get('/api/filaments')).json()
    expect(library.some((f: any) => f.uuid === 'e2e-foreign-uuid')).toBe(true)
  })

  test('an invalid project file shows an error', async ({ page }, testInfo) => {
    const file = testInfo.outputPath('broken.json')
    fs.writeFileSync(file, '{ not json')
    await openApp(page)
    await byTestId(page, 'file-menu-load-input').setInputFiles(file)
    await expect(byTestId(page, 'file-menu-error')).toBeVisible()
    // It used to stay on screen for the rest of the session.
    await byTestId(page, 'file-menu-error-dismiss').click()
    await expect(byTestId(page, 'file-menu-error')).toHaveCount(0)
    await byTestId(page, 'file-menu-load-input').setInputFiles(file)
    await expect(byTestId(page, 'file-menu-error')).toBeVisible()
    await byTestId(page, 'file-menu-btn').click()
    await expect(byTestId(page, 'file-menu-error')).toHaveCount(0)
  })

  test('Export is disabled without a finished result', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'file-menu-btn').click()
    await expect(byTestId(page, 'file-menu-export')).toBeDisabled()
  })
})
