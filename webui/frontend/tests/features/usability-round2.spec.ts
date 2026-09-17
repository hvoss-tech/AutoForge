import fs from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import {
  activeLibraryItem,
  activePalette,
  byTestId,
  createFilament,
  historyLabels,
  openApp,
  resetBackend,
  setActive,
  showTab,
  typeInto,
  uploadAndWaitForPreview,
  uploadImage,
  waitForSnapshot,
} from '../helpers'

// Second usability pass: resizable panes, library summary, swatch picker,
// linked band selection, band reordering, session restore, project names.

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

async function setSliders(page: Page, sliders: object[]) {
  const state = await (await page.request.get('/api/project/state')).json()
  await page.request.post('/api/project/state', { data: { ...state, color_sliders: sliders } })
}

const band = (layer: number, filament_uuid: string, td = 2) => ({ td, layer, depth_mm: 0, filament_uuid, enabled: true })
const height = async (page: Page, id: string) => (await byTestId(page, id).boundingBox())!.height

test.describe('Layout', () => {
  test('the color layers panel resizes, keeps a minimum and remembers its height', async ({ page }) => {
    await openApp(page)
    const start = await height(page, 'bottom-panel-container')
    expect(start).toBeCloseTo(240, 0)

    await byTestId(page, 'bottom-panel-resizer').focus()
    await page.keyboard.press('Shift+ArrowUp')
    await expect.poll(() => height(page, 'bottom-panel-container')).toBeCloseTo(start + 64, 0)

    // Dragged far down it stops at the minimum instead of disappearing.
    const box = (await byTestId(page, 'bottom-panel-resizer').boundingBox())!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width / 2, box.y + 2000, { steps: 5 })
    await page.mouse.up()
    await expect.poll(() => height(page, 'bottom-panel-container')).toBeCloseTo(140, 0)

    await page.reload()
    await expect.poll(() => height(page, 'bottom-panel-container')).toBeCloseTo(140, 0)
    await byTestId(page, 'bottom-panel-resizer').dblclick()
    await expect.poll(() => height(page, 'bottom-panel-container')).toBeCloseTo(240, 0)
  })

  test('the library can be widened and collapsed, and stays that way', async ({ page }) => {
    await openApp(page)
    const width = async () => (await byTestId(page, 'sidebar').boundingBox())!.width
    const before = await width()
    await byTestId(page, 'sidebar-resizer').focus()
    await page.keyboard.press('ArrowRight')
    await expect.poll(width).toBeCloseTo(before + 16, 0)

    await byTestId(page, 'sidebar-collapse-btn').click()
    await expect(byTestId(page, 'sidebar-collapsed')).toBeVisible()
    await expect(byTestId(page, 'filament-list')).toHaveCount(0)
    await page.reload()
    await expect(byTestId(page, 'sidebar-collapsed')).toBeVisible()
    await byTestId(page, 'sidebar-expand-btn').click()
    await expect(byTestId(page, 'filament-list')).toBeVisible()
  })

  test('the history drawer sits between the bars, toasts sit bottom right', async ({ page, request }) => {
    const f = await createFilament(request)
    await openApp(page)
    await byTestId(page, 'history-open-btn').click()
    const drawer = (await byTestId(page, 'history-drawer').boundingBox())!
    const topBar = (await byTestId(page, 'top-bar').boundingBox())!
    const statusBar = (await byTestId(page, 'global-params').boundingBox())!
    expect(drawer.y).toBeCloseTo(topBar.y + topBar.height, 0)
    expect(drawer.y + drawer.height).toBeCloseTo(statusBar.y, 0)
    await byTestId(page, 'history-close-btn').click()

    await page.route('**/api/filaments/active', (route) => (route.request().method() === 'POST' ? route.abort() : route.continue()))
    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    const toast = (await byTestId(page, 'toast-container').boundingBox())!
    const viewport = page.viewportSize()!
    expect(toast.y).toBeGreaterThan(viewport.height / 2)
    expect(toast.y + toast.height).toBeLessThanOrEqual(statusBar.y)
  })

  test('Max layers is in the print bar next to the height readout', async ({ page }) => {
    await openApp(page)
    await typeInto(page, 'global-max-layers', '40')
    await expect(byTestId(page, 'global-params')).toContainText('Max print height:')
    await waitForSnapshot(page)
    const state = await (await page.request.get('/api/project/state')).json()
    expect(state.settings.max_layers).toBe(40)
  })
})

test.describe('Filament library', () => {
  test('a one-line summary replaces the active list, with a filter for just the active ones', async ({ page, request }) => {
    const pla = await createFilament(request, { name: 'E2E Summary PLA' })
    const petg = await createFilament(request, { name: 'E2E Summary PETG', filament_type: 'PETG' })
    await setActive(request, [pla, petg])
    await openApp(page)

    await expect(byTestId(page, 'active-filaments-summary')).toHaveAttribute('data-count', '2')
    await showTab(page, 'PLA')
    await expect(byTestId(page, `filament-${petg.uuid}`)).toHaveCount(0)

    await byTestId(page, 'active-only-toggle').click()
    // Every active filament, whatever the open tab, and no tabs meanwhile.
    await expect(activeLibraryItem(page, pla.uuid)).toBeVisible()
    await expect(activeLibraryItem(page, petg.uuid)).toBeVisible()
    await expect(byTestId(page, 'tab-PETG')).toHaveCount(0)
    await expect(byTestId(page, 'tab-label')).toContainText('Active • 2')
  })

  test('with a single type there is no tab row', async ({ page }) => {
    await page.route('**/api/filaments/types', (route) => route.fulfill({ json: ['PLA'] }))
    await openApp(page)
    await expect(byTestId(page, 'tab-PLA')).toHaveCount(0)
    await expect(byTestId(page, 'tab-label')).toContainText('PLA')
  })

  test('hover buttons edit a filament and mark it owned; the drag hint can be dismissed', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Hover' })
    await openApp(page)
    const row = byTestId(page, `filament-${f.uuid}`)

    await row.hover()
    await byTestId(page, `toggle-owned-${f.uuid}`).click()
    await expect(byTestId(page, `owned-badge-${f.uuid}`)).toBeVisible()
    await expect.poll(async () => (await (await request.get('/api/filaments')).json()).find((x: any) => x.uuid === f.uuid).owned).toBe(true)

    await row.hover()
    await byTestId(page, `edit-filament-${f.uuid}`).click()
    await expect(byTestId(page, 'edit-filament-modal')).toBeVisible()
    await page.keyboard.press('Escape')

    await byTestId(page, 'drag-hint-dismiss').click()
    await page.reload()
    await expect(byTestId(page, 'filament-list')).toBeVisible()
    await expect(byTestId(page, 'drag-hint')).toHaveCount(0)
  })
})

test.describe('Color layers', () => {
  test('the filament picker shows swatches, searches, and activates library picks', async ({ page, request }) => {
    const [black] = await activePalette(request)
    const other = await createFilament(request, { name: 'E2E Not Active', color: '#00ff00' })
    await setSliders(page, [band(4, black.uuid)])
    await openApp(page)

    await byTestId(page, 'filament-select-0').click()
    await expect(byTestId(page, `filament-option-${black.uuid}`)).toBeVisible()
    await byTestId(page, 'filament-picker-search').fill('Not Active')
    await expect(byTestId(page, `filament-option-${black.uuid}`)).toHaveCount(0)
    await expect(byTestId(page, `filament-option-${other.uuid}`)).toBeVisible()
    await page.keyboard.press('Enter')

    await expect(byTestId(page, 'filament-picker')).toHaveCount(0)
    await expect(byTestId(page, 'filament-label-0')).toHaveText('E2E Not Active')
    await expect(byTestId(page, 'color-indicator-0')).toHaveCSS('background-color', 'rgb(0, 255, 0)')
    await expect(activeLibraryItem(page, other.uuid)).toBeVisible()
  })

  test('selecting a band in the color column, the table or the strip selects it everywhere', async ({ page, request }) => {
    const [a, b, c] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid), band(12, c.uuid)])
    await openApp(page)

    await byTestId(page, 'color-core-handle-1').click()
    await expect(byTestId(page, 'slider-column-1')).toHaveAttribute('data-selected', 'true')

    await byTestId(page, 'slider-column-2').click({ position: { x: 40, y: 10 } })
    await expect(byTestId(page, 'color-core-handle-2')).toHaveAttribute('data-selected', 'true')
    await expect(byTestId(page, 'stack-overview').locator('[data-selected="true"]')).toHaveCount(1)

    await byTestId(page, 'slider-column-0').hover({ position: { x: 40, y: 10 } })
    await expect(byTestId(page, 'color-core-handle-0').locator('div').first()).toHaveClass(/ring-cyan-400/)
  })

  test('color column handles have tooltips, zoom and arrow keys', async ({ page, request }) => {
    const [a, b, c] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid), band(12, c.uuid)])
    await openApp(page)

    await byTestId(page, 'color-core-handle-1').hover()
    await expect(byTestId(page, 'color-core-tooltip')).toContainText(b.name)
    await expect(byTestId(page, 'color-core-tooltip')).toContainText('Layers 4–8')

    await byTestId(page, 'color-core-handle-1').focus()
    await page.keyboard.press('ArrowUp')
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('9')
    // Same bounds as dragging: not past the next band.
    await page.keyboard.press('Shift+ArrowUp')
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('12')

    await byTestId(page, 'color-core-zoom-in').click()
    await expect(byTestId(page, 'color-core-zoom-level')).toHaveText('1.5×')
    await byTestId(page, 'color-core-zoom-level').click()
    await expect(byTestId(page, 'color-core-zoom-level')).toHaveText('1×')
  })

  test('readouts: single-layer bands, heights in mm, a band TD that differs from its filament', async ({ page, request }) => {
    const [a, b] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid, a.td), band(4, b.uuid, 9)])
    await openApp(page)

    await expect(byTestId(page, 'band-range-1')).toHaveText('4')
    await expect(byTestId(page, 'depth-0')).toHaveText('0.12 mm')
    await expect(byTestId(page, 'td-reset-0')).toHaveCount(0)
    await expect(byTestId(page, 'stack-overview').locator('..')).toContainText('Top (4)')

    await byTestId(page, 'td-reset-1').click()
    await expect(byTestId(page, 'td-input-1')).toHaveValue(String(b.td))
    await expect(byTestId(page, 'td-reset-1')).toHaveCount(0)
  })

  test('bands can be dragged to another place in the stack, keeping their thickness', async ({ page, request }) => {
    const [a, b, c] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid), band(12, c.uuid)])
    await openApp(page)

    await byTestId(page, 'band-grip-0').dragTo(byTestId(page, 'slider-column-2'))
    await expect(byTestId(page, 'filament-label-0')).toHaveText(b.name)
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('5')
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('9')
    await expect(byTestId(page, 'filament-label-2')).toHaveText(a.name)
    await expect(byTestId(page, 'layer-input-2')).toHaveValue('12')
    await waitForSnapshot(page)
    expect((await historyLabels(page)).at(-1)).toBe('▶Moved band 1 to position 3')
  })

  test('a band can be inserted above another', async ({ page, request }) => {
    const [a, b] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid)])
    await openApp(page)

    await byTestId(page, 'insert-band-0').click()
    await expect(page.locator('[data-testid^="slider-column-"]')).toHaveCount(3)
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('6')
    await expect(byTestId(page, 'filament-label-1')).toHaveText('Empty')
    await expect(byTestId(page, 'layer-input-2')).toHaveValue('11')
    await expect(byTestId(page, 'slider-column-1')).toHaveAttribute('data-selected', 'true')
  })

  test('sliders dragged past each other can be sorted back into print order', async ({ page, request }) => {
    const [a, b] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid)])
    await openApp(page)
    await expect(byTestId(page, 'sort-bands-btn')).toHaveCount(0)

    // Sliders span every layer, so a band can pass its neighbour.
    await typeInto(page, 'layer-input-0', '10')
    await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
    await expect(byTestId(page, 'sort-bands-btn')).toBeVisible()
    await byTestId(page, 'sort-bands-btn').click()
    await expect(byTestId(page, 'filament-label-0')).toHaveText(b.name)
    await expect(byTestId(page, 'layer-input-1')).toHaveValue('10')
    await expect(byTestId(page, 'sort-bands-btn')).toHaveCount(0)
  })
})

test.describe('Project', () => {
  test('a reload resumes the image without rebuilding the preview; New project clears it and Undo brings it back', async ({ page, request }) => {
    // Waits for a real heightmap init, which alone can take ~30s under load.
    test.setTimeout(120_000)
    await activePalette(request)
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await waitForSnapshot(page)

    let initRuns = 0
    page.on('request', (r) => { if (r.url().includes('/api/init/run')) initRuns += 1 })
    await page.reload()
    await expect(byTestId(page, 'input-image')).toBeVisible()
    await expect(byTestId(page, 'session-restored-banner')).toBeVisible()
    await expect(byTestId(page, 'top-start-btn')).toBeEnabled()
    await page.waitForTimeout(1500)
    expect(initRuns).toBe(0)

    await byTestId(page, 'start-new-project-btn').click()
    await byTestId(page, 'confirm-ok').click()
    await expect(byTestId(page, 'image-drop-zone')).toBeVisible()
    await expect(byTestId(page, 'session-restored-banner')).toHaveCount(0)
    await expect(byTestId(page, 'run-disabled-reason')).toHaveText('Upload an input image first')

    await byTestId(page, 'undo-btn').click()
    await expect(byTestId(page, 'input-image')).toBeVisible()
  })

  test('the project name names the saved file; unsaved changes are marked; Ctrl+S saves', async ({ page, request }, testInfo) => {
    const [a] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid)])
    await openApp(page)
    await expect(byTestId(page, 'project-dirty-indicator')).toBeVisible()

    await byTestId(page, 'project-name-input').fill('E2E My Print')
    await byTestId(page, 'td-input-0').click()
    const [download] = await Promise.all([page.waitForEvent('download'), page.keyboard.press('Control+s')])
    expect(download.suggestedFilename()).toBe('E2E-My-Print.json')
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'))
    expect(saved.name).toBe('E2E My Print')
    await expect(byTestId(page, 'project-dirty-indicator')).toHaveCount(0)

    await typeInto(page, 'td-input-0', '4')
    await expect(byTestId(page, 'project-dirty-indicator')).toBeVisible()

    const file = testInfo.outputPath('named.json')
    fs.writeFileSync(file, JSON.stringify({ ...saved, name: 'E2E Loaded Name' }))
    await byTestId(page, 'file-menu-load-input').setInputFiles(file)
    await expect(byTestId(page, 'project-name-input')).toHaveValue('E2E Loaded Name')
    await expect(byTestId(page, 'project-dirty-indicator')).toHaveCount(0)
  })

  test('swap instructions export before any run; model files say why they need one', async ({ page, request }) => {
    const [a, b] = await activePalette(request)
    await setSliders(page, [band(3, a.uuid), band(8, b.uuid)])
    await openApp(page)
    await byTestId(page, 'file-menu-btn').click()
    await expect(byTestId(page, 'download-stl')).toBeDisabled()
    await expect(byTestId(page, 'download-stl')).toHaveAttribute('title', /optimizer run/)
    const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'download-instructions').click()])
    expect(download.suggestedFilename()).toBe('swap_instructions.txt')
  })

  test('workflow steps are shortcuts to where each step happens', async ({ page }) => {
    await openApp(page)
    const chooser = page.waitForEvent('filechooser')
    await byTestId(page, 'workflow-step-btn-image').click()
    await chooser

    await byTestId(page, 'workflow-step-btn-filaments').click()
    await expect(byTestId(page, 'filter-input')).toBeFocused()

    await byTestId(page, 'workflow-step-btn-export').click()
    await expect(byTestId(page, 'file-menu-dropdown')).toBeVisible()
  })
})

test.describe('Image', () => {
  test('the image zooms with the wheel and resets on double-click; upload errors can be dismissed', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'image-file-input').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hi') })
    await expect(byTestId(page, 'image-upload-error')).toBeVisible()
    await byTestId(page, 'image-upload-error-dismiss').click()
    await expect(byTestId(page, 'image-upload-error')).toHaveCount(0)

    await uploadImage(page)
    const box = (await byTestId(page, 'input-image').boundingBox())!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.wheel(0, -300)
    await expect(byTestId(page, 'image-zoom-reset')).toBeVisible()
    await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2)
    await expect(byTestId(page, 'image-zoom-reset')).toHaveCount(0)
  })
})
