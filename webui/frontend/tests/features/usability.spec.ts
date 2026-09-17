import fs from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import {
  activePalette,
  byTestId,
  createFilament,
  historyLabels,
  activeLibraryItem,
  openApp,
  pickFilament,
  resetBackend,
  setActive,
  typeInto,
  uploadAndWaitForPreview,
  uploadImage,
  waitForSnapshot,
} from '../helpers'

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

async function setSliders(page: Page, sliders: object[]) {
  const state = await (await page.request.get('/api/project/state')).json()
  await page.request.post('/api/project/state', { data: { ...state, color_sliders: sliders } })
}

const band = (layer: number, filament_uuid: string, td = 2) => ({ td, layer, depth_mm: 0, filament_uuid, enabled: true })

test.describe('Workflow steps', () => {
  test('the step indicator follows the user through the workflow', async ({ page, request }) => {
    const f = await createFilament(request)
    await openApp(page)
    const state = (id: string) => byTestId(page, `workflow-step-${id}`)
    await expect(state('image')).toHaveAttribute('data-state', 'current')

    await uploadImage(page)
    await expect(state('image')).toHaveAttribute('data-state', 'done')
    await expect(state('filaments')).toHaveAttribute('data-state', 'current')

    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(state('filaments')).toHaveAttribute('data-state', 'done')
    await expect(state('run')).toHaveAttribute('data-state', 'current')
    await expect(state('export')).toHaveAttribute('data-state', 'upcoming')
  })

  test('Run waits for the preview to be built instead of competing with it', async ({ page, request }) => {
    // Waits for a real heightmap init, which alone can take ~30s under load.
    test.setTimeout(120_000)
    await activePalette(request)
    await page.route('**/api/init/run', async (route) => {
      await new Promise((r) => setTimeout(r, 2500))
      await route.continue()
    })
    await openApp(page)
    await uploadImage(page)
    await expect(byTestId(page, 'top-start-btn')).toBeDisabled()
    await expect(byTestId(page, 'run-disabled-reason')).toHaveText('Preparing the preview…')
    await expect(byTestId(page, 'top-start-btn')).toBeEnabled({ timeout: 60000 })
  })
})

test.describe('Color layers', () => {
  test('a fresh project starts empty with guidance, and bands can be added by hand', async ({ page, request }) => {
    const black = await createFilament(request, { name: 'E2E Band Black', color: '#000000', td: 0.6 })
    await setSliders(page, [])
    await setActive(request, [black])
    await openApp(page)

    await expect(byTestId(page, 'color-sliders-panel')).toContainText('No color layers yet')
    await expect(byTestId(page, 'color-core-empty')).toBeVisible()

    await byTestId(page, 'add-band-btn').click()
    await expect(byTestId(page, 'slider-column-0')).toBeVisible()
    await expect(byTestId(page, 'band-range-0')).toHaveText('1–5')
    await pickFilament(page, 0, black.uuid)
    await expect(byTestId(page, 'filament-label-0')).toHaveText('E2E Band Black')
    await expect(byTestId(page, 'td-input-0')).toHaveValue('0.6')

    await byTestId(page, 'add-band-btn').click()
    await expect(byTestId(page, 'band-range-1')).toHaveText('6–10')
    await byTestId(page, 'remove-band-1').click()
    await expect(byTestId(page, 'slider-column-1')).toHaveCount(0)
    await waitForSnapshot(page)
    expect((await historyLabels(page)).at(-1)).toBe('▶Removed band 2')
  })

  test('dragging a filament onto the empty panel creates a band', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Dragged' })
    await setSliders(page, [])
    await openApp(page)
    await page.locator(`[data-testid="filament-${f.uuid}"]`).dragTo(byTestId(page, 'color-sliders-panel'))
    await expect(byTestId(page, 'filament-label-0')).toHaveText('E2E Dragged')
    await expect(activeLibraryItem(page, f.uuid)).toBeVisible()
  })

  test('bands show their layer range and can be filtered by filament', async ({ page, request }) => {
    const a = await createFilament(request, { name: 'E2E Filter A', color: '#ff0000' })
    const b = await createFilament(request, { name: 'E2E Filter B', color: '#00ff00' })
    await setActive(request, [a, b])
    await setSliders(page, [band(3, a.uuid), band(7, b.uuid), band(12, a.uuid)])
    await openApp(page)

    await expect(byTestId(page, 'band-range-0')).toHaveText('1–3')
    await expect(byTestId(page, 'band-range-1')).toHaveText('4–7')
    await expect(byTestId(page, 'band-range-2')).toHaveText('8–12')

    await byTestId(page, 'band-filter').selectOption(a.uuid)
    await expect(page.locator('[data-testid^="slider-column-"]')).toHaveCount(2)
    await expect(byTestId(page, 'slider-column-1')).toHaveCount(0)
    await byTestId(page, 'band-filter').selectOption('all')
    await expect(page.locator('[data-testid^="slider-column-"]')).toHaveCount(3)

    // The overview strip has one segment per band, sized by its layers.
    await expect(byTestId(page, 'stack-overview').locator('button')).toHaveCount(3)
  })

  test('hide/show and delete buttons are labelled for screen readers', async ({ page, request }) => {
    const a = await createFilament(request)
    await setSliders(page, [band(3, a.uuid)])
    await openApp(page)
    await expect(page.getByRole('button', { name: 'Hide band 1' })).toBeVisible()
    await page.getByRole('button', { name: 'Hide band 1' }).click()
    await expect(page.getByRole('button', { name: 'Show band 1' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Delete band 1' })).toBeVisible()
  })

  test('running over hand-made colors asks first', async ({ page, request }) => {
    // Waits for a real heightmap init, which alone can take ~30s under load.
    test.setTimeout(120_000)
    const [black] = await activePalette(request)
    await setSliders(page, [])
    let started = 0
    await page.route('**/api/optimize/start', (route) => {
      started += 1
      return route.fulfill({ status: 409, json: { detail: 'mocked' } })
    })
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await byTestId(page, 'add-band-btn').click()
    await pickFilament(page, 0, black.uuid)

    await byTestId(page, 'top-start-btn').click()
    await expect(byTestId(page, 'confirm-dialog')).toContainText('Replace your color layers?')
    await byTestId(page, 'confirm-cancel').click()
    expect(started).toBe(0)

    await byTestId(page, 'top-start-btn').click()
    await byTestId(page, 'confirm-ok').click()
    await expect(byTestId(page, 'start-error')).toHaveText('mocked')
    expect(started).toBe(1)
  })
})

test.describe('Print plan', () => {
  test('shows counts and the swap list for the layers as edited, and exports them', async ({ page, request }, testInfo) => {
    const a = await createFilament(request, { brand: 'E2E Plan', name: 'Black', color: '#000000' })
    const b = await createFilament(request, { brand: 'E2E Plan', name: 'White', color: '#ffffff' })
    await setActive(request, [a, b])
    await setSliders(page, [band(2, a.uuid), band(5, a.uuid), band(9, b.uuid), band(12, a.uuid)])
    await openApp(page)

    await expect(byTestId(page, 'tab-print-plan')).toContainText('2 swaps')
    await byTestId(page, 'tab-print-plan').click()
    await expect(byTestId(page, 'plan-colors')).toContainText('2')
    await expect(byTestId(page, 'plan-swaps')).toContainText('2')
    await expect(byTestId(page, 'plan-layers')).toContainText('12')
    await expect(byTestId(page, 'plan-height')).toContainText('0.72 mm')
    const rows = page.locator('[data-testid="plan-swap-row"]')
    await expect(rows).toHaveCount(3)
    await expect(rows.nth(1)).toContainText('7')
    await expect(rows.nth(1)).toContainText('0.48 mm')
    await expect(rows.nth(1)).toContainText('E2E Plan White')

    const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'plan-download-btn').click()])
    expect(download.suggestedFilename()).toBe('swap_instructions.txt')
    const text = fs.readFileSync(await download.path(), 'utf8')
    expect(text).toContain('At layer #7 (0.48mm) swap to E2E Plan - White')
    expect(text).toContain('For the rest, use E2E Plan - Black')

    // Edits are reflected immediately.
    await byTestId(page, 'tab-color-layers').click()
    await byTestId(page, 'toggle-2').click()
    await expect(byTestId(page, 'tab-print-plan')).toContainText('0 swaps')
    void testInfo
  })

  test('copy puts the instructions on the clipboard', async ({ page, request, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write'])
    const a = await createFilament(request, { brand: 'E2E Copy', name: 'Red' })
    await setSliders(page, [band(4, a.uuid)])
    await openApp(page)
    await byTestId(page, 'tab-print-plan').click()
    await byTestId(page, 'plan-copy-btn').click()
    await expect(byTestId(page, 'toast-info')).toHaveText('Swap instructions copied')
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('swap to E2E Copy - Red')
  })

  test('with no layers the plan explains how to get some', async ({ page }) => {
    await setSliders(page, [])
    await openApp(page)
    await byTestId(page, 'tab-print-plan').click()
    await expect(byTestId(page, 'print-plan')).toContainText('run the optimizer or add bands by hand')
  })
})

test.describe('Library', () => {
  test('active filaments are marked with a check and labelled buttons', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Check Me' })
    await openApp(page)
    const toggle = byTestId(page, `toggle-filament-${f.uuid}`)
    await expect(toggle).toHaveAccessibleName('Add E2E Check Me to active filaments')
    await toggle.click()
    await expect(toggle).toHaveAccessibleName('Remove E2E Check Me from active filaments')
    await expect(byTestId(page, `filament-${f.uuid}`)).toHaveAttribute('data-active', 'true')
  })

  test('owned-only filter and sorting', async ({ page, request }) => {
    const owned = await createFilament(request, { brand: 'E2E Sort', name: 'Zeta Owned', owned: true, td: 9, color: '#ff0000' })
    const notOwned = await createFilament(request, { brand: 'E2E Sort', name: 'Alpha', td: 1, color: '#0000ff' })
    await openApp(page)
    await expect(byTestId(page, `owned-badge-${owned.uuid}`)).toBeVisible()

    await byTestId(page, 'owned-only-toggle').check()
    await expect(byTestId(page, `filament-${notOwned.uuid}`)).toHaveCount(0)
    await expect(byTestId(page, `filament-${owned.uuid}`)).toBeVisible()
    await byTestId(page, 'owned-only-toggle').uncheck()

    const order = async () => (await page.locator('[data-testid="brand-E2E Sort"] + div [data-testid^="filament-"]').evaluateAll((els) => els.map((e) => e.textContent ?? '')))
    await byTestId(page, 'filament-sort').selectOption('name')
    expect((await order())[0]).toContain('Alpha')
    await byTestId(page, 'filament-sort').selectOption('td')
    expect((await order())[0]).toContain('Alpha')
    await byTestId(page, 'filament-sort').selectOption('color')
    expect((await order())[0]).toContain('Zeta Owned')
  })

  test('import explains the formats and offers example files', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'import-btn').click()
    await expect(byTestId(page, 'import-formats')).toContainText('Brand, Type, Color, Name, TD, Owned')
    const [csv] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'import-example-csv').click()])
    expect(csv.suggestedFilename()).toBe('filaments-example.csv')
    // The example can be imported as-is.
    await byTestId(page, 'file-input').setInputFiles({ name: 'filaments-example.csv', mimeType: 'text/csv', buffer: fs.readFileSync(await csv.path()) })
    await expect(byTestId(page, 'import-mode-choice')).toBeVisible()
  })
})

test.describe('Details', () => {
  test('dropping a non-image file explains what went wrong', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'image-file-input').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hi') })
    await expect(byTestId(page, 'image-upload-error')).toContainText('"notes.txt" isn\'t an image')
  })

  test('dialogs have exactly one close button', async ({ page }) => {
    await openApp(page)
    for (const [open, dialog] of [['new-filament-btn', 'new-filament-modal'], ['import-btn', 'import-modal'], ['settings-button', 'settings-modal']]) {
      await byTestId(page, open).click()
      // One ✕ icon (a built-in one used to sit next to each dialog's own).
      const closeIcons = byTestId(page, dialog).getByRole('button', { name: /^close/i }).filter({ has: page.locator('svg') })
      await expect(closeIcons).toHaveCount(1)
      await closeIcons.click()
      await expect(byTestId(page, dialog)).toHaveCount(0)
    }
  })

  test('mesh height readout says what the numbers are', async ({ page }) => {
    await openApp(page)
    await expect(byTestId(page, 'mesh-height-label')).toContainText('Top color layer:')
    await expect(byTestId(page, 'mesh-height-label')).toContainText('Max print height:')
  })

  test('the page itself never scrolls (the top bar stays in place)', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'settings-button').click()
    await byTestId(page, 'close-settings').click()
    await page.mouse.wheel(0, 600)
    expect(await page.evaluate(() => [window.scrollY, document.documentElement.scrollHeight <= window.innerHeight + 1])).toEqual([0, true])
  })
})

test.describe('Layout and themes', () => {
  test('nothing important is cut off on a 1280×720 laptop screen', async ({ browser, baseURL }) => {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, baseURL })
    await openApp(page)
    for (const id of ['settings-button', 'theme-toggle-btn', 'top-pruning-btn', 'top-start-btn', 'filter-input', 'new-filament-btn', 'tab-print-plan', 'global-stl-size']) {
      const box = await byTestId(page, id).boundingBox()
      expect(box, id).not.toBeNull()
      expect(box!.x + box!.width, id).toBeLessThanOrEqual(1280)
      expect(box!.y + box!.height, id).toBeLessThanOrEqual(720)
    }
    await page.close()
  })

  test('light theme: panels are light and text has enough contrast', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'theme-toggle-btn').click()
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')

    const report = await page.evaluate(() => {
      const parse = (c: string) => (c.match(/[\d.]+/g) ?? []).map(Number)
      const lum = ([r, g, b]: number[]) => {
        const f = (v: number) => {
          const s = v / 255
          return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
        }
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
      }
      const bgOf = (el: Element | null): number[] => {
        for (let e = el; e; e = e.parentElement) {
          const c = parse(getComputedStyle(e).backgroundColor)
          if (c.length >= 3 && (c[3] === undefined || c[3] > 0.5)) return c
        }
        return [255, 255, 255]
      }
      const ids = ['top-bar', 'filament-list', 'bottom-panel', 'global-params']
      const panels = ids.map((id) => [id, lum(bgOf(document.querySelector(`[data-testid="${id}"]`)))])
      const lowContrast: string[] = []
      for (const el of Array.from(document.querySelectorAll('span, p, label, button, h2, h3'))) {
        const text = (el as HTMLElement).innerText?.trim()
        if (!text || el.children.length > 0 || (el as HTMLElement).offsetParent === null) continue
        if ((el as HTMLElement).closest('[disabled], [aria-hidden="true"]')) continue
        const style = getComputedStyle(el)
        if (Number(style.opacity) < 0.9) continue
        const fg = lum(parse(style.color))
        const bg = lum(bgOf(el))
        const ratio = (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05)
        if (ratio < 3) lowContrast.push(`${text.slice(0, 30)} (${ratio.toFixed(2)})`)
      }
      return { panels, lowContrast }
    })
    for (const [id, l] of report.panels) expect(l as number, `${id} background`).toBeGreaterThan(0.6)
    expect(report.lowContrast).toEqual([])
  })
})
