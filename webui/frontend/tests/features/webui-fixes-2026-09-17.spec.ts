import { test, expect, type APIRequestContext } from '@playwright/test'
import { byTestId, createFilament, openApp, resetBackend, setActive, typeInto, waitForSnapshot } from '../helpers'

/**
 * The 2026-09-17 round of WebUI fixes, in the browser:
 *
 *  1  the base/background color shown as a (height-locked) row of the color layers
 *  2  File ▸ New project
 *  4  the mouse wheel scrolls the layer list instead of dragging a band
 *  5  History stays on the right image and the right layers across many steps
 *  6  the first-run tutorial, reopenable from the ? button
 *  7  pruning limits start at the current result
 *  8  the "run it more than once" hint
 * 12  "STL size", and the warning above 200 mm
 * 13  the filament library's auto-save toggle
 *
 * (Items 3, 9, 10 and 11 are backend behaviour — see
 * tests/webui/test_fixes_2026_09_17.py — with their UI halves covered here
 * where they're reachable without a GPU run.)
 */

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

// --- 1. The base color is part of the color layers --------------------------

test('the base color is shown as the bottom row of the color layers', async ({ page }) => {
  await openApp(page)

  const base = byTestId(page, 'base-band-row')
  await expect(base).toBeVisible()
  // Its thickness is the Base setting (0.24 mm by default), not something
  // the row itself can change.
  await expect(byTestId(page, 'base-height')).toHaveText('0.24 mm')
  await expect(byTestId(page, 'base-fixed-note')).toHaveText('Fixed')
  // No layer slider, no number field, no delete: nothing to drag it with.
  await expect(base.locator('input[type="range"]')).toHaveCount(0)
  await expect(base.locator('input[type="number"]')).toHaveCount(0)
})

test('the base row follows the Base height setting', async ({ page }) => {
  await openApp(page)
  const backgroundHeight = byTestId(page, 'global-background-height')
  await backgroundHeight.fill('0.6')
  await backgroundHeight.dispatchEvent('change')
  await expect(byTestId(page, 'base-height')).toHaveText('0.60 mm')
  await expect(byTestId(page, 'base-band-range')).toHaveText('base (15)')
})

test('the only thing you can change on the base row is its filament', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E Base Blue', color: '#2244aa', td: 1 })
  await openApp(page)

  await byTestId(page, 'base-filament-select').click()
  await byTestId(page, `filament-option-${filament.uuid}`).click()

  await expect(byTestId(page, 'base-band-row')).toHaveAttribute('data-base-color', '#2244aa')
  // Choosing by hand turns auto-selection off, or the next run would quietly
  // replace the choice with the closest match to the image again.
  await expect(byTestId(page, 'base-auto-note')).toHaveText('manual')
  await byTestId(page, 'settings-button').click()
  await expect(byTestId(page, 'setting-background_color')).toHaveValue('#2244aa')
})

test('the stack overview starts at the base', async ({ page }) => {
  await openApp(page)
  await expect(byTestId(page, 'stack-overview-base')).toBeVisible()
})

// --- 4. The wheel scrolls the list, it does not move bands ------------------

test('the mouse wheel over a band slider scrolls the list instead of moving the band', async ({ page }) => {
  await openApp(page)

  const slider = byTestId(page, 'slider-0')
  const list = byTestId(page, 'slider-columns')
  await expect(slider).toBeVisible()
  const before = await slider.inputValue()

  // Hover the slider and scroll, the way you would passing over the list.
  await slider.hover()
  await page.mouse.wheel(0, 240)
  await page.waitForTimeout(200)

  // The band did not move...
  await expect(slider).toHaveValue(before)
  // ...and the list scrolled, as it would anywhere else in the panel.
  expect(await list.evaluate((el) => el.scrollTop)).toBeGreaterThan(0)
})

test('scrolling over a slider leaves the layer number alone too', async ({ page }) => {
  await openApp(page)
  const layer = byTestId(page, 'layer-input-0')
  const before = await layer.inputValue()
  await byTestId(page, 'slider-0').hover()
  await page.mouse.wheel(0, -400)
  await page.mouse.wheel(0, 400)
  await page.waitForTimeout(200)
  await expect(layer).toHaveValue(before)
})

test('dragging a band slider still moves it', async ({ page }) => {
  await openApp(page)
  const slider = byTestId(page, 'slider-0')
  const box = await slider.boundingBox()
  expect(box).not.toBeNull()
  await page.mouse.move(box!.x + box!.width * 0.1, box!.y + box!.height / 2)
  await page.mouse.down()
  await page.mouse.move(box!.x + box!.width * 0.8, box!.y + box!.height / 2, { steps: 5 })
  await page.mouse.up()
  await expect(byTestId(page, 'layer-input-0')).not.toHaveValue('8')
})

// --- 2. New project ---------------------------------------------------------

test('File ▸ New project clears the image, the layers and the result', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E New Project' })
  await setActive(request, [filament])
  await openApp(page)

  // Give the project something to clear.
  await typeInto(page, 'project-name-input', 'Something')
  await byTestId(page, 'add-band-btn').click()
  await waitForSnapshot(page)

  await byTestId(page, 'file-menu-btn').click()
  await byTestId(page, 'file-menu-new').click()
  await byTestId(page, 'confirm-ok').click()

  await expect(byTestId(page, 'image-drop-zone')).toBeVisible()
  await expect(byTestId(page, 'project-name-input')).toHaveValue('')
  await expect(page.locator('[data-testid^="slider-column-"]')).toHaveCount(0)
  // Active filaments are explicitly kept — only the project is new.
  await expect(byTestId(page, 'active-filaments-summary')).toHaveAttribute('data-count', '1')
})

test('New project is undoable', async ({ page }) => {
  await openApp(page)
  const columns = page.locator('[data-testid^="slider-column-"]')
  // Undo needs a step to go back *to*, so record one first.
  await typeInto(page, 'td-input-0', '3')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  await waitForSnapshot(page)
  const before = await columns.count()
  expect(before).toBeGreaterThan(0)

  await byTestId(page, 'file-menu-btn').click()
  await byTestId(page, 'file-menu-new').click()
  await byTestId(page, 'confirm-ok').click()
  await expect(columns).toHaveCount(0)

  await waitForSnapshot(page)
  await byTestId(page, 'undo-btn').click()
  await expect(columns).toHaveCount(before)
  await expect(byTestId(page, 'td-input-0')).toHaveValue('3')
})

// --- 5. History across many steps and several images ------------------------

interface SeedStep {
  label: string
  image: string
  sliders: { td: number; layer: number; depth_mm: number; filament_uuid: string; enabled: boolean }[]
}

/** Writes snapshots straight to the backend, the same shape the store posts.
 * Seeding beats making 50+ real edits: it's deterministic, it needs no GPU,
 * and it's the *reload* path — the one that hydrates the undo stack from the
 * server — which is where the image/slider mix-ups actually showed up. */
async function seedHistory(request: APIRequestContext, steps: SeedStep[]) {
  await request.delete('/api/state/history')
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i]
    const res = await request.post('/api/state/snapshot', {
      data: {
        timestamp: 1_700_000_000 + i,
        label: step.label,
        activeFilaments: [],
        colorSliders: step.sliders,
        settings: { input_image: step.image, layer_height: 0.04, background_height: 0.24, max_layers: 75 },
        // Deliberately absent: the display URL. Real snapshots often have a
        // stale blob: or nothing at all, and the restore has to work from
        // settings.input_image alone.
        inputImage: null,
        currentJobId: null,
        optimizationResultId: null,
        jobStatus: null,
        sliderLayerRange: { min: 0, max: 75 },
      },
    })
    expect(res.ok()).toBeTruthy()
  }
}

const band = (layer: number, uuid: string) => ({ td: 2, layer, depth_mm: layer * 0.04, filament_uuid: uuid, enabled: true })

test('history keeps each step on its own image and its own layers across 50+ steps', async ({ page, request }) => {
  test.setTimeout(180_000)

  const images = ['hist_a.png', 'hist_b.png', 'hist_c.png']
  const filaments = [
    await createFilament(request, { name: 'E2E Hist A', color: '#aa1111' }),
    await createFilament(request, { name: 'E2E Hist B', color: '#11aa11' }),
    await createFilament(request, { name: 'E2E Hist C', color: '#1111aa' }),
  ]
  // Three images, ~18 steps each: the "several optimizations, several
  // pictures" session that used to end up showing one image's photo next to
  // another image's colors.
  const steps: SeedStep[] = []
  for (let i = 0; i < 54; i++) {
    const group = Math.floor(i / 18)
    steps.push({
      label: `${images[group]} step ${i}`,
      image: images[group],
      sliders: [band(5 + (i % 18), filaments[group].uuid)],
    })
  }
  await seedHistory(request, steps)

  await openApp(page)
  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  // The client keeps the newest 50 (MAX_HISTORY); the server caps at the same
  // number, so both ends agree on which steps still exist.
  await expect(entries).toHaveCount(50)

  const seeded = steps.slice(steps.length - 50)

  // Jump around deliberately — including back and forth across image
  // boundaries, which is the case that used to get confused.
  for (const index of [0, 49, 25, 1, 48, 17, 18, 35, 34, 2, 47]) {
    await entries.nth(index).click()
    const expected = seeded[index]
    await expect(byTestId(page, 'input-image')).toHaveAttribute('src', `/uploads/${expected.image}`)
    await expect(byTestId(page, 'layer-input-0')).toHaveValue(String(expected.sliders[0].layer))
    await expect(byTestId(page, `filament-select-0`)).toHaveAttribute('data-value', expected.sliders[0].filament_uuid)
  }
})

test('undo and redo walk the same steps the history list shows', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E Walk' })
  const steps: SeedStep[] = [
    { label: 'a', image: 'walk_a.png', sliders: [band(4, filament.uuid)] },
    { label: 'b', image: 'walk_b.png', sliders: [band(9, filament.uuid)] },
    { label: 'c', image: 'walk_c.png', sliders: [band(14, filament.uuid)] },
  ]
  await seedHistory(request, steps)
  await openApp(page)

  // The page opens on the *project state*, which a reset leaves empty; step
  // onto the newest entry first, the way a user picking up a session does.
  await byTestId(page, 'history-open-btn').click()
  await page.locator('[data-testid="history-entry-2"]').click()
  await byTestId(page, 'history-close-btn').click()

  const image = byTestId(page, 'input-image')
  await expect(image).toHaveAttribute('src', '/uploads/walk_c.png')

  await byTestId(page, 'undo-btn').click()
  await expect(image).toHaveAttribute('src', '/uploads/walk_b.png')
  await expect(byTestId(page, 'layer-input-0')).toHaveValue('9')

  await byTestId(page, 'undo-btn').click()
  await expect(image).toHaveAttribute('src', '/uploads/walk_a.png')
  await expect(byTestId(page, 'layer-input-0')).toHaveValue('4')

  await byTestId(page, 'redo-btn').click()
  await expect(image).toHaveAttribute('src', '/uploads/walk_b.png')
  await expect(byTestId(page, 'layer-input-0')).toHaveValue('9')
})

test('a step without an image shows no image, whatever the step before it had', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E Blank' })
  await seedHistory(request, [
    { label: 'with image', image: 'blank_a.png', sliders: [band(4, filament.uuid)] },
    { label: 'new project', image: '', sliders: [] },
  ])
  await openApp(page)

  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  await entries.nth(0).click()
  await expect(byTestId(page, 'input-image')).toHaveAttribute('src', '/uploads/blank_a.png')
  await entries.nth(1).click()
  await expect(byTestId(page, 'input-image')).toHaveCount(0)
  await expect(byTestId(page, 'image-drop-zone')).toBeVisible()
})

// --- The tutorial covers the auto-repeat toggle -----------------------------

test('the tutorial explains that pruning can repeat itself', async ({ page }) => {
  await openApp(page)
  await byTestId(page, 'tutorial-open-btn').click()
  for (let i = 0; i < 3; i++) await byTestId(page, 'tutorial-next-btn').click()
  await expect(byTestId(page, 'tutorial-body')).toHaveAttribute('data-step', 'prune')
  await expect(byTestId(page, 'tutorial-body')).toContainText('Keep pruning until it stops improving')
})

// --- 6. The tutorial --------------------------------------------------------

test.describe('first-run tutorial', () => {
  // Opt out of the suite-wide "already seen" flag: this is the first visit.
  test.use({ storageState: { cookies: [], origins: [] } })

  test('opens by itself the first time, and not again afterwards', async ({ page }) => {
    await page.goto('/')
    await expect(byTestId(page, 'tutorial-modal')).toBeVisible()

    await byTestId(page, 'tutorial-close-x').click()
    await expect(byTestId(page, 'tutorial-modal')).toHaveCount(0)

    await page.reload()
    await expect(byTestId(page, 'app')).toBeVisible()
    await expect(byTestId(page, 'tutorial-modal')).toHaveCount(0)
  })

  test('walks through the whole workflow, pruning and STL size included', async ({ page }) => {
    await page.goto('/')
    await expect(byTestId(page, 'tutorial-modal')).toBeVisible()
    await expect(byTestId(page, 'tutorial-progress')).toHaveText('1 of 6')

    const seen: string[] = []
    for (let i = 0; i < 6; i++) {
      seen.push(await byTestId(page, 'tutorial-body').innerText())
      if (i < 5) await byTestId(page, 'tutorial-next-btn').click()
    }
    const text = seen.join('\n')

    // It has to actually explain the job, not just name the buttons.
    expect(text).toMatch(/active filament/i)
    expect(text).toMatch(/\bRun\b/)
    expect(text).toMatch(/prun/i)
    expect(text).toMatch(/more than once|several times|again/i)
    expect(text).toMatch(/swap/i)
    expect(text).toMatch(/STL size/i)
    expect(text).toMatch(/VRAM/)
    expect(text).toMatch(/350/)

    await byTestId(page, 'tutorial-done-btn').click()
    await expect(byTestId(page, 'tutorial-modal')).toHaveCount(0)
  })

  test('Back returns to the previous step', async ({ page }) => {
    await page.goto('/')
    await byTestId(page, 'tutorial-next-btn').click()
    await expect(byTestId(page, 'tutorial-progress')).toHaveText('2 of 6')
    await byTestId(page, 'tutorial-back-btn').click()
    await expect(byTestId(page, 'tutorial-progress')).toHaveText('1 of 6')
    await expect(byTestId(page, 'tutorial-back-btn')).toBeDisabled()
  })
})

test('the ? button in the top bar reopens the tutorial from the start', async ({ page }) => {
  await openApp(page)
  await expect(byTestId(page, 'tutorial-modal')).toHaveCount(0)

  await byTestId(page, 'tutorial-open-btn').click()
  await expect(byTestId(page, 'tutorial-modal')).toBeVisible()
  await expect(byTestId(page, 'tutorial-progress')).toHaveText('1 of 6')

  await byTestId(page, 'tutorial-next-btn').click()
  await byTestId(page, 'tutorial-close-x').click()
  // Reopening starts over rather than resuming somewhere in the middle.
  await byTestId(page, 'tutorial-open-btn').click()
  await expect(byTestId(page, 'tutorial-progress')).toHaveText('1 of 6')
})

// --- 12. STL size -----------------------------------------------------------

test('the size field is labelled STL size and warns above 200 mm', async ({ page }) => {
  await openApp(page)

  await expect(byTestId(page, 'global-params')).toContainText('STL size')
  await expect(byTestId(page, 'stl-size-warning')).toHaveCount(0)

  await typeInto(page, 'global-stl-size', '250')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })

  const warning = byTestId(page, 'stl-size-warning')
  await expect(warning).toBeVisible()
  const title = await warning.getAttribute('title')
  expect(title).toContain('VRAM')
  expect(title).toContain('350')
  // The hover text explains the "you can scale it up anyway" part.
  expect(title).toContain('double')

  await typeInto(page, 'global-stl-size', '150')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  await expect(byTestId(page, 'stl-size-warning')).toHaveCount(0)
})

test('exactly 200 mm is not a warning', async ({ page }) => {
  await openApp(page)
  await typeInto(page, 'global-stl-size', '200')
  await byTestId(page, 'global-params').click({ position: { x: 5, y: 5 } })
  await expect(byTestId(page, 'stl-size-warning')).toHaveCount(0)
})

// --- 13. Filament library auto save ----------------------------------------

test('library auto save is on by default and writes edits as they happen', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E AutoSave', td: 2 })
  await openApp(page)

  await expect(byTestId(page, 'library-auto-save-toggle')).toBeChecked()

  await byTestId(page, `edit-filament-${filament.uuid}`).click()
  await expect(byTestId(page, 'edit-filament-modal')).toBeVisible()
  await typeInto(page, 'edit-filament-td', '4.5')

  // No Save click: the change lands in the library on its own.
  await expect(byTestId(page, 'edit-filament-autosave-state')).toContainText('Saved')
  await expect
    .poll(async () => {
      const all = await (await request.get('/api/filaments')).json()
      return all.find((f: { uuid: string }) => f.uuid === filament.uuid)?.td
    })
    .toBe(4.5)

  // And closing without saving keeps it.
  await byTestId(page, 'close-edit-filament-btn').click()
  await expect(byTestId(page, 'edit-filament-modal')).toHaveCount(0)
  const saved = await (await request.get('/api/filaments')).json()
  expect(saved.find((f: { uuid: string }) => f.uuid === filament.uuid).td).toBe(4.5)
})

test('turning auto save off puts the edit back behind the Save button', async ({ page, request }) => {
  const filament = await createFilament(request, { name: 'E2E ManualSave', td: 2 })
  await openApp(page)

  await byTestId(page, 'library-auto-save-toggle').uncheck()
  await expect(byTestId(page, 'library-auto-save-state')).toHaveText('— off')

  await byTestId(page, `edit-filament-${filament.uuid}`).click()
  await typeInto(page, 'edit-filament-td', '7')
  await expect(byTestId(page, 'edit-filament-autosave-state')).toHaveCount(0)
  await page.waitForTimeout(800)

  const unchanged = await (await request.get('/api/filaments')).json()
  expect(unchanged.find((f: { uuid: string }) => f.uuid === filament.uuid).td).toBe(2)

  await byTestId(page, 'save-filament-btn').click()
  await expect(byTestId(page, 'edit-filament-modal')).toHaveCount(0)
  const after = await (await request.get('/api/filaments')).json()
  expect(after.find((f: { uuid: string }) => f.uuid === filament.uuid).td).toBe(7)
})

test('the auto save choice survives a reload', async ({ page }) => {
  await openApp(page)
  await byTestId(page, 'library-auto-save-toggle').uncheck()
  await page.reload()
  await expect(byTestId(page, 'app')).toBeVisible()
  await expect(byTestId(page, 'library-auto-save-toggle')).not.toBeChecked()
})
