import fs from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import { byTestId, openApp, resetBackend, uploadImage } from '../helpers'

// The image editor: adjustment layers over the uploaded picture, each for
// the whole picture or a selection, applied as the picture the optimizer
// uses. The pixel work itself is covered by tests/image-edit.test.mjs.
// No filaments are active here, so no auto-preview (GPU) runs.

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

async function settings(page: Page) {
  return (await (await page.request.get('/api/project/state')).json()).settings
}

/** The project settings once the app has saved them in the state `ready`
 * describes (the app saves its state to the server shortly after a change). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function settingsWhen(page: Page, ready: (s: any) => boolean) {
  await expect.poll(async () => ready(await settings(page)), { timeout: 15_000 }).toBe(true)
  return settings(page)
}

async function uploaded(page: Page): Promise<string> {
  await uploadImage(page)
  return (await settingsWhen(page, (s) => !!s.input_image)).input_image
}

/** An uploaded picture's pixels (RGBA), decoded in the page. */
async function pixels(page: Page, filename: string): Promise<{ w: number; h: number; data: number[] }> {
  return page.evaluate(async (url) => {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`)
    const bitmap = await createImageBitmap(await response.blob())
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(bitmap, 0, 0)
    return { w: bitmap.width, h: bitmap.height, data: Array.from(ctx.getImageData(0, 0, bitmap.width, bitmap.height).data) }
  }, `/uploads/${encodeURIComponent(filename)}`)
}

const mean = (p: { data: number[] }, channel = -1) => {
  let sum = 0
  let n = 0
  for (let i = 0; i < p.data.length; i += 4) {
    sum += channel < 0 ? (p.data[i] + p.data[i + 1] + p.data[i + 2]) / 3 : p.data[i + channel]
    n++
  }
  return sum / n
}

/** How colourful the pixels with x in [x0, x1) are (mean max-min spread). */
const spread = (p: { w: number; data: number[] }, x0: number, x1: number) => {
  let sum = 0
  let n = 0
  for (let i = 0; i < p.data.length; i += 4) {
    const x = (i / 4) % p.w
    if (x < x0 || x >= x1) continue
    sum += Math.max(p.data[i], p.data[i + 1], p.data[i + 2]) - Math.min(p.data[i], p.data[i + 1], p.data[i + 2])
    n++
  }
  return sum / n
}

async function openEditor(page: Page) {
  await byTestId(page, 'edit-image-btn').click()
  await expect(byTestId(page, 'image-editor')).toHaveAttribute('data-ready', 'true', { timeout: 15_000 })
}

async function applyEdit(page: Page) {
  await byTestId(page, 'editor-apply').click()
  await expect(byTestId(page, 'image-editor')).toHaveCount(0, { timeout: 20_000 })
}

/** A slider moved with the keyboard, like a user would (one undo step). */
async function press(page: Page, testId: string, key: string) {
  await byTestId(page, testId).focus()
  await page.keyboard.press(key)
}

/** Drag across the picture between two points given as fractions of it. */
async function dragOnPicture(page: Page, from: [number, number], to: [number, number]) {
  const box = (await byTestId(page, 'editor-canvas').boundingBox())!
  await page.mouse.move(box.x + box.width * from[0], box.y + box.height * from[1])
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * to[0], box.y + box.height * to[1], { steps: 6 })
  await page.mouse.up()
}

test('an edit of the whole picture is applied, kept, and undoable', async ({ page }) => {
  await openApp(page)
  const source = await uploaded(page)
  const before = await pixels(page, source)

  await openEditor(page)
  await press(page, 'adjust-exposure', 'End')
  await expect(byTestId(page, 'adjust-exposure-value')).toHaveText('+3.00 EV')
  await applyEdit(page)

  // The optimizer now gets the edited picture; the upload stays as it was.
  const s = await settingsWhen(page, (s) => s.image_edit?.source === source)
  expect(s.input_image).not.toBe(source)
  expect(s.image_edit.layers[0].adjust.exposure).toBe(3)
  expect(mean(await pixels(page, s.input_image))).toBeGreaterThan(mean(before) + 20)
  expect(await pixels(page, source)).toEqual(before)
  await expect(byTestId(page, 'edit-image-dot')).toBeVisible()

  // Holding the eye shows the unedited picture.
  await byTestId(page, 'show-unedited-btn').hover()
  await page.mouse.down()
  await expect(byTestId(page, 'input-image')).toHaveAttribute('src', `/uploads/${source}`)
  await page.mouse.up()
  await expect(byTestId(page, 'input-image')).not.toHaveAttribute('src', `/uploads/${source}`)

  // Opening the editor again continues from the edit.
  await openEditor(page)
  await expect(byTestId(page, 'adjust-exposure-value')).toHaveText('+3.00 EV')
  await byTestId(page, 'editor-cancel').click()
  await expect(byTestId(page, 'image-editor')).toHaveCount(0)

  // The app's undo goes back to the unedited picture.
  await byTestId(page, 'undo-btn').click()
  await expect.poll(async () => (await settings(page)).input_image).toBe(source)
  expect((await settings(page)).image_edit ?? null).toBeNull()
  await expect(byTestId(page, 'edit-image-dot')).toHaveCount(0)
})

test('a layer on a selection changes only the selected part', async ({ page }) => {
  await openApp(page)
  await uploaded(page)
  await openEditor(page)

  await byTestId(page, 'editor-scope-selection').click()
  await expect(byTestId(page, 'editor-empty-selection')).toBeVisible()
  await byTestId(page, 'editor-tool-rect').click()
  // The left half, starting above the picture's edge (shapes are clipped).
  await dragOnPicture(page, [0.001, -0.05], [0.5, 1.05])
  await expect(byTestId(page, 'editor-empty-selection')).toHaveCount(0)
  await expect(byTestId(page, 'editor-layer-0').getByTestId('editor-layer-scope')).toHaveText('50%')
  // Shown as an outline around the left half (a rectangle: four straight
  // edges), not as a tint over the picture.
  const outline = byTestId(page, 'editor-selection-outline')
  await expect(outline).toHaveAttribute('data-segments', '4')
  await expect(page.locator('[data-testid="editor-canvas"] canvas')).toHaveCount(1)
  await byTestId(page, 'editor-show-selection').click()
  await expect(outline).toHaveCount(0)
  await byTestId(page, 'editor-show-selection').click()
  await expect(outline).toBeVisible()
  // Hard edges, so the halves are clean.
  await press(page, 'editor-feather', 'Home')
  await press(page, 'adjust-saturation', 'Home')
  await applyEdit(page)

  const s = await settingsWhen(page, (s) => !!s.image_edit)
  expect(s.image_edit.layers[0].selection).toBe('mask')
  expect(s.image_edit.layers[0].mask).not.toBe('')
  const edited = await pixels(page, s.input_image)
  const original = await pixels(page, s.image_edit.source)
  expect(spread(edited, 0, edited.w / 2 - 2)).toBeLessThan(3)
  expect(spread(edited, edited.w / 2 + 2, edited.w)).toBeCloseTo(spread(original, original.w / 2 + 2, original.w), 0)
})

test('magic wand, fewer colours, layers and the editor\'s own undo', async ({ page }) => {
  await openApp(page)
  await uploaded(page)
  await openEditor(page)

  // A second layer, limited to what the magic wand picks.
  await byTestId(page, 'editor-add-layer').click()
  await expect(page.locator('[data-testid^="editor-layer-"][data-selected="true"]')).toHaveAttribute('data-testid', 'editor-layer-1')
  await byTestId(page, 'editor-tool-wand').click()
  await byTestId(page, 'editor-wand-contiguous').uncheck()
  const box = (await byTestId(page, 'editor-canvas').boundingBox())!
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5)
  const scope = byTestId(page, 'editor-layer-1').getByTestId('editor-layer-scope')
  await expect(scope).not.toHaveText('All')
  // Raising the tolerance afterwards grows that click's selection.
  const share = async () => Number((await scope.innerText()).replace('%', ''))
  const clicked = await share()
  await press(page, 'editor-wand-tolerance', 'End')
  await expect.poll(share).toBeGreaterThan(clicked)

  // Fewer colours on it: the palette shows at most the chosen number.
  await byTestId(page, 'editor-section-reduce-toggle').click()
  await byTestId(page, 'reduce-enabled').check()
  const palette = byTestId(page, 'reduce-palette')
  await expect(palette).toBeVisible({ timeout: 15_000 })
  expect(Number(await palette.getAttribute('data-count'))).toBeLessThanOrEqual(16)
  await press(page, 'reduce-colors', 'Home')
  await expect(palette).toHaveAttribute('data-count', '2')

  // Undo inside the editor steps back through those changes.
  await byTestId(page, 'editor-undo').click()
  await expect(palette).not.toHaveAttribute('data-count', '2')
  await byTestId(page, 'editor-undo').click()
  await expect(byTestId(page, 'reduce-enabled')).not.toBeChecked()
  await byTestId(page, 'editor-redo').click()
  await expect(byTestId(page, 'reduce-enabled')).toBeChecked()

  // Hiding the layer shows the picture without it.
  await byTestId(page, 'editor-layer-1').getByTestId('editor-layer-visible').click()
  await expect(byTestId(page, 'editor-layer-1')).toContainText('Layer')

  // Cancelling asks first, and keeps the picture as it was.
  const before = (await settings(page)).input_image
  await byTestId(page, 'editor-cancel').click()
  await expect(byTestId(page, 'confirm-dialog')).toContainText('Discard your changes?')
  await byTestId(page, 'confirm-cancel').click()
  await expect(byTestId(page, 'image-editor')).toBeVisible()
  await byTestId(page, 'editor-cancel').click()
  await byTestId(page, 'confirm-ok').click()
  await expect(byTestId(page, 'image-editor')).toHaveCount(0)
  expect((await settings(page)).input_image).toBe(before)
})

test('removing every edit goes back to the picture as uploaded', async ({ page }) => {
  await openApp(page)
  const source = await uploaded(page)
  await openEditor(page)
  await press(page, 'adjust-contrast', 'End')
  await applyEdit(page)
  await settingsWhen(page, (s) => !!s.image_edit)

  await openEditor(page)
  await byTestId(page, 'editor-layer-0').hover()
  await byTestId(page, 'editor-layer-0').getByTestId('editor-layer-delete').click()
  await expect(byTestId(page, 'adjust-contrast-value')).toHaveText('0')
  await applyEdit(page)
  await expect.poll(async () => (await settings(page)).input_image).toBe(source)
  expect((await settings(page)).image_edit ?? null).toBeNull()
})

test('a saved project recreates the edited picture on another computer', async ({ page, baseURL }, testInfo) => {
  await openApp(page)
  await uploaded(page)
  await openEditor(page)
  // Two layers: a whole-picture curve and colour change, and a selection
  // (lasso, then a brush stroke added) with fewer colours and soft edges.
  await byTestId(page, 'editor-section-curves-toggle').click()
  const curve = (await byTestId(page, 'curve-canvas').boundingBox())!
  await page.mouse.click(curve.x + curve.width * 0.5, curve.y + curve.height * 0.3)
  await press(page, 'adjust-temperature', 'End')
  await byTestId(page, 'editor-add-layer').click()
  await byTestId(page, 'editor-tool-lasso').click()
  const box = (await byTestId(page, 'editor-canvas').boundingBox())!
  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height * 0.1)
  await page.mouse.down()
  for (const [fx, fy] of [[0.6, 0.15], [0.55, 0.8], [0.15, 0.7]]) await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy, { steps: 5 })
  await page.mouse.up()
  await byTestId(page, 'editor-tool-brush').click()
  await dragOnPicture(page, [0.7, 0.5], [0.95, 0.6])
  await byTestId(page, 'editor-section-reduce-toggle').click()
  await byTestId(page, 'reduce-enabled').check()
  await press(page, 'reduce-colors', 'Home')
  await applyEdit(page)

  const saved = await settingsWhen(page, (s) => !!s.image_edit)
  expect(saved.image_edit.layers).toHaveLength(2)
  const editedBefore = await pixels(page, saved.input_image)
  const originalBefore = await pixels(page, saved.image_edit.source)

  // Save the project file.
  await byTestId(page, 'file-menu-btn').click()
  const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'file-menu-save').click()])
  const file = testInfo.outputPath('edited-project.json')
  await download.saveAs(file)

  // Another computer: none of the files the project names exist there. Every
  // name in the file is changed to one this server has never seen, so all
  // that can be used is what the file itself carries.
  let text = fs.readFileSync(file, 'utf-8')
  const names = [saved.input_image, saved.image_edit.source, ...saved.image_edit.layers.map((l: { mask: string }) => l.mask).filter(Boolean)]
  names.forEach((name: string, i: number) => {
    text = text.split(name).join(`elsewhere-${i}-${name}`)
  })
  fs.writeFileSync(file, text)
  await resetBackend(baseURL!)
  await openApp(page)
  await byTestId(page, 'file-menu-load-input').setInputFiles(file)

  // The edited picture is the same, pixel for pixel.
  await expect.poll(async () => (await settings(page)).image_edit?.layers?.length ?? 0, { timeout: 20_000 }).toBe(2)
  const loaded = await settings(page)
  for (const name of [loaded.input_image, loaded.image_edit.source, ...loaded.image_edit.layers.map((l: { mask: string }) => l.mask).filter(Boolean)]) {
    expect(name).not.toContain('elsewhere-')
  }
  expect(await pixels(page, loaded.input_image)).toEqual(editedBefore)
  expect(await pixels(page, loaded.image_edit.source)).toEqual(originalBefore)
  expect(loaded.image_edit.layers.map((l: { adjust: unknown }) => l.adjust)).toEqual(saved.image_edit.layers.map((l: { adjust: unknown }) => l.adjust))

  // And the edit itself came along: the editor opens with both layers and
  // the selection, and applying it to the embedded original makes the
  // same picture again.
  await openEditor(page)
  await expect(page.locator('[data-testid^="editor-layer-"][data-selected]')).toHaveCount(2)
  await expect(byTestId(page, 'editor-layer-1').getByTestId('editor-layer-scope')).not.toHaveText('All')
  // Uploads are named by their content, so the edited picture made again
  // lands under the very same name: the same bytes, not just similar ones.
  const uploads: string[] = []
  page.on('response', async (response) => {
    if (response.url().endsWith('/api/images/upload') && response.ok()) uploads.push((await response.json()).filename)
  })
  await applyEdit(page)
  await expect.poll(() => uploads.length).toBeGreaterThan(0)
  expect(uploads[uploads.length - 1]).toBe(saved.input_image)
  expect(await pixels(page, (await settings(page)).input_image)).toEqual(editedBefore)
})
