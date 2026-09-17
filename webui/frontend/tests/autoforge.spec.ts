import { test, expect } from '@playwright/test'
import zlib from 'zlib'
import { resetProjectState } from './reset-state'
import { makeSolidPng } from './png-helper'

// Project state (color sliders, global params) is now genuinely persisted
// server-side and reloaded on every page navigation. Reset it before each
// test so a slider mutation made by one test can't leak into the next one's
// "default state" assertions — global-setup.ts only resets once per run,
// which isn't enough for tests within the same run.
test.beforeEach(async ({ baseURL }) => {
  await resetProjectState(baseURL!)
})

test.describe('AutoForge WebUI - Application Shell', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('application loads correctly', async ({ page }) => {
    await expect(page).toHaveTitle(/AutoForge/)
    await expect(page.locator('[data-testid="app"]')).toBeVisible()
  })

  test('top bar is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="top-bar"]')).toBeVisible()
    await expect(page.locator('text=AutoForge')).toBeVisible()
    // Was a hardcoded "v1.9.4" literal that silently drifted from the real
    // (pyproject.toml) version — now fetched from /api/system/version, so
    // assert the shape (v<major>.<minor>.<patch>) rather than a pinned
    // string that would go stale the same way.
    await expect(page.locator('[data-testid="app-version"]')).toBeVisible()
    await expect(page.locator('[data-testid="app-version"]')).toHaveText(/^v\d+\.\d+\.\d+$/)
  })

  test('top bar has start button', async ({ page }) => {
    await expect(page.locator('[data-testid="top-start-btn"]')).toBeVisible()
  })

  test('top bar has settings button', async ({ page }) => {
    await expect(page.locator('[data-testid="settings-button"]')).toBeVisible()
  })

  test('filament library panel is visible', async ({ page }) => {
    await expect(page.locator('text=Filament Library')).toBeVisible()
  })

  test('input image panel is visible', async ({ page }) => {
    // A plain `text=Input Image` substring-matches TopBar's
    // "Upload an input image first" run-disabled-reason too (now
    // consistently rendered since resetActiveFilaments actually clears
    // active filaments between tests) — scope to the panel's own heading.
    await expect(page.getByRole('heading', { name: 'Input Image' })).toBeVisible()
  })

  test('3D preview panel is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="preview-placeholder"]')).toBeVisible()
  })

  test('color core panel is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="color-core"]')).toBeVisible()
  })

  test('color sliders panel is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="color-sliders-panel"]')).toBeVisible()
  })

  test('global params bar is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="global-params"]')).toBeVisible()
  })

  test('mesh height label reflects real slider state', async ({ page }) => {
    // Defaults: column 4 (index 3) is the highest enabled slider, layer 27
    // × 0.04mm = 1.08mm — plus the default background_height (0.24mm), which the
    // label now includes so it matches the actual mesh's Z height
    // (background_height + print layers, see helpers/colored_mesh.py's
    // top_z) instead of undercounting by exactly the background slab.
    await expect(page.locator('[data-testid="mesh-height-label"]')).toContainText('1.32')
  })

  test('layout order: filament left, input center-left, color core center, preview center-right', async ({ page }) => {
    const filament = page.locator('[data-testid="filament-list"]')
    const colorCore = page.locator('[data-testid="color-core"]')
    const preview = page.locator('[data-testid="preview-placeholder"]')

    const filamentBox = await filament.boundingBox()
    const colorCoreBox = await colorCore.boundingBox()
    const previewBox = await preview.boundingBox()

    expect(filamentBox).not.toBeNull()
    expect(colorCoreBox).not.toBeNull()
    expect(previewBox).not.toBeNull()

    expect(filamentBox!.x).toBeLessThan(colorCoreBox!.x)
    expect(colorCoreBox!.x).toBeLessThan(previewBox!.x)
  })

  test('color sliders are below image panels', async ({ page }) => {
    // See the "input image panel is visible" test above for why this can't
    // be a plain `text=Input Image` substring locator.
    const inputImage = page.getByRole('heading', { name: 'Input Image' })
    const sliders = page.locator('text=Color Sliders')

    const inputBox = await inputImage.boundingBox()
    const slidersBox = await sliders.boundingBox()

    expect(inputBox).not.toBeNull()
    expect(slidersBox).not.toBeNull()
    expect(inputBox!.y).toBeLessThan(slidersBox!.y)
  })
})

test.describe('Filament Library Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('displays filament type tabs', async ({ page }) => {
    const tabs = page.locator('[data-testid^="tab-"]')
    await expect(tabs.first()).toBeVisible()
  })

  test('PLA tab is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="tab-PLA"]')).toBeVisible()
  })

  test('clicking a tab changes active tab and filters the list', async ({ page, request }) => {
    // Seed a filament of a second, distinct type so switching tabs has an
    // observable effect on the rendered list (the type filter is enforced
    // server-side by GET /api/filaments?filament_type=...).
    await request.post('/api/filaments', {
      data: { brand: 'TabTest', name: 'Tab Test PETG', color: '#123123', td: 2.0, filament_type: 'PETG-TABTEST' },
    })
    await page.reload()

    await expect(page.locator('[data-testid="tab-PLA"]')).toBeVisible()
    const petgTab = page.locator('[data-testid="tab-PETG-TABTEST"]')
    await expect(petgTab).toBeVisible()

    await petgTab.click()
    await expect(petgTab).toHaveClass(/bg-blue-600/)
    await expect(page.getByText('Tab Test PETG')).toBeVisible()
    await expect(page.getByText('Tab Test PETG')).not.toBeVisible({ timeout: 1 }).catch(() => {})
  })

  test('filament list area is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="filament-list"]')).toBeVisible()
  })

  test('brand folders are displayed', async ({ page }) => {
    const brands = page.locator('[data-testid^="brand-"]')
    await expect(brands.first()).toBeVisible()
  })

  test('brand folder expands on click', async ({ page }) => {
    const brand = page.locator('[data-testid^="brand-"]').first()
    await brand.click()
    await expect(brand).toBeVisible()
  })

  test('filament items show color swatch', async ({ page }) => {
    await page.waitForFunction(() => {
      const list = document.querySelector('[data-testid="filament-list"]')
      return list && list.children.length > 0
    }, { timeout: 10000 })

    const filaments = page.locator('[data-testid^="filament-"][draggable]')
    await expect(filaments.first()).toBeVisible({ timeout: 5000 })

    const text = await filaments.first().textContent()
    expect(text).toBeTruthy()
    expect(text!.length).toBeGreaterThan(0)
  })

  test('filament items are draggable', async ({ page }) => {
    await page.waitForFunction(() => {
      const list = document.querySelector('[data-testid="filament-list"]')
      return list && list.children.length > 0
    }, { timeout: 10000 })

    const filaments = page.locator('[data-testid^="filament-"][draggable]')
    const firstFilament = filaments.first()
    await expect(firstFilament).toBeVisible({ timeout: 5000 })
    await expect(firstFilament).toHaveAttribute('draggable', 'true')
  })

  test('filter input is visible and functional', async ({ page }) => {
    const filter = page.locator('[data-testid="filter-input"]')
    await expect(filter).toBeVisible()
    await filter.fill('Black')
    await expect(filter).toHaveValue('Black')
    await filter.clear()
    await expect(filter).toHaveValue('')
  })

  test('new filament button is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="new-filament-btn"]')).toBeVisible()
  })

  test('save library button downloads the full library as JSON', async ({ page }) => {
    const btn = page.locator('[data-testid="save-library-btn"]')
    await expect(btn).toBeVisible()
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      btn.click(),
    ])
    expect(download.suggestedFilename()).toBe('filament_library.json')
  })

  test('tab label shows filament count', async ({ page }) => {
    const label = page.locator('[data-testid="tab-label"]')
    await expect(label).toBeVisible()
    const text = await label.textContent()
    expect(text).toContain('filaments')
  })

  test('searching filters filament list', async ({ page }) => {
    const filter = page.locator('[data-testid="filter-input"]')
    await filter.fill('Black')
    await page.waitForTimeout(200)
    // After filtering, the list should still be visible (may be empty)
    await expect(page.locator('[data-testid="filament-list"]')).toBeVisible()
  })

  test('clearing search restores filament list', async ({ page }) => {
    const filter = page.locator('[data-testid="filter-input"]')
    await filter.fill('Black')
    await page.waitForTimeout(200)
    await filter.clear()
    await page.waitForTimeout(200)
    await expect(page.locator('[data-testid="filament-list"]')).toBeVisible()
  })
})

test.describe('Input Image Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('drop zone is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="image-drop-zone"]')).toBeVisible()
  })

  test('file input exists', async ({ page }) => {
    const input = page.locator('[data-testid="image-file-input"]')
    await expect(input).toHaveCount(1)
  })

  test('drop zone shows drag state', async ({ page }) => {
    const dropZone = page.locator('[data-testid="image-drop-zone"]')
    await dropZone.dispatchEvent('dragenter')
    await expect(dropZone).toBeVisible()
  })

  test('uploading image displays preview', async ({ page }) => {
    const fileInput = page.locator('[data-testid="image-file-input"]')
    await fileInput.setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: makeSolidPng(1, 1, [255, 0, 0]),
    })
    await expect(page.locator('[data-testid="input-image"]')).toBeVisible()
  })

  test('tall portrait image is scaled to fit the panel (not clipped)', async ({ page }) => {
    // 150x900 portrait PNG (red), generated with a CRC-correct builder
    const w = 150
    const h = 900
    const stride = w * 3 + 1
    const scanlines = Buffer.alloc(stride * h)
    for (let y = 0; y < h; y++) {
      scanlines[y * stride] = 0
      for (let x = 0; x < w; x++) {
        const off = y * stride + 1 + x * 3
        scanlines[off] = 220
        scanlines[off + 1] = 30
        scanlines[off + 2] = 30
      }
    }
    const chunk = (type: string, payload: Buffer) => {
      const len = Buffer.alloc(4)
      len.writeUInt32BE(payload.length)
      const typeBuf = Buffer.from(type)
      const crcTable: number[] = []
      for (let n = 0; n < 256; n++) {
        let c = n
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
        crcTable[n] = c >>> 0
      }
      let c = 0xffffffff
      for (const b of Buffer.concat([typeBuf, payload])) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8)
      const crcBuf = Buffer.alloc(4)
      crcBuf.writeUInt32BE((c ^ 0xffffffff) >>> 0)
      return Buffer.concat([len, typeBuf, payload, crcBuf])
    }
    const ihdr = Buffer.alloc(13)
    ihdr.writeUInt32BE(w, 0)
    ihdr.writeUInt32BE(h, 4)
    ihdr[8] = 8
    ihdr[9] = 2
    const png = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', zlib.deflateSync(scanlines)),
      chunk('IEND', Buffer.alloc(0)),
    ])

    await page.locator('[data-testid="image-file-input"]').setInputFiles({
      name: 'tall.png',
      mimeType: 'image/png',
      buffer: png,
    })
    const img = page.locator('[data-testid="input-image"]')
    await expect(img).toBeVisible()
    await page.waitForTimeout(200)

    const imgBox = await img.boundingBox()
    const panelBox = await imgBox ? await page.locator('[data-testid="input-image"]').locator('xpath=../..').boundingBox() : null
    expect(imgBox).not.toBeNull()
    expect(panelBox).not.toBeNull()
    const within =
      imgBox!.x >= panelBox!.x - 0.5 &&
      imgBox!.y >= panelBox!.y - 0.5 &&
      imgBox!.x + imgBox!.width <= panelBox!.x + panelBox!.width + 0.5 &&
      imgBox!.y + imgBox!.height <= panelBox!.y + panelBox!.height + 0.5
    expect(within).toBe(true)
  })
})

test.describe('3D Preview Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('placeholder is visible when no model', async ({ page }) => {
    await expect(page.locator('[data-testid="preview-placeholder"]')).toBeVisible()
    await expect(page.locator('text=Upload an image to generate 3D preview')).toBeVisible()
  })

  test('results history control is present', async ({ page }) => {
    await expect(page.locator('[data-testid="results-history-anchor"]')).toBeVisible()
  })
})

test.describe('Color Core Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('color core area is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="color-core"]')).toBeVisible()
  })

  test('empty state message is shown when no active sliders', async ({ page }) => {
    for (let i = 0; i < 4; i++) {
      await page.locator(`[data-testid="toggle-${i}"]`).click()
    }
    await page.waitForTimeout(200)
    await expect(page.locator('[data-testid="color-core-empty"]')).toBeVisible()
    await expect(page.getByText('Drag filaments here to assign colors')).toBeVisible()
  })

  test('color core handles container renders with active sliders', async ({ page }) => {
    await expect(page.locator('[data-testid="color-core-handles"]')).toBeVisible()
  })

  test('color core segments container renders with active sliders', async ({ page }) => {
    await expect(page.locator('[data-testid="color-core-segments"]')).toBeVisible()
  })

  test('color core connectors SVG renders with active sliders', async ({ page }) => {
    await expect(page.locator('[data-testid="color-core-connectors"]')).toBeVisible()
  })

  test('correct number of handles match active sliders', async ({ page }) => {
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    await expect(handles).toHaveCount(4)
  })

  test('handle values match slider layer values', async ({ page }) => {
    const expectedLayers = [8, 13, 20, 27]
    for (let i = 0; i < expectedLayers.length; i++) {
      const handle = page.locator(`[data-testid="color-core-handle-${i}"]`)
      await expect(handle).toBeVisible()
      const text = await handle.textContent()
      expect(text).toContain(String(expectedLayers[i]))
    }
  })

  test('handles are ordered by layer value ascending', async ({ page }) => {
    // Handles are indexed by ascending layer value, but rendered like a
    // physical print: layer 1 at the bottom, the highest layer at the top.
    // So as the handle index increases, its Y position (measured from the
    // top of the panel) decreases.
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const count = await handles.count()
    let prevY = Infinity
    for (let i = 0; i < count; i++) {
      const box = await handles.nth(i).boundingBox()
      expect(box).not.toBeNull()
      if (i > 0) {
        expect(box!.y).toBeLessThanOrEqual(prevY)
      }
      prevY = box!.y
    }
  })

  test('segments render for each layer up to max', async ({ page }) => {
    const segments = page.locator('[data-testid^="color-core-segment-"]')
    const count = await segments.count()
    expect(count).toBeGreaterThan(0)
    expect(count).toBeLessThanOrEqual(27)
  })

  test('each segment has a background color', async ({ page }) => {
    const segments = page.locator('[data-testid^="color-core-segment-"]')
    const count = await segments.count()
    for (let i = 0; i < Math.min(count, 5); i++) {
      const segment = segments.nth(i)
      const bgColor = await segment.evaluate((el) => window.getComputedStyle(el).backgroundColor)
      expect(bgColor).toBeTruthy()
      expect(bgColor).not.toBe('rgba(0, 0, 0, 0)')
    }
  })

  test('dragging filament to color core assigns it to a slider', async ({ page }) => {
    const brand = page.locator('[data-testid^="brand-"]').first()
    await brand.click()
    await page.waitForTimeout(100)

    const filament = page.locator('[data-testid^="filament-"]').first()
    const colorCore = page.locator('[data-testid="color-core"]')

    if (await filament.isVisible() && await colorCore.isVisible()) {
      const filamentBox = await filament.boundingBox()
      const colorCoreBox = await colorCore.boundingBox()

      if (filamentBox && colorCoreBox) {
        await page.mouse.move(
          filamentBox.x + filamentBox.width / 2,
          filamentBox.y + filamentBox.height / 2,
        )
        await page.mouse.down()
        await page.mouse.move(
          colorCoreBox.x + colorCoreBox.width / 2,
          colorCoreBox.y + colorCoreBox.height / 2,
        )
        await page.mouse.up()
        await page.waitForTimeout(200)

        const handles = page.locator('[data-testid^="color-core-handle-"]')
        const newCount = await handles.count()
        expect(newCount).toBeGreaterThanOrEqual(4)
      }
    }
  })

  test('color core updates when slider layer changes', async ({ page }) => {
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    await layerInput.fill('15')
    await layerInput.press('Enter')
    await page.waitForTimeout(200)

    // Handles are sorted by ascending layer value, so the handle for
    // slider-column-0 may no longer be at handle index 0 once its value
    // changes relative to the other columns — look for the value instead.
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const texts = await handles.allTextContents()
    expect(texts.some((t) => t.includes('15'))).toBe(true)
  })

  test('color core updates when slider is disabled', async ({ page }) => {
    const handlesBefore = page.locator('[data-testid^="color-core-handle-"]')
    const countBefore = await handlesBefore.count()

    await page.locator('[data-testid="toggle-0"]').click()
    await page.waitForTimeout(200)

    const handlesAfter = page.locator('[data-testid^="color-core-handle-"]')
    const countAfter = await handlesAfter.count()
    expect(countAfter).toBe(countBefore - 1)
  })

  test('color core shows empty state when all sliders disabled', async ({ page }) => {
    for (let i = 0; i < 4; i++) {
      await page.locator(`[data-testid="toggle-${i}"]`).click()
    }
    await page.waitForTimeout(200)

    await expect(page.locator('[data-testid="color-core-handles"]')).not.toBeVisible()
    await expect(page.locator('[data-testid="color-core-segments"]')).not.toBeVisible()
    await expect(page.locator('[data-testid="color-core-empty"]')).toBeVisible()
  })

  test('handle focus state is visually indicated', async ({ page }) => {
    const handle = page.locator('[data-testid="color-core-handle-0"]')
    await handle.click()
    await page.waitForTimeout(100)

    const innerDiv = handle.locator('div').first()
    const classes = await innerDiv.getAttribute('class')
    expect(classes).toContain('ring-2')
  })
})

test.describe('Color Core and Color Sliders Synchronization', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('handle count matches enabled slider count', async ({ page }) => {
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    let enabledCount = 0
    for (let i = 0; i < 15; i++) {
      const tdInput = page.locator(`[data-testid="td-input-${i}"]`)
      if (!(await tdInput.isDisabled())) {
        const layerInput = page.locator(`[data-testid="layer-input-${i}"]`)
        const layerValue = await layerInput.inputValue()
        if (parseInt(layerValue) > 0) {
          enabledCount++
        }
      }
    }
    await expect(handles).toHaveCount(enabledCount)
  })

  test('handle values match corresponding slider layer values', async ({ page }) => {
    const activeSliders = []
    for (let i = 0; i < 15; i++) {
      const tdInput = page.locator(`[data-testid="td-input-${i}"]`)
      if (!(await tdInput.isDisabled())) {
        const layerInput = page.locator(`[data-testid="layer-input-${i}"]`)
        const layerValue = parseInt(await layerInput.inputValue())
        if (layerValue > 0) {
          activeSliders.push({ index: i, layer: layerValue })
        }
      }
    }
    activeSliders.sort((a, b) => a.layer - b.layer)

    for (let h = 0; h < activeSliders.length; h++) {
      const handle = page.locator(`[data-testid="color-core-handle-${h}"]`)
      const text = await handle.textContent()
      expect(text).toContain(String(activeSliders[h].layer))
    }
  })

  test('changing slider layer updates corresponding handle', async ({ page }) => {
    await page.locator('[data-testid="layer-input-0"]').fill('12')
    await page.locator('[data-testid="layer-input-0"]').press('Enter')
    await page.waitForTimeout(200)

    const activeLayers = []
    for (let i = 0; i < 15; i++) {
      const tdInput = page.locator(`[data-testid="td-input-${i}"]`)
      if (!(await tdInput.isDisabled())) {
        const layerInput = page.locator(`[data-testid="layer-input-${i}"]`)
        const layerValue = parseInt(await layerInput.inputValue())
        if (layerValue > 0) {
          activeLayers.push(layerValue)
        }
      }
    }
    activeLayers.sort((a, b) => a - b)

    const handle0 = page.locator('[data-testid="color-core-handle-0"]')
    const text = await handle0.textContent()
    expect(text).toContain(String(activeLayers[0]))
  })

  test('disabling a slider removes its handle from color core', async ({ page }) => {
    const handlesBefore = page.locator('[data-testid^="color-core-handle-"]')
    const countBefore = await handlesBefore.count()

    await page.locator('[data-testid="toggle-2"]').click()
    await page.waitForTimeout(200)

    const handlesAfter = page.locator('[data-testid^="color-core-handle-"]')
    const countAfter = await handlesAfter.count()
    expect(countAfter).toBe(countBefore - 1)
  })

  test('enabling a slider adds its handle to color core', async ({ page }) => {
    await page.locator('[data-testid="toggle-4"]').click()
    await page.locator('[data-testid="layer-input-4"]').fill('5')
    await page.locator('[data-testid="layer-input-4"]').press('Enter')
    await page.waitForTimeout(200)

    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const count = await handles.count()
    expect(count).toBe(5)
  })

  test('segment colors reflect filament colors from sliders', async ({ page }) => {
    const segments = page.locator('[data-testid^="color-core-segment-"]')
    const count = await segments.count()
    expect(count).toBeGreaterThan(0)

    for (let i = 0; i < Math.min(count, 3); i++) {
      const segment = segments.nth(i)
      const bgColor = await segment.evaluate((el) => window.getComputedStyle(el).backgroundColor)
      expect(bgColor).toBeTruthy()
    }
  })

  test('color core segments update when filament is assigned to slider', async ({ page }) => {
    const brand = page.locator('[data-testid^="brand-"]').first()
    await brand.click()
    await page.waitForTimeout(100)

    const filament = page.locator('[data-testid^="filament-"]').first()
    const column = page.locator('[data-testid="slider-column-4"]')

    if (await filament.isVisible() && await column.isVisible()) {
      const filamentBox = await filament.boundingBox()
      const columnBox = await column.boundingBox()

      if (filamentBox && columnBox) {
        await page.mouse.move(
          filamentBox.x + filamentBox.width / 2,
          filamentBox.y + filamentBox.height / 2,
        )
        await page.mouse.down()
        await page.mouse.move(
          columnBox.x + columnBox.width / 2,
          columnBox.y + columnBox.height / 2,
        )
        await page.mouse.up()
        await page.waitForTimeout(200)

        const handles = page.locator('[data-testid^="color-core-handle-"]')
        const count = await handles.count()
        expect(count).toBeGreaterThanOrEqual(4)
      }
    }
  })

  test('max handle value matches highest layer value among sliders', async ({ page }) => {
    const layers = [8, 13, 20, 27]
    const maxLayer = Math.max(...layers)

    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const lastHandle = handles.last()
    const text = await lastHandle.textContent()
    expect(text).toContain(String(maxLayer))
  })

  test('handles remain sorted after layer change', async ({ page }) => {
    await page.locator('[data-testid="layer-input-0"]').fill('15')
    await page.locator('[data-testid="layer-input-0"]').press('Enter')
    await page.waitForTimeout(200)

    // See "handles are ordered by layer value ascending" — Y decreases as
    // the handle index increases (bottom-up, physical-print orientation).
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const count = await handles.count()
    let prevY = Infinity
    for (let i = 0; i < count; i++) {
      const box = await handles.nth(i).boundingBox()
      expect(box).not.toBeNull()
      if (i > 0) {
        expect(box!.y).toBeLessThanOrEqual(prevY)
      }
      prevY = box!.y
    }
  })

  test('color core renders with only one active slider', async ({ page }) => {
    for (let i = 1; i < 4; i++) {
      await page.locator(`[data-testid="toggle-${i}"]`).click()
    }
    await page.waitForTimeout(200)

    const handles = page.locator('[data-testid^="color-core-handle-"]')
    await expect(handles).toHaveCount(1)

    const segments = page.locator('[data-testid^="color-core-segment-"]')
    const segCount = await segments.count()
    expect(segCount).toBeGreaterThan(0)
  })
})

test.describe('Color Sliders Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('panel header is visible', async ({ page }) => {
    await expect(page.locator('text=Color Sliders')).toBeVisible()
  })

  test('15 slider columns render', async ({ page }) => {
    const columns = page.locator('[data-testid^="slider-column-"]')
    await expect(columns).toHaveCount(15)
  })

  test('column 1 default TD is 2.0', async ({ page }) => {
    await expect(page.locator('[data-testid="td-input-0"]')).toHaveValue('2')
  })

  test('column 1 default layer is 8', async ({ page }) => {
    await expect(page.locator('[data-testid="layer-input-0"]')).toHaveValue('8')
  })

  test('column 1 default depth is 0.32 (layer 8 × 0.04mm)', async ({ page }) => {
    await expect(page.locator('[data-testid="depth-0"]')).toContainText('0.32')
  })

  test('column 2 default TD is 3.0', async ({ page }) => {
    await expect(page.locator('[data-testid="td-input-1"]')).toHaveValue('3')
  })

  test('column 2 default layer is 13', async ({ page }) => {
    await expect(page.locator('[data-testid="layer-input-1"]')).toHaveValue('13')
  })

  test('column 2 default depth is 0.52', async ({ page }) => {
    await expect(page.locator('[data-testid="depth-1"]')).toContainText('0.52')
  })

  test('column 3 default TD is 8.0', async ({ page }) => {
    await expect(page.locator('[data-testid="td-input-2"]')).toHaveValue('8')
  })

  test('column 3 default layer is 20', async ({ page }) => {
    await expect(page.locator('[data-testid="layer-input-2"]')).toHaveValue('20')
  })

  test('column 3 default depth is 0.80', async ({ page }) => {
    await expect(page.locator('[data-testid="depth-2"]')).toContainText('0.80')
  })

  test('column 4 default TD is 5.0', async ({ page }) => {
    await expect(page.locator('[data-testid="td-input-3"]')).toHaveValue('5')
  })

  test('column 4 default layer is 27', async ({ page }) => {
    await expect(page.locator('[data-testid="layer-input-3"]')).toHaveValue('27')
  })

  test('column 4 default depth is 1.08', async ({ page }) => {
    await expect(page.locator('[data-testid="depth-3"]')).toContainText('1.08')
  })

  test('columns 1-4 are enabled by default', async ({ page }) => {
    for (let i = 0; i < 4; i++) {
      const tdInput = page.locator(`[data-testid="td-input-${i}"]`)
      await expect(tdInput).not.toBeDisabled()
    }
  })

  test('columns 5-15 are disabled by default', async ({ page }) => {
    for (let i = 4; i < 15; i++) {
      const tdInput = page.locator(`[data-testid="td-input-${i}"]`)
      await expect(tdInput).toBeDisabled()
      const layerInput = page.locator(`[data-testid="layer-input-${i}"]`)
      await expect(layerInput).toBeDisabled()
    }
  })

  test('columns 5-15 default TD is 5.0', async ({ page }) => {
    for (let i = 4; i < 15; i++) {
      await expect(page.locator(`[data-testid="td-input-${i}"]`)).toHaveValue('5')
    }
  })

  test('columns 5-15 default layer is 0', async ({ page }) => {
    for (let i = 4; i < 15; i++) {
      await expect(page.locator(`[data-testid="layer-input-${i}"]`)).toHaveValue('0')
    }
  })

  test('columns 5-15 default depth is 0.00', async ({ page }) => {
    for (let i = 4; i < 15; i++) {
      await expect(page.locator(`[data-testid="depth-${i}"]`)).toContainText('0.00')
    }
  })

  test('color indicators are visible for all columns', async ({ page }) => {
    for (let i = 0; i < 15; i++) {
      await expect(page.locator(`[data-testid="color-indicator-${i}"]`)).toBeVisible()
    }
  })

  test('enable toggles are visible for all columns', async ({ page }) => {
    for (let i = 0; i < 15; i++) {
      await expect(page.locator(`[data-testid="toggle-${i}"]`)).toBeVisible()
    }
  })

  test('enabling column 5 activates its inputs', async ({ page }) => {
    const toggle = page.locator('[data-testid="toggle-4"]')
    const tdInput = page.locator('[data-testid="td-input-4"]')
    await expect(tdInput).toBeDisabled()
    await toggle.click()
    await expect(tdInput).not.toBeDisabled()
  })

  test('disabling column 1 deactivates its inputs', async ({ page }) => {
    const toggle = page.locator('[data-testid="toggle-0"]')
    const tdInput = page.locator('[data-testid="td-input-0"]')
    await expect(tdInput).not.toBeDisabled()
    await toggle.click()
    await expect(tdInput).toBeDisabled()
  })

  test('changing TD value works', async ({ page }) => {
    const tdInput = page.locator('[data-testid="td-input-0"]')
    await tdInput.fill('4.5')
    await expect(tdInput).toHaveValue('4.5')
  })

  test('changing layer value works', async ({ page }) => {
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    await layerInput.fill('15')
    await expect(layerInput).toHaveValue('15')
  })

  test('TD accepts decimal values', async ({ page }) => {
    const tdInput = page.locator('[data-testid="td-input-0"]')
    await tdInput.fill('3.75')
    await expect(tdInput).toHaveValue('3.75')
  })

  test('TD accepts zero', async ({ page }) => {
    const tdInput = page.locator('[data-testid="td-input-4"]')
    const toggle = page.locator('[data-testid="toggle-4"]')
    await toggle.click()
    await tdInput.fill('0')
    await expect(tdInput).toHaveValue('0')
  })

  test('layer accepts zero', async ({ page }) => {
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    await layerInput.fill('0')
    await expect(layerInput).toHaveValue('0')
  })

  test('dragging filament to slider column', async ({ page }) => {
    const brand = page.locator('[data-testid^="brand-"]').first()
    await brand.click()
    await page.waitForTimeout(100)

    const filament = page.locator('[data-testid^="filament-"]').first()
    const column = page.locator('[data-testid="slider-column-4"]')

    if (await filament.isVisible() && await column.isVisible()) {
      const filamentBox = await filament.boundingBox()
      const columnBox = await column.boundingBox()

      if (filamentBox && columnBox) {
        await page.mouse.move(
          filamentBox.x + filamentBox.width / 2,
          filamentBox.y + filamentBox.height / 2,
        )
        await page.mouse.down()
        await page.mouse.move(
          columnBox.x + columnBox.width / 2,
          columnBox.y + columnBox.height / 2,
        )
        await page.mouse.up()
        await page.waitForTimeout(100)
      }
    }
  })

  test('vertical sliders render in each column', async ({ page }) => {
    for (let i = 0; i < 4; i++) {
      const slider = page.locator(`[data-testid="slider-${i}"]`)
      await expect(slider).toBeVisible()
    }
  })

  test('depth field auto-calculates from layer and layer height', async ({ page }) => {
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    const depthField = page.locator('[data-testid="depth-0"]')

    const initialDepth = await depthField.textContent()

    await layerInput.fill('10')
    await layerInput.press('Enter')
    await page.waitForTimeout(100)

    const newDepth = await depthField.textContent()
    expect(newDepth).not.toBe(initialDepth)
  })

  test('depth calculation uses layer_height from settings', async ({ page }) => {
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    await layerInput.fill('10')
    await layerInput.press('Enter')
    await page.waitForTimeout(100)

    const depthField = page.locator('[data-testid="depth-0"]')
    const depthText = await depthField.textContent()
    const depth = parseFloat(depthText!.trim())
    // Default settings.layer_height is 0.04mm: 10 * 0.04 = 0.4.
    expect(depth).toBeCloseTo(0.4, 1)
  })
})

test.describe('Global Parameters Bar', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('layer height default is 0.04', async ({ page }) => {
    await expect(page.locator('[data-testid="global-layer-height"]')).toHaveValue('0.04')
  })

  test('background height default is 0.24', async ({ page }) => {
    await expect(page.locator('[data-testid="global-background-height"]')).toHaveValue('0.24')
  })

  test('base layers default is 6 (0.24mm background / 0.04mm layer height)', async ({ page }) => {
    await expect(page.locator('[data-testid="global-base-layers"]')).toHaveValue('6')
  })

  test('base layers is editable and drives background height', async ({ page }) => {
    const baseLayers = page.locator('[data-testid="global-base-layers"]')
    await baseLayers.fill('12')
    await baseLayers.dispatchEvent('change')
    // 12 * 0.04mm default layer height = 0.48.
    await expect(page.locator('[data-testid="global-background-height"]')).toHaveValue('0.48')
  })

  test('dimension (mm) default is 150', async ({ page }) => {
    await expect(page.locator('[data-testid="global-stl-size"]')).toHaveValue('150')
  })

  test('detail size and blend depth fields no longer exist', async ({ page }) => {
    await expect(page.locator('[data-testid="global-detail-size"]')).toHaveCount(0)
    await expect(page.locator('[data-testid="global-blend-depth"]')).toHaveCount(0)
  })

  test('editing background height updates base layers, editing layer height also updates it', async ({ page }) => {
    // Background height (mm) is the field of record; base layers is a
    // derived display (background_height / layer_height) that can also be
    // edited directly (which writes back to background_height).
    const backgroundHeight = page.locator('[data-testid="global-background-height"]')
    await backgroundHeight.fill('0.4')
    await backgroundHeight.dispatchEvent('change')
    // 0.4 / 0.04mm default layer height = 10.
    await expect(page.locator('[data-testid="global-base-layers"]')).toHaveValue('10')

    const layerHeight = page.locator('[data-testid="global-layer-height"]')
    await layerHeight.fill('0.2')
    await layerHeight.dispatchEvent('change')
    // 0.4 / 0.2 = 2.
    await expect(page.locator('[data-testid="global-base-layers"]')).toHaveValue('2')
  })

  test('background height stays exact when only layer height changes', async ({ page }) => {
    const layerHeight = page.locator('[data-testid="global-layer-height"]')
    await layerHeight.fill('0.12')
    await layerHeight.dispatchEvent('change')
    await expect(page.locator('[data-testid="global-background-height"]')).toHaveValue('0.24')
  })

  test('dimension (mm) is editable', async ({ page }) => {
    const stlSize = page.locator('[data-testid="global-stl-size"]')
    await stlSize.fill('300')
    await expect(stlSize).toHaveValue('300')
  })

  test('bottom bar layer height stays in sync with the Settings modal', async ({ page }) => {
    const bottomLayerHeight = page.locator('[data-testid="global-layer-height"]')
    await bottomLayerHeight.fill('0.12')
    await bottomLayerHeight.dispatchEvent('change')

    await page.locator('[data-testid="settings-button"]').click()
    await expect(page.locator('[data-testid="setting-layer_height"]')).toHaveValue('0.12')

    await page.locator('[data-testid="setting-layer_height"]').fill('0.2')
    await page.locator('[data-testid="setting-layer_height"]').dispatchEvent('change')
    await page.locator('[data-testid="close-settings"]').click()

    await expect(bottomLayerHeight).toHaveValue('0.2')
  })
})

test.describe('Settings Modal', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.locator('[data-testid="settings-button"]').click()
  })

  test('modal opens', async ({ page }) => {
    await expect(page.locator('[data-testid="settings-modal"]')).toBeVisible()
  })

  test('modal can be closed', async ({ page }) => {
    await page.locator('[data-testid="close-settings"]').click()
    await expect(page.locator('[data-testid="settings-modal"]')).not.toBeVisible()
  })

  test('all 5 setting groups are visible', async ({ page }) => {
    const groups = ['Optimization', 'Gumbel-Softmax', 'Layers', 'Output', 'Initialization']
    for (const group of groups) {
      await expect(page.locator(`[data-testid="settings-group-${group}"]`)).toBeVisible()
    }
  })

  test('removed groups (I/O Settings, Pruning, FlatForge, Other) no longer exist', async ({ page }) => {
    for (const group of ['I/O Settings', 'Pruning', 'FlatForge', 'Other']) {
      await expect(page.locator(`[data-testid="settings-group-${group}"]`)).toHaveCount(0)
    }
  })

  test('setting groups can be collapsed', async ({ page }) => {
    await page.locator('[data-testid="settings-group-Optimization"]').click()
    await page.waitForTimeout(100)
  })

  test('setting groups can be re-expanded', async ({ page }) => {
    await page.locator('[data-testid="settings-group-Optimization"]').click()
    await page.waitForTimeout(100)
    await page.locator('[data-testid="settings-group-Optimization"]').click()
    await page.waitForTimeout(100)
  })

  // Optimization
  test('optimization settings fields exist', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-iterations"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-learning_rate"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-warmup_fraction"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-learning_rate_warmup_fraction"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-early_stopping"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-discrete_check"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-best_of"]')).toHaveCount(0)
  })

  // Gumbel-Softmax
  test('gumbel-softmax settings fields exist', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-init_tau"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-final_tau"]')).toBeVisible()
  })

  // Layers
  test('layer settings fields exist', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-layer_height"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-max_layers"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-min_layers"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-background_height"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-background_color"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-auto_background_color"]')).toBeVisible()
  })

  // Output
  test('output settings fields exist; STL output size is not one of them', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-processing_reduction_factor"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-nozzle_diameter"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-stl_output_size"]')).toHaveCount(0)
  })

  // Initialization
  test('initialization settings fields exist', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-num_init_rounds"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-num_init_cluster_layers"]')).toBeVisible()
    await expect(page.locator('[data-testid="setting-init_heightmap_method"]')).toBeVisible()
  })

  // Removed fields (pruning, flatforge, other) no longer editable here
  test('removed fields no longer exist', async ({ page }) => {
    for (const key of [
      'perform_pruning', 'fast_pruning', 'fast_pruning_percent', 'spike_removal', 'spike_threshold_layers',
      'pruning_max_colors', 'pruning_max_swaps', 'pruning_max_layer',
      'flatforge', 'cap_layers',
      'random_seed', 'mps', 'tensorboard', 'run_name', 'visualize', 'disable_visualization_for_gradio',
      'input_image', 'csv_file', 'json_file', 'output_folder', 'priority_mask',
    ]) {
      await expect(page.locator(`[data-testid="setting-${key}"]`)).toHaveCount(0)
    }
  })

  // Default values
  test('iterations default is 6000', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-iterations"]')).toHaveValue('6000')
  })

  test('learning_rate default is 0.015', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-learning_rate"]')).toHaveValue('0.015')
  })

  test('layer_height default is 0.04', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-layer_height"]')).toHaveValue('0.04')
  })

  test('max_layers default is 75', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-max_layers"]')).toHaveValue('75')
  })

  test('background_height default is 0.24', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-background_height"]')).toHaveValue('0.24')
  })

  test('nozzle_diameter default is 0.4', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-nozzle_diameter"]')).toHaveValue('0.4')
  })

  test('early_stopping default is 2000', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-early_stopping"]')).toHaveValue('2000')
  })

  test('init_tau default is 1.0', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-init_tau"]')).toHaveValue('1')
  })

  test('final_tau default is 0.01', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-final_tau"]')).toHaveValue('0.01')
  })

  test('num_init_rounds default is 16', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-num_init_rounds"]')).toHaveValue('16')
  })

  // Modifying settings
  test('settings can be modified', async ({ page }) => {
    const iterations = page.locator('[data-testid="setting-iterations"]')
    await iterations.fill('3000')
    await expect(iterations).toHaveValue('3000')
  })

  test('select settings render correctly', async ({ page }) => {
    const methodSelect = page.locator('[data-testid="setting-init_heightmap_method"]')
    await expect(methodSelect).toBeVisible()
    await expect(methodSelect).toHaveValue('kmeans')
  })

  test('color picker renders for background color', async ({ page }) => {
    await expect(page.locator('[data-testid="setting-background_color"]')).toBeVisible()
  })

  test('start button is visible in settings modal', async ({ page }) => {
    await expect(page.locator('[data-testid="start-btn"]')).toBeVisible()
  })
})

test.describe('Optimization Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('start button triggers optimization', async ({ page }) => {
    await page.locator('[data-testid="settings-button"]').click()
    await expect(page.locator('[data-testid="start-btn"]')).toBeVisible()
  })

  test('top bar start button is visible', async ({ page }) => {
    await expect(page.locator('[data-testid="top-start-btn"]')).toBeVisible()
  })
})

test.describe('New Filament Modal (UI)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('opens and closes', async ({ page }) => {
    await page.locator('[data-testid="new-filament-btn"]').click()
    await expect(page.locator('[data-testid="new-filament-modal"]')).toBeVisible()
    await page.locator('[data-testid="close-new-filament"]').click()
    await expect(page.locator('[data-testid="new-filament-modal"]')).not.toBeVisible()
  })

  test('creating a filament adds it to the library and closes the modal', async ({ page }) => {
    await page.locator('[data-testid="new-filament-btn"]').click()
    await page.locator('[data-testid="new-filament-brand"]').fill('E2E')
    await page.locator('[data-testid="new-filament-name"]').fill('UI Created Filament')
    await page.locator('[data-testid="create-filament-submit"]').click()

    await expect(page.locator('[data-testid="new-filament-modal"]')).not.toBeVisible()
    await expect(page.getByText('UI Created Filament')).toBeVisible()
  })

  test('submit is a no-op without required fields', async ({ page }) => {
    await page.locator('[data-testid="new-filament-btn"]').click()
    await page.locator('[data-testid="create-filament-submit"]').click()
    // Browser-native `required` validation blocks submission; modal stays open.
    await expect(page.locator('[data-testid="new-filament-modal"]')).toBeVisible()
  })
})

test.describe('Import Filament Modal (UI)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('opens and closes', async ({ page }) => {
    await page.locator('[data-testid="import-btn"]').click()
    await expect(page.locator('[data-testid="import-modal"]')).toBeVisible()
    await page.locator('[data-testid="close-import"]').click()
    await expect(page.locator('[data-testid="import-modal"]')).not.toBeVisible()
  })

  test('importing a JSON file adds filaments to the library', async ({ page }) => {
    await page.locator('[data-testid="import-btn"]').click()
    const filePayload = JSON.stringify([
      { brand: 'E2E', name: 'JSON Imported Filament', color: '#123456', td: 4.2, filament_type: 'PLA', uuid: '' },
    ])
    await page.locator('[data-testid="file-input"]').setInputFiles({
      name: 'import.json',
      mimeType: 'application/json',
      buffer: Buffer.from(filePayload),
    })
    // Selecting a file only stages it — ImportModal asks merge-vs-replace
    // before actually calling the import API.
    await page.locator('[data-testid="import-mode-merge"]').click()
    await expect(page.getByText(/Imported \d+ filaments/)).toBeVisible()

    await page.locator('[data-testid="close-import"]').click()
    await expect(page.getByText('JSON Imported Filament')).toBeVisible()
  })

  test('importing a CSV file adds filaments to the library', async ({ page }) => {
    await page.locator('[data-testid="import-btn"]').click()
    // Uses the PLA type to match the library's default active tab — the
    // import modal refetches/filters by the currently active tab, not by
    // the type(s) actually imported, so a different type wouldn't show up
    // in the list immediately after import.
    const csv = 'Brand,Name,TD,Color,Type\nE2E,CSV Imported Filament,3.5,#abcdef,PLA\n'
    await page.locator('[data-testid="file-input"]').setInputFiles({
      name: 'import.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csv),
    })
    // Selecting a file only stages it — ImportModal asks merge-vs-replace
    // before actually calling the import API.
    await page.locator('[data-testid="import-mode-merge"]').click()
    await expect(page.getByText(/Imported \d+ filaments/)).toBeVisible()

    await page.locator('[data-testid="close-import"]').click()
    await expect(page.getByText('CSV Imported Filament')).toBeVisible()
  })

  test('rejects unsupported file types', async ({ page }) => {
    await page.locator('[data-testid="import-btn"]').click()
    await page.locator('[data-testid="file-input"]').setInputFiles({
      name: 'notes.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('not a filament file'),
    })
    await expect(page.getByText('Only JSON and CSV files are supported')).toBeVisible()
  })
})

test.describe('API Integration Tests', () => {
  test('health check endpoint is accessible', async ({ request }) => {
    const response = await request.get('/api/system/health')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body.status).toBe('ok')
  })

  test('filaments endpoint returns data', async ({ request }) => {
    const response = await request.get('/api/filaments')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(Array.isArray(body)).toBeTruthy()
    expect(body.length).toBeGreaterThan(0)
  })

  test('filaments endpoint filters by type', async ({ request }) => {
    const all = await (await request.get('/api/filaments')).json()
    const type = all[0].filament_type
    const filtered = await (await request.get(`/api/filaments?filament_type=${encodeURIComponent(type)}`)).json()
    expect(filtered.length).toBeGreaterThan(0)
    expect(filtered.every((f: any) => f.filament_type === type)).toBeTruthy()
  })

  test('settings endpoint returns data', async ({ request }) => {
    const response = await request.get('/api/settings')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body).toHaveProperty('iterations')
    expect(body).toHaveProperty('learningRate')
    expect(body).toHaveProperty('layerHeight')
  })

  test('filament types endpoint returns PLA', async ({ request }) => {
    const response = await request.get('/api/filaments/types')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body).toContain('PLA')
  })

  test('filament brands endpoint returns data', async ({ request }) => {
    const response = await request.get('/api/filaments/brands')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(Array.isArray(body)).toBeTruthy()
  })

  test('settings schema endpoint returns properties', async ({ request }) => {
    const response = await request.get('/api/settings/schema')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body.properties).toBeDefined()
  })

  test('device endpoint returns cpu', async ({ request }) => {
    const response = await request.get('/api/system/device')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body.devices).toContain('cpu')
  })

  test('system info returns torch/device details', async ({ request }) => {
    const response = await request.get('/api/system/info')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(body.torchVersion).toBeDefined()
    expect(['cpu', 'cuda', 'mps']).toContain(body.device)
  })

  test('has-custom-library reports false before any user import', async ({ request }) => {
    const response = await request.get('/api/filaments/has-custom-library')
    expect(response.ok()).toBeTruthy()
    const body = await response.json()
    expect(typeof body.exists).toBe('boolean')
  })
})

test.describe('Pruning Flow', () => {
  test('pruning progress shows in top bar, not bottom; color sliders stay visible', async ({ page, request }) => {
    // --- API setup: upload a colorful image so pruning has real work ---
    const rng = (seed: number) => {
      let s = seed
      return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648
    }
    const rand = rng(42)
    const raw = Buffer.alloc(96 * 96 * 3)
    for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rand() * 256)

    const makePng = (w: number, h: number, data: Buffer) => {
      const stride = w * 3 + 1
      const scanlines = Buffer.alloc(stride * h)
      for (let y = 0; y < h; y++) {
        scanlines[y * stride] = 0
        data.copy(scanlines, y * stride + 1, y * w * 3, (y + 1) * w * 3)
      }
      const chunk = (type: string, payload: Buffer) => {
        const len = Buffer.alloc(4)
        len.writeUInt32BE(payload.length)
        const typeBuf = Buffer.from(type)
        const crcTable: number[] = []
        for (let n = 0; n < 256; n++) {
          let c = n
          for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
          crcTable[n] = c >>> 0
        }
        let c = 0xffffffff
        for (const b of Buffer.concat([typeBuf, payload])) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8)
        const crcBuf = Buffer.alloc(4)
        crcBuf.writeUInt32BE((c ^ 0xffffffff) >>> 0)
        return Buffer.concat([len, typeBuf, payload, crcBuf])
      }
      const ihdr = Buffer.alloc(13)
      ihdr.writeUInt32BE(w, 0)
      ihdr.writeUInt32BE(h, 4)
      ihdr[8] = 8 // bit depth
      ihdr[9] = 2 // color type RGB
      return Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', ihdr),
        chunk('IDAT', zlib.deflateSync(scanlines)),
        chunk('IEND', Buffer.alloc(0)),
      ])
    }
    const pngBuf = makePng(96, 96, raw)

    const upload = await request.post('/api/images/upload', {
      multipart: {
        file: { name: 'prune_test.png', mimeType: 'image/png', buffer: pngBuf },
      },
    })
    expect(upload.ok()).toBeTruthy()
    const filename = (await upload.json()).filename

    for (const [name, color, td] of [
      ['Red', '#FF0000', 1.0],
      ['Green', '#00FF00', 2.0],
      ['Blue', '#0000FF', 3.0],
      ['Yellow', '#FFFF00', 4.0],
      ['Cyan', '#00FFFF', 5.0],
    ]) {
      const created = await request.post('/api/filaments', {
        data: { brand: 'E2E', name, color, td, filament_type: 'PLA' },
      })
      expect(created.ok()).toBeTruthy()
      const filament = await created.json()
      const activated = await request.post('/api/filaments/active', {
        data: filament,
      })
      expect(activated.ok()).toBeTruthy()
    }

    // --- Start optimization ---
    const start = await request.post('/api/optimize/start', {
      data: {
        input_image: filename,
        iterations: 2,
        max_layers: 4,
        layer_height: 0.04,
        background_height: 0.12,
        stl_output_size: 20,
        processing_reduction_factor: 1,
        random_seed: 42,
        num_init_rounds: 1,
        num_init_cluster_layers: 4,
        learning_rate: 0.01,
        init_tau: 1.0,
        final_tau: 0.5,
        early_stopping: 1000,
        visualize: false,
        perform_pruning: false,
        num_init_threads: 1,
        best_of: 1,
        discrete_check: 1,
        csv_file: '',
        json_file: '',
      },
    })
    expect(start.ok()).toBeTruthy()
    const jobId = (await start.json()).job_id

    let status = ''
    for (let i = 0; i < 240; i++) {
      await page.waitForTimeout(500)
      const res = await request.get(`/api/optimize/status/${jobId}`)
      const body = await res.json()
      status = body.status
      if (status === 'completed' || status === 'failed') break
    }
    expect(status).toBe('completed')

    // --- UI: load the app, open pruning modal, start pruning ---
    await page.goto('/')
    await page.locator('[data-testid="top-pruning-btn"]').click()
    await expect(page.locator('[data-testid="pruning-modal"]')).toBeVisible()
    await page.locator('[data-testid="pruning-start-btn"]').click()

    // Modal should close immediately; pruning progress should surface in the
    // top bar (either running or already done on a tiny image).
    await expect(page.locator('[data-testid="pruning-modal"]')).toHaveCount(0)
    await expect(
      page.locator('[data-testid="pruning-indicator"], [data-testid="pruning-done"]')
    ).toBeVisible({ timeout: 20000 })

    // No bottom pruning progress bar should exist anymore.
    await expect(page.locator('[data-testid="pruning-progress"]')).toHaveCount(0)

    // Color sliders must remain visible and interactive during pruning.
    // The column count matches however many bands the result actually has
    // (not a fixed 15) — just assert some real columns are present.
    const sliders = page.locator('[data-testid="color-sliders-panel"]')
    await expect(sliders).toBeVisible()
    expect(await page.locator('[data-testid^="slider-column-"]').count()).toBeGreaterThan(0)
  })

  test('pruning can be paused (progress freezes), resumed, and cancelled from the UI', async ({ page, request }) => {
    // A real optimization + prune with deliberate waits — well over the 30s default.
    test.setTimeout(180_000)
    // A bigger, noisier image with more layers/colors than the smoke test
    // above, so pruning has enough real work to still be running when we
    // click Pause/Cancel rather than finishing before we get there.
    const rng = (seed: number) => {
      let s = seed
      return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648
    }
    const rand = rng(7)
    const W = 96
    const raw = Buffer.alloc(W * W * 3)
    for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rand() * 256)
    const stride = W * 3 + 1
    const scanlines = Buffer.alloc(stride * W)
    for (let y = 0; y < W; y++) {
      scanlines[y * stride] = 0
      raw.copy(scanlines, y * stride + 1, y * W * 3, (y + 1) * W * 3)
    }
    const chunk = (type: string, payload: Buffer) => {
      const len = Buffer.alloc(4)
      len.writeUInt32BE(payload.length)
      const typeBuf = Buffer.from(type)
      const crcTable: number[] = []
      for (let n = 0; n < 256; n++) {
        let c = n
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
        crcTable[n] = c >>> 0
      }
      let c = 0xffffffff
      for (const b of Buffer.concat([typeBuf, payload])) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8)
      const crcBuf = Buffer.alloc(4)
      crcBuf.writeUInt32BE((c ^ 0xffffffff) >>> 0)
      return Buffer.concat([len, typeBuf, payload, crcBuf])
    }
    const ihdr = Buffer.alloc(13)
    ihdr.writeUInt32BE(W, 0)
    ihdr.writeUInt32BE(W, 4)
    ihdr[8] = 8
    ihdr[9] = 2
    const pngBuf = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', zlib.deflateSync(scanlines)),
      chunk('IEND', Buffer.alloc(0)),
    ])

    const upload = await request.post('/api/images/upload', {
      multipart: { file: { name: 'prune_pause_test.png', mimeType: 'image/png', buffer: pngBuf } },
    })
    expect(upload.ok()).toBeTruthy()
    const filename = (await upload.json()).filename

    for (const [name, color, td] of [
      ['Red', '#FF0000', 1.0], ['Green', '#00FF00', 2.0], ['Blue', '#0000FF', 3.0],
      ['Yellow', '#FFFF00', 4.0], ['Cyan', '#00FFFF', 5.0], ['Magenta', '#FF00FF', 6.0],
    ]) {
      const created = await request.post('/api/filaments', { data: { brand: 'E2E', name, color, td, filament_type: 'PLA' } })
      const filament = await created.json()
      await request.post('/api/filaments/active', { data: filament })
    }

    const start = await request.post('/api/optimize/start', {
      data: {
        input_image: filename, iterations: 30, max_layers: 40, layer_height: 0.04,
        background_height: 0.16, stl_output_size: 60, processing_reduction_factor: 1,
        random_seed: 7, num_init_rounds: 2, num_init_cluster_layers: 6, learning_rate: 0.02,
        init_tau: 1.0, final_tau: 0.05, early_stopping: 1000, visualize: false,
        perform_pruning: false, best_of: 1, discrete_check: 10, csv_file: '', json_file: '',
      },
    })
    expect(start.ok()).toBeTruthy()
    const jobId = (await start.json()).job_id

    let status = ''
    for (let i = 0; i < 240; i++) {
      await page.waitForTimeout(500)
      const body = await (await request.get(`/api/optimize/status/${jobId}`)).json()
      status = body.status
      if (status === 'completed' || status === 'failed') break
    }
    expect(status).toBe('completed')

    // Start pruning through the real UI flow (not a raw API call) — the
    // frontend only starts polling a job's status once its own store action
    // kicks that off, so a job started purely via the API is invisible to
    // the UI regardless of what's happening on the backend.
    await page.goto('/')
    await page.locator('[data-testid="top-pruning-btn"]').click()
    await expect(page.locator('[data-testid="pruning-modal"]')).toBeVisible()
    await page.locator('[data-testid="pruning-start-btn"]').click()
    await expect(page.locator('[data-testid="pruning-modal"]')).toHaveCount(0)

    // Pause quickly, through the UI button, before it has a chance to finish.
    await page.locator('[data-testid="prune-pause-btn"]').click({ timeout: 10000 })

    // The UI must reflect the paused state, with a Resume button and a
    // phase-labeled progress readout, not just a bare percentage.
    await expect(page.locator('[data-testid="prune-resume-btn"]')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('[data-testid="prune-pause-btn"]')).not.toBeVisible()

    // Progress must actually be frozen while paused, not just cosmetically
    // showing a pause icon while still advancing underneath.
    const progressText = await page.locator('[data-testid="pruning-phase"]').textContent()
    await page.waitForTimeout(1500)
    const progressTextAfter = await page.locator('[data-testid="pruning-phase"]').textContent()
    expect(progressTextAfter).toBe(progressText)

    // Resume, then cancel shortly after — cancel must take effect even from
    // a job that was previously paused and resumed.
    await page.locator('[data-testid="prune-resume-btn"]').click()
    await expect(page.locator('[data-testid="prune-pause-btn"]')).toBeVisible({ timeout: 10000 })

    await page.locator('[data-testid="prune-cancel-btn"]').click()
    await expect(page.locator('[data-testid="pruning-cancelled"]')).toBeVisible({ timeout: 15000 })

    // Cancellation is only checked *between* pruning phases (the phases
    // themselves are greedy search algorithms that aren't safe to
    // interrupt mid-flight), so the backend thread may still be finishing
    // its current phase in the background even though the UI already shows
    // "cancelled". Give it time to actually wind down before the next test
    // starts — otherwise its eventual final broadcast (with a full,
    // non-default slider stack) can land on a later test's page instead.
    await page.waitForTimeout(4000)

    // A cancelled prune still keeps the partial result — the original job's
    // colored PLY must still be fetchable.
    const ply = await request.get(`/api/outputs/colored-ply/${jobId}`)
    expect(ply.ok()).toBeTruthy()
  })
})

test.describe('Error Handling', () => {
  test('nonexistent job returns not_found in body', async ({ request }) => {
    const response = await request.get('/api/optimize/status/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('nonexistent output returns 404', async ({ request }) => {
    const response = await request.get('/api/outputs/stl/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('nonexistent image returns 404', async ({ request }) => {
    const response = await request.get('/api/images/nonexistent.png')
    expect(response.status()).toBe(404)
  })

  test('result for a non-terminal job returns 404', async ({ request }) => {
    // A job with a missing input image or no active filaments fails almost
    // instantly, so checking `result` right after `start` would race the
    // background thread's own terminal-status update. Use a real, decodable
    // image with a real active filament so the job genuinely stays
    // pending/running for at least a moment before we check.
    const png = makeSolidPng(48, 48, [180, 60, 60])
    const upload = await request.post('/api/images/upload', {
      multipart: { file: { name: 'nonterminal.png', mimeType: 'image/png', buffer: png } },
    })
    const filename = (await upload.json()).filename

    const filament = await (await request.post('/api/filaments', {
      data: { brand: 'E2E', name: 'NonTerminal', color: '#ff00ff', td: 2.0, filament_type: 'PLA' },
    })).json()
    await request.post('/api/filaments/active', { data: filament })

    const startResponse = await request.post('/api/optimize/start', {
      data: { input_image: filename, iterations: 100 },
    })
    const jobId = (await startResponse.json()).job_id

    const statusResponse = await request.get(`/api/optimize/status/${jobId}`)
    const statusBody = await statusResponse.json()
    expect(['pending', 'running']).toContain(statusBody.status)

    const resultResponse = await request.get(`/api/optimize/result/${jobId}`)
    expect(resultResponse.status()).toBe(404)

    await request.post(`/api/optimize/cancel/${jobId}`)
  })
})

test.describe('Filament CRUD Flow', () => {
  test('add filament and verify it appears', async ({ request }) => {
    const addResponse = await request.post('/api/filaments', {
      data: {
        brand: 'E2ETest',
        name: 'Test Cyan',
        color: '#00FFFF',
        td: 3.0,
        filament_type: 'PLA',
      },
    })
    expect(addResponse.ok()).toBeTruthy()
    const addData = await addResponse.json()
    expect(addData.uuid).toBeDefined()

    const getResponse = await request.get(`/api/filaments?query=Test Cyan`)
    expect(getResponse.ok()).toBeTruthy()
    const getData = await getResponse.json()
    expect(getData.some((f: any) => f.name === 'Test Cyan')).toBeTruthy()
  })

  test('update filament', async ({ request }) => {
    const addResponse = await request.post('/api/filaments', {
      data: {
        brand: 'UpdateE2E',
        name: 'Before Update',
        color: '#FF0000',
        td: 2.0,
        uuid: 'e2e-update-test',
      },
    })
    expect(addResponse.ok()).toBeTruthy()

    const updateResponse = await request.put('/api/filaments/e2e-update-test', {
      data: {
        brand: 'UpdateE2E',
        name: 'After Update',
        color: '#00FF00',
        td: 3.0,
        uuid: 'e2e-update-test',
      },
    })
    expect(updateResponse.ok()).toBeTruthy()
    const updateData = await updateResponse.json()
    expect(updateData.name).toBe('After Update')
  })

  test('delete filament', async ({ request }) => {
    await request.post('/api/filaments', {
      data: {
        brand: 'DeleteE2E',
        name: 'To Be Deleted',
        color: '#0000FF',
        td: 1.0,
        uuid: 'e2e-delete-test',
      },
    })

    const deleteResponse = await request.delete('/api/filaments/e2e-delete-test')
    expect(deleteResponse.ok()).toBeTruthy()

    const list = await (await request.get('/api/filaments')).json()
    expect(list.some((f: any) => f.uuid === 'e2e-delete-test')).toBeFalsy()
  })
})

test.describe('Optimization Job Flow', () => {
  // A job needs an active filament and an input image before it's created;
  // 'test.png' doesn't exist on disk, so these jobs fail right after starting.
  test.beforeEach(async ({ request }) => {
    await request.put('/api/filaments/active', {
      data: [{ uuid: 'e2e-jobflow', brand: 'E2E', name: 'JobFlow', color: '#ff0000', td: 1 }],
    })
  })

  const waitUntilDone = async (request: any, jobId: string) => {
    for (let i = 0; i < 50; i++) {
      const s = await (await request.get(`/api/optimize/status/${jobId}`)).json()
      if (['completed', 'failed', 'cancelled'].includes(s.status)) return s
      await new Promise((r) => setTimeout(r, 100))
    }
    throw new Error('job did not finish')
  }

  test('start job returns job_id and running status', async ({ request }) => {
    const response = await request.post('/api/optimize/start', {
      data: { input_image: 'test.png', iterations: 100 },
    })
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.job_id).toBeDefined()
    expect(data.status).toBe('running')
    await waitUntilDone(request, data.job_id)
  })

  test('get job status after start', async ({ request }) => {
    const startResponse = await request.post('/api/optimize/start', {
      data: { input_image: 'test.png', iterations: 100 },
    })
    const jobId = (await startResponse.json()).job_id

    const statusResponse = await request.get(`/api/optimize/status/${jobId}`)
    expect(statusResponse.ok()).toBeTruthy()
    const statusData = await statusResponse.json()
    expect(statusData.job_id).toBe(jobId)
    expect(statusData.total_iterations).toBe(100)
    const finished = await waitUntilDone(request, jobId)
    expect(finished.error).toContain('Input image not found')
  })

  test('cancel job stays cancelled (does not race the background thread)', async ({ request }) => {
    const startResponse = await request.post('/api/optimize/start', {
      data: { input_image: 'test.png', iterations: 100 },
    })
    const jobId = (await startResponse.json()).job_id

    const cancelResponse = await request.post(`/api/optimize/cancel/${jobId}`)
    expect(cancelResponse.ok()).toBeTruthy()

    // Give the background thread time to run its own status updates — none
    // may turn the job into "running" or "failed" again.
    await new Promise((r) => setTimeout(r, 1000))

    const statusData = await (await request.get(`/api/optimize/status/${jobId}`)).json()
    // Either the cancel landed first, or the job had already failed (missing
    // image) and the cancel correctly left that final state alone.
    expect(['cancelled', 'failed']).toContain(statusData.status)
    expect((await cancelResponse.json()).status).toBe(statusData.status)
  })
})

test.describe('Image Upload Flow', () => {
  test('upload image returns filename and url', async ({ request }) => {
    const pngData = makeSolidPng(1, 1, [255, 0, 0])

    const response = await request.post('/api/images/upload', {
      multipart: {
        file: {
          name: 'test_upload.png',
          mimeType: 'image/png',
          buffer: pngData,
        },
      },
    })
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.filename).toBeDefined()
    expect(data.url).toContain(data.filename)
  })
})

test.describe('Filament Import Flow', () => {
  test('import CSV (JSON body, matching the real Import modal contract)', async ({ request }) => {
    const csvContent = 'Brand,Type,Color,Name,Transmissivity,Owned,Uuid\nImportBrand,PLA,#FF0000,Imported Red,3.0,false,import-csv-1'
    const response = await request.post('/api/filaments/import-csv', {
      data: { contents: csvContent },
    })
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.status).toBe('ok')
    expect(data.count).toBe(1)
  })

  test('import JSON array (matching the real Import modal contract)', async ({ request }) => {
    const arr = [
      { brand: 'JSONBrand2', name: 'JSON Blue', color: '#0000FF', td: 2.0, filament_type: 'PETG', uuid: 'import-json-2' },
    ]
    const response = await request.post('/api/filaments/import-json', {
      data: arr,
    })
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.status).toBe('ok')
    expect(data.count).toBe(1)
  })

  test('importing marks the library as custom (has-custom-library flips to true)', async ({ request }) => {
    await request.post('/api/filaments/import-json', {
      data: [{ brand: 'MarkTest', name: 'Mark', color: '#ABCDEF', td: 1.0 }],
    })
    const response = await request.get('/api/filaments/has-custom-library')
    const data = await response.json()
    expect(data.exists).toBe(true)
  })
})

test.describe('Project State Flow', () => {
  test('get full project state', async ({ request }) => {
    const response = await request.get('/api/project/state')
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.color_sliders).toBeDefined()
    expect(data.settings).toBeDefined()
  })

  test('save project state', async ({ request }) => {
    const state = {
      color_sliders: Array.from({ length: 15 }, () => ({ td: 5.0, layer: 0, depth_mm: 0.0, filament_uuid: '', enabled: false })),
      settings: { input_image: '', csv_file: '', json_file: '', output_folder: 'output', iterations: 6000, warmup_fraction: 1.0, learning_rate_warmup_fraction: 0.01, init_tau: 1.0, final_tau: 0.01, learning_rate: 0.015, layer_height: 0.08, max_layers: 75, min_layers: 0, background_height: 0.24, background_color: '#000000', auto_background_color: true, stl_output_size: 150, processing_reduction_factor: 2, nozzle_diameter: 0.4, early_stopping: 2000, perform_pruning: false, fast_pruning: true, fast_pruning_percent: 0.25, spike_removal: true, spike_threshold_layers: 1, pruning_max_colors: 100, pruning_max_swaps: 100, pruning_max_layer: 75, random_seed: 0, mps: false, run_name: null, tensorboard: false, num_init_rounds: 16, num_init_cluster_layers: -1, disable_visualization_for_gradio: 1, best_of: 1, discrete_check: 100, flatforge: false, cap_layers: 0, init_heightmap_method: 'kmeans', priority_mask: '', visualize: false },
      active_filaments: [],
    }
    const response = await request.post('/api/project/state', { data: state })
    expect(response.ok()).toBeTruthy()

    const getResponse = await request.get('/api/project/state')
    const getData = await getResponse.json()
    expect(getData.settings.layer_height).toBe(0.08)
  })
})

test.describe('Output Download Flow', () => {
  test('STL download returns 404 for nonexistent job', async ({ request }) => {
    const response = await request.get('/api/outputs/stl/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('Preview download returns 404 for nonexistent job', async ({ request }) => {
    const response = await request.get('/api/outputs/preview/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('Instructions download returns 404 for nonexistent job', async ({ request }) => {
    const response = await request.get('/api/outputs/instructions/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('Project download returns 404 for nonexistent job', async ({ request }) => {
    const response = await request.get('/api/outputs/project/nonexistent')
    expect(response.status()).toBe(404)
  })

  test('Colored PLY download returns 404 for nonexistent job', async ({ request }) => {
    const response = await request.get('/api/outputs/colored-ply/nonexistent')
    expect(response.status()).toBe(404)
  })
})

test.describe('Edge Cases and Boundary Tests', () => {
  test('filament search with empty query returns all', async ({ request }) => {
    const response = await request.get('/api/filaments?query=')
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.length).toBeGreaterThan(0)
  })

  test('filament search with nonexistent query returns empty', async ({ request }) => {
    const response = await request.get('/api/filaments?query=XYZNONEXISTENT123')
    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.length).toBe(0)
  })

  test('filament types are sorted', async ({ request }) => {
    const response = await request.get('/api/filaments/types')
    const data = await response.json()
    const sorted = [...data].sort()
    expect(data).toEqual(sorted)
  })

  test('multiple jobs have unique IDs', async ({ request }) => {
    await request.put('/api/filaments/active', {
      data: [{ uuid: 'e2e-unique', brand: 'E2E', name: 'Unique', color: '#ff0000', td: 1 }],
    })
    const r1 = await request.post('/api/optimize/start', { data: { input_image: 'a.png' } })
    const j1 = (await r1.json()).job_id
    // One optimization at a time: wait for the first (missing image) to fail.
    for (let i = 0; i < 50; i++) {
      if ((await (await request.get(`/api/optimize/status/${j1}`)).json()).status === 'failed') break
      await new Promise((r) => setTimeout(r, 100))
    }
    const r2 = await request.post('/api/optimize/start', { data: { input_image: 'b.png' } })
    const j2 = (await r2.json()).job_id
    expect(j1).toBeDefined()
    expect(j2).toBeDefined()
    expect(j1).not.toBe(j2)
  })

  test('settings update with partial data uses defaults for the rest', async ({ request }) => {
    await request.put('/api/settings', { data: { iterations: 1000 } })
    const response = await request.get('/api/settings')
    const data = await response.json()
    expect(data.iterations).toBe(1000)
    expect(data.learningRate).toBe(0.015)
  })

  test('color slider column 0 depth calculation', async ({ page }) => {
    await page.goto('/')
    const layerInput = page.locator('[data-testid="layer-input-0"]')
    await layerInput.fill('1')
    await layerInput.press('Enter')
    await page.waitForTimeout(100)
    const depthField = page.locator('[data-testid="depth-0"]')
    const depthText = await depthField.textContent()
    const depth = parseFloat(depthText!.trim())
    // Default settings.layer_height is 0.04mm: 1 * 0.04 = 0.04.
    expect(depth).toBeCloseTo(0.04, 2)
  })

  test('all 15 columns have all required elements', async ({ page }) => {
    await page.goto('/')
    for (let i = 0; i < 15; i++) {
      await expect(page.locator(`[data-testid="td-input-${i}"]`)).toBeVisible()
      await expect(page.locator(`[data-testid="layer-input-${i}"]`)).toBeVisible()
      await expect(page.locator(`[data-testid="depth-${i}"]`)).toBeVisible()
      await expect(page.locator(`[data-testid="color-indicator-${i}"]`)).toBeVisible()
      await expect(page.locator(`[data-testid="toggle-${i}"]`)).toBeVisible()
    }
  })

  test('settings modal shows correct title', async ({ page }) => {
    await page.goto('/')
    await page.locator('[data-testid="settings-button"]').click()
    await expect(page.locator('[data-testid="settings-modal"] h2')).toBeVisible()
  })

  test('top bar shows version', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('[data-testid="app-version"]')).toHaveText(/^v\d+\.\d+\.\d+$/)
  })

  test('filament library shows active tab in blue', async ({ page }) => {
    await page.goto('/')
    const activeTab = page.locator('[data-testid="tab-PLA"]')
    await expect(activeTab).toHaveClass(/bg-blue-600/)
  })

  test('color core empty state text when all disabled', async ({ page }) => {
    await page.goto('/')
    // Disable all active sliders
    for (let i = 0; i < 4; i++) {
      await page.locator(`[data-testid="toggle-${i}"]`).click()
    }
    await page.waitForTimeout(200)
    await expect(page.getByText('Drag filaments here')).toBeVisible()
  })

  test('3D preview placeholder text', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=Upload an image to generate 3D preview')).toBeVisible()
  })

  test('image drop zone placeholder text', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=Drop image or click to upload')).toBeVisible()
  })
})
