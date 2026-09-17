import { test, expect } from '@playwright/test'
import { makeSolidPng } from './png-helper'
import { resetProjectState, cleanupTestFilaments } from './reset-state'

// Regression tests for two webui bugs found in the 2026-09-16 audit:
//
// 1. InputImagePanel silently treated a failed `/api/images/upload` as a
//    success — it always called `setInputImage()` with a local blob: URL
//    preview, even when the request errored or the server rejected it, and
//    never set `settings.input_image`. Since the Run button's enabled state
//    is driven purely by `!!inputImage`, this made Run look ready with no
//    real uploaded file behind it — clicking it either failed with
//    "Upload an input image before running optimization" (confusing, given
//    a preview was visibly showing) or silently reran whatever the
//    previous successful upload had been.
//
// 2. Persisted state (undo/redo snapshots via POST /api/state/snapshot, and
//    "Save Project" files) stored `inputImage` as the raw `blob:` object
//    URL created at upload time. That URL is only valid in the browser tab
//    that created it — after a reload, on another machine, or when the
//    saved project file is loaded elsewhere, the image reference is dead
//    even though the server still has the real uploaded file at
//    `/uploads/<filename>`.

// A real, decodable PNG — the hand-written bytes used before had a corrupt
// IDAT chunk, which the backend now (correctly) rejects on upload.
const TINY_PNG = makeSolidPng(1, 1, [255, 0, 0])

test.describe('Input image upload failure handling', () => {
  test.beforeEach(async ({ baseURL, request }) => {
    await resetProjectState(baseURL!)
    await request.post('/api/filaments', {
      data: { brand: 'BugfixTest', name: 'Upload Guard', color: '#123456', td: 5.0, filamentType: 'PLA' },
    })
  })

  test.afterEach(async ({ baseURL }) => {
    await cleanupTestFilaments(baseURL!)
  })

  test('a failed upload shows an error and does not enable Run', async ({ page }) => {
    await page.route('**/api/images/upload', (route) =>
      route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'disk full' }) }),
    )

    await page.goto('/')
    // Make the active-filament precondition for Run true, so the only thing
    // left gating it is the (about to fail) image upload.
    const created = await page.request.post('/api/filaments', {
      data: { brand: 'BugfixTest2', name: 'Guard2', color: '#654321', td: 4.0, filamentType: 'PLA' },
    })
    const filament = await created.json()
    await page.request.post('/api/filaments/active', { data: filament })
    await page.reload()

    await page.locator('[data-testid="image-file-input"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: TINY_PNG,
    })

    await expect(page.locator('[data-testid="image-upload-error"]')).toBeVisible()
    // No phantom preview from a local blob: URL with nothing uploaded behind it.
    await expect(page.locator('[data-testid="input-image"]')).toHaveCount(0)
    // Run must stay disabled — it wasn't gated only by a truthy `inputImage`.
    await expect(page.locator('[data-testid="top-start-btn"]')).toBeDisabled()
    await expect(page.locator('[data-testid="run-disabled-reason"]')).toHaveText(/upload an input image/i)
  })

  test('a successful upload clears any previous error and shows the preview', async ({ page }) => {
    await page.goto('/')
    await page.locator('[data-testid="image-file-input"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: TINY_PNG,
    })
    await expect(page.locator('[data-testid="input-image"]')).toBeVisible()
    await expect(page.locator('[data-testid="image-upload-error"]')).toHaveCount(0)
  })
})

test.describe('Persisted input image survives a browser-local blob: URL', () => {
  test.beforeEach(async ({ baseURL }) => {
    await resetProjectState(baseURL!)
  })

  test('undo/redo snapshots store the durable /uploads path, not a blob: URL', async ({ page, baseURL }) => {
    await page.goto('/')
    await page.locator('[data-testid="image-file-input"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: TINY_PNG,
    })
    await expect(page.locator('[data-testid="input-image"]')).toBeVisible()

    // Snapshot capture is debounced 500ms after the settings/image change.
    await page.waitForTimeout(1000)

    const history = await page.request.get('/api/state/history')
    expect(history.ok()).toBeTruthy()
    const snapshots = await history.json()
    expect(Array.isArray(snapshots)).toBeTruthy()
    expect(snapshots.length).toBeGreaterThan(0)

    const latest = [...snapshots].sort((a, b) => b.timestamp - a.timestamp)[0]
    expect(latest.inputImage).toBeTruthy()
    expect(latest.inputImage.startsWith('blob:')).toBeFalsy()
    expect(latest.inputImage).toMatch(/^\/uploads\//)
  })
})
