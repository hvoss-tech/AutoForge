import { test, expect } from '@playwright/test'
import {
  activePalette,
  byTestId,
  createFilament,
  historyLabels,
  openApp,
  presetSettings,
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

test.describe('Input image and auto-preview', () => {
  test('uploading with no active filaments shows the image and asks for filaments', async ({ page }) => {
    await openApp(page)
    await expect(byTestId(page, 'preview-placeholder')).toBeVisible()
    await expect(byTestId(page, 'run-disabled-reason')).toHaveText('Upload an input image first')

    await uploadImage(page)
    await expect(byTestId(page, 'no-filaments-warning')).toBeVisible()
    await expect(byTestId(page, 'run-disabled-reason')).toHaveText('Add at least one active filament first')
    await expect(byTestId(page, 'top-start-btn')).toBeDisabled()
  })

  test('image + filament builds the 3D auto-preview and enables Run', async ({ page, request }) => {
    await activePalette(request)
    await presetSettings(request, { max_layers: 40 })
    await openApp(page)
    await uploadAndWaitForPreview(page)
    await expect(byTestId(page, 'top-start-btn')).toBeEnabled()
    await expect(byTestId(page, 'run-disabled-reason')).toHaveCount(0)
    // The slider track range follows the real heightmap, not the 0-75 placeholder.
    const range = await (await request.get('/api/init/status')).json()
    expect(range.max_layer).toBeLessThanOrEqual(40)
    await expect(byTestId(page, 'slider-0')).toHaveAttribute('max', String(range.max_layer))
  })

  test('the upload is recorded as an "Input image changed" undo step', async ({ page, request }) => {
    await activePalette(request)
    await openApp(page)
    await uploadImage(page)
    await waitForSnapshot(page)
    expect(await historyLabels(page)).toContain('▶Input image changed')
  })

  test('Change swaps the image', async ({ page }) => {
    await openApp(page)
    await uploadImage(page)
    const first = await byTestId(page, 'input-image').getAttribute('src')
    await byTestId(page, 'image-file-input').setInputFiles({
      name: 'other.png',
      mimeType: 'image/png',
      buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==', 'base64'),
    })
    await expect(byTestId(page, 'input-image')).not.toHaveAttribute('src', first!)
  })

  test('a corrupt image file is rejected with an explanation', async ({ page, request }) => {
    await activePalette(request)
    await openApp(page)
    await byTestId(page, 'image-file-input').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('definitely not a png') })
    await expect(byTestId(page, 'image-upload-error')).toContainText("isn't an image")
    await expect(byTestId(page, 'input-image')).toHaveCount(0)
    await expect(byTestId(page, 'top-start-btn')).toBeDisabled()
  })

  test('a server-side upload failure is reported and Run stays disabled', async ({ page, request }) => {
    await activePalette(request)
    await page.route('**/api/images/upload', (route) => route.fulfill({ status: 500, body: 'boom' }))
    await openApp(page)
    await byTestId(page, 'image-file-input').setInputFiles(await import('../helpers').then((h) => h.FIXTURE_IMAGE))
    await expect(byTestId(page, 'image-upload-error')).toContainText('HTTP 500')
    await expect(byTestId(page, 'top-start-btn')).toBeDisabled()
  })
})

test.describe('Job feedback', () => {
  const jobMessage = (status: string, extra: Record<string, unknown> = {}) =>
    JSON.stringify({ job_id: 'e2e-mock-job', status, progress: 40, iteration: 4, total_iterations: 10, loss: 0.1234, error: null, started_at: null, completed_at: null, preview_image: null, phase: null, ...extra })

  test('a failed run shows a toast with the summary and the error panel with details', async ({ page, request }) => {
    await activePalette(request)
    await page.route('**/api/optimize/start', (route) => route.fulfill({ json: { job_id: 'e2e-mock-job', total_iterations: 10, status: 'running' } }))
    await page.routeWebSocket(/\/ws\/optimize\//, (ws) => {
      ws.send(jobMessage('running'))
      setTimeout(() => ws.send(jobMessage('failed', { error: 'The GPU ran out of memory.\n\nCUDA out of memory. Tried to allocate 2 GiB' })), 400)
    })
    await openApp(page)
    await uploadImage(page)
    await byTestId(page, 'top-start-btn').click()

    await expect(byTestId(page, 'toast-error')).toHaveText('Optimization failed: The GPU ran out of memory.')
    await expect(byTestId(page, 'optimization-error')).toContainText('The GPU ran out of memory.')
    await expect(byTestId(page, 'optimization-error').getByText('CUDA out of memory')).toBeHidden()
    await byTestId(page, 'optimization-error').getByText('Show details').click()
    await expect(byTestId(page, 'optimization-error').getByText(/CUDA out of memory/)).toBeVisible()
    // Dismissable, and Run is available again.
    await byTestId(page, 'toast-error').getByRole('button', { name: 'Dismiss' }).click()
    await expect(byTestId(page, 'toast-error')).toHaveCount(0)
    await expect(byTestId(page, 'top-start-btn')).toBeEnabled()
  })

  test('progress, iteration count and loss are shown while running', async ({ page, request }) => {
    await activePalette(request)
    await page.route('**/api/optimize/start', (route) => route.fulfill({ json: { job_id: 'e2e-mock-job', total_iterations: 10, status: 'running' } }))
    await page.routeWebSocket(/\/ws\/optimize\//, (ws) => {
      ws.send(jobMessage('running'))
    })
    await openApp(page)
    await uploadImage(page)
    await byTestId(page, 'top-start-btn').click()
    await expect(byTestId(page, 'job-iteration-count')).toHaveText('(4/10)')
    await expect(byTestId(page, 'top-bar')).toContainText('40.0%')
    await expect(byTestId(page, 'top-bar')).toContainText('Loss: 0.1234')
    await expect(byTestId(page, 'top-cancel-btn')).toBeVisible()
  })
})

test.describe('Color sliders', () => {
  test('dropping a filament assigns it, activates it and uses its TD', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Drop Blue', color: '#0033ff', td: 3.5 })
    await openApp(page)
    await page.locator(`[data-testid="filament-${f.uuid}"]`).dragTo(byTestId(page, 'slider-column-1'))
    await expect(byTestId(page, 'filament-label-1')).toHaveText('E2E Drop Blue')
    await expect(byTestId(page, 'td-input-1')).toHaveValue('3.5')
    await expect(byTestId(page, 'color-indicator-1')).toHaveCSS('background-color', 'rgb(0, 51, 255)')
    await expect(byTestId(page, `active-filament-${f.uuid}`)).toBeVisible()
  })

  test('TD and layer can be typed key by key', async ({ page }) => {
    await openApp(page)
    // Used to coerce every keystroke (`parseFloat(v) || 0`), so partial input
    // like "1." or an emptied field snapped to another value mid-typing.
    await typeInto(page, 'td-input-0', '1.25')
    await expect(byTestId(page, 'td-input-0')).toHaveValue('1.25')
    await typeInto(page, 'layer-input-0', '12')
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('12')
    await expect(byTestId(page, 'depth-0')).toHaveText('0.48')
    await byTestId(page, 'td-input-0').click()
    await expect(byTestId(page, 'color-core-handle-0')).toHaveAttribute('data-layer', /\d+/)
  })

  test('an invalid layer reverts or clamps when leaving the field', async ({ page }) => {
    await openApp(page)
    await typeInto(page, 'layer-input-0', '999')
    await byTestId(page, 'td-input-0').click()
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('75')

    await typeInto(page, 'layer-input-0', '')
    await byTestId(page, 'td-input-0').click()
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('75')
  })

  test('the mouse wheel moves a slider one layer at a time within range', async ({ page }) => {
    await openApp(page)
    const slider = byTestId(page, 'slider-0')
    await slider.hover()
    await page.mouse.wheel(0, -100)
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('9')
    await page.mouse.wheel(0, 100)
    await page.mouse.wheel(0, 100)
    await expect(byTestId(page, 'layer-input-0')).toHaveValue('7')
  })

  test('toggling a column disables its inputs and removes its color-core handle', async ({ page }) => {
    await openApp(page)
    const handles = page.locator('[data-testid^="color-core-handle-"]')
    const before = await handles.count()
    await byTestId(page, 'toggle-0').click()
    await expect(byTestId(page, 'td-input-0')).toBeDisabled()
    await expect(byTestId(page, 'layer-input-0')).toBeDisabled()
    await expect(handles).toHaveCount(before - 1)
    await byTestId(page, 'toggle-0').click()
    await expect(handles).toHaveCount(before)
  })

  test('two sliders on the same layer: only the right-most one counts', async ({ page }) => {
    await openApp(page)
    await typeInto(page, 'layer-input-1', '8')
    await byTestId(page, 'td-input-1').click()
    await expect(byTestId(page, 'slider-column-0')).toHaveAttribute('data-overlap-disabled', 'true')
    await expect(byTestId(page, 'slider-column-1')).not.toHaveAttribute('data-overlap-disabled', 'true')
  })

  test('mesh height label follows the highest slider and the base height', async ({ page }) => {
    await openApp(page)
    // Defaults: top slider at layer 27 × 0.04mm + 0.24mm base. (The default
    // columns used to carry depths that didn't match their layers: 2.48.)
    await expect(byTestId(page, 'mesh-height-label')).toContainText('1.32/')
    await expect(byTestId(page, 'depth-3')).toHaveText('1.08')
    await typeInto(page, 'layer-input-3', '30')
    await byTestId(page, 'td-input-0').click()
    await expect(byTestId(page, 'mesh-height-label')).toContainText('1.44/')
    // Depths follow a layer height change instead of keeping stale values.
    await typeInto(page, 'global-layer-height', '0.08')
    await expect(byTestId(page, 'mesh-height-label')).toContainText('2.64/')
    await expect(byTestId(page, 'depth-3')).toHaveText('2.40')
  })
})

test.describe('Color core', () => {
  test('dragging a handle changes that slider’s layer', async ({ page }) => {
    await openApp(page)
    const handle = byTestId(page, 'color-core-handle-3')
    const before = Number(await handle.getAttribute('data-layer'))
    const box = (await handle.boundingBox())!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 + 60, { steps: 6 })
    await page.mouse.up()
    const after = Number(await byTestId(page, 'color-core-handle-3').getAttribute('data-layer'))
    expect(after).toBeLessThan(before)
    await expect(byTestId(page, 'layer-input-3')).toHaveValue(String(after))
  })

  test('dropping a filament on the core fills the first free column', async ({ page, request }) => {
    const f = await createFilament(request, { name: 'E2E Core Drop' })
    await openApp(page)
    await page.locator(`[data-testid="filament-${f.uuid}"]`).dragTo(byTestId(page, 'color-core'))
    await expect(byTestId(page, 'filament-label-4')).toHaveText('E2E Core Drop')
    await expect(byTestId(page, 'toggle-4')).toHaveText('-')
  })
})

test.describe('Global parameters and settings', () => {
  test('layer height can be typed key by key and is shared with the Settings dialog', async ({ page }) => {
    await openApp(page)
    // Used to turn the first "0" into 0.01 (`|| 0.01`), ending at "0.018".
    await typeInto(page, 'global-layer-height', '0.08')
    await expect(byTestId(page, 'global-layer-height')).toHaveValue('0.08')
    await byTestId(page, 'settings-button').click()
    await expect(byTestId(page, 'setting-layer_height')).toHaveValue('0.08')
  })

  test('base layers and background height stay in sync', async ({ page }) => {
    await openApp(page)
    await typeInto(page, 'global-base-layers', '10')
    await expect(byTestId(page, 'global-background-height')).toHaveValue('0.4')
    await typeInto(page, 'global-background-height', '0.12')
    await expect(byTestId(page, 'global-base-layers')).toHaveValue('3')
  })

  test('dimension below the minimum is clamped when leaving the field', async ({ page }) => {
    await openApp(page)
    await typeInto(page, 'global-stl-size', '5')
    await byTestId(page, 'global-layer-height').click()
    await expect(byTestId(page, 'global-stl-size')).toHaveValue('10')
  })

  test('integer settings round and persist across reloads', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'settings-button').click()
    await typeInto(page, 'setting-iterations', '1234.6')
    await byTestId(page, 'setting-learning_rate').click()
    await expect(byTestId(page, 'setting-iterations')).toHaveValue('1235')
    await typeInto(page, 'setting-learning_rate', '0.02')
    await byTestId(page, 'close-settings').click()
    await waitForSnapshot(page)

    await page.reload()
    await byTestId(page, 'settings-button').click()
    await expect(byTestId(page, 'setting-iterations')).toHaveValue('1235')
    await expect(byTestId(page, 'setting-learning_rate')).toHaveValue('0.02')
  })

  test('boolean, select and color settings apply', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'settings-button').click()
    await byTestId(page, 'setting-init_heightmap_method').selectOption('depth')
    await byTestId(page, 'setting-auto_background_color').click()
    await byTestId(page, 'setting-background_color').fill('#336699')
    await byTestId(page, 'close-settings').click()
    await waitForSnapshot(page)
    const state = await (await page.request.get('/api/project/state')).json()
    expect(state.settings).toMatchObject({ init_heightmap_method: 'depth', auto_background_color: false, background_color: '#336699' })
  })

  test('settings groups collapse', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'settings-button').click()
    await expect(byTestId(page, 'setting-iterations')).toBeVisible()
    await byTestId(page, 'settings-group-Optimization').click()
    await expect(byTestId(page, 'setting-iterations')).toHaveCount(0)
  })
})

test.describe('App shell', () => {
  test('theme toggle persists across reloads', async ({ page }) => {
    await openApp(page)
    const initial = await page.locator('html').getAttribute('data-theme')
    await byTestId(page, 'theme-toggle-btn').click()
    const toggled = initial === 'dark' ? 'light' : 'dark'
    await expect(page.locator('html')).toHaveAttribute('data-theme', toggled)
    await page.reload()
    await expect(page.locator('html')).toHaveAttribute('data-theme', toggled)
  })

  test('shows the server version', async ({ page, request }) => {
    const { version } = await (await request.get('/api/system/version')).json()
    await openApp(page)
    await expect(byTestId(page, 'app-version')).toHaveText(`v${version}`)
  })

  test('Pruning needs a finished result first', async ({ page }) => {
    await openApp(page)
    await expect(byTestId(page, 'top-pruning-btn')).toBeDisabled()
    await expect(byTestId(page, 'top-pruning-btn')).toHaveAttribute('title', 'Run an optimization first')
  })

  test('a failing Run explains why and the message clears once fixed', async ({ page, request }) => {
    const f = await createFilament(request)
    await setActive(request, [f])
    await page.route('**/api/optimize/start', (route) => route.fulfill({ status: 409, json: { detail: 'An optimization is already running.' } }))
    await openApp(page)
    await uploadImage(page)
    await byTestId(page, 'top-start-btn').click()
    await expect(byTestId(page, 'start-error')).toHaveText('An optimization is already running.')

    await byTestId(page, `remove-filament-${f.uuid}`).click()
    await expect(byTestId(page, 'start-error')).toHaveCount(0)
    await expect(byTestId(page, 'run-disabled-reason')).toHaveText('Add at least one active filament first')
  })
})
