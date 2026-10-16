import { test, expect } from '@playwright/test'
import { activePalette, byTestId, openApp, resetBackend, typeInto, uploadAndWaitForPreview } from '../helpers'

// "Best of" runs (settings.best_of): set in Settings, and a run of more than
// one is confirmed first. The start request is intercepted, so no
// optimization runs; tests/webui/test_best_of.py covers the runs themselves.

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

test('best of more than one asks before the run starts', async ({ page, request }) => {
  await activePalette(request)
  await openApp(page)
  await uploadAndWaitForPreview(page)

  await byTestId(page, 'settings-button').click()
  await typeInto(page, 'setting-best_of', '3')
  await byTestId(page, 'setting-best_of').blur()
  await byTestId(page, 'close-settings').click()
  await expect.poll(async () => (await (await request.get('/api/project/state')).json()).settings.best_of).toBe(3)

  const started: Record<string, unknown>[] = []
  await page.route('**/api/optimize/start', async (route) => {
    started.push(route.request().postDataJSON())
    await route.fulfill({ status: 409, json: { detail: 'Not started in this test.' } })
  })

  // Cancel: nothing starts.
  await byTestId(page, 'top-start-btn').click()
  const dialog = byTestId(page, 'confirm-dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('Run the optimization 3 times?')
  await expect(dialog).toContainText('3× as long')
  await byTestId(page, 'confirm-cancel').click()
  await expect(dialog).toBeHidden()
  expect(started).toHaveLength(0)

  // Confirm: the run starts with the setting.
  await byTestId(page, 'top-start-btn').click()
  await expect(dialog).toBeVisible()
  await expect(byTestId(page, 'confirm-ok')).toHaveText('Run 3 times')
  await byTestId(page, 'confirm-ok').click()
  await expect.poll(() => started.length).toBe(1)
  expect(started[0].best_of).toBe(3)
})
