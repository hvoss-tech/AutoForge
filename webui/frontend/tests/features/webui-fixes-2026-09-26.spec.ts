import { test, expect } from '@playwright/test'
import { activeLibraryItem, byTestId, createFilament, openApp, resetBackend, setActive } from '../helpers'

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

// fetch() only throws on a network failure, so a server that *rejected*
// activating or deactivating a filament went unnoticed: the list on screen
// and the one /api/optimize/start runs with silently diverged.
test.describe('Active filament changes the server rejects', () => {
  test('a rejected activation is reported', async ({ page, request }) => {
    const f = await createFilament(request)
    await page.route('**/api/filaments/active', (route) =>
      route.request().method() === 'POST'
        ? route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })
        : route.continue(),
    )
    await openApp(page)
    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(byTestId(page, 'toast-error')).toContainText(f.name)
  })

  test('a rejected deactivation is reported', async ({ page, request }) => {
    const f = await createFilament(request)
    await setActive(request, [f])
    await page.route(`**/api/filaments/active/${f.uuid}`, (route) =>
      route.request().method() === 'DELETE'
        ? route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })
        : route.continue(),
    )
    await openApp(page)
    await expect(activeLibraryItem(page, f.uuid)).toBeVisible()
    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(byTestId(page, 'toast-error')).toContainText('Failed to remove active filament')
  })

  test('deactivating a filament the server already dropped is not an error', async ({ page, request }) => {
    const f = await createFilament(request)
    await setActive(request, [f])
    await openApp(page)
    await expect(activeLibraryItem(page, f.uuid)).toBeVisible()
    await setActive(request, [])
    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(activeLibraryItem(page, f.uuid)).toHaveCount(0)
    await expect(byTestId(page, 'toast-error')).toHaveCount(0)
  })
})
