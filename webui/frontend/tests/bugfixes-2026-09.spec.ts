import { test, expect } from '@playwright/test'
import { resetProjectState, cleanupTestFilaments } from './reset-state'

// Coverage for the 2026-09-16 bug-fix batch:
//  1. Double-clicking a filament (library OR active list) opens the edit modal.
//  2/10. Dragging a filament from the library onto a slider both assigns it
//        AND adds it to Active Filaments (it previously only did the former).
//  9. The New/Edit Filament dialogs can add a brand-new type category, not
//     just pick from the fixed PLA/PETG/... list.
//  7. A failed action surfaces a toast, not just a console.error.
test.beforeEach(async ({ page, baseURL }) => {
  await resetProjectState(baseURL!)
  await cleanupTestFilaments(baseURL!)
  await page.goto('/')
})

test.afterEach(async ({ baseURL }) => {
  await cleanupTestFilaments(baseURL!)
})

const activeRows = '[data-testid="filament-list"] [data-testid^="filament-"][data-active="true"]'

test.describe('Active filaments — double-click and drag (bug #1, #2)', () => {
  test('double-clicking an active filament opens the edit modal', async ({ page }) => {
    // Activate the first library filament via its own "+" toggle (deterministic,
    // doesn't depend on drag simulation working).
    const firstToggle = page.locator('[data-testid="filament-list"] [data-testid^="toggle-filament-"]').first()
    await firstToggle.click()
    await page.locator('[data-testid="active-only-toggle"]').click()

    const activeRow = page.locator(activeRows).first()
    await expect(activeRow).toBeVisible()

    await activeRow.dblclick()
    await expect(page.locator('[data-testid="edit-filament-modal"]')).toBeVisible()
    // The delete button proves this is the full edit modal (change-or-remove),
    // not just a read-only popup.
    await expect(page.locator('[data-testid="delete-filament-btn"]')).toBeVisible()
  })

  test('active filament rows are draggable (drag source for slider assignment)', async ({ page }) => {
    const firstToggle = page.locator('[data-testid="filament-list"] [data-testid^="toggle-filament-"]').first()
    await firstToggle.click()
    await expect(page.locator(activeRows)).toHaveCount(1)
    const activeRow = page.locator(activeRows).first()
    await expect(activeRow).toHaveAttribute('draggable', 'true')
  })
})

test.describe('Dragging a filament onto a slider (bug #2, #10)', () => {
  test('dropping a library filament onto a slider assigns it AND activates it', async ({ page }) => {
    const libraryItem = page.locator('[data-testid="filament-list"] [data-testid^="filament-"]').first()
    const uuid = await libraryItem.getAttribute('data-testid').then((v) => v!.replace('filament-', ''))

    // Sanity: not active yet.
    await expect(page.locator(`[data-testid="filament-${uuid}"][data-active="true"]`)).toHaveCount(0)

    const targetSlider = page.locator('[data-testid="slider-column-0"]')
    await libraryItem.dragTo(targetSlider)

    // Assigned to the slider...
    await expect(page.locator('[data-testid="filament-label-0"]')).not.toHaveText('')
    // ...and now also in Active Filaments, not just referenced by the slider —
    // this is the part that was missing: the optimizer only ever sees
    // active_filaments, so a slider-only assignment was invisible to it.
    await expect(page.locator(`[data-testid="filament-${uuid}"][data-active="true"]`)).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Custom filament type category (bug #9)', () => {
  test('New Filament dialog can add a brand-new category', async ({ page }) => {
    await page.locator('[data-testid="new-filament-btn"]').click()
    await page.locator('[data-testid="new-filament-brand"]').fill('E2ECustomType')
    await page.locator('[data-testid="new-filament-name"]').fill('Custom Type Filament')

    await page.locator('[data-testid="new-filament-type"]').selectOption({ label: '+ Add new category…' })
    await expect(page.locator('[data-testid="new-filament-custom-type"]')).toBeVisible()
    await page.locator('[data-testid="new-filament-custom-type"]').fill('E2E-CustomCategory')

    await page.locator('[data-testid="create-filament-submit"]').click()
    await expect(page.locator('[data-testid="new-filament-modal"]')).not.toBeVisible()

    // The new category's tab appears without a reload.
    await expect(page.locator('[data-testid="tab-E2E-CustomCategory"]')).toBeVisible({ timeout: 5000 })
  })

  test('Edit Filament dialog can add a brand-new category', async ({ page }) => {
    const libraryItem = page.locator('[data-testid="filament-list"] [data-testid^="filament-"]').first()
    await libraryItem.dblclick()
    await expect(page.locator('[data-testid="edit-filament-modal"]')).toBeVisible()

    await page.locator('[data-testid="edit-filament-type"]').selectOption({ label: '+ Add new category…' })
    await expect(page.locator('[data-testid="edit-filament-custom-type"]')).toBeVisible()
    await page.locator('[data-testid="edit-filament-custom-type"]').fill('E2E-EditedCategory')
    await page.locator('[data-testid="save-filament-btn"]').click()

    await expect(page.locator('[data-testid="edit-filament-modal"]')).not.toBeVisible()
    await expect(page.locator('[data-testid="tab-E2E-EditedCategory"]')).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Error toasts (bug #7)', () => {
  test('a failed filament creation shows a toast, not just a silent failure', async ({ page }) => {
    await page.route('**/api/filaments', (route) => {
      if (route.request().method() === 'POST') return route.abort('failed')
      return route.continue()
    })

    await page.locator('[data-testid="new-filament-btn"]').click()
    await page.locator('[data-testid="new-filament-brand"]').fill('E2EToastTest')
    await page.locator('[data-testid="new-filament-name"]').fill('Should Fail')
    await page.locator('[data-testid="create-filament-submit"]').click()

    await expect(page.locator('[data-testid="toast-error"]')).toBeVisible()
    await expect(page.locator('[data-testid="toast-error"]')).toContainText('Failed to create filament')
  })

  test('toasts can be dismissed', async ({ page }) => {
    await page.route('**/api/filaments', (route) => {
      if (route.request().method() === 'POST') return route.abort('failed')
      return route.continue()
    })
    await page.locator('[data-testid="new-filament-btn"]').click()
    await page.locator('[data-testid="new-filament-brand"]').fill('E2EToastTest2')
    await page.locator('[data-testid="new-filament-name"]').fill('Should Fail 2')
    await page.locator('[data-testid="create-filament-submit"]').click()

    const toast = page.locator('[data-testid="toast-error"]')
    await expect(toast).toBeVisible()
    await page.locator('[data-testid^="toast-dismiss-"]').first().click()
    await expect(toast).toHaveCount(0)
  })
})
