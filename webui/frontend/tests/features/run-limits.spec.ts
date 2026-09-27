import { test, expect, type APIRequestContext } from '@playwright/test'
import { activePalette, byTestId, openApp, resetBackend, typeInto } from '../helpers'

// Color / swap limits set before a run (settings max_colors / max_swaps).
// No optimization runs here; tests/jobs/run-limits.spec.ts runs real ones.

const savedLimits = async (request: APIRequestContext) => {
  const state = await (await request.get('/api/project/state')).json()
  return { colors: state.settings.max_colors ?? null, swaps: state.settings.max_swaps ?? null }
}

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

test('limits are unlimited by default and set from the button next to Run', async ({ page, request }) => {
  await activePalette(request)
  await openApp(page)

  const button = byTestId(page, 'run-limits-btn')
  await expect(button).toBeVisible()
  await expect(button).toHaveAttribute('data-limited', 'false')
  await expect(byTestId(page, 'run-limits-summary')).toHaveText('No limits')
  expect(await savedLimits(request)).toEqual({ colors: null, swaps: null })

  await button.click()
  const popover = byTestId(page, 'run-limits-popover')
  await expect(popover).toBeVisible()
  await expect(byTestId(page, 'run-limit-max_colors-unlimited')).toHaveAttribute('aria-checked', 'true')
  await expect(byTestId(page, 'run-limit-max_colors-value')).toHaveCount(0)

  // "At most" switches the limit on with a sensible starting value to edit.
  await byTestId(page, 'run-limit-max_colors-limited').click()
  await expect(byTestId(page, 'run-limit-max_colors-limited')).toHaveAttribute('aria-checked', 'true')
  await expect(byTestId(page, 'run-limit-max_colors-value')).toHaveValue('4')
  await typeInto(page, 'run-limit-max_colors-value', '3')
  await byTestId(page, 'run-limit-max_swaps-limited').click()
  await typeInto(page, 'run-limit-max_swaps-value', '5')
  await byTestId(page, 'run-limit-max_swaps-value').blur()

  await expect(byTestId(page, 'run-limits-summary')).toHaveText('≤ 3 colors · 5 swaps')
  await expect(button).toHaveAttribute('data-limited', 'true')
  await expect.poll(() => savedLimits(request)).toEqual({ colors: 3, swaps: 5 })

  // Escape closes the popover; the limits stay.
  await page.keyboard.press('Escape')
  await expect(popover).toBeHidden()

  // They survive a reload and show up in Settings as well.
  await page.reload()
  await expect(byTestId(page, 'run-limits-summary')).toHaveText('≤ 3 colors · 5 swaps')
  await byTestId(page, 'settings-button').click()
  const section = byTestId(page, 'settings-run-limits')
  await expect(section).toBeVisible()
  await expect(section.getByTestId('run-limit-max_colors-value')).toHaveValue('3')
  await expect(section.getByTestId('run-limit-max_swaps-value')).toHaveValue('5')

  // "No limit" clears it again, from either place.
  await section.getByTestId('run-limit-max_swaps-unlimited').click()
  await expect(section.getByTestId('run-limit-max_swaps-value')).toHaveCount(0)
  await expect.poll(() => savedLimits(request)).toEqual({ colors: 3, swaps: null })
  await byTestId(page, 'close-settings').click()
  await expect(byTestId(page, 'run-limits-summary')).toHaveText('≤ 3 colors')
})

test('limits are kept sensible: at least base + one color, and a warning when they cannot bite', async ({ page, request }) => {
  await activePalette(request) // 4 filaments
  await openApp(page)
  await byTestId(page, 'run-limits-btn').click()
  await byTestId(page, 'run-limit-max_colors-limited').click()

  // 1 color would leave nothing but the base; it's raised to 2.
  await typeInto(page, 'run-limit-max_colors-value', '1')
  await byTestId(page, 'run-limit-max_colors-value').blur()
  await expect.poll(async () => (await savedLimits(request)).colors).toBe(2)

  // With 4 filaments active (the base is one of them) a limit of 4 or more can't change anything.
  await expect(byTestId(page, 'run-limit-max_colors-note')).toHaveCount(0)
  await typeInto(page, 'run-limit-max_colors-value', '6')
  await expect(byTestId(page, 'run-limit-max_colors-note')).toContainText('4 filaments active')

  // Zero swaps is a real limit (one color band).
  await byTestId(page, 'run-limit-max_swaps-limited').click()
  await typeInto(page, 'run-limit-max_swaps-value', '0')
  await byTestId(page, 'run-limit-max_swaps-value').blur()
  await expect.poll(async () => (await savedLimits(request)).swaps).toBe(0)
  await expect(byTestId(page, 'run-limits-summary')).toHaveText('≤ 6 colors · 0 swaps')
})

test('the backend rejects limits that make no sense', async ({ request }) => {
  const res = await request.post('/api/optimize/start', { data: { input_image: 'x.png', max_colors: 1 } })
  expect(res.status()).toBe(422)
  const res2 = await request.post('/api/optimize/start', { data: { input_image: 'x.png', max_swaps: -1 } })
  expect(res2.status()).toBe(422)
})
