import { test, expect } from '@playwright/test'
import { resetBackend } from './helpers'

// Regression for: a fresh page load resurrected the single most recent job
// ever run against the backend (GET /api/optimize/latest) as `currentJob`
// unconditionally — including a *failed* one from a completely unrelated
// past session. In the wild this showed "Optimization failed: Input image
// not found: whatever.png ..." on every load, even though the user hadn't
// uploaded anything or clicked Run this session.
test('a stale failed job in history is not resurrected as the current job on load', async ({ page, request, baseURL }) => {
  // Earlier specs leave slider assignments behind, which show a color-stack
  // preview instead of the placeholder this test looks for.
  await resetBackend(baseURL!)
  // Seed exactly that: the newest job is a failed one.
  await request.put('/api/filaments/active', {
    data: [{ uuid: 'e2e-stale', brand: 'E2E', name: 'Stale', color: '#ff0000', td: 1 }],
  })
  const { job_id } = await (await request.post('/api/optimize/start', { data: { input_image: 'whatever.png', iterations: 1 } })).json()
  await expect.poll(async () => (await (await request.get(`/api/optimize/status/${job_id}`)).json()).status).toBe('failed')
  expect((await (await request.get('/api/optimize/latest')).json()).job_id).toBe(job_id)
  await request.put('/api/filaments/active', { data: [] })

  await page.goto('/')
  // Give the app's mount-time loadCurrentJob() a moment to run.
  await page.waitForTimeout(1000)

  await expect(page.locator('[data-testid="optimization-error"]')).toHaveCount(0)
  await expect(page.getByText(/whatever\.png/)).toHaveCount(0)
  await expect(page.locator('[data-testid="toast-error"]')).toHaveCount(0)
  await expect(page.locator('[data-testid="preview-placeholder"]')).toBeVisible()
})
