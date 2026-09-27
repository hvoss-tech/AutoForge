import { test, expect } from '@playwright/test'
import {
  FAST_SETTINGS,
  activePalette,
  byTestId,
  openApp,
  presetSettings,
  resetBackend,
  uploadAndWaitForPreview,
  waitForJob,
} from '../helpers'

// The job's status socket can take many seconds to deliver anything (a
// browser holds a new WebSocket back while another handshake to the host is
// pending). The page used to sit on "Preparing the optimization…" until then;
// it now follows the job by polling until the socket's first status arrives.
test.setTimeout(240_000)

test('a run is followed even when its status socket never says anything', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, { ...FAST_SETTINGS, iterations: 1500 })
  // Every job socket connects to a mock that never sends a status.
  await page.routeWebSocket(/\/ws\/optimize\//, () => {})
  await openApp(page)
  await uploadAndWaitForPreview(page)

  const started = page.waitForResponse((r) => r.url().endsWith('/api/optimize/start') && r.request().method() === 'POST')
  await byTestId(page, 'top-start-btn').click()
  const jobId = (await (await started).json()).job_id

  // Progress shows up from polling, well before the run ends.
  await expect(page.getByText('Preparing the optimization…')).toBeHidden({ timeout: 20_000 })
  await expect
    .poll(async () => Number((await (await request.get(`/api/optimize/status/${jobId}`)).json()).iteration), { timeout: 20_000 })
    .toBeGreaterThan(0)

  const done = await waitForJob(request, jobId, 200_000)
  expect(done.status, done.error ?? '').toBe('completed')
  // The page notices the end without the socket, too.
  await expect(byTestId(page, 'job-done-badge')).toBeVisible({ timeout: 15_000 })
})
