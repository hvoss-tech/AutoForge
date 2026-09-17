import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, expect, type APIRequestContext } from '@playwright/test'
import { FAST_SETTINGS, FIXTURE_IMAGE, activePalette, byTestId, openApp, presetSettings, resetBackend, waitForJob } from '../helpers'

/**
 * Working on one image after another (items 10 and 11).
 *
 * 10  optimize + prune, step through History, then load a different image:
 *     the 3D panel used to keep showing the previous image's mesh, because
 *     the finished result stayed selected and the server was still serving
 *     the old auto-preview.
 * 11  VRAM climbed with every image, because every run's pipeline result —
 *     a whole optimizer, on the device — was kept forever.
 */

test.describe.configure({ mode: 'serial' })
test.setTimeout(420_000)

/** A second upload under its own name. The pixels are deliberately the same
 * as the fixture's: what matters here is that it is a *different file* going
 * through the upload path, and a synthetic image risks giving the optimizer a
 * degenerate input that fails for reasons unrelated to what's being tested.
 *
 * Written to the OS temp directory rather than next to the real fixture, so a
 * test run doesn't leave an untracked file in the repo. */
function secondImageFile(): string {
  const file = path.join(os.tmpdir(), 'autoforge-e2e-second-image.png')
  if (!fs.existsSync(file)) fs.copyFileSync(FIXTURE_IMAGE, file)
  return file
}

async function latestJobId(request: APIRequestContext): Promise<string> {
  const res = await request.get('/api/optimize/latest')
  expect(res.ok()).toBeTruthy()
  return (await res.json()).job_id
}

test('loading a different image after a run, a prune and a history jump clears the old preview', async ({ page, request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)

  await openApp(page)
  await byTestId(page, 'image-file-input').setInputFiles(FIXTURE_IMAGE)
  await expect(byTestId(page, 'init-three-d-view')).toBeVisible({ timeout: 120_000 })

  await byTestId(page, 'top-start-btn').click()
  await expect.poll(async () => {
    const res = await request.get('/api/optimize/latest')
    return res.ok() ? (await res.json()).status : null
  }, { timeout: 60_000 }).not.toBe(null)
  const jobId = await latestJobId(request)
  const done = await waitForJob(request, jobId)
  expect(done.status, done.error ?? '').toBe('completed')
  await expect(byTestId(page, 'three-d-view')).toBeVisible({ timeout: 60_000 })

  // Prune it, so the job has edited outputs on disk too.
  const prune = await request.post('/api/pruning/start', {
    data: { pruning_max_colors: 4, pruning_max_swaps: 6, pruning_max_layer: 18, job_id: jobId },
  })
  expect(prune.ok(), await prune.text()).toBeTruthy()
  await waitForJob(request, (await prune.json()).job_id)

  // Walk through History, the way the report describes.
  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  await expect(entries.first()).toBeVisible()
  await entries.nth(0).click()
  await entries.nth((await entries.count()) - 1).click()
  await byTestId(page, 'history-close-btn').click()

  // Now load a different picture.
  await byTestId(page, 'image-file-input').setInputFiles(secondImageFile())
  await expect(byTestId(page, 'input-image')).toBeVisible()

  // The finished result belonged to the *previous* image, so it must not
  // still be on screen — this is exactly what "the old image is shown in the
  // 3D preview" meant.
  await expect(byTestId(page, 'three-d-view')).toHaveCount(0)
  await expect(byTestId(page, 'result-image')).toHaveCount(0)
  await expect(byTestId(page, 'top-pruning-btn')).toBeDisabled()

  // And the new image gets its own heightmap preview, not the old one.
  await expect(byTestId(page, 'init-three-d-view')).toBeVisible({ timeout: 120_000 })
  expect((await (await request.get('/api/init/status')).json()).status).toBe('ready')
})

test('the backend keeps only the newest run, so several images do not pile up', async ({ request, baseURL }) => {
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)

  const ids: string[] = []
  for (const image of [FIXTURE_IMAGE, secondImageFile()]) {
    const upload = await request.post('/api/images/upload', {
      multipart: { file: { name: path.basename(image), mimeType: 'image/png', buffer: fs.readFileSync(image) } },
    })
    expect(upload.ok()).toBeTruthy()
    const start = await request.post('/api/optimize/start', {
      data: { ...FAST_SETTINGS, input_image: (await upload.json()).filename },
    })
    expect(start.ok(), await start.text()).toBeTruthy()
    const jobId = (await start.json()).job_id
    ids.push(jobId)
    const done = await waitForJob(request, jobId)
    expect(done.status, done.error ?? '').toBe('completed')
  }

  // The older run's pipeline state was released when the newer run started,
  // so pruning it finds nothing to prune from; the newest one still works.
  const stale = await request.post('/api/pruning/start', {
    data: { pruning_max_colors: 4, pruning_max_swaps: 4, pruning_max_layer: 18, job_id: ids[0] },
  })
  expect(stale.ok(), await stale.text()).toBeTruthy()
  const staleJob = await waitForJob(request, (await stale.json()).job_id)
  expect(staleJob.status).toBe('failed')
  expect(staleJob.error ?? '').toContain('No optimization pipeline result')

  const fresh = await request.post('/api/pruning/start', {
    data: { pruning_max_colors: 4, pruning_max_swaps: 6, pruning_max_layer: 18, job_id: ids[1] },
  })
  expect(fresh.ok(), await fresh.text()).toBeTruthy()
  const freshJob = await waitForJob(request, (await fresh.json()).job_id)
  expect(freshJob.status, freshJob.error ?? '').toBe('completed')
})
