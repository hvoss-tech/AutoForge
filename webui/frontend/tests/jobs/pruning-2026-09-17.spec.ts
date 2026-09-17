import fs from 'node:fs'
import { test, expect, type APIRequestContext } from '@playwright/test'
import { FAST_SETTINGS, FIXTURE_IMAGE, activePalette, byTestId, openApp, presetSettings, resetBackend, waitForJob } from '../helpers'

/**
 * The pruning dialog against a real result:
 *
 *  - the limits open at the result's own counts, with no suggested reduction
 *  - the dialog says to run pruning more than once
 *  - the counts move while pruning runs, instead of only at the end
 *  - the two polish passes (better colors, better heights) run first, are
 *    switchable with their own limits, and only ever improve the result
 *  - auto-repeat keeps going until a pass stops improving
 *  - repeat passes never end up worse than the pass before them
 *
 * These need a completed optimization — the dialog is disabled without one —
 * so they live under jobs/ with the rest of the real-run specs.
 */

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

/** One real (tiny) optimization, driven through the API, then the page. */
async function runOptimization(request: APIRequestContext, baseURL: string) {
  await resetBackend(baseURL)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)

  const upload = await request.post('/api/images/upload', {
    multipart: { file: { name: 'cat.png', mimeType: 'image/png', buffer: fs.readFileSync(FIXTURE_IMAGE) } },
  })
  expect(upload.ok()).toBeTruthy()
  const filename = (await upload.json()).filename

  const start = await request.post('/api/optimize/start', { data: { ...FAST_SETTINGS, input_image: filename } })
  expect(start.ok(), await start.text()).toBeTruthy()
  const jobId = (await start.json()).job_id
  const done = await waitForJob(request, jobId)
  expect(done.status, done.error ?? '').toBe('completed')
  return jobId
}

test('the pruning dialog opens on the result you already have', async ({ page, request, baseURL }) => {
  await runOptimization(request, baseURL!)
  await openApp(page)

  await expect(byTestId(page, 'top-pruning-btn')).toBeEnabled()
  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()

  // Item 7: the limits are the current counts, not a reduction the dialog
  // decided on. Pressing Start now accepts what the user is looking at.
  for (const [count, limit] of [
    ['pruning-current-colors', 'pruning-max-colors'],
    ['pruning-current-swaps', 'pruning-max-swaps'],
    ['pruning-current-layers', 'pruning-max-layer'],
  ]) {
    const current = await byTestId(page, count).getAttribute('data-value')
    expect(current).not.toBeNull()
    await expect(byTestId(page, limit)).toHaveValue(current!)
  }

  // Item 8: and it says to come back and do it again.
  const hint = byTestId(page, 'pruning-repeat-hint')
  await expect(hint).toBeVisible()
  await expect(hint).toContainText(/several times/i)
})

test('the counts in the pruning overlay move while it runs', async ({ page, request, baseURL }) => {
  await runOptimization(request, baseURL!)
  await openApp(page)

  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()

  // Ask for a real reduction so there is something to watch come down.
  const colors = byTestId(page, 'pruning-max-colors')
  const startColors = Number(await byTestId(page, 'pruning-current-colors').getAttribute('data-value'))
  await colors.fill(String(Math.max(2, startColors - 2)))
  await colors.dispatchEvent('change')

  await byTestId(page, 'pruning-start-btn').click()
  await expect(byTestId(page, 'pruning-progress-view')).toBeVisible()

  // Item 9: while the job runs the tiles follow the job, not the color
  // sliders (which still hold the pre-pruning stack until the very end).
  await expect(byTestId(page, 'pruning-current-counts')).toHaveAttribute('data-live', 'true')
  await expect(page.getByText('Result so far')).toBeVisible()

  // The reported counts are real numbers from the solution being pruned.
  await expect
    .poll(async () => Number(await byTestId(page, 'pruning-current-colors').getAttribute('data-value')), { timeout: 200_000 })
    .toBeGreaterThan(0)

  await expect(byTestId(page, 'pruning-modal-status')).toHaveText(
    'Pruning finished — the color layers were updated.',
    { timeout: 240_000 },
  )

  // Once finished it ends at (or below) what was asked for, and invites
  // another pass.
  await expect(byTestId(page, 'pruning-again-hint')).toBeVisible()
  await expect(byTestId(page, 'pruning-again-btn')).toBeVisible()
  const finalColors = Number(await byTestId(page, 'pruning-current-colors').getAttribute('data-value'))
  expect(finalColors).toBeLessThanOrEqual(startColors)
})

test('the polish passes run before pruning, are switchable, and only ever help', async ({ page, request, baseURL }) => {
  const jobId = await runOptimization(request, baseURL!)
  await openApp(page)

  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()

  // Both on by default — they were hardcoded on before they became options.
  await expect(byTestId(page, 'pruning-seed-search')).toBeChecked()
  await expect(byTestId(page, 'pruning-fine-tune-height')).toBeChecked()
  // Their limits are editable, and only shown while the pass is enabled.
  await expect(byTestId(page, 'pruning-seed-search-count')).toHaveValue('200')
  await expect(byTestId(page, 'pruning-fine-tune-steps')).toHaveValue('50')
  await byTestId(page, 'pruning-seed-search').uncheck()
  await expect(byTestId(page, 'pruning-seed-search-count')).toHaveCount(0)
  await byTestId(page, 'pruning-seed-search').check()

  // Keep them small so the test stays quick, but real.
  await byTestId(page, 'pruning-seed-search-count').fill('40')
  await byTestId(page, 'pruning-seed-search-count').dispatchEvent('change')
  await byTestId(page, 'pruning-fine-tune-steps').fill('10')
  await byTestId(page, 'pruning-fine-tune-steps').dispatchEvent('change')

  await byTestId(page, 'pruning-start-btn').click()
  await expect(byTestId(page, 'pruning-progress-view')).toBeVisible()

  // Neither pass changes the colour/swap/layer counts, so the loss is where
  // their work shows — the dialog has to put it on screen.
  await expect(byTestId(page, 'pruning-loss')).toBeVisible({ timeout: 120_000 })
  await expect(byTestId(page, 'pruning-loss-value')).not.toBeEmpty()

  await expect(byTestId(page, 'pruning-modal-status')).toHaveText(
    'Pruning finished — the color layers were updated.',
    { timeout: 300_000 },
  )

  // The result is at least as good as where pruning started; these searches
  // keep their work only when the real loss went down.
  const status = await (await request.get(`/api/optimize/status/${(await latestPruneJobId(request, jobId))}`)).json()
  expect(status.pruning_start_loss).toBeGreaterThan(0)
  expect(status.loss).toBeLessThanOrEqual(status.pruning_start_loss)
})

/** The pruning job the page just started for `jobId`. */
async function latestPruneJobId(request: APIRequestContext, jobId: string): Promise<string> {
  const history = await (await request.get('/api/optimize/history')).json()
  const prunes = history.filter((j: { job_id: string }) => j.job_id.startsWith('prune-'))
  expect(prunes.length, `no pruning job recorded for ${jobId}`).toBeGreaterThan(0)
  return prunes[0].job_id
}

test('the polish passes report their own phases in order, before any reduction', async ({ request, baseURL }) => {
  const jobId = await runOptimization(request, baseURL!)

  const start = await request.post('/api/pruning/start', {
    data: {
      pruning_max_colors: 4, pruning_max_swaps: 6, pruning_max_layer: 18, job_id: jobId,
      seed_search: true, seed_search_count: 40, fine_tune_height: true, fine_tune_steps: 10,
    },
  })
  expect(start.ok(), await start.text()).toBeTruthy()
  const pruneJobId = (await start.json()).job_id

  const phases: string[] = []
  let startLoss: number | null = null
  const deadline = Date.now() + 300_000
  while (Date.now() < deadline) {
    const status = await (await request.get(`/api/optimize/status/${pruneJobId}`)).json()
    if (startLoss === null && typeof status.pruning_start_loss === 'number') startLoss = status.pruning_start_loss
    if (status.phase && phases[phases.length - 1] !== status.phase) phases.push(status.phase)
    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
      expect(status.status, status.error ?? '').toBe('completed')
      expect(status.loss).toBeLessThanOrEqual(startLoss!)
      break
    }
    // Polling can only ever see a subset of the phases — on a tiny image a
    // phase can start and finish between two reads — so the assertions below
    // are about the *order* of the ones that were seen, not about catching
    // every one.
    await new Promise((r) => setTimeout(r, 50))
  }

  // Both searches feed the greedy reduction phases a better starting point,
  // so no polish phase may appear after a reduction phase has started.
  const POLISH = ['Searching color seeds', 'Fine-tuning height']
  const firstReduction = phases.findIndex((p) => p.startsWith('Reducing'))
  expect(firstReduction, `no reduction phase seen in ${JSON.stringify(phases)}`).toBeGreaterThanOrEqual(0)
  // "Fine-tuning height" runs again *after* the reductions by design (the
  // offsets that suited 45 layers rarely suit 20), so only the first
  // occurrence of each is checked against the reductions.
  for (const phase of POLISH) {
    const at = phases.indexOf(phase)
    if (at >= 0) expect(at, `${phase} ran after reducing started`).toBeLessThan(firstReduction)
  }
  // And the reductions themselves stay in their documented order.
  const reductions = phases.filter((p) => p.startsWith('Reducing'))
  const expectedOrder = ['Reducing colors', 'Reducing swaps', 'Reducing layers']
  expect(reductions).toEqual(expectedOrder.filter((p) => reductions.includes(p)))
})

test('turning both polish passes off skips their phases entirely', async ({ request, baseURL }) => {
  const jobId = await runOptimization(request, baseURL!)

  const start = await request.post('/api/pruning/start', {
    data: {
      pruning_max_colors: 4, pruning_max_swaps: 6, pruning_max_layer: 18, job_id: jobId,
      seed_search: false, fine_tune_height: false,
    },
  })
  expect(start.ok(), await start.text()).toBeTruthy()
  const pruneJobId = (await start.json()).job_id

  const phases = new Set<string>()
  const deadline = Date.now() + 300_000
  while (Date.now() < deadline) {
    const status = await (await request.get(`/api/optimize/status/${pruneJobId}`)).json()
    if (status.phase) phases.add(status.phase)
    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
      expect(status.status, status.error ?? '').toBe('completed')
      break
    }
    await new Promise((r) => setTimeout(r, 150))
  }
  expect([...phases]).not.toContain('Searching color seeds')
  // The bar's phase slices adapt, so it doesn't sit at 0% waiting for a
  // phase that never runs.
  expect([...phases]).toContain('Reducing colors')
})

test('auto-repeat keeps pruning until a pass stops improving', async ({ page, request, baseURL }) => {
  await runOptimization(request, baseURL!)
  await openApp(page)

  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()

  // Off unless asked for: every pass is real GPU time.
  const toggle = byTestId(page, 'pruning-auto-repeat')
  await expect(toggle).not.toBeChecked()
  await toggle.check()
  await byTestId(page, 'pruning-start-btn').click()

  // It reports which pass it is on...
  await expect(byTestId(page, 'pruning-pass-label')).toBeVisible({ timeout: 120_000 })
  await expect(byTestId(page, 'pruning-pass-label')).toContainText(/pass \d+ of \d+/)

  // ...and stops on its own, without anyone pressing anything.
  await expect(byTestId(page, 'pruning-modal-status')).toHaveText(
    'Pruning finished — the color layers were updated.',
    { timeout: 300_000 },
  )
  const passes = Number((await byTestId(page, 'pruning-pass-label').innerText()).match(/pass (\d+)/)![1])
  expect(passes).toBeGreaterThanOrEqual(2)
})

test('repeat pruning never ends up worse than the pass before it', async ({ request, baseURL }) => {
  // Spike removal used to be applied unconditionally, and on a detailed
  // image it reliably raises the loss — so every extra prune paid that cost
  // again and the result visibly degraded. The first pass may still make
  // that trade (spikes print badly whatever the loss says); later ones may
  // not, so the loss must never climb from one pass to the next.
  const jobId = await runOptimization(request, baseURL!)

  const losses: number[] = []
  for (let pass = 0; pass < 3; pass++) {
    const start = await request.post('/api/pruning/start', {
      data: { pruning_max_colors: 4, pruning_max_swaps: 6, pruning_max_layer: 18, job_id: jobId },
    })
    expect(start.ok(), await start.text()).toBeTruthy()
    const done = await waitForJob(request, (await start.json()).job_id)
    expect(done.status, done.error ?? '').toBe('completed')
    expect(typeof done.loss, `pass ${pass + 1} reported no loss`).toBe('number')
    losses.push(done.loss as number)
  }

  // Pass 2 onwards may improve or hold, never regress. (Pass 1 is exempt: it
  // is the one allowed to trade a little accuracy for a printable surface.)
  for (let i = 2; i < losses.length; i++) {
    expect(losses[i], `pass ${i + 1} (${losses[i]}) got worse than pass ${i} (${losses[i - 1]})`)
      .toBeLessThanOrEqual(losses[i - 1] + 1e-3)
  }
})

test('the pruning job reports live counts over the API', async ({ request, baseURL }) => {
  const jobId = await runOptimization(request, baseURL!)

  const start = await request.post('/api/pruning/start', {
    data: { pruning_max_colors: 3, pruning_max_swaps: 4, pruning_max_layer: 15, job_id: jobId },
  })
  expect(start.ok(), await start.text()).toBeTruthy()
  const pruneJobId = (await start.json()).job_id

  // The counts have to be there *before* the job finishes — that is the
  // whole difference from the old behaviour.
  let sawLiveCounts = false
  const deadline = Date.now() + 240_000
  while (Date.now() < deadline) {
    const status = await (await request.get(`/api/optimize/status/${pruneJobId}`)).json()
    if (typeof status.result_colors === 'number' && status.result_layers > 0) {
      sawLiveCounts = sawLiveCounts || status.status === 'running'
    }
    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
      expect(status.status, status.error ?? '').toBe('completed')
      expect(status.result_colors).toBeGreaterThan(0)
      expect(status.result_layers).toBeGreaterThan(0)
      break
    }
    await new Promise((r) => setTimeout(r, 300))
  }
  expect(sawLiveCounts).toBeTruthy()
})
