import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test, expect, type Page } from '@playwright/test'
import {
  FAST_SETTINGS,
  FIXTURE_IMAGE,
  activePalette,
  byTestId,
  openApp,
  presetSettings,
  resetBackend,
  typeInto,
  uploadAndWaitForPreview,
  waitForSnapshot,
} from '../helpers'

/**
 * Regression coverage for "pruning runs in place and destroys some of the
 * history": a long, real session (two images, two optimizations, four
 * pruning passes of varying strength, several hand slider edits) followed
 * by jumping to every history step in random order and checking that each
 * one still shows *exactly* what was on screen when it was captured.
 *
 * This needs real optimizer/pruning runs (pruning mutates the optimizer's
 * state and the job's output files in place, which is the whole point of
 * the regression), so it lives under jobs/ with the rest of the real-run
 * specs.
 */

test.describe.configure({ mode: 'serial' })

function secondImageFile(): string {
  const file = path.join(os.tmpdir(), 'autoforge-e2e-history-prune-second-image.png')
  if (!fs.existsSync(file)) fs.copyFileSync(FIXTURE_IMAGE, file)
  return file
}

interface BandState {
  layer: string
  td: string
  filamentUuid: string | null
  enabled: boolean
}

interface CapturedState {
  image: string | null
  bands: BandState[]
}

/** Everything a history step promises to restore: the input image and the
 * full color-slider stack (layer, TD, filament assignment, enabled). Not
 * included: the resolved base color and the 3D mesh/STL bytes — neither is
 * part of Snapshot (see src/types/index.ts), so they are not something
 * History claims to reproduce exactly.
 *
 * The image is identified by its *filename* (from the persisted project
 * state), not the <img> element's literal `src`: a freshly uploaded image
 * displays from a one-off `blob:` URL, while a step reached by navigating
 * history resolves to the durable `/uploads/<name>` path for the same file
 * (see restoredImage() in src/lib/history.ts) — comparing the raw src would
 * flag that legitimate, intentional difference as if it were a corruption. */
async function captureState(page: Page): Promise<CapturedState> {
  const projectState = await (await page.request.get('/api/project/state')).json()
  const image = projectState?.settings?.input_image ?? null
  const columns = page.locator('[data-testid^="slider-column-"]')
  const n = await columns.count()
  const bands: BandState[] = []
  for (let i = 0; i < n; i++) {
    const layer = await byTestId(page, `layer-input-${i}`).inputValue()
    const td = await byTestId(page, `td-input-${i}`).inputValue()
    const filamentUuid = await byTestId(page, `filament-select-${i}`).getAttribute('data-value')
    const enabled = (await byTestId(page, `toggle-${i}`).getAttribute('data-enabled')) === 'true'
    bands.push({ layer, td, filamentUuid, enabled })
  }
  return { image, bands }
}

async function clickStartAndConfirm(page: Page) {
  await byTestId(page, 'top-start-btn').click()
  const confirm = byTestId(page, 'confirm-dialog')
  try {
    await confirm.waitFor({ state: 'visible', timeout: 2000 })
    await byTestId(page, 'confirm-ok').click()
  } catch {
    // No hand-edit confirmation needed this time.
  }
}

/** Edits band 0's layer and TD to new, deliberately different values (so
 * each call is guaranteed to change something and create its own history
 * step), blurring after each so NumberInput settles/commits it. */
async function moveSliders(page: Page, layer: number, td: number) {
  await typeInto(page, 'layer-input-0', String(layer))
  await byTestId(page, 'td-input-0').click()
  await typeInto(page, 'td-input-0', String(td))
  await byTestId(page, 'layer-input-0').click()
}

/** Runs one pruning pass through the real dialog, with limits pulled down
 * from the *current* counts by the given amounts — so consecutive passes
 * ask for genuinely different (varying) reductions, not the same one twice. */
async function runPruningPass(
  page: Page,
  delta: { colors: number; swaps: number; layers: number },
) {
  const curColors = Number(await byTestId(page, 'pruning-current-colors').getAttribute('data-value'))
  const curSwaps = Number(await byTestId(page, 'pruning-current-swaps').getAttribute('data-value'))
  const curLayers = Number(await byTestId(page, 'pruning-current-layers').getAttribute('data-value'))

  const colors = byTestId(page, 'pruning-max-colors')
  await colors.fill(String(Math.max(1, curColors - delta.colors)))
  await colors.dispatchEvent('change')
  const swaps = byTestId(page, 'pruning-max-swaps')
  await swaps.fill(String(Math.max(0, curSwaps - delta.swaps)))
  await swaps.dispatchEvent('change')
  const layers = byTestId(page, 'pruning-max-layer')
  await layers.fill(String(Math.max(1, curLayers - delta.layers)))
  await layers.dispatchEvent('change')

  await byTestId(page, 'pruning-start-btn').click()
  await expect(byTestId(page, 'pruning-modal-status')).toHaveText(
    'Pruning finished — the color layers were updated.',
    { timeout: 240_000 },
  )
}

/** Deterministic Fisher-Yates so a failure is reproducible. */
function shuffled(n: number, seed: number): number[] {
  let s = seed
  const rand = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648
  const arr = Array.from({ length: n }, (_, i) => i)
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

test('every history step survives repeated in-place pruning and random-order navigation', async ({ page, request, baseURL }) => {
  // The per-test `{ timeout }` options form is silently ignored by this
  // Playwright version (confirmed by probing test.info().timeout — it stays
  // at the 30s project default), and test.setTimeout() *overwrites* rather
  // than extends: uploadAndWaitForPreview() below calls it internally
  // (120_000), so it has to be called again *after* every helper that also
  // calls it, not just once up front.
  test.setTimeout(7_200_000)
  await resetBackend(baseURL!)
  await activePalette(request)
  await presetSettings(request, FAST_SETTINGS)
  await openApp(page)

  const checkpoints: { label: string; state: CapturedState }[] = []
  const record = async (label: string) => {
    checkpoints.push({ label, state: await captureState(page) })
  }

  // --- Image 1: optimize, then prune it twice at different strengths ---
  await uploadAndWaitForPreview(page)
  test.setTimeout(7_200_000) // uploadAndWaitForPreview() just reset this to 120_000
  await waitForSnapshot(page)
  await record('upload image 1')

  await clickStartAndConfirm(page)
  await expect(page.getByText('Done', { exact: true })).toBeVisible({ timeout: 60_000 })
  await waitForSnapshot(page)
  await record('optimization 1 done')

  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()
  await runPruningPass(page, { colors: 1, swaps: 1, layers: 2 })
  await byTestId(page, 'pruning-close-btn').click()
  await waitForSnapshot(page)
  await record('prune 1a (mild)')

  // Reopening after a close resets the dialog to a fresh limits form (not
  // the "prune further" progress view — that only stays reachable while
  // the dialog was never closed in between), seeded with suggestions based
  // on the now-pruned result. Same start button, no "again" button needed.
  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()
  await runPruningPass(page, { colors: 0, swaps: 2, layers: 4 })
  await byTestId(page, 'pruning-close-btn').click()
  await waitForSnapshot(page)
  await record('prune 1b (aggressive)')

  // --- Hand-move some sliders on the pruned result ---
  await moveSliders(page, 9, 3.3)
  await waitForSnapshot(page)
  await record('slider move 1a')

  await moveSliders(page, 6, 1.7)
  await waitForSnapshot(page)
  await record('slider move 1b')

  // --- Image 2: a different picture, its own optimization, its own edit ---
  await byTestId(page, 'image-file-input').setInputFiles(secondImageFile())
  await expect(byTestId(page, 'input-image')).toBeVisible()
  await waitForSnapshot(page)
  await record('upload image 2')

  await expect(byTestId(page, 'init-three-d-view')).toBeVisible({ timeout: 120_000 })
  await clickStartAndConfirm(page)
  await expect(page.getByText('Done', { exact: true })).toBeVisible({ timeout: 60_000 })
  await waitForSnapshot(page)
  await record('optimization 2 done')

  await moveSliders(page, 8, 2.2)
  await waitForSnapshot(page)
  await record('slider move 2a')

  // --- Prune image 2's result twice more, again at different strengths ---
  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()
  await runPruningPass(page, { colors: 1, swaps: 1, layers: 3 })
  await byTestId(page, 'pruning-close-btn').click()
  await waitForSnapshot(page)
  await record('prune 2a (mild)')

  await byTestId(page, 'top-pruning-btn').click()
  await expect(byTestId(page, 'pruning-modal')).toBeVisible()
  await runPruningPass(page, { colors: 0, swaps: 1, layers: 5 })
  await byTestId(page, 'pruning-close-btn').click()
  await waitForSnapshot(page)
  await record('prune 2b (aggressive)')

  // --- Every step is now on the books. Confirm the drawer agrees, then
  //     visit every one of them in random order and check it still shows
  //     exactly what was captured — not what pruning most recently did to
  //     the job it happens to share an id with. ---
  console.log('checkpoints:', JSON.stringify(checkpoints, null, 2))
  await byTestId(page, 'history-open-btn').click()
  const entries = page.locator('[data-testid^="history-entry-"]')
  await expect(entries).toHaveCount(checkpoints.length)

  const order = shuffled(checkpoints.length, 20260924)
  for (const index of order) {
    const { label, state: expected } = checkpoints[index]
    await entries.nth(index).click()
    try {
      // Generous: jumping to a step on a *different* image than the one
      // currently shown drops the server's cached auto-preview and (if no
      // completed result covers the new image yet) rebuilds it with a real
      // heightmap init before the restore's own await chain resolves — by
      // design, only one image's preview/optimizer is kept in memory at a
      // time (see the concurrent-init fix). That is genuinely slow, not a
      // sign the restore is stuck — confirmed twice on this machine: the
      // state always matched once given enough time, it just occasionally
      // took longer than 600s for this specific transition under load
      // (other tests' leftover browser/backend processes competing for
      // CPU during the heightmap rebuild).
      await expect
        .poll(async () => JSON.stringify(await captureState(page)), {
          timeout: 900_000,
          message: `history step "${label}" (index ${index}) did not restore to its captured state`,
        })
        .toBe(JSON.stringify(expected))
    } catch (e) {
      const actual = await captureState(page)
      console.log(`MISMATCH at "${label}" (index ${index})`)
      console.log('expected:', JSON.stringify(expected, null, 2))
      console.log('actual:  ', JSON.stringify(actual, null, 2))
      throw e
    }
  }
})
