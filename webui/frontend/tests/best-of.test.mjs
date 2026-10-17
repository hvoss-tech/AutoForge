// Test runner: node --test tests/best-of.test.mjs
//
// "Best of" runs (settings.best_of): the warning confirmed before a run and
// how the runs are shown. The dialog itself is covered by
// tests/features/best-of.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  BEST_OF_MAX,
  bestOfImageUrl,
  bestOfRuns,
  bestOfWarning,
  describeBestOfRuns,
  describeLossGap,
  galleryGrid,
  isChoosingBestOf,
  lowestLossRun,
  needsInitialPrune,
  stepRun,
  toggleCompareRun,
} from '../src/lib/bestOf.ts'
import { staleReasons } from '../src/lib/staleResult.ts'

globalThis.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {} }

test('a single run needs no warning', () => {
  assert.equal(bestOfWarning(1), null)
  assert.equal(bestOfWarning(undefined), null)
  assert.equal(bestOfWarning(0), null)
})

test('more than one run asks first, naming how many runs it takes', () => {
  const warning = bestOfWarning(3)
  assert.ok(warning)
  assert.match(warning.message, /3 times/)
  assert.match(warning.message, /3× as long/)
  // The user picks the run, it isn't chosen for them.
  assert.match(warning.message, /pick the one to keep/)
  assert.equal(warning.confirmLabel, 'Run 3 times')
})

test('the run count is a whole number within the server limit', () => {
  assert.equal(bestOfRuns(2.7), 2)
  assert.equal(bestOfRuns('4'), 4)
  assert.equal(bestOfRuns(500), BEST_OF_MAX)
  assert.equal(bestOfRuns(Number.NaN), 1)
})

test('changing best of does not put the result out of date', () => {
  const run = { settings: { best_of: 1, iterations: 2000 }, filamentUuids: ['a'] }
  assert.deepEqual(staleReasons(run, { best_of: 3, iterations: 2000 }, ['a']), [])
})

test('finished runs are listed with the kept one marked', () => {
  assert.equal(describeBestOfRuns(null), null)
  assert.equal(describeBestOfRuns([]), null)
  assert.deepEqual(describeBestOfRuns([3, 1.23456], 1), ['Run 1: 3.0000', 'Run 2: 1.2346 (kept)'])
})

test('a series waits for a choice only while its job is still going', () => {
  const job = { status: 'running', best_of_choosing: true, best_of_losses: [2, 1] }
  assert.equal(isChoosingBestOf(job), true)
  assert.equal(isChoosingBestOf({ ...job, status: 'paused' }), true)
  // The server keeps the flag on a job cancelled while choosing.
  assert.equal(isChoosingBestOf({ ...job, status: 'cancelled' }), false)
  assert.equal(isChoosingBestOf({ ...job, status: 'completed' }), false)
  assert.equal(isChoosingBestOf({ ...job, best_of_choosing: false }), false)
  assert.equal(isChoosingBestOf({ ...job, best_of_losses: [1] }), false)
  assert.equal(isChoosingBestOf(null), false)
})

test('the gallery is one row for up to three runs, then square-ish', () => {
  assert.deepEqual(galleryGrid(2), { columns: 2, rows: 1 })
  assert.deepEqual(galleryGrid(3), { columns: 3, rows: 1 })
  assert.deepEqual(galleryGrid(4), { columns: 2, rows: 2 })
  assert.deepEqual(galleryGrid(5), { columns: 3, rows: 2 })
  assert.deepEqual(galleryGrid(9), { columns: 3, rows: 3 })
  assert.deepEqual(galleryGrid(10), { columns: 4, rows: 3 })
  assert.deepEqual(galleryGrid(BEST_OF_MAX), { columns: 5, rows: 4 })
})

test('each run says how far its loss is behind the lowest', () => {
  const losses = [2.2, 2.0, 2.0005, 3.0]
  assert.equal(lowestLossRun(losses), 1)
  assert.equal(lowestLossRun([]), -1)
  assert.equal(describeLossGap(losses, 1), 'Lowest loss')
  assert.equal(describeLossGap(losses, 0), '+10.0%')
  assert.equal(describeLossGap(losses, 2), '+0.03%')
  assert.equal(describeLossGap(losses, 3), '+50.0%')
  // A tie with the lowest is just as good.
  assert.equal(describeLossGap([1, 1], 1), 'Lowest loss')
  // A zero loss has no percentage to speak of.
  assert.equal(describeLossGap([0, 0.5], 1), '+0.5000')
})

test('at most two runs are marked for A/B, the oldest mark giving way', () => {
  assert.deepEqual(toggleCompareRun([], 3), [3])
  assert.deepEqual(toggleCompareRun([3], 1), [3, 1])
  assert.deepEqual(toggleCompareRun([3, 1], 4), [1, 4])
  // Clicking a marked run unmarks it.
  assert.deepEqual(toggleCompareRun([3, 1], 3), [1])
})

test('flipping through runs wraps around both ways', () => {
  assert.equal(stepRun(0, -1, 4), 3)
  assert.equal(stepRun(3, 1, 4), 0)
  assert.equal(stepRun(1, 1, 4), 2)
  assert.equal(stepRun(0, 1, 0), 0)
})

test('run images are addressed by job and run', () => {
  assert.equal(bestOfImageUrl('abc', 2), '/api/optimize/best-of/abc/image/2')
})

test('a best-of series pruned run by run is not pruned again', () => {
  assert.equal(needsInitialPrune({ job_id: 'a' }, true), true)
  assert.equal(needsInitialPrune({ job_id: 'a', best_of_pruned: false }, true), true)
  assert.equal(needsInitialPrune({ job_id: 'a', best_of_pruned: true }, true), false)
  assert.equal(needsInitialPrune({ job_id: 'prune-a' }, true), false)
  assert.equal(needsInitialPrune({ job_id: 'a' }, false), false)
})
