// Test runner: node --test tests/best-of.test.mjs
//
// "Best of" runs (settings.best_of): the warning confirmed before a run and
// how the runs are shown. The dialog itself is covered by
// tests/features/best-of.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import { BEST_OF_MAX, bestOfRuns, bestOfWarning, describeBestOfRuns } from '../src/lib/bestOf.ts'
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
