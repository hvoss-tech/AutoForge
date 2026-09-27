// Test runner: node --test tests/run-limits.test.mjs
//
// Pure logic behind the color/swap limits set before a run (settings
// max_colors / max_swaps). The UI and a real limited run are covered by
// tests/jobs/run-limits.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  RUN_LIMIT_DEFAULT,
  colorLimitNote,
  describeRunLimits,
  isLimited,
  normalizeLimit,
  withinRunLimits,
} from '../src/lib/runLimits.ts'
import { staleReasons } from '../src/lib/staleResult.ts'
import { settingName } from '../src/lib/history.ts'

globalThis.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {} }

test('null and undefined mean unlimited', () => {
  assert.equal(isLimited(null), false)
  assert.equal(isLimited(undefined), false)
  assert.equal(isLimited(0), true)
  assert.equal(isLimited(Number.NaN), false)
})

test('limits are stored as whole numbers at or above their minimum', () => {
  assert.equal(normalizeLimit('max_colors', 1), 2) // base + one color is the least
  assert.equal(normalizeLimit('max_colors', 4.6), 5)
  assert.equal(normalizeLimit('max_swaps', -3), 0)
  assert.equal(normalizeLimit('max_swaps', 0), 0) // zero swaps is a real limit
  assert.equal(normalizeLimit('max_swaps', null), null)
  assert.ok(RUN_LIMIT_DEFAULT.max_colors >= 2 && RUN_LIMIT_DEFAULT.max_swaps >= 0)
})

test('the summary names only what is limited', () => {
  assert.equal(describeRunLimits({ max_colors: null, max_swaps: null }), null)
  assert.equal(describeRunLimits({}), null)
  assert.equal(describeRunLimits({ max_colors: 4, max_swaps: null }), 'at most 4 colors')
  assert.equal(describeRunLimits({ max_colors: null, max_swaps: 1 }), 'at most 1 swap')
  assert.equal(describeRunLimits({ max_colors: 3, max_swaps: 0 }), 'at most 3 colors · 0 swaps')
})

test('a color limit at or above the active filaments (plus base) is flagged as having no effect', () => {
  assert.equal(colorLimitNote(null, 4), null)
  assert.equal(colorLimitNote(4, 4), null)
  assert.match(colorLimitNote(5, 4), /4 filaments active/)
  assert.match(colorLimitNote(9, 1), /1 filament active/)
  assert.equal(colorLimitNote(5, 0), null) // nothing active yet: nothing to say
})

test('results are checked against the limits, colors including the base', () => {
  assert.equal(withinRunLimits({ max_colors: 3, max_swaps: 2 }, { colors: 3, swaps: 2 }), true)
  assert.equal(withinRunLimits({ max_colors: 3, max_swaps: 2 }, { colors: 4, swaps: 2 }), false)
  assert.equal(withinRunLimits({ max_colors: 3, max_swaps: 2 }, { colors: 3, swaps: 3 }), false)
  assert.equal(withinRunLimits({ max_colors: null, max_swaps: null }, { colors: 12, swaps: 40 }), true)
})

test('changing a limit makes an existing result out of date, and is named readably', () => {
  const run = { settings: { max_colors: null, max_swaps: 10 }, filamentUuids: [] }
  assert.deepEqual(staleReasons(run, { max_colors: 4, max_swaps: 10 }, []), ['max_colors'])
  assert.deepEqual(staleReasons(run, { max_colors: null, max_swaps: 10 }, []), [])
  assert.equal(settingName('max_colors'), 'color limit')
  assert.equal(settingName('max_swaps'), 'swap limit')
})
