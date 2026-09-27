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

test('a color limit the active filaments can never exceed is flagged as having no effect', () => {
  assert.equal(colorLimitNote(null, 4), null)
  // The base is one of the 4 filaments: at most 4 colors anyway.
  assert.equal(colorLimitNote(3, 4), null)
  assert.match(colorLimitNote(4, 4), /4 filaments active/)
  assert.match(colorLimitNote(9, 1), /1 filament active/)
  // A custom base color is one color more.
  assert.equal(colorLimitNote(4, 4, false), null)
  assert.match(colorLimitNote(5, 4, false), /4 filaments active/)
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

test('the base filament is one of the colors: a band reusing it is not counted twice', async () => {
  const { resultCounts } = await import('../src/lib/pruning.ts')
  const { buildPrintPlan } = await import('../src/lib/printPlan.ts')
  const sliders = [
    { td: 1, layer: 3, depth_mm: 0, filament_uuid: 'black', enabled: true },
    { td: 1, layer: 6, depth_mm: 0, filament_uuid: 'red', enabled: true },
    { td: 1, layer: 9, depth_mm: 0, filament_uuid: 'white', enabled: true },
  ]
  const plan = buildPrintPlan(sliders, [], { layer_height: 0.04, background_height: 0.24 })
  assert.equal(plan.colors, 3)
  // Base in black, which a band also uses: black, red, white = 3 colors.
  assert.equal(resultCounts(plan, 'black').colors, 3)
  // Base in a fourth filament: 4 colors.
  assert.equal(resultCounts(plan, 'grey').colors, 4)
  // Base unknown / a custom color: one more than the bands.
  assert.equal(resultCounts(plan).colors, 4)
})
