// Test runner: node --test tests/usability-logic.test.mjs
// Pure logic behind the 2026-09 usability rework (src/lib/*).

import test from 'node:test'
import assert from 'node:assert/strict'
import { buildPrintPlan, getPlanBands, printPlanText } from '../src/lib/printPlan.ts'
import { getWorkflowSteps } from '../src/lib/workflow.ts'
import { describeSettingsChange, describeSliderEdit, mergeHistoryLabels } from '../src/lib/history.ts'
import { suggestPruningLimits } from '../src/lib/pruning.ts'
import { sortFilaments } from '../src/lib/library.ts'
import { QUALITY_PRESETS, presetForIterations } from '../src/lib/settingsPresets.ts'

const settings = { layer_height: 0.04, background_height: 0.24, background_color: '#000000' }
const black = { uuid: 'k', brand: 'B', name: 'Black', color: '#000000', td: 0.6, owned: true, filament_type: 'PLA' }
const white = { uuid: 'w', brand: 'B', name: 'White', color: '#ffffff', td: 5, owned: false, filament_type: 'PLA' }
const s = (layer, uuid, extra = {}) => ({ td: 1, layer, depth_mm: 0, filament_uuid: uuid, enabled: true, ...extra })

test('print plan', async (t) => {
  await t.test('bands own the layers above the previous band, in print order', () => {
    const bands = getPlanBands([s(10, 'w'), s(4, 'k')], [black, white], settings)
    assert.deepEqual(bands.map((b) => [b.filament.name, b.startLayer, b.endLayer]), [['Black', 1, 4], ['White', 5, 10]])
    assert.equal(bands[1].endHeightMm, 0.64)
  })

  await t.test('hidden bands and the losing side of a shared layer are ignored', () => {
    const bands = getPlanBands([s(4, 'k'), s(4, 'w'), s(8, 'k', { enabled: false })], [black, white], settings)
    assert.deepEqual(bands.map((b) => b.filament.name), ['White'])
  })

  await t.test('counts: adjacent bands of the same filament are no swap', () => {
    const plan = buildPrintPlan([s(2, 'k'), s(5, 'k'), s(9, 'w'), s(12, 'k')], [black, white], settings)
    assert.equal(plan.colors, 2)
    assert.equal(plan.swaps, 2)
    assert.equal(plan.topLayer, 12)
    assert.equal(plan.totalHeightMm, 0.72)
    assert.deepEqual(plan.swapsList.map((x) => [x.layerNumber, x.heightMm, x.filament.name]), [[2, 0.28, 'Black'], [7, 0.48, 'White'], [11, 0.64, 'Black']])
  })

  await t.test('counts work without the filament list (keyed by uuid)', () => {
    assert.equal(buildPrintPlan([s(2, 'k'), s(5, 'k'), s(9, 'w')], [], settings).swaps, 1)
  })

  await t.test('instructions text matches the optimizer wording', () => {
    const plan = buildPrintPlan([s(2, 'k'), s(9, 'w')], [black, white], settings)
    const text = printPlanText(plan, settings)
    assert.match(text, /^Print at 100% infill with a layer height of 0\.04mm with a base layer of 0\.24mm/)
    assert.match(text, /At layer #2 \(0\.28mm\) swap to B - Black/)
    assert.match(text, /At layer #4 \(0\.36mm\) swap to B - White/)
    assert.match(text, /For the rest, use B - White$/)
    assert.equal(printPlanText(buildPrintPlan([], [], settings), settings), 'No layers printed.')
  })
})

test('workflow steps', async (t) => {
  const states = (input) => Object.fromEntries(getWorkflowSteps(input).map((st) => [st.id, st.state]))

  await t.test('fresh start: image is the current step', () => {
    assert.deepEqual(states({ hasImage: false, activeFilamentCount: 0, jobStatus: null }), { image: 'current', filaments: 'upcoming', run: 'upcoming', adjust: 'upcoming', export: 'upcoming' })
  })
  await t.test('image and filaments done: run is next', () => {
    assert.equal(states({ hasImage: true, activeFilamentCount: 2, jobStatus: null }).run, 'current')
  })
  await t.test('filaments before image still points at the image first', () => {
    const st = states({ hasImage: false, activeFilamentCount: 2, jobStatus: null })
    assert.equal(st.image, 'current')
    assert.equal(st.filaments, 'done')
    assert.equal(st.run, 'upcoming')
  })
  await t.test('a result makes adjusting and exporting available', () => {
    assert.deepEqual(states({ hasImage: false, activeFilamentCount: 0, jobStatus: 'completed' }), { image: 'done', filaments: 'done', run: 'done', adjust: 'current', export: 'current' })
  })
})

test('history labels', async (t) => {
  await t.test('additions in one debounce window are listed together', () => {
    let label
    for (const next of ['Added Black', 'Added Blue', 'Added Black', 'Added Beige']) label = mergeHistoryLabels(label, next)
    assert.equal(label, 'Added Black, Blue, Beige')
  })
  await t.test('unrelated edits: the latest wins', () => {
    assert.equal(mergeHistoryLabels('Added Black', 'Moved band 2 to layer 5'), 'Moved band 2 to layer 5')
  })
  await t.test('slider edits describe what changed', () => {
    assert.equal(describeSliderEdit(1, { layer: 14 }), 'Moved band 2 to layer 14')
    assert.equal(describeSliderEdit(0, { td: 2.5 }), 'Changed TD of band 1 to 2.5')
    assert.equal(describeSliderEdit(2, { filament_uuid: 'x', enabled: true, td: 1 }, 'White'), 'Assigned White to band 3')
    assert.equal(describeSliderEdit(0, { enabled: false }), 'Disabled band 1')
  })
  await t.test('settings changes name the setting', () => {
    assert.equal(describeSettingsChange({ layer_height: 0.04, iterations: 1 }, { layer_height: 0.08, iterations: 1 }), 'Changed layer height')
    assert.equal(describeSettingsChange({ stl_output_size: 1 }, { stl_output_size: 2 }), 'Changed dimension')
    assert.equal(describeSettingsChange({ a: 1, b: 1 }, { a: 2, b: 2 }), 'Changed settings')
  })
})

test('pruning limits start at the current result, not a guessed reduction', () => {
  // The dialog used to pre-fill a reduction nobody asked for (one colour
  // fewer, a third fewer swaps), so pressing Start accepted a target the
  // user never chose. It now opens on exactly what the result has.
  assert.deepEqual(suggestPruningLimits({ colors: 5, swaps: 38, layers: 75 }), { colors: 5, swaps: 38, layers: 75 })
  // Still never below what the pruner accepts (layers >= 1, colors >= 1).
  assert.deepEqual(suggestPruningLimits({ colors: 2, swaps: 0, layers: 0 }), { colors: 2, swaps: 0, layers: 1 })
  assert.deepEqual(suggestPruningLimits({ colors: 0, swaps: 0, layers: 0 }), { colors: 1, swaps: 0, layers: 1 })
})

test('filament sorting', () => {
  const fs = [
    { name: 'Red', color: '#e53935', td: 4 },
    { name: 'Black', color: '#000000', td: 0.6 },
    { name: 'Blue', color: '#1e40af', td: 4 },
    { name: 'White', color: '#ffffff', td: 5 },
  ]
  assert.deepEqual(sortFilaments(fs, 'name').map((f) => f.name), ['Black', 'Blue', 'Red', 'White'])
  assert.deepEqual(sortFilaments(fs, 'td').map((f) => f.name), ['Black', 'Blue', 'Red', 'White'])
  // Grays by lightness first, then around the hue wheel.
  assert.deepEqual(sortFilaments(fs, 'color').map((f) => f.name), ['Black', 'White', 'Red', 'Blue'])
})

test('quality presets', () => {
  assert.equal(presetForIterations(6000), 'balanced')
  assert.equal(presetForIterations(1234), 'custom')
  assert.ok(QUALITY_PRESETS.every((p, i, all) => i === 0 || p.iterations > all[i - 1].iterations))
})
