// Test runner: node --test tests/usability-round2.test.mjs
// Pure logic behind the second usability pass (band reordering, stale results,
// project files, loss history, pane sizes).

import test from 'node:test'
import assert from 'node:assert/strict'
import { insertBandAbove, isInPrintOrder, moveBand, printOrder, sortBandsByLayer } from '../src/lib/bandOps.ts'
import { staleReasons } from '../src/lib/staleResult.ts'
import { hasUnsavedChanges, projectFileName, projectFingerprint, projectNameFromFile } from '../src/lib/project.ts'
import { appendLossPoint, sparklinePath } from '../src/lib/lossHistory.ts'
import { clampSize, zoomAt, IDENTITY_ZOOM } from '../src/lib/layout.ts'
import { describePruningChange, resultCounts } from '../src/lib/pruning.ts'

const band = (layer, uuid, extra = {}) => ({ td: 1, layer, depth_mm: 0, filament_uuid: uuid, enabled: true, ...extra })
const summary = (sliders) => sliders.map((s) => `${s.filament_uuid}@${s.layer}`)

test('band order', async (t) => {
  await t.test('print order sorts by layer, keeps ties in store order, unplaced last', () => {
    const sliders = [band(8, 'a'), band(0, 'x', { enabled: false }), band(3, 'b'), band(8, 'c')]
    assert.deepEqual(printOrder(sliders), [2, 0, 3, 1])
    assert.equal(isInPrintOrder(sliders), false)
    assert.equal(isInPrintOrder(sortBandsByLayer(sliders).sliders), true)
  })

  await t.test('sorting follows the band the caller cares about', () => {
    const edit = sortBandsByLayer([band(9, 'a'), band(2, 'b')], 0)
    assert.deepEqual(summary(edit.sliders), ['b@2', 'a@9'])
    assert.equal(edit.index, 1)
  })

  await t.test('the default test columns (positioned, then unplaced) are already in order', () => {
    const sliders = [band(8, ''), band(13, ''), band(20, ''), band(27, ''), band(0, '', { enabled: false }), band(0, '', { enabled: false })]
    assert.equal(isInPrintOrder(sliders), true)
  })
})

test('moving a band keeps every thickness', async (t) => {
  // a: 1-3 (3 thick), b: 4-8 (5), c: 9-10 (2)
  const stack = [band(3, 'a'), band(8, 'b'), band(10, 'c')]

  await t.test('moving the bottom band to the top', () => {
    const edit = moveBand(stack, 0, 2, 0.04)
    assert.deepEqual(summary(edit.sliders), ['b@5', 'c@7', 'a@10'])
    assert.equal(edit.index, 2)
    assert.equal(edit.sliders[2].depth_mm, 0.4)
  })

  await t.test('moving the top band down one', () => {
    assert.deepEqual(summary(moveBand(stack, 2, 1, 0.04).sliders), ['a@3', 'c@5', 'b@10'])
  })

  await t.test('works on rows that were out of order', () => {
    const shuffled = [band(8, 'b'), band(10, 'c'), band(3, 'a')]
    // Store index 2 is "a" (first in print); move it where "c" (store 1) is.
    assert.deepEqual(summary(moveBand(shuffled, 2, 1, 0.04).sliders), ['b@5', 'c@7', 'a@10'])
  })

  await t.test('unplaced bands are left alone', () => {
    const withUnplaced = [...stack, band(0, 'x', { enabled: false })]
    assert.deepEqual(summary(moveBand(withUnplaced, 3, 0, 0.04).sliders), summary(withUnplaced))
  })
})

test('inserting a band', async (t) => {
  const stack = [band(3, 'a'), band(8, 'b')]

  await t.test('shifts the bands above up when there is room', () => {
    const edit = insertBandAbove(stack, 0, { filament_uuid: 'n', td: 5 }, 75, 0.04)
    assert.deepEqual(summary(edit.sliders), ['a@3', 'n@6', 'b@11'])
    assert.equal(edit.index, 1)
    assert.equal(edit.sliders[1].enabled, true)
  })

  await t.test('uses what room there is', () => {
    assert.deepEqual(summary(insertBandAbove(stack, 0, { filament_uuid: 'n', td: 5 }, 9, 0.04).sliders), ['a@3', 'n@4', 'b@9'])
  })

  await t.test('splits the band below when the stack is full', () => {
    assert.deepEqual(summary(insertBandAbove(stack, 1, { filament_uuid: 'n', td: 5 }, 8, 0.04).sliders), ['a@3', 'b@6', 'n@8'])
  })

  await t.test('gives up when there is not a layer to spare', () => {
    assert.equal(insertBandAbove([band(1, 'a')], 0, { filament_uuid: 'n', td: 5 }, 1, 0.04), null)
  })
})

test('stale results', async (t) => {
  const run = { settings: { layer_height: 0.04, max_layers: 75, pruning_max_colors: 100 }, filamentUuids: ['a', 'b'] }

  await t.test('nothing changed, or unknown run', () => {
    assert.deepEqual(staleReasons(run, { layer_height: 0.04, max_layers: 75, pruning_max_colors: 100 }, ['b', 'a']), [])
    assert.deepEqual(staleReasons(undefined, { layer_height: 0.08 }, []), [])
  })

  await t.test('names changed settings and filaments, ignoring pruning limits', () => {
    assert.deepEqual(staleReasons(run, { layer_height: 0.08, max_layers: 75, pruning_max_colors: 5 }, ['a']), ['layer_height', 'active_filaments'])
  })
})

test('project files', async (t) => {
  await t.test('file names come from the project name', () => {
    assert.equal(projectFileName('My Cat / v2'), 'My-Cat-v2.json')
    assert.equal(projectFileName('  '), 'autoforge-project.json')
    assert.equal(projectNameFromFile('Stored', 'x.json'), 'Stored')
    assert.equal(projectNameFromFile(undefined, 'Sunset.json'), 'Sunset')
    assert.equal(projectNameFromFile(undefined, 'autoforge-project-1789.json'), '')
  })

  await t.test('unsaved changes compare against the saved fingerprint', () => {
    const content = { name: 'n', inputImage: '/uploads/a.png', settings: { a: 1 }, colorSliders: [band(3, 'a')], activeFilamentUuids: ['b', 'a'] }
    const saved = projectFingerprint(content)
    assert.equal(projectFingerprint({ ...content, activeFilamentUuids: ['a', 'b'] }), saved)
    assert.notEqual(projectFingerprint({ ...content, settings: { a: 2 } }), saved)
    assert.equal(hasUnsavedChanges(saved, saved, true), false)
    assert.equal(hasUnsavedChanges(saved, 'other', true), true)
    assert.equal(hasUnsavedChanges(null, 'x', false), false)
    assert.equal(hasUnsavedChanges(null, 'x', true), true)
  })
})

test('loss history', async (t) => {
  await t.test('skips repeats and restarts on a new run', () => {
    let points = appendLossPoint([], 10, 0.5)
    points = appendLossPoint(points, 10, 0.4)
    points = appendLossPoint(points, 20, 0.3)
    assert.deepEqual(points.map((p) => p.iteration), [10, 20])
    assert.deepEqual(appendLossPoint(points, 5, 0.9).map((p) => p.iteration), [5])
  })

  await t.test('thins out old points past the limit but keeps the newest', () => {
    let points = []
    for (let i = 1; i <= 50; i++) points = appendLossPoint(points, i, 1 / i, 20)
    assert.ok(points.length <= 20)
    assert.equal(points[points.length - 1].iteration, 50)
    assert.equal(points[0].iteration, 1)
  })

  await t.test('sparkline spans the box, lower loss drawn lower', () => {
    const d = sparklinePath([{ iteration: 0, loss: 1 }, { iteration: 10, loss: 0 }], 100, 40)
    assert.equal(d, 'M0.0,0.0 L100.0,40.0')
    assert.equal(sparklinePath([{ iteration: 0, loss: 1 }], 100, 40), '')
  })
})

test('layout', async (t) => {
  await t.test('pane sizes clamp, min wins', () => {
    assert.equal(clampSize(50, 140, 600), 140)
    assert.equal(clampSize(900, 140, 600), 600)
    assert.equal(clampSize(300, 140, 100), 140)
    assert.equal(clampSize(NaN, 140, 600), 140)
  })

  await t.test('zoom keeps the point under the cursor and resets at 1x', () => {
    const z = zoomAt(IDENTITY_ZOOM, 2, 100, 50)
    assert.deepEqual(z, { scale: 2, x: -100, y: -50 })
    // The cursor point maps back to the same content point.
    assert.equal((100 - z.x) / z.scale, 100)
    assert.deepEqual(zoomAt(z, 0.25, 10, 10), IDENTITY_ZOOM)
  })
})

test('pruning summary', () => {
  const before = resultCounts({ colors: 25, swaps: 25, topLayer: 60 })
  assert.deepEqual(before, { colors: 26, swaps: 25, layers: 60 })
  assert.equal(describePruningChange(before, { colors: 12, swaps: 11, layers: 60 }), '26 → 12 colors, 25 → 11 swaps, 60 layers')
})
