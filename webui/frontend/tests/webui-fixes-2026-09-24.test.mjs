// Test runner: node --test tests/webui-fixes-2026-09-24.test.mjs
//
// Pure logic behind the 2026-09-24 WebUI fixes. The HueForge library import
// is covered end to end by tests/features/webui-fixes-2026-09-24.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import { inheritRunInputs, staleReasons } from '../src/lib/staleResult.ts'
import { sliderRenderPaused } from '../src/lib/history.ts'

// localStorage is only a convenience for storeRunInputs; node has none.
globalThis.localStorage ??= { getItem: () => null, setItem: () => {}, removeItem: () => {} }

const runSettings = { layer_height: 0.04, max_layers: 75, iterations: 6000 }

test('a pruned result inherits the run inputs of the job it was pruned from', async (t) => {
  const all = { 'job-1': { settings: runSettings, filamentUuids: ['a', 'b'] } }

  await t.test('without them, a pruned result could never be flagged out of date', () => {
    const changed = { ...runSettings, max_layers: 40 }
    assert.deepEqual(staleReasons(all['prune-1'], changed, ['a', 'b']), [])
    const next = inheritRunInputs(all, 'job-1', 'prune-1')
    assert.deepEqual(staleReasons(next['prune-1'], changed, ['a', 'b']), ['max_layers'])
    assert.deepEqual(staleReasons(next['prune-1'], runSettings, ['a']), ['active_filaments'])
    assert.deepEqual(staleReasons(next['prune-1'], runSettings, ['b', 'a']), [])
  })

  await t.test('chains through repeated prunes', () => {
    const once = inheritRunInputs(all, 'job-1', 'prune-1')
    const twice = inheritRunInputs(once, 'prune-1', 'prune-2')
    assert.deepEqual(twice['prune-2'], all['job-1'])
  })

  await t.test('leaves the map alone when the source run is unknown', () => {
    assert.equal(inheritRunInputs(all, 'nope', 'prune-1'), all)
    assert.equal(inheritRunInputs(all, 'job-1', 'job-1'), all)
  })

  await t.test('does not mutate the input map', () => {
    inheritRunInputs(all, 'job-1', 'prune-9')
    assert.deepEqual(Object.keys(all), ['job-1'])
  })
})

test('slider edits are not re-rendered while pruning rewrites the optimizer', () => {
  const done = { status: 'completed' }
  assert.equal(sliderRenderPaused(done, null), false)
  assert.equal(sliderRenderPaused(done, { status: 'completed' }), false)
  assert.equal(sliderRenderPaused(done, { status: 'cancelled' }), false)
  assert.equal(sliderRenderPaused(done, { status: 'failed' }), false)
  for (const status of ['pending', 'running', 'paused']) {
    assert.equal(sliderRenderPaused(done, { status }), true, `pruning ${status}`)
    assert.equal(sliderRenderPaused({ status }, null), true, `optimization ${status}`)
  }
  assert.equal(sliderRenderPaused(null, undefined), false)
})
