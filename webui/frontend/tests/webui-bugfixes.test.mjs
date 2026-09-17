// Test runner: node --test tests/webui-bugfixes.test.mjs
//
// Regressions for the 2026-09-17 webui audit. Unlike appStore-logic.test.mjs
// these import the real modules: src/lib/* has no DOM side effects and only
// type-only imports, which Node (>= 23.6) strips natively.

import test from 'node:test'
import assert from 'node:assert/strict'
import { slidersNeedRender } from '../src/lib/sliderDiff.ts'
import { jobStatusFromControlResponse } from '../src/lib/jobControl.ts'

const slider = (layer, overrides = {}) => ({ td: 5, layer, depth_mm: layer * 0.04, filament_uuid: 'f', enabled: true, ...overrides })

test('slidersNeedRender', async (t) => {
  // Regression: ColorSliders compared each new column against
  // `JSON.parse(last)[i]` without a length check. An optimizer/pruner
  // broadcast with more columns than before (e.g. 20 bands replacing the 10
  // default columns while the auto-preview was ready) read `.td` of
  // undefined inside a useEffect and unmounted the whole app.
  await t.test('does not throw when the stack grows', () => {
    assert.equal(slidersNeedRender([slider(1)], [slider(1), slider(2), slider(3)]), true)
  })

  await t.test('treats a removed column as a change', () => {
    assert.equal(slidersNeedRender([slider(1), slider(2)], [slider(1)]), true)
  })

  await t.test('identical stacks need no render', () => {
    assert.equal(slidersNeedRender([slider(1), slider(2)], [slider(1), slider(2)]), false)
  })

  await t.test('depth_mm alone is not a render-affecting change', () => {
    assert.equal(slidersNeedRender([slider(1)], [slider(1, { depth_mm: 9 })]), false)
  })

  for (const [field, value] of [['td', 1], ['layer', 7], ['enabled', false], ['filament_uuid', 'other']]) {
    await t.test(`a changed ${field} needs a render`, () => {
      assert.equal(slidersNeedRender([slider(1)], [slider(1, { [field]: value })]), true)
    })
  }
})

test('jobStatusFromControlResponse', async (t) => {
  // Regression: pause/resume/cancel set the requested status locally no
  // matter what, so cancelling a job that had just completed showed its
  // result as "cancelled" (and pausing one showed Resume forever).
  await t.test('uses the status the backend reports', () => {
    assert.equal(jobStatusFromControlResponse({ status: 'completed' }, 'cancelled'), 'completed')
  })

  await t.test('falls back when the body is missing or unrecognised', () => {
    assert.equal(jobStatusFromControlResponse(null, 'paused'), 'paused')
    assert.equal(jobStatusFromControlResponse({ status: 'resumed' }, 'running'), 'running')
    assert.equal(jobStatusFromControlResponse({}, 'cancelled'), 'cancelled')
  })
})

import { resolveRestoredJob, historyShortcut } from '../src/lib/history.ts'

const job = (job_id, status) => ({ job_id, status })

test('resolveRestoredJob', async (t) => {
  await t.test('never touches a running job (undo used to cancel it)', () => {
    assert.deepEqual(resolveRestoredJob(job('j', 'running'), { optimizationResultId: null }), { kind: 'keep' })
    assert.deepEqual(resolveRestoredJob(job('j', 'paused'), { optimizationResultId: 'x' }), { kind: 'keep' })
  })

  await t.test('a snapshot taken while a job ran (no result yet) restores no job', () => {
    // Regression: currentJobId was used, so this restored the since-completed
    // job with the pre-result sliders and re-rendered the result with them.
    assert.deepEqual(resolveRestoredJob(job('j', 'completed'), { currentJobId: 'j', optimizationResultId: null }), { kind: 'none' })
  })

  await t.test('keeps the shown result when the snapshot has the same one', () => {
    assert.deepEqual(resolveRestoredJob(job('j', 'completed'), { optimizationResultId: 'j' }), { kind: 'keep' })
  })

  await t.test('fetches a different result', () => {
    assert.deepEqual(resolveRestoredJob(job('b', 'completed'), { optimizationResultId: 'a' }), { kind: 'fetch', jobId: 'a' })
    assert.deepEqual(resolveRestoredJob(null, { optimizationResultId: 'a' }), { kind: 'fetch', jobId: 'a' })
  })
})

test('historyShortcut', async (t) => {
  const ev = (key, mods = {}) => ({ key, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, target: { tagName: 'BODY' }, ...mods })

  await t.test('Ctrl+Z undoes', () => assert.equal(historyShortcut(ev('z', { ctrlKey: true })), 'undo'))

  await t.test('Ctrl+Shift+Z redoes (key arrives upper-case)', () => {
    // Regression: `e.shiftKey && e.key === 'z'` never matched 'Z'.
    assert.equal(historyShortcut(ev('Z', { ctrlKey: true, shiftKey: true })), 'redo')
  })

  await t.test('Ctrl+Y redoes, Cmd works on macOS', () => {
    assert.equal(historyShortcut(ev('y', { ctrlKey: true })), 'redo')
    assert.equal(historyShortcut(ev('z', { metaKey: true })), 'undo')
  })

  await t.test('leaves text fields their native undo', () => {
    assert.equal(historyShortcut(ev('z', { ctrlKey: true, target: { tagName: 'INPUT', type: 'number' } })), null)
    assert.equal(historyShortcut(ev('z', { ctrlKey: true, target: { tagName: 'TEXTAREA' } })), null)
    assert.equal(historyShortcut(ev('z', { ctrlKey: true, target: { tagName: 'INPUT', type: 'range' } })), 'undo')
  })

  await t.test('ignores unmodified keys', () => assert.equal(historyShortcut(ev('z')), null))
})

import { parseNumberDraft, settleNumberDraft } from '../src/lib/numberDraft.ts'
import { chooseActiveTab } from '../src/lib/library.ts'

test('parseNumberDraft', async (t) => {
  await t.test('typing "0.08" one key at a time never commits a coerced value', () => {
    // Regression: `parseFloat(v) || 0.01` turned the first "0" into 0.01, so
    // the field read "0.018" by the time the user finished typing.
    const commits = ['0', '0.', '0.0', '0.08'].map((d) => parseNumberDraft(d, { min: 0.01 }))
    assert.deepEqual(commits, [null, null, null, 0.08])
  })

  await t.test('empty and partial input commits nothing', () => {
    for (const d of ['', ' ', '-', '.', '-.']) assert.equal(parseNumberDraft(d), null)
  })

  await t.test('respects integer and range rules', () => {
    assert.equal(parseNumberDraft('1.5', { integer: true }), null)
    assert.equal(parseNumberDraft('150', { min: 1, max: 200, integer: true }), 150)
    assert.equal(parseNumberDraft('250', { max: 200 }), null)
  })
})

test('settleNumberDraft', async (t) => {
  await t.test('reverts garbage to the last valid value', () => {
    assert.equal(settleNumberDraft('', 0.04), 0.04)
    assert.equal(settleNumberDraft('-', 7), 7)
  })
  await t.test('clamps and rounds on blur', () => {
    assert.equal(settleNumberDraft('0', 0.04, { min: 0.01 }), 0.01)
    assert.equal(settleNumberDraft('300', 75, { max: 200, integer: true }), 200)
    assert.equal(settleNumberDraft('7.6', 1, { integer: true }), 8)
  })
})

test('chooseActiveTab', async (t) => {
  await t.test('a library without the default PLA tab opens on its first type', () => {
    assert.equal(chooseActiveTab(['PETG', 'TPU'], 'PLA'), 'PETG')
  })
  await t.test('keeps a still-valid tab', () => {
    assert.equal(chooseActiveTab(['PETG', 'PLA'], 'PLA'), 'PLA')
  })
  await t.test('keeps the tab while types are unknown', () => {
    assert.equal(chooseActiveTab([], 'PLA'), 'PLA')
  })
})
