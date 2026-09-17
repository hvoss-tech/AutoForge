// Test runner: node --test tests/appStore-logic.test.mjs
//
// Unit tests for two small pure functions exported from
// src/store/appStore.ts. Importing that module directly isn't viable here
// (it has module-scope side effects that read `document`/`localStorage` for
// initial theme detection, which need a real DOM — there's no jsdom/browser
// shim wired into this project's plain `node --test` setup), so — mirroring
// the existing convention in plyParser.test.mjs — the logic is mirrored
// here instead. Keep these in sync with appStore.ts if either changes.

import test from 'node:test'
import assert from 'node:assert/strict'

// Mirrors shouldRestoreJobOnLoad() in src/store/appStore.ts.
function shouldRestoreJobOnLoad(status) {
  return status === 'running' || status === 'paused' || status === 'completed'
}

// Mirrors durableInputImageUrl() in src/store/appStore.ts.
function durableInputImageUrl(inputImage, settings) {
  if (inputImage && inputImage.startsWith('blob:') && settings.input_image) {
    return `/uploads/${settings.input_image}`
  }
  return inputImage
}

test('shouldRestoreJobOnLoad', async (t) => {
  // Regression: /api/optimize/latest returning an old 'failed' job (from a
  // completely unrelated session — in the wild this included a batch of
  // automated-test job records that ended up in the real
  // checkpoints/history.json) used to be restored as `currentJob` on every
  // fresh page load, showing "Optimization failed: ..." before the user
  // had uploaded anything or clicked Run.
  await t.test('does not restore a failed job', () => {
    assert.equal(shouldRestoreJobOnLoad('failed'), false)
  })

  await t.test('does not restore a cancelled job', () => {
    assert.equal(shouldRestoreJobOnLoad('cancelled'), false)
  })

  await t.test('restores a running job (worth reconnecting to)', () => {
    assert.equal(shouldRestoreJobOnLoad('running'), true)
  })

  await t.test('restores a paused job', () => {
    assert.equal(shouldRestoreJobOnLoad('paused'), true)
  })

  await t.test('restores a completed job (result still worth showing)', () => {
    assert.equal(shouldRestoreJobOnLoad('completed'), true)
  })

  await t.test('does not restore an unrecognised status', () => {
    assert.equal(shouldRestoreJobOnLoad('pending'), false)
  })
})

test('durableInputImageUrl', async (t) => {
  await t.test('rewrites a blob: URL to the durable server path when one is known', () => {
    const result = durableInputImageUrl('blob:http://localhost:3000/abc-123', { input_image: 'xyz.png' })
    assert.equal(result, '/uploads/xyz.png')
  })

  await t.test('leaves a blob: URL alone when no server-side filename is known yet', () => {
    const result = durableInputImageUrl('blob:http://localhost:3000/abc-123', { input_image: '' })
    assert.equal(result, 'blob:http://localhost:3000/abc-123')
  })

  await t.test('leaves an already-durable /uploads/ path untouched', () => {
    const result = durableInputImageUrl('/uploads/xyz.png', { input_image: 'xyz.png' })
    assert.equal(result, '/uploads/xyz.png')
  })

  await t.test('passes through null', () => {
    assert.equal(durableInputImageUrl(null, { input_image: '' }), null)
  })
})
