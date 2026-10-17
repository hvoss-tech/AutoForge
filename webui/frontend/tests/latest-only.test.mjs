// Test runner: node --test tests/latest-only.test.mjs
//
// The image editor renders through LatestOnly: dragging a slider asks for a
// render at every position, and each one used to run in turn (a wide soften
// took most of a second each), so the preview lagged far behind the slider.

import test from 'node:test'
import assert from 'node:assert/strict'
import { LatestOnly } from '../src/lib/latestOnly.ts'

const later = (ms) => new Promise((r) => setTimeout(r, ms))

test('while one job runs, only the newest of the jobs asked for meanwhile runs', async () => {
  const queue = new LatestOnly()
  const ran = []
  const job = (n) => async () => {
    ran.push(n)
    await later(20)
    return n
  }
  const results = await Promise.all([1, 2, 3, 4, 5].map((n) => queue.run(job(n))))
  assert.deepEqual(ran, [1, 5])
  assert.deepEqual(results, [1, null, null, null, 5])
})

test('one job at a time, and a failure does not block the next', async () => {
  const queue = new LatestOnly()
  let running = 0
  let most = 0
  const job = (fail) => async () => {
    running++
    most = Math.max(most, running)
    await later(10)
    running--
    if (fail) throw new Error('boom')
    return 'ok'
  }
  const first = queue.run(job(true))
  const second = queue.run(job(false))
  await assert.rejects(first, /boom/)
  assert.equal(await second, 'ok')
  assert.equal(most, 1)
  // Idle again: the next job runs straight away.
  assert.equal(await queue.run(job(false)), 'ok')
})
