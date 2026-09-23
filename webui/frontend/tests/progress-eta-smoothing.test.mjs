// Test runner: node --test tests/progress-eta-smoothing.test.mjs
//
// Regression coverage for "the 'x s left' estimate jumps around every
// iteration": createProgressTracker()'s smoothing used to blend a fixed
// fraction of each new sample into the running rate regardless of how much
// wall-clock time the sample actually covered (`smoothedRate * 0.7 +
// sampleRate * 0.3` every qualifying call). That implicitly assumes the
// caller samples on an even cadence; TopBar's effect used to re-run (and
// re-sample) on every websocket message instead of on a steady timer, so
// the "0.3 weight" was sometimes applied to a sample covering 100ms and
// sometimes to one covering 2s, each treated as equally trustworthy. Fixed
// two ways: TopBar now drives the tracker off a steady 1s interval
// (src/components/TopBar.tsx, not unit-testable without a DOM), and the
// tracker itself was made robust to whatever cadence it's actually called
// at, by turning the fixed decay into a time-constant-based one
// (`alpha = 1 - exp(-dt / TAU)`) — this file covers that second half.
//
// Note: ProgressState.rate is progress-per-*second*, but update() is fed
// progress/time as plain (progress, ms-timestamp) pairs — so a rate
// expressed "per ms" here is multiplied by 1000 before comparing against
// `state.rate`.

import test from 'node:test'
import assert from 'node:assert/strict'
import { createProgressTracker, formatDuration } from '../src/lib/progress.ts'

test('createProgressTracker rate estimate is stable regardless of call cadence', async (t) => {
  await t.test('a steady true rate converges to the same smoothed rate whether sampled once a second or in frequent small steps', () => {
    const RATE_PER_MS = 0.00005 // -> 0.05 progress/second, reaches 1.0 in 20_000ms

    // Tracker A: one call per second, for 20 seconds.
    const trackerA = createProgressTracker()
    let stateA
    for (let t = 1000; t <= 20_000; t += 1000) {
      stateA = trackerA.update(RATE_PER_MS * t, t)
    }

    // Tracker B: the exact same underlying progress curve, but sampled
    // every 100ms (10x the call frequency) over the same 20s window.
    const trackerB = createProgressTracker()
    let stateB
    for (let t = 100; t <= 20_000; t += 100) {
      stateB = trackerB.update(RATE_PER_MS * t, t)
    }

    const truePerSecond = RATE_PER_MS * 1000
    // Both must have converged close to the true rate, and therefore close
    // to each other — the old fixed-decay blend was systematically biased
    // by how often update() happened to be called, not just by the
    // underlying signal.
    assert.ok(Math.abs(stateA.rate - truePerSecond) / truePerSecond < 0.05, `rate A ${stateA.rate} vs true ${truePerSecond}`)
    assert.ok(Math.abs(stateB.rate - truePerSecond) / truePerSecond < 0.05, `rate B ${stateB.rate} vs true ${truePerSecond}`)
  })

  await t.test('a single short, anomalously-fast-looking sample does not swing the rate by a large factor', () => {
    // Establish a stable rate over several normal, evenly-spaced samples.
    // (Regression-sensitive: with the old fixed `smoothedRate*0.7 +
    // sampleRate*0.3` blend, a sample covering only 200ms got the same 30%
    // weight as one covering a full second — turning a single oddly-timed
    // websocket message (e.g. a burst of iterations reported together, or
    // one delayed by a GC pause) into a several-x jump in the displayed
    // rate/ETA. The time-constant blend weights a 200ms sample by how
    // little of the last ~4s (RATE_TIME_CONSTANT_MS) it actually covers,
    // instead of by a flat per-call fraction.)
    const tracker = createProgressTracker()
    let t = 0
    let progress = 0
    const ratePerMs = 0.0002 // -> 0.2 progress/second
    let state
    for (let i = 0; i < 10; i++) {
      t += 1000
      progress += ratePerMs * 1000
      state = tracker.update(progress, t)
    }
    const rateBefore = state.rate
    assert.ok(Math.abs(rateBefore - 0.2) / 0.2 < 0.05, `expected the rate to have converged near 0.2, got ${rateBefore}`)

    // One short (200ms — past TIME_THRESHOLD, so it isn't gated out
    // entirely), anomalous sample reporting an instantaneous rate 10x the
    // established one.
    t += 200
    progress += 2.0 * 0.2 // dp such that dp / (200/1000) == 2.0 progress/second
    state = tracker.update(progress, t)

    const ratio = state.rate / rateBefore
    assert.ok(ratio > 0.5 && ratio < 2, `expected a bounded change, got rate ${rateBefore} -> ${state.rate} (x${ratio})`)
  })

  await t.test('the tracker recovers a stable ETA reading over further normal samples after a burst', () => {
    const tracker = createProgressTracker()
    let t = 0
    let progress = 0
    const ratePerMs = 0.00002
    for (let i = 0; i < 10; i++) {
      t += 1000
      progress += ratePerMs * 1000
      tracker.update(progress, t)
    }
    // A short burst of rapid, small-progress samples.
    for (let i = 0; i < 5; i++) {
      t += 20
      progress += ratePerMs * 20
      tracker.update(progress, t)
    }
    // Back to a steady 1s cadence at the same underlying rate — the ETA
    // readings across these should not keep jumping around.
    const etas = []
    for (let i = 0; i < 5; i++) {
      t += 1000
      progress += ratePerMs * 1000
      etas.push(tracker.update(progress, t).eta)
    }
    for (let i = 1; i < etas.length; i++) {
      const ratio = etas[i] / etas[i - 1]
      assert.ok(ratio > 0.8 && ratio < 1.25, `eta jumped from ${etas[i - 1]} to ${etas[i]} (x${ratio})`)
    }
  })
})

test('formatDuration', () => {
  assert.equal(formatDuration(0), '0s')
  assert.equal(formatDuration(45_000), '45s')
  assert.equal(formatDuration(65_000), '1m 5s')
  assert.equal(formatDuration(NaN), '...')
  assert.equal(formatDuration(-5), '...')
})
