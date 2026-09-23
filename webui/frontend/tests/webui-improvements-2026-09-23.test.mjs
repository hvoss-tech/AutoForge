// Test runner: node --test tests/webui-improvements-2026-09-23.test.mjs
//
// Pure logic behind the 2026-09-23 usability round: the Differences view's
// color math, the focus-area mask helpers and the color column's marker
// spacing. The DOM side is in tests/features/webui-improvements-2026-09-23.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  CLEARLY_DIFFERENT,
  DIFF_SCALE_FLOOR,
  boxBlur,
  comparisonSize,
  deltaE,
  describeMeanDifference,
  diffHeatmap,
  heatColor,
  percentile,
  rgbToLab,
} from '../src/lib/imageDiff.ts'
import {
  alphaToGreyscale,
  brushDiameter,
  clientToMask,
  greyscaleToAlpha,
  maskCanvasSize,
  maskHasPaint,
  paintedShare,
  sliderFromStrength,
  strengthFromSlider,
} from '../src/lib/focusMask.ts'
import { FULL_HANDLE_SPACING, compactHandleHeight, nearestNeighborGaps } from '../src/lib/colorStack.ts'

const rgba = (pixels) => new Uint8ClampedArray(pixels.flat())

// --- Differences view --------------------------------------------------------

test('rgbToLab matches reference values', () => {
  const [l0] = rgbToLab(0, 0, 0)
  const [l1, a1, b1] = rgbToLab(255, 255, 255)
  assert.ok(Math.abs(l0) < 1e-6)
  assert.ok(Math.abs(l1 - 100) < 0.01)
  assert.ok(Math.abs(a1) < 0.01 && Math.abs(b1) < 0.01)
  // sRGB red is about L 53.2, a 80.1, b 67.2.
  const [l, a, b] = rgbToLab(255, 0, 0)
  assert.ok(Math.abs(l - 53.24) < 0.1 && Math.abs(a - 80.09) < 0.2 && Math.abs(b - 67.2) < 0.2)
})

test('deltaE: identical colors are 0, black to white is 100', () => {
  assert.equal(deltaE(rgbToLab(10, 20, 30), rgbToLab(10, 20, 30)), 0)
  assert.ok(Math.abs(deltaE(rgbToLab(0, 0, 0), rgbToLab(255, 255, 255)) - 100) < 0.01)
})

test('heatColor runs from dark to light and clamps', () => {
  const lum = ([r, g, b]) => 0.299 * r + 0.587 * g + 0.114 * b
  let prev = -1
  for (let t = 0; t <= 1.0001; t += 0.1) {
    const l = lum(heatColor(t))
    assert.ok(l > prev, `brightness rises at t=${t.toFixed(1)}`)
    prev = l
  }
  assert.deepEqual(heatColor(-1), heatColor(0))
  assert.deepEqual(heatColor(2), heatColor(1))
})

test('diffHeatmap: identical images are a perfect match', () => {
  const img = rgba([[200, 30, 30, 255], [30, 200, 30, 255], [30, 30, 200, 255], [240, 240, 240, 255]])
  const { stats, rgba: out } = diffHeatmap(img, img, 2, 2)
  assert.equal(stats.mean, 0)
  assert.equal(stats.clearlyDifferent, 0)
  assert.equal(stats.scaleMax, DIFF_SCALE_FLOOR)
  assert.equal(out.length, 16)
})

test('diffHeatmap: counts clearly different pixels and skips transparent ones', () => {
  const original = rgba([[0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 255]])
  const result = rgba([[255, 255, 255, 255], [0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255]])
  const { stats, rgba: out } = diffHeatmap(original, result, 4, 1)
  // 3 compared pixels, one of them black-vs-white (ΔE 100).
  assert.ok(Math.abs(stats.mean - 100 / 3) < 0.01)
  assert.ok(Math.abs(stats.clearlyDifferent - 1 / 3) < 1e-9)
  assert.equal(out[2 * 4 + 3], 0, 'transparent in the original stays transparent')
  // The differing pixel is drawn brighter than the matching ones.
  const bright = (i) => out[i * 4] + out[i * 4 + 1] + out[i * 4 + 2]
  assert.ok(bright(0) > bright(1))
})

test('percentile and boxBlur', () => {
  const values = new Float32Array(Array.from({ length: 100 }, (_, i) => i))
  const p95 = percentile(values, 0.95)
  assert.ok(p95 >= 94 && p95 <= 96, `p95 ${p95}`)
  assert.equal(percentile(new Float32Array([NaN, NaN]), 0.5), 0)

  const img = new Float32Array([0, 0, 9, 0, 0])
  boxBlur(img, 5, 1, 1)
  assert.deepEqual(Array.from(img), [0, 3, 3, 3, 0])
  const withHole = new Float32Array([NaN, 6, 0])
  boxBlur(withHole, 3, 1, 1)
  assert.ok(Number.isNaN(withHole[0]), 'transparent stays out')
  assert.equal(withHole[1], 3)
})

test('describeMeanDifference and comparisonSize', () => {
  assert.equal(describeMeanDifference(1), 'Very close match')
  assert.equal(describeMeanDifference(CLEARLY_DIFFERENT + 1), 'Large differences')
  assert.deepEqual(comparisonSize(400, 200), { width: 400, height: 200 })
  assert.deepEqual(comparisonSize(1800, 900), { width: 900, height: 450 })
})

// --- Focus areas ---------------------------------------------------------------

test('maskCanvasSize keeps the aspect and caps the longest side', () => {
  assert.deepEqual(maskCanvasSize(640, 480), { width: 640, height: 480 })
  assert.deepEqual(maskCanvasSize(4096, 2048), { width: 1024, height: 512 })
})

test('brushDiameter is a share of the longest side', () => {
  assert.equal(brushDiameter(10, 1000, 500), 100)
  assert.equal(brushDiameter(0, 10, 10), 1)
})

test('clientToMask maps through object-fit: contain letterboxing', () => {
  // A 200x100 mask shown in a 400x400 box: scale 2, 100px bars top and bottom.
  const rect = { left: 10, top: 20, width: 400, height: 400 }
  const p = clientToMask(10 + 200, 20 + 100 + 100, rect, 200, 100)
  assert.deepEqual(p, { x: 100, y: 50, scale: 2 })
  const corner = clientToMask(10, 20 + 100, rect, 200, 100)
  assert.equal(corner.x, 0)
  assert.equal(corner.y, 0)
  assert.equal(clientToMask(0, 0, { left: 0, top: 0, width: 0, height: 0 }, 10, 10), null)
})

test('mask round trip: painted alpha to opaque greyscale and back', () => {
  const painted = rgba([[255, 255, 255, 255], [255, 255, 255, 0], [255, 255, 255, 128]])
  assert.equal(maskHasPaint(painted), true)
  assert.ok(Math.abs(paintedShare(painted) - 2 / 3) < 1e-9)
  const grey = alphaToGreyscale(new Uint8ClampedArray(painted))
  assert.deepEqual(Array.from(grey), [255, 255, 255, 255, 0, 0, 0, 255, 128, 128, 128, 255])
  const back = greyscaleToAlpha(new Uint8ClampedArray(grey))
  assert.deepEqual(Array.from(back), Array.from(painted))
  // An eraser's anti-aliased fringe is not paint.
  assert.equal(maskHasPaint(rgba([[255, 255, 255, 0], [255, 255, 255, 5], [255, 255, 255, 100]])), false)
})

test('focus strength slider: logarithmic 2x-100x, default 10x round-trips', () => {
  assert.equal(strengthFromSlider(0), 2)
  assert.equal(strengthFromSlider(100), 100)
  assert.equal(strengthFromSlider(sliderFromStrength(10)), 10)
  assert.equal(strengthFromSlider(-5), 2)
  assert.equal(strengthFromSlider(500), 100)
  // Monotonic, whole numbers, rounded to 5 above 20.
  let prev = 0
  for (let p = 0; p <= 100; p++) {
    const s = strengthFromSlider(p)
    assert.ok(s >= prev && Number.isInteger(s))
    if (s > 20) assert.equal(s % 5, 0)
    prev = s
  }
  // An unset strength shows as the default.
  assert.equal(sliderFromStrength(undefined), sliderFromStrength(10))
})

// --- Color column markers --------------------------------------------------------

test('nearestNeighborGaps and compactHandleHeight', () => {
  assert.deepEqual(nearestNeighborGaps([]), [])
  assert.deepEqual(nearestNeighborGaps([50]), [Infinity])
  assert.deepEqual(nearestNeighborGaps([100, 90, 60]), [10, 10, 30])
  assert.ok(10 < FULL_HANDLE_SPACING && 30 >= FULL_HANDLE_SPACING)
  // Leaves a pixel between neighbours, never vanishes, never grows past 12.
  assert.equal(compactHandleHeight(10), 9)
  assert.equal(compactHandleHeight(1.5), 3)
  assert.equal(compactHandleHeight(19), 12)
})
