// Test runner: node --test tests/image-edit.test.mjs
//
// The image editor's pixel work: colour spaces, adjustments, selections,
// colour reduction and the layer compositing. The editor UI is covered by
// tests/features/image-editor.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import { gaussianBlur, labToRgb, resizeChannel, rgbToLab } from '../src/lib/imageMath.ts'
import {
  applyAdjustments,
  autoLevels,
  buildLuts,
  curveLut,
  defaultAdjustments,
  histogram,
  isIdentity,
} from '../src/lib/imageAdjust.ts'
import {
  brushStroke,
  combine,
  ellipsePoints,
  featherMask,
  growMask,
  invertMask,
  magicWand,
  maskOutline,
  maskShare,
  polygonMask,
  rectPoints,
} from '../src/lib/selection.ts'
import { analysisLab, applyPalette, cleanupIslands, cutPalette, prepareReduce } from '../src/lib/colorReduce.ts'
import { ReduceCache, editChangesNothing, fitPixels, newLayer, nextLayerName, parseEdit, renderLayers, savedEdit } from '../src/lib/imageEdit.ts'

/** A w×h RGBA picture from a function of (x, y) -> [r, g, b]. */
function picture(w, h, fn) {
  const rgba = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fn(x, y)
      const p = (y * w + x) * 4
      rgba[p] = r
      rgba[p + 1] = g
      rgba[p + 2] = b
      rgba[p + 3] = 255
    }
  }
  return rgba
}

const px = (rgba, w, x, y) => Array.from(rgba.slice((y * w + x) * 4, (y * w + x) * 4 + 3))

/** Deterministic noise, -amp..amp. */
function noise(seed, amp) {
  let s = seed
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return Math.round(((s / 0x7fffffff) * 2 - 1) * amp)
  }
}

// ---- colour spaces ----

test('Lab of black, white and back again', () => {
  const lab = [0, 0, 0]
  rgbToLab(255, 255, 255, lab)
  assert.ok(Math.abs(lab[0] - 100) < 0.01 && Math.abs(lab[1]) < 0.01 && Math.abs(lab[2]) < 0.01)
  rgbToLab(0, 0, 0, lab)
  assert.ok(Math.abs(lab[0]) < 0.01)
  for (const c of [[12, 200, 77], [255, 0, 0], [128, 128, 128], [3, 4, 250]]) {
    rgbToLab(c[0], c[1], c[2], lab)
    const back = [0, 0, 0]
    labToRgb(lab[0], lab[1], lab[2], back)
    assert.deepEqual(back, c)
  }
})

test('a blur keeps a flat picture flat and spreads a dot', () => {
  const flat = new Float32Array(25).fill(7)
  gaussianBlur(flat, 5, 5, 2)
  assert.ok(flat.every((v) => Math.abs(v - 7) < 1e-4))
  const dot = new Float32Array(81)
  dot[40] = 100
  gaussianBlur(dot, 9, 9, 1)
  assert.ok(dot[40] < 100 && dot[41] > 0 && dot[0] < dot[41])
})

test('a mask resized up keeps its shape', () => {
  const small = new Uint8Array([0, 255, 0, 255])
  const big = resizeChannel(small, 2, 2, 4, 4)
  assert.equal(big[0], 0)
  assert.equal(big[3], 255)
})

// ---- adjustments ----

test('default adjustments change nothing', () => {
  const a = defaultAdjustments()
  assert.equal(isIdentity(a), true)
  const [r, g, b] = buildLuts(a)
  for (let i = 0; i < 256; i++) assert.ok(r[i] === i && g[i] === i && b[i] === i)
  const src = picture(4, 4, (x, y) => [x * 60, y * 60, 100])
  assert.deepEqual(Array.from(applyAdjustments(src, 4, 4, a)), Array.from(src))
})

test('each light slider moves the tones the way it says', () => {
  const lut = (changes) => buildLuts({ ...defaultAdjustments(), ...changes })[0]
  assert.ok(lut({ exposure: 1 })[100] > 120)
  assert.ok(lut({ exposure: -1 })[100] < 80)
  assert.ok(lut({ brightness: 50 })[128] > 140)
  // Contrast: darks darker and lights lighter, mid grey stays.
  const c = lut({ contrast: 60 })
  assert.ok(c[50] < 50 && c[200] > 200 && Math.abs(c[128] - 128) <= 2)
  // Shadows lift the darks but hardly the lights; highlights the other way.
  const s = lut({ shadows: 80 })
  assert.ok(s[40] > 50 && s[230] - 230 <= 2)
  const hl = lut({ highlights: -80 })
  assert.ok(hl[230] < 220 && Math.abs(hl[30] - 30) <= 2)
  // Black and white points.
  assert.ok(lut({ blacks: -100 })[30] < 10)
  assert.ok(lut({ whites: 100 })[220] === 255)
  // No slider ever turns a darker value into a lighter one.
  for (const changes of [{ contrast: 100 }, { shadows: 100, highlights: -100 }, { blacks: 100, whites: -100, brightness: -100 }]) {
    const l = lut(changes)
    for (let i = 1; i < 256; i++) assert.ok(l[i] >= l[i - 1], JSON.stringify(changes))
  }
})

test('levels and curves', () => {
  const a = defaultAdjustments()
  a.levels = { inBlack: 50, inWhite: 200, gamma: 1, outBlack: 0, outWhite: 255 }
  const [r] = buildLuts(a)
  assert.equal(r[50], 0)
  assert.equal(r[200], 255)
  assert.ok(Math.abs(r[125] - 128) <= 1)
  // A curve through a raised midpoint, smooth and without overshoot.
  const lut = curveLut([[0, 0], [128, 180], [255, 255]])
  assert.ok(Math.abs(lut[128] - 180) < 0.5)
  for (let i = 1; i < 256; i++) assert.ok(lut[i] >= lut[i - 1] && lut[i] <= 255)
  // An inverted curve is allowed (a curve may do that on purpose).
  const b = defaultAdjustments()
  b.curves.master = [[0, 255], [255, 0]]
  assert.equal(buildLuts(b)[0][0], 255)
  assert.equal(buildLuts(b)[0][255], 0)
})

test('saturation, hue and vibrance', () => {
  const src = picture(1, 1, () => [200, 60, 40])
  const grey = applyAdjustments(src, 1, 1, { ...defaultAdjustments(), saturation: -100 })
  assert.ok(Math.abs(grey[0] - grey[1]) <= 1 && Math.abs(grey[1] - grey[2]) <= 1)
  const turned = applyAdjustments(src, 1, 1, { ...defaultAdjustments(), hue: 180 })
  assert.ok(turned[2] > turned[0], 'red turns blue-ish')
  // Vibrance boosts a dull colour more than a strong one.
  const dull = picture(1, 1, () => [140, 120, 110])
  const strong = picture(1, 1, () => [240, 20, 10])
  const spread = (p) => Math.max(p[0], p[1], p[2]) - Math.min(p[0], p[1], p[2])
  const vib = { ...defaultAdjustments(), vibrance: 80 }
  const dullGain = spread(applyAdjustments(dull, 1, 1, vib)) / spread(dull)
  const strongGain = spread(applyAdjustments(strong, 1, 1, vib)) / spread(strong)
  assert.ok(dullGain > strongGain)
})

test('soften smooths noise, sharpen adds edge contrast, alpha is kept', () => {
  const rnd = noise(1, 40)
  const src = picture(32, 32, () => [128 + rnd(), 128, 128])
  src[3] = 7
  const varOf = (p) => {
    let m = 0
    for (let i = 0; i < p.length; i += 4) m += p[i]
    m /= p.length / 4
    let v = 0
    for (let i = 0; i < p.length; i += 4) v += (p[i] - m) ** 2
    return v
  }
  const soft = applyAdjustments(src, 32, 32, { ...defaultAdjustments(), soften: 60 }, undefined, 1)
  assert.ok(varOf(soft) < varOf(src) / 3)
  assert.equal(soft[3], 7)
  const edge = picture(16, 1, (x) => (x < 8 ? [100, 100, 100] : [150, 150, 150]))
  const sharp = applyAdjustments(edge, 16, 1, { ...defaultAdjustments(), sharpen: 100 })
  assert.ok(sharp[7 * 4] < 100 && sharp[8 * 4] > 150)
})

test('auto levels stretch to the picture\'s own range', () => {
  const src = picture(10, 10, (x) => [60 + x * 10, 60 + x * 10, 60 + x * 10])
  const l = autoLevels(histogram(src).l)
  assert.ok(l.inBlack >= 55 && l.inBlack <= 65)
  assert.ok(l.inWhite >= 145 && l.inWhite <= 155)
})

// ---- selections ----

test('shapes select their inside', () => {
  const rect = polygonMask(10, 10, rectPoints(2, 2, 6, 5))
  assert.equal(rect.filter((v) => v === 255).length, 12)
  assert.equal(rect[2 * 10 + 2], 255)
  assert.equal(rect[5 * 10 + 6], 0)
  const ellipse = polygonMask(100, 100, ellipsePoints(10, 20, 90, 80))
  const area = ellipse.filter((v) => v === 255).length
  assert.ok(Math.abs(area - Math.PI * 40 * 30) < 60)
})

test('replace, add and subtract', () => {
  const a = polygonMask(10, 1, rectPoints(0, 0, 5, 1))
  const b = polygonMask(10, 1, rectPoints(3, 0, 8, 1))
  const t = a.slice()
  combine(t, b, 'add')
  assert.equal(maskShare(t), 0.8)
  combine(t, a, 'subtract')
  assert.deepEqual(Array.from(t), [0, 0, 0, 0, 0, 255, 255, 255, 0, 0])
  combine(t, a, 'replace')
  assert.deepEqual(t, a)
  invertMask(t)
  assert.equal(maskShare(t), 0.5)
})

test('the brush paints a soft line and the eraser removes it', () => {
  const m = new Uint8Array(40 * 20)
  brushStroke(m, 40, 20, 5, 10, 35, 10, 4, 0.5, false)
  assert.equal(m[10 * 40 + 20], 255)
  assert.ok(m[13 * 40 + 20] > 0 && m[13 * 40 + 20] < 255, 'soft edge')
  assert.equal(m[0], 0)
  brushStroke(m, 40, 20, 5, 10, 35, 10, 4, 1, true)
  assert.equal(m[10 * 40 + 20], 0)
})

test('the magic wand picks similar colours, joined or anywhere', () => {
  // Two red squares apart on blue, one with a slightly different red.
  const src = picture(20, 10, (x, y) => (y >= 2 && y < 8 && ((x >= 1 && x < 7) || (x >= 12 && x < 18)) ? (x < 10 ? [220, 30, 30] : [225, 35, 28]) : [20, 40, 200]))
  const lab = analysisLab(src, 20, 10, 0)
  const joined = magicWand(lab, 20, 10, 3, 4, 10, true)
  assert.equal(joined[4 * 20 + 3], 255)
  assert.equal(joined[4 * 20 + 14], 0, 'the other square is not joined')
  assert.equal(joined[0], 0)
  const anywhere = magicWand(lab, 20, 10, 3, 4, 10, false)
  assert.equal(anywhere[4 * 20 + 14], 255)
  assert.equal(anywhere[0], 0)
  // A tight tolerance leaves the slightly different red out.
  const tight = magicWand(lab, 20, 10, 3, 4, 0.5, false)
  assert.equal(tight[4 * 20 + 14], 0)
})

test('the outline traces the selection\'s edge, straight runs as one line', () => {
  const rect = polygonMask(10, 10, rectPoints(2, 2, 6, 5))
  const { path, segments } = maskOutline(rect, 10, 10)
  assert.equal(segments, 4)
  for (const part of ['M2 2H6', 'M2 5H6', 'M2 2V5', 'M6 2V5']) assert.ok(path.includes(part), part)
  // A selection touching the picture's border is outlined along it too.
  const all = new Uint8Array(4 * 3).fill(255)
  assert.equal(maskOutline(all, 4, 3).segments, 4)
  // Soft selections are outlined where they're half selected.
  const soft = new Uint8Array([0, 100, 200, 255])
  assert.equal(maskOutline(soft, 4, 1).path, 'M2 0H4M2 1H4M2 0V1M4 0V1')
  assert.deepEqual(maskOutline(new Uint8Array(9), 3, 3), { path: '', segments: 0 })
})

test('grow, shrink and soften a selection', () => {
  const m = polygonMask(30, 30, rectPoints(10, 10, 20, 20))
  assert.ok(maskShare(growMask(m, 30, 30, 4)) > maskShare(m))
  assert.ok(maskShare(growMask(m, 30, 30, -4)) < maskShare(m))
  const soft = featherMask(m, 30, 30, 2)
  assert.ok(soft[15 * 30 + 10] > 0 && soft[15 * 30 + 10] < 255)
  assert.ok(soft[15 * 30 + 15] > 240)
})

// ---- colour reduction ----

/** Three flat colours with noise: left red, middle green, right grey. */
const threeColours = () => {
  const rnd = noise(3, 6)
  return picture(60, 30, (x) => {
    const base = x < 20 ? [200, 40, 40] : x < 40 ? [40, 160, 60] : [128, 128, 128]
    return base.map((v) => v + rnd())
  })
}

test('three noisy colours come back as three clean ones', () => {
  const src = threeColours()
  const lab = analysisLab(src, 60, 30, 0.06)
  const prep = prepareReduce(lab, null, 60 * 30)
  const palette = cutPalette(prep, 3)
  assert.equal(palette.count, 3)
  const out = src.slice()
  applyPalette(src, 60, 30, palette, null, out, 0.06, lab)
  // Every pixel of an area got the same colour, near the original.
  for (const [x0, x1, colour] of [[0, 20, [200, 40, 40]], [20, 40, [40, 160, 60]], [40, 60, [128, 128, 128]]]) {
    const first = px(out, 60, x0 + 2, 15)
    for (let x = x0 + 2; x < x1 - 2; x++) for (let y = 2; y < 28; y++) assert.deepEqual(px(out, 60, x, y), first)
    assert.ok(first.every((v, i) => Math.abs(v - colour[i]) < 12), `${first} vs ${colour}`)
  }
  // The grey stays grey and is listed as one.
  assert.equal(palette.gray.filter(Boolean).length, 1)
})

test('fewer colours merge the closest ones; the tree is cut, not rebuilt', () => {
  const src = threeColours()
  const prep = prepareReduce(analysisLab(src, 60, 30, 0.06), null, 60 * 30)
  // Two colours: the grey is never averaged into a colour.
  const two = cutPalette(prep, 2)
  assert.equal(two.count, 2)
  assert.equal(two.gray.filter(Boolean).length, 1)
  const many = cutPalette(prep, 64)
  assert.ok(many.count >= 3)
  assert.ok(Math.abs(many.share.reduce((a, b) => a + b, 0) - 1) < 1e-6)
})

test('only the selection is reduced', () => {
  const src = threeColours()
  const weights = new Uint8Array(60 * 30)
  for (let y = 0; y < 30; y++) for (let x = 0; x < 20; x++) weights[y * 60 + x] = 255
  const lab = analysisLab(src, 60, 30, 0.06)
  const palette = cutPalette(prepareReduce(lab, weights, 60 * 30), 4)
  // Only red was looked at.
  assert.ok(palette.count >= 1)
  for (let c = 0; c < palette.count; c++) assert.ok(palette.rgb[c * 3] > 150)
  const out = src.slice()
  applyPalette(src, 60, 30, palette, weights, out, 0.06, lab)
  assert.deepEqual(px(out, 60, 30, 10), px(src, 60, 30, 10))
})

test('a one-pixel speck joins its surroundings', () => {
  const labels = new Int16Array(25).fill(1)
  labels[12] = 0
  labels[0] = -1
  cleanupIslands(labels, 5, 5, 3)
  assert.equal(labels[12], 1)
  assert.equal(labels[0], -1, 'unselected pixels stay out')
})

// ---- layers ----

test('layers: whole picture, a selection, soft edges, hidden', () => {
  const src = picture(20, 20, () => [100, 100, 100])
  const all = newLayer('Light')
  all.adjust.exposure = 1
  const { rgba: brighter } = renderLayers({ rgba: src, w: 20, h: 20, layers: [all], masks: new Map() })
  assert.ok(brighter[0] > 120)
  assert.equal(src[0], 100, 'the source is untouched')

  const sel = newLayer('Left', 'mask')
  sel.adjust.exposure = -2
  sel.feather = 0
  const mask = polygonMask(20, 20, rectPoints(0, 0, 10, 20))
  const { rgba: half } = renderLayers({ rgba: src, w: 20, h: 20, layers: [sel], masks: new Map([[sel.id, mask]]) })
  assert.ok(px(half, 20, 2, 5)[0] < 60)
  assert.deepEqual(px(half, 20, 15, 5), [100, 100, 100])

  // Soft edges are sized relative to the picture (so the preview matches
  // the full picture): on a 400 px one, half softness is a few pixels.
  const big = picture(400, 40, () => [100, 100, 100])
  const bigMask = polygonMask(400, 40, rectPoints(0, 0, 200, 40))
  sel.feather = 50
  const { rgba: soft } = renderLayers({ rgba: big, w: 400, h: 40, layers: [sel], masks: new Map([[sel.id, bigMask]]) })
  const edge = px(soft, 400, 200, 20)[0]
  assert.ok(edge > 40 && edge < 100, `soft edge ${edge}`)
  assert.ok(px(soft, 400, 20, 20)[0] < 60)

  sel.visible = false
  const { rgba: hidden } = renderLayers({ rgba: src, w: 20, h: 20, layers: [sel], masks: new Map([[sel.id, mask]]) })
  assert.deepEqual(Array.from(hidden), Array.from(src))
})

test('layers stack in order', () => {
  const src = picture(4, 4, () => [100, 100, 100])
  const a = newLayer('A')
  a.adjust.saturation = -100
  const b = newLayer('B')
  b.adjust.temperature = 60
  const ab = renderLayers({ rgba: src, w: 4, h: 4, layers: [a, b], masks: new Map() }).rgba
  const ba = renderLayers({ rgba: src, w: 4, h: 4, layers: [b, a], masks: new Map() }).rgba
  assert.ok(ab[0] > ab[2], 'warmed after desaturating: still warm')
  assert.ok(Math.abs(ba[0] - ba[2]) <= 1, 'desaturated last: grey')
})

test('a reducing layer reports its palette and reuses a given one', () => {
  const src = threeColours()
  const layer = newLayer('Flat')
  layer.adjust.reduce = { enabled: true, colors: 3 }
  const cache = new ReduceCache()
  const first = renderLayers({ rgba: src, w: 60, h: 30, layers: [layer], masks: new Map(), cache })
  const palette = first.palettes.get(layer.id)
  assert.equal(palette.count, 3)
  // The full picture gets the preview's palette.
  const again = renderLayers({ rgba: src, w: 60, h: 30, layers: [layer], masks: new Map(), palettes: first.palettes })
  assert.equal(again.palettes.get(layer.id), palette)
  assert.deepEqual(Array.from(again.rgba), Array.from(first.rgba))
})

test('the saved edit: nothing to save, names, and reading it back', () => {
  const layer = newLayer('Light')
  assert.equal(editChangesNothing([layer]), true)
  assert.equal(savedEdit('a.png', [layer]), null)
  layer.adjust.contrast = 20
  const saved = savedEdit('a.png', [layer])
  assert.equal(saved.source, 'a.png')
  // A copy: editing on doesn't change what was saved.
  layer.adjust.contrast = 50
  assert.equal(saved.layers[0].adjust.contrast, 20)
  assert.deepEqual(parseEdit(JSON.parse(JSON.stringify(saved))), saved)
  assert.equal(parseEdit(null), null)
  assert.equal(parseEdit({ version: 2, source: 'a', layers: [] }), null)
  // Older or partial layers get the defaults they lack.
  const partial = parseEdit({ version: 1, source: 'a.png', layers: [{ id: 'x', adjust: { contrast: 5 } }] })
  assert.equal(partial.layers[0].adjust.contrast, 5)
  assert.equal(partial.layers[0].adjust.levels.inWhite, 255)
  assert.equal(nextLayerName([{ name: 'Layer 1' }, { name: 'Layer 2' }], 'Layer'), 'Layer 3')
  assert.equal(nextLayerName([{ name: 'Layer 2' }], 'Layer'), 'Layer 1')
  assert.deepEqual(fitPixels(4000, 3000, 1_200_000), { w: 1265, h: 949 })
})
