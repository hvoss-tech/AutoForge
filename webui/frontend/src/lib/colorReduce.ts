/** Colour reduction: the picture (or a selection of it) in a few flat
 * colours, chosen by how the eye groups them.
 *
 * 1. The picture in Lab, lightly blurred, so noise doesn't become colours.
 * 2. Its colours gathered into small Lab bins, each weighted by how many
 *    pixels it holds (and by the selection, if there is one).
 * 3. Greys and colours kept apart: a grey is never averaged into a colour.
 * 4. The bins clustered bottom-up (weighted average linkage) into one merge
 *    tree, built once; any number of colours is then a cheap cut of it.
 * 5. Colours that would cover almost nothing join their nearest neighbour.
 * 6. Every pixel takes its nearest colour, then a vote among its neighbours
 *    (weighed by how well each fits) settles noisy edges, and specks smaller
 *    than a few pixels join the area around them.
 */

import { clamp, gaussianBlur, labToRgb, rgbaToLab } from './imageMath.ts'

export const REDUCE_MIN_COLORS = 2
export const REDUCE_MAX_COLORS = 64

/** Bins the merge tree is built from (its cost grows with the square). */
const MAX_BINS = 1800
/** Lab chroma below which a colour counts as grey. */
const GRAY_CHROMA = 7
/** A colour covering less than this share of the picture joins its nearest. */
const MIN_SHARE = 0.002

export interface Palette {
  /** Lab, 3 per colour. */
  lab: Float32Array
  /** sRGB, 3 per colour. */
  rgb: Uint8Array
  gray: boolean[]
  /** Share of the (selected) picture each colour covers. */
  share: number[]
  count: number
}

export interface Prepared {
  binLab: Float32Array
  binWeight: Float64Array
  binGray: Uint8Array
  bins: number
  totalWeight: number
  /** Every merge of the tree, lowest first: bins a and b's clusters join at height h. */
  merges: { a: number; b: number; h: number }[]
  populations: number
}

/** The picture in Lab as the analysis sees it: blurred so sensor noise and
 * compression artefacts don't turn into colours of their own. `scale` is
 * the picture's size relative to a 1000 px one. */
export function analysisLab(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number, scale = 1): Float32Array {
  const lab = rgbaToLab(rgba, w * h)
  gaussianBlur(lab, w, h, clamp(1.0 * scale, 0.5, 3), 3)
  return lab
}

/** Bins and the merge tree of the pixels with weight > 0 (`weights` 0..255
 * per pixel, or null for all of them). */
export function prepareReduce(lab: Float32Array, weights: Uint8Array | null, count: number): Prepared {
  let step = 2
  let keys: Map<number, number>
  let sums: number[]
  for (;;) {
    keys = new Map()
    sums = []
    for (let i = 0; i < count; i++) {
      const wt = weights ? weights[i] / 255 : 1
      if (wt <= 0) continue
      const l = lab[i * 3]
      const a = lab[i * 3 + 1]
      const b = lab[i * 3 + 2]
      const key = (Math.floor(l / step) * 4096 + Math.floor((a + 160) / step)) * 4096 + Math.floor((b + 160) / step)
      let bin = keys.get(key)
      if (bin === undefined) {
        bin = keys.size
        keys.set(key, bin)
        sums.push(0, 0, 0, 0)
        if (keys.size > MAX_BINS) break
      }
      sums[bin * 4] += l * wt
      sums[bin * 4 + 1] += a * wt
      sums[bin * 4 + 2] += b * wt
      sums[bin * 4 + 3] += wt
    }
    if (keys.size <= MAX_BINS) break
    step *= 1.3
  }
  const bins = keys.size
  const binLab = new Float32Array(bins * 3)
  const binWeight = new Float64Array(bins)
  const binGray = new Uint8Array(bins)
  let totalWeight = 0
  for (let k = 0; k < bins; k++) {
    const wt = sums[k * 4 + 3]
    binWeight[k] = wt
    totalWeight += wt
    binLab[k * 3] = sums[k * 4] / wt
    binLab[k * 3 + 1] = sums[k * 4 + 1] / wt
    binLab[k * 3 + 2] = sums[k * 4 + 2] / wt
    binGray[k] = Math.hypot(binLab[k * 3 + 1], binLab[k * 3 + 2]) < GRAY_CHROMA ? 1 : 0
  }
  const grays: number[] = []
  const colours: number[] = []
  for (let k = 0; k < bins; k++) (binGray[k] ? grays : colours).push(k)
  const merges = [...mergeTree(binLab, binWeight, grays), ...mergeTree(binLab, binWeight, colours)].sort((p, q) => p.h - q.h)
  return { binLab, binWeight, binGray, bins, totalWeight, merges, populations: (grays.length > 0 ? 1 : 0) + (colours.length > 0 ? 1 : 0) }
}

/** Weighted average-linkage clustering of the bins `idx` (nearest-neighbour
 * chain: O(n²) time and memory). Merges name bins by their global index. */
function mergeTree(binLab: Float32Array, binWeight: Float64Array, idx: number[]): { a: number; b: number; h: number }[] {
  const m = idx.length
  if (m < 2) return []
  const d = new Float32Array(m * m)
  for (let i = 0; i < m; i++) {
    const pi = idx[i] * 3
    for (let j = i + 1; j < m; j++) {
      const pj = idx[j] * 3
      const v = Math.hypot(binLab[pi] - binLab[pj], binLab[pi + 1] - binLab[pj + 1], binLab[pi + 2] - binLab[pj + 2])
      d[i * m + j] = v
      d[j * m + i] = v
    }
  }
  const size = new Float64Array(m)
  for (let i = 0; i < m; i++) size[i] = binWeight[idx[i]]
  const active = new Uint8Array(m).fill(1)
  const chain = new Int32Array(m)
  let top = 0
  let remaining = m
  let firstActive = 0
  const merges: { a: number; b: number; h: number }[] = []
  while (remaining > 1) {
    if (top === 0) {
      while (!active[firstActive]) firstActive++
      chain[top++] = firstActive
    }
    const a = chain[top - 1]
    const prev = top > 1 ? chain[top - 2] : -1
    let best = prev
    let bestD = prev >= 0 ? d[a * m + prev] : Infinity
    const row = a * m
    for (let c = 0; c < m; c++) {
      if (!active[c] || c === a) continue
      if (d[row + c] < bestD) {
        bestD = d[row + c]
        best = c
      }
    }
    if (best === prev) {
      top -= 2
      merges.push({ a: idx[a], b: idx[prev], h: bestD })
      const wa = size[a]
      const wb = size[prev]
      for (let c = 0; c < m; c++) {
        if (!active[c] || c === a || c === prev) continue
        const v = (wa * d[a * m + c] + wb * d[prev * m + c]) / (wa + wb)
        d[a * m + c] = v
        d[c * m + a] = v
      }
      size[a] = wa + wb
      active[prev] = 0
      remaining--
    } else {
      chain[top++] = best
    }
  }
  return merges
}

/** The palette of `colors` colours (fewer if the picture has fewer), cut
 * from the merge tree. */
export function cutPalette(prep: Prepared, colors: number): Palette {
  const parent = new Int32Array(prep.bins)
  for (let i = 0; i < prep.bins; i++) parent[i] = i
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  const target = Math.max(Math.round(clamp(colors, REDUCE_MIN_COLORS, REDUCE_MAX_COLORS)), prep.populations)
  let clusters = prep.bins
  for (const { a, b } of prep.merges) {
    if (clusters <= target) break
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) {
      parent[rb] = ra
      clusters--
    }
  }
  // Each cluster's weight and weighted mean colour.
  const index = new Map<number, number>()
  const acc: number[] = []
  const gray: boolean[] = []
  for (let k = 0; k < prep.bins; k++) {
    const root = find(k)
    let c = index.get(root)
    if (c === undefined) {
      c = index.size
      index.set(root, c)
      acc.push(0, 0, 0, 0)
      gray.push(prep.binGray[k] === 1)
    }
    const wt = prep.binWeight[k]
    acc[c * 4] += prep.binLab[k * 3] * wt
    acc[c * 4 + 1] += prep.binLab[k * 3 + 1] * wt
    acc[c * 4 + 2] += prep.binLab[k * 3 + 2] * wt
    acc[c * 4 + 3] += wt
  }
  let list = Array.from({ length: index.size }, (_, c) => ({
    l: acc[c * 4] / acc[c * 4 + 3],
    a: acc[c * 4 + 1] / acc[c * 4 + 3],
    b: acc[c * 4 + 2] / acc[c * 4 + 3],
    w: acc[c * 4 + 3],
    gray: gray[c],
  }))
  // Colours covering almost nothing join their nearest neighbour.
  const minWeight = prep.totalWeight * MIN_SHARE
  for (;;) {
    if (list.length <= 1) break
    let tiny = -1
    for (let i = 0; i < list.length; i++) if (list[i].w < minWeight && (tiny < 0 || list[i].w < list[tiny].w)) tiny = i
    if (tiny < 0) break
    const t = list[tiny]
    let near = -1
    let nearD = Infinity
    for (let i = 0; i < list.length; i++) {
      if (i === tiny) continue
      const dd = Math.hypot(list[i].l - t.l, list[i].a - t.a, list[i].b - t.b)
      if (dd < nearD) {
        nearD = dd
        near = i
      }
    }
    const n = list[near]
    const w = n.w + t.w
    list[near] = { l: (n.l * n.w + t.l * t.w) / w, a: (n.a * n.w + t.a * t.w) / w, b: (n.b * n.w + t.b * t.w) / w, w, gray: n.gray }
    list = list.filter((_, i) => i !== tiny)
  }
  // Darkest first, so the swatches read in order.
  list.sort((p, q) => p.l - q.l)
  const count = list.length
  const lab = new Float32Array(count * 3)
  const rgb = new Uint8Array(count * 3)
  list.forEach((c, i) => {
    lab[i * 3] = c.l
    lab[i * 3 + 1] = c.a
    lab[i * 3 + 2] = c.b
    labToRgb(c.l, c.a, c.b, rgb, i * 3)
  })
  return { lab, rgb, gray: list.map((c) => c.gray), share: list.map((c) => (prep.totalWeight > 0 ? c.w / prep.totalWeight : 0)), count }
}

/** Each pixel's nearest palette colour (-1 where `weights` is 0). */
export function assignLabels(lab: Float32Array, count: number, palette: Palette, weights: Uint8Array | null): Int16Array {
  const labels = new Int16Array(count)
  const k = palette.count
  const p = palette.lab
  for (let i = 0; i < count; i++) {
    if (weights && weights[i] === 0) {
      labels[i] = -1
      continue
    }
    const l = lab[i * 3]
    const a = lab[i * 3 + 1]
    const b = lab[i * 3 + 2]
    let best = 0
    let bestD = Infinity
    for (let c = 0; c < k; c++) {
      const dl = l - p[c * 3]
      const da = a - p[c * 3 + 1]
      const db = b - p[c * 3 + 2]
      const dd = dl * dl + da * da + db * db
      if (dd < bestD) {
        bestD = dd
        best = c
      }
    }
    labels[i] = best
  }
  return labels
}

/** Each pixel re-decided by its neighbours: their labels, weighed by
 * distance and by how well each label's colour fits this pixel. Smooths
 * the salt-and-pepper a hard nearest-colour choice leaves on gradients. */
export function voteLabels(labels: Int16Array, lab: Float32Array, w: number, h: number, palette: Palette, scale = 1): Int16Array {
  const sigmaS = clamp(1.0 * scale, 0.8, 2)
  const radius = Math.min(2, Math.ceil(sigmaS * 1.5))
  const sigmaC = 12
  const spatial: number[] = []
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) spatial.push(Math.exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS)))
  // Each candidate label's neighbourhood vote at the current pixel.
  const score = new Float32Array(palette.count)
  const touched = new Int32Array((radius * 2 + 1) ** 2)
  const out = new Int16Array(labels.length)
  const p = palette.lab
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (labels[i] < 0) {
        out[i] = -1
        continue
      }
      // Inside an area of one colour there is nothing to decide (most of
      // the picture, after the nearest-colour pass).
      const own = labels[i]
      let uniform = true
      for (let dy = -radius; dy <= radius && uniform; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= h) continue
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx
          if (xx < 0 || xx >= w) continue
          const lbl = labels[yy * w + xx]
          if (lbl !== own && lbl >= 0) {
            uniform = false
            break
          }
        }
      }
      if (uniform) {
        out[i] = own
        continue
      }
      let nTouched = 0
      let s = 0
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy
        for (let dx = -radius; dx <= radius; dx++, s++) {
          const xx = x + dx
          if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue
          const lbl = labels[yy * w + xx]
          if (lbl < 0) continue
          if (score[lbl] === 0) touched[nTouched++] = lbl
          score[lbl] += spatial[s]
        }
      }
      let best = labels[i]
      let bestScore = -1
      const l = lab[i * 3]
      const a = lab[i * 3 + 1]
      const b = lab[i * 3 + 2]
      for (let t = 0; t < nTouched; t++) {
        const c = touched[t]
        const dl = l - p[c * 3]
        const da = a - p[c * 3 + 1]
        const db = b - p[c * 3 + 2]
        const fit = Math.exp(-(dl * dl + da * da + db * db) / (2 * sigmaC * sigmaC))
        const sc = score[c] * fit
        if (sc > bestScore) {
          bestScore = sc
          best = c
        }
        score[c] = 0
      }
      out[i] = best
    }
  }
  return out
}

/** Specks of one colour smaller than `minSize` pixels take the colour most
 * of their surroundings have. In place. */
export function cleanupIslands(labels: Int16Array, w: number, h: number, minSize: number): void {
  if (minSize <= 1) return
  const seen = new Uint8Array(labels.length)
  const stack = new Int32Array(labels.length)
  const members: number[] = []
  const counts = new Map<number, number>()
  for (let start = 0; start < labels.length; start++) {
    if (seen[start] || labels[start] < 0) continue
    const lbl = labels[start]
    members.length = 0
    counts.clear()
    let top = 0
    stack[top++] = start
    seen[start] = 1
    while (top > 0) {
      const i = stack[--top]
      members.push(i)
      const x = i % w
      const y = (i - x) / w
      for (let n = 0; n < 4; n++) {
        const j = n === 0 ? (x > 0 ? i - 1 : -1) : n === 1 ? (x < w - 1 ? i + 1 : -1) : n === 2 ? (y > 0 ? i - w : -1) : y < h - 1 ? i + w : -1
        if (j < 0) continue
        const lj = labels[j]
        if (lj === lbl) {
          if (!seen[j]) {
            seen[j] = 1
            stack[top++] = j
          }
        } else if (lj >= 0 && members.length < minSize) {
          counts.set(lj, (counts.get(lj) ?? 0) + 1)
        }
      }
    }
    if (members.length >= minSize || counts.size === 0) continue
    let best = lbl
    let bestCount = 0
    for (const [c, n] of counts) {
      if (n > bestCount) {
        bestCount = n
        best = c
      }
    }
    for (const i of members) labels[i] = best
  }
}

/** The smallest speck kept, in pixels, for a picture of this scale. */
export const minIsland = (scale: number) => Math.max(2, Math.round(10 * scale * scale))

/** The palette painted into `out` (RGBA) where a pixel has a label. */
export function paintLabels(labels: Int16Array, palette: Palette, out: Uint8ClampedArray): void {
  for (let i = 0; i < labels.length; i++) {
    const c = labels[i]
    if (c < 0) continue
    out[i * 4] = palette.rgb[c * 3]
    out[i * 4 + 1] = palette.rgb[c * 3 + 1]
    out[i * 4 + 2] = palette.rgb[c * 3 + 2]
  }
}

/** The pixels with weight > 0 in the palette's colours, written into `out`
 * (which otherwise keeps its pixels): assign, vote, clean up. */
export function applyPalette(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  palette: Palette,
  weights: Uint8Array | null,
  out: Uint8ClampedArray,
  scale = 1,
  lab: Float32Array = analysisLab(rgba, w, h, scale),
): void {
  let labels = assignLabels(lab, w * h, palette, weights)
  labels = voteLabels(labels, lab, w, h, palette, scale)
  cleanupIslands(labels, w, h, minIsland(scale))
  paintLabels(labels, palette, out)
}
