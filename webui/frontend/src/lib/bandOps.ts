import type { ColorSliderConfig } from '../types'

/** Pure edits of the band list (the color sliders) that keep layers and
 * depths consistent. A band is "positioned" once it has a top layer; bands at
 * layer 0 are unplaced columns and always stay at the end. */

const positioned = (s: ColorSliderConfig) => s.layer > 0

function withLayer(s: ColorSliderConfig, layer: number, layerHeight: number): ColorSliderConfig {
  return { ...s, layer, depth_mm: parseFloat((layer * layerHeight).toFixed(2)) }
}

/** Store indices in print order: positioned bands by layer (ties keep their
 * store order, so the tie winner stays the winner), then unplaced ones. */
export function printOrder(sliders: ColorSliderConfig[]): number[] {
  const indices = sliders.map((_, i) => i)
  const placed = indices.filter((i) => positioned(sliders[i])).sort((a, b) => sliders[a].layer - sliders[b].layer || a - b)
  return [...placed, ...indices.filter((i) => !positioned(sliders[i]))]
}

/** True when the rows already list the bands in print order. Dragging a
 * slider over its neighbours is allowed, and leaves the rows out of order. */
export function isInPrintOrder(sliders: ColorSliderConfig[]): boolean {
  return printOrder(sliders).every((storeIndex, position) => storeIndex === position)
}

export interface BandEdit {
  sliders: ColorSliderConfig[]
  /** Where the band the edit was about ended up. */
  index: number
}

/** Rows sorted into print order; `index` follows the band at `followIndex`. */
export function sortBandsByLayer(sliders: ColorSliderConfig[], followIndex = 0): BandEdit {
  const order = printOrder(sliders)
  return { sliders: order.map((i) => sliders[i]), index: Math.max(0, order.indexOf(followIndex)) }
}

/** Moves a band to another position in the print order, keeping every band's
 * thickness: the bands in between shift down (or up) to make room. */
export function moveBand(sliders: ColorSliderConfig[], from: number, to: number, layerHeight: number): BandEdit {
  const sorted = sortBandsByLayer(sliders, from)
  const target = sortBandsByLayer(sliders, to).index
  const list = sorted.sliders
  const placedCount = list.filter(positioned).length
  const fromPos = sorted.index
  if (fromPos >= placedCount || target >= placedCount || fromPos === target) return sorted

  let previous = 0
  const pieces = list.slice(0, placedCount).map((s) => {
    const thickness = Math.max(0, s.layer - previous)
    previous = s.layer
    return { band: s, thickness }
  })
  const [moved] = pieces.splice(fromPos, 1)
  pieces.splice(target, 0, moved)

  let top = 0
  const restacked = pieces.map(({ band, thickness }) => {
    top += thickness
    return withLayer(band, top, layerHeight)
  })
  return { sliders: [...restacked, ...list.slice(placedCount)], index: target }
}

/** Adds a band directly above the band at `index` (in print order). The bands
 * above move up by `thickness` layers while there's room below `maxLayer`;
 * with no room left, the band below gives up its upper half instead. Returns
 * null when there isn't a single layer to spare. */
export function insertBandAbove(
  sliders: ColorSliderConfig[],
  index: number,
  newBand: Pick<ColorSliderConfig, 'filament_uuid' | 'td'>,
  maxLayer: number,
  layerHeight: number,
  thickness = 3,
): BandEdit | null {
  const sorted = sortBandsByLayer(sliders, index)
  const list = sorted.sliders
  const pos = sorted.index
  const below = list[pos]
  if (!below || !positioned(below)) return null
  const placed = list.filter(positioned)
  const top = placed.length ? placed[placed.length - 1].layer : 0
  const room = Math.max(0, maxLayer - top)
  const base = { ...newBand, enabled: true }

  if (room > 0) {
    const shift = Math.min(thickness, room)
    const result = list.map((s, i) => (i > pos && positioned(s) ? withLayer(s, s.layer + shift, layerHeight) : s))
    const inserted = withLayer({ ...base, layer: 0, depth_mm: 0 }, below.layer + shift, layerHeight)
    result.splice(pos + 1, 0, inserted)
    return { sliders: result, index: pos + 1 }
  }

  const previousTop = pos > 0 ? list[pos - 1].layer : 0
  const half = Math.floor((below.layer - previousTop) / 2)
  if (half < 1) return null
  const result = [...list]
  result[pos] = withLayer(below, below.layer - half, layerHeight)
  result.splice(pos + 1, 0, withLayer({ ...base, layer: 0, depth_mm: 0 }, below.layer, layerHeight))
  return { sliders: result, index: pos + 1 }
}
