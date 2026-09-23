import type { ColorSliderConfig, Filament } from '../types'

export interface StackHandle {
  storeIndex: number
  value: number
  color: string
}

export interface StackSegment {
  layerIndex: number
  color: string
}

function lerpColor(c1: string, c2: string, t: number): string {
  const r1 = parseInt(c1.slice(1, 3), 16)
  const g1 = parseInt(c1.slice(3, 5), 16)
  const b1 = parseInt(c1.slice(5, 7), 16)
  const r2 = parseInt(c2.slice(1, 3), 16)
  const g2 = parseInt(c2.slice(3, 5), 16)
  const b2 = parseInt(c2.slice(5, 7), 16)
  const r = Math.round(r1 + (r2 - r1) * t)
  const g = Math.round(g1 + (g2 - g1) * t)
  const b = Math.round(b1 + (b2 - b1) * t)
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

export function getFilamentColor(filaments: Filament[], uuid: string): string {
  if (!uuid) return '#555555'
  const f = filaments.find((f) => f.uuid === uuid)
  return f ? f.color : '#555555'
}

/** Enabled sliders with a real layer value, sorted ascending by layer —
 * shared by ColorCore (2D) and the pre-optimization 3D stack preview so
 * they always agree on which colors go where. */
export function getStackHandles(colorSliders: ColorSliderConfig[], filaments: Filament[]): StackHandle[] {
  return colorSliders
    .map((slider, idx) => ({ ...slider, storeIndex: idx }))
    .filter((s) => s.enabled && s.layer > 0)
    .sort((a, b) => a.layer - b.layer)
    .map((slider) => ({
      storeIndex: slider.storeIndex,
      value: slider.layer,
      color: getFilamentColor(filaments, slider.filament_uuid),
    }))
}

/** Sliders are allowed to share a layer (dragging one over another is a
 * normal interaction, not an error) — but only one of them can actually own
 * that print layer for coloring/compositing purposes. When two or more
 * handles land on the same layer, the one further right in the column order
 * (the highest `storeIndex`) wins; the rest are silently excluded here. */
export function filterActiveHandles(handles: StackHandle[]): StackHandle[] {
  const winnerByValue = new Map<number, number>()
  for (const h of handles) {
    const current = winnerByValue.get(h.value)
    if (current === undefined || h.storeIndex > current) winnerByValue.set(h.value, h.storeIndex)
  }
  return handles.filter((h) => winnerByValue.get(h.value) === h.storeIndex)
}

/** Store indices of sliders that currently lose a layer-overlap tie (see
 * `filterActiveHandles`) — used to dim/mark them in the UI without touching
 * their `enabled` flag, so they silently stop affecting the print until
 * moved off the shared layer. */
export function getOverlapDisabledIndices(colorSliders: ColorSliderConfig[], filaments: Filament[]): Set<number> {
  const handles = getStackHandles(colorSliders, filaments)
  const activeIndices = new Set(filterActiveHandles(handles).map((h) => h.storeIndex))
  return new Set(handles.filter((h) => !activeIndices.has(h.storeIndex)).map((h) => h.storeIndex))
}

/** One entry per print layer (1..maxLayer), color-blended between the
 * handle below and at each layer — same interpolation ColorCore uses. */
export function getStackSegments(handles: StackHandle[]): StackSegment[] {
  if (handles.length === 0) return []
  const maxLayer = handles[handles.length - 1].value
  const result: StackSegment[] = []

  for (let layer = 1; layer <= maxLayer; layer++) {
    let owningHandleIdx = handles.length - 1
    for (let h = 0; h < handles.length; h++) {
      if (layer <= handles[h].value) {
        owningHandleIdx = h
        break
      }
    }

    let color: string
    if (owningHandleIdx === 0) {
      color = handles[0].color
    } else {
      const prevHandle = handles[owningHandleIdx - 1]
      const currHandle = handles[owningHandleIdx]
      const rangeStart = prevHandle.value
      const rangeEnd = currHandle.value
      const t = rangeEnd === rangeStart ? 0 : (layer - rangeStart) / (rangeEnd - rangeStart)
      color = lerpColor(prevHandle.color, currHandle.color, t)
    }

    result.push({ layerIndex: layer, color })
  }

  return result
}

/** Pixel height a full color-column handle (arrow plus layer number) needs
 * to be drawn without overlapping the next one. */
export const FULL_HANDLE_SPACING = 20

/** For handles at these vertical positions (in order), the distance to the
 * nearest neighbour above or below — Infinity for a lone handle. */
export function nearestNeighborGaps(positions: number[]): number[] {
  return positions.map((y, i) => {
    const gaps = [positions[i - 1], positions[i + 1]].filter((n): n is number => n !== undefined).map((n) => Math.abs(n - y))
    return gaps.length ? Math.min(...gaps) : Infinity
  })
}

/** Height of the slim marker drawn instead of a full handle when the
 * neighbours are closer than FULL_HANDLE_SPACING: as tall as the gap allows
 * (leaving a pixel between markers), but never an invisible sliver. */
export function compactHandleHeight(gap: number): number {
  return Math.max(3, Math.min(12, Math.floor(gap) - 1))
}
