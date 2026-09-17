import type { ColorSliderConfig } from '../types'

// Whether a slider stack differs from the last rendered one in a way that
// changes the rendered result (depth_mm is derived from layer, so it's not
// compared). The stacks may differ in length — an optimizer/pruner result
// routinely has more or fewer columns than the stack it replaces — and a
// column that was added or removed is itself a change.
export function slidersNeedRender(prev: ColorSliderConfig[], next: ColorSliderConfig[]): boolean {
  if (prev.length !== next.length) return true
  return next.some((s, i) => {
    const p = prev[i]
    return s.td !== p.td || s.layer !== p.layer || s.enabled !== p.enabled || s.filament_uuid !== p.filament_uuid
  })
}
