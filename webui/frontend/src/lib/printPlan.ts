import type { ColorSliderConfig, Filament } from '../types'

export interface PlanBand {
  /** Index into colorSliders (for editing). */
  storeIndex: number
  filament: Filament | null
  /** The slider's filament_uuid ('' when unassigned), even if the filament
   * isn't in the list that was passed in. */
  filamentUuid: string
  td: number
  /** 1-based print layers above the base, inclusive. */
  startLayer: number
  endLayer: number
  startHeightMm: number
  endHeightMm: number
}

export interface PlanSwap {
  /** Layer number as a slicer counts it (the base is layer 1). */
  layerNumber: number
  heightMm: number
  filament: Filament | null
}

export interface PrintPlan {
  bands: PlanBand[]
  /** Distinct filaments used by the layer bands (the base is extra). */
  colors: number
  /** Those filaments' uuids ('unassigned' for bands without one). */
  colorKeys: string[]
  /** Filament changes from the base up, the same count the optimizer and
   * the pruner limit: the base is band 0, so the change from the base to the
   * first band is a swap unless that band is in the base filament. */
  swaps: number
  topLayer: number
  totalHeightMm: number
  swapsList: PlanSwap[]
}

export interface PlanSettings {
  layer_height: number
  background_height: number
  background_color?: string
  /** The base's filament, when it is one ('' / absent: a custom color). */
  base_filament_uuid?: string
}

const round2 = (n: number) => Math.round(n * 100) / 100

/** Enabled bands in print order. A slider owns the layers from just above
 * the previous slider's layer up to its own; when two share a layer the
 * right-most column wins (same rule as the color core and the renderer). */
export function getPlanBands(
  sliders: ColorSliderConfig[],
  filaments: Filament[],
  settings: PlanSettings,
): PlanBand[] {
  const lh = settings.layer_height || 0.04
  const base = settings.background_height || 0
  const winner = new Map<number, number>()
  sliders.forEach((s, i) => {
    if (!s.enabled || s.layer <= 0) return
    const current = winner.get(s.layer)
    if (current === undefined || i > current) winner.set(s.layer, i)
  })
  const ordered = [...winner.entries()].sort((a, b) => a[0] - b[0])
  let previous = 0
  return ordered.map(([layer, storeIndex]) => {
    const s = sliders[storeIndex]
    const band: PlanBand = {
      storeIndex,
      filament: filaments.find((f) => f.uuid === s.filament_uuid) ?? null,
      filamentUuid: s.filament_uuid,
      td: s.td,
      startLayer: previous + 1,
      endLayer: layer,
      startHeightMm: round2(base + previous * lh),
      endHeightMm: round2(base + layer * lh),
    }
    previous = layer
    return band
  })
}

export function buildPrintPlan(sliders: ColorSliderConfig[], filaments: Filament[], settings: PlanSettings): PrintPlan {
  const bands = getPlanBands(sliders, filaments, settings)
  const lh = settings.layer_height || 0.04
  const base = settings.background_height || 0
  // A single constant key for every unassigned band, not one keyed by
  // storeIndex — adjacent unassigned bands are visually and physically the
  // same (no filament, no swap needed between them), but distinct per-band
  // keys made them always compare unequal, inflating the swap count.
  const key = (b: PlanBand) => b.filamentUuid || 'unassigned'

  // Adjacent bands with the same filament are one continuous color: no swap.
  const swapsList: PlanSwap[] = []
  const baseKey = settings.base_filament_uuid || null
  bands.forEach((b, i) => {
    if (i > 0 && key(bands[i - 1]) === key(b)) return
    // A first band in the base filament just carries on printing it.
    if (i === 0 && baseKey && key(b) === baseKey) return
    const printLayer = b.startLayer
    swapsList.push({ layerNumber: printLayer + 1, heightMm: round2(printLayer * lh + base), filament: b.filament })
  })

  const topLayer = bands.length ? bands[bands.length - 1].endLayer : 0
  return {
    bands,
    colors: new Set(bands.map(key)).size,
    colorKeys: [...new Set(bands.map(key))],
    swaps: swapsList.length,
    topLayer,
    totalHeightMm: round2(base + topLayer * lh),
    swapsList,
  }
}

const filamentLabel = (f: Filament | null) => (f ? [f.brand, f.name].filter(Boolean).join(' - ') : 'an unassigned filament')

/** Same wording as the optimizer's swap_instructions.txt, but for the
 * slider stack as currently edited. */
export function printPlanText(plan: PrintPlan, settings: PlanSettings): string {
  const lh = settings.layer_height || 0.04
  const base = settings.background_height || 0
  if (plan.bands.length === 0) return 'No layers printed.'
  const lines = [
    `Print at 100% infill with a layer height of ${lh.toFixed(2)}mm with a base layer of ${base.toFixed(2)}mm`,
    '',
    `Start with your background color${settings.background_color ? ` (${settings.background_color})` : ''}, with a layer height of ${base.toFixed(2)}mm for the first layer.`,
    ...plan.swapsList.map((s) => `At layer #${s.layerNumber} (${s.heightMm.toFixed(2)}mm) swap to ${filamentLabel(s.filament)}`),
    `For the rest, use ${filamentLabel(plan.bands[plan.bands.length - 1].filament)}`,
  ]
  return lines.join('\n')
}
