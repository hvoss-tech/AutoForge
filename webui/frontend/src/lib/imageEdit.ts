/** The image editor's document: adjustment layers over the uploaded
 * picture, each with its own selection. Saved in the settings
 * (settings.image_edit) next to the edited picture it produced, so an edit
 * can be changed again later, undone, and travels with the project.
 *
 * Layers apply bottom to top: each one adjusts the picture as the layers
 * below left it, then blends that in through its selection. */

import { applyAdjustments, defaultAdjustments, isIdentity, type Adjustments } from './imageAdjust.ts'
import { analysisLab, applyPalette, cutPalette, prepareReduce, type Palette, type Prepared } from './colorReduce.ts'
import { featherMask } from './selection.ts'

export interface EditLayer {
  id: string
  name: string
  visible: boolean
  /** The whole picture, or the layer's own mask. */
  selection: 'all' | 'mask'
  /** The mask as an uploaded grey PNG (uploads/), '' until first saved. */
  mask: string
  /** Soft edges of the selection, 0..100. */
  feather: number
  adjust: Adjustments
}

export interface ImageEdit {
  version: 1
  /** The unedited picture (uploads/). */
  source: string
  layers: EditLayer[]
}

/** The editor works on the picture at about this many pixels; the edit is
 * then applied to the full picture (up to OUTPUT_MAX_SIDE). */
export const WORK_PIXELS = 1_200_000
export const OUTPUT_MAX_SIDE = 4096

/** A size scaled to fit `pixels` (never enlarged). */
export function fitPixels(w: number, h: number, pixels: number): { w: number; h: number } {
  const f = Math.min(1, Math.sqrt(pixels / Math.max(1, w * h)))
  return { w: Math.max(1, Math.round(w * f)), h: Math.max(1, Math.round(h * f)) }
}

/** A size scaled so its long side is at most `side`. */
export function fitSide(w: number, h: number, side: number): { w: number; h: number } {
  const f = Math.min(1, side / Math.max(w, h))
  return { w: Math.max(1, Math.round(w * f)), h: Math.max(1, Math.round(h * f)) }
}

/** A picture's size relative to a 1000 px one: radii (soften, sharpen,
 * soft edges, colour reduction's clean-up) follow it, so the preview and
 * the full picture look alike. */
export const sizeScale = (w: number, h: number) => Math.max(w, h) / 1000

let counter = 0
export function newLayer(name: string, selection: 'all' | 'mask' = 'all'): EditLayer {
  counter += 1
  return {
    id: `layer-${Date.now().toString(36)}-${counter}`,
    name,
    visible: true,
    selection,
    mask: '',
    feather: 10,
    adjust: defaultAdjustments(),
  }
}

/** A numbered name for the next layer ("Layer 3") that isn't taken yet. */
export function nextLayerName(layers: { name: string }[], base: string): string {
  const names = new Set(layers.map((l) => l.name))
  for (let i = 1; ; i++) if (!names.has(`${base} ${i}`)) return `${base} ${i}`
}

/** True when the layers change nothing (no edit to keep). */
export function editChangesNothing(layers: EditLayer[]): boolean {
  return layers.every((l) => !l.visible || isIdentity(l.adjust))
}

/** The edit as saved in the settings, or null when it changes nothing. */
export function savedEdit(source: string, layers: EditLayer[]): ImageEdit | null {
  if (editChangesNothing(layers)) return null
  return { version: 1, source, layers: layers.map((l) => ({ ...l, adjust: structuredCloneSafe(l.adjust) })) }
}

function structuredCloneSafe<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

/** A settings value read back as an edit, or null if it isn't one. */
export function parseEdit(value: unknown): ImageEdit | null {
  const v = value as Partial<ImageEdit> | null
  if (!v || typeof v !== 'object' || v.version !== 1 || typeof v.source !== 'string' || !v.source || !Array.isArray(v.layers)) return null
  const defaults = defaultAdjustments()
  const layers = v.layers
    .filter((l): l is EditLayer => !!l && typeof l === 'object' && typeof (l as EditLayer).id === 'string')
    .map((l) => ({
      id: l.id,
      name: typeof l.name === 'string' ? l.name : 'Layer',
      visible: l.visible !== false,
      selection: l.selection === 'mask' ? ('mask' as const) : ('all' as const),
      mask: typeof l.mask === 'string' ? l.mask : '',
      feather: Number.isFinite(l.feather) ? l.feather : 10,
      adjust: {
        ...defaults,
        ...(l.adjust ?? {}),
        levels: { ...defaults.levels, ...(l.adjust?.levels ?? {}) },
        curves: { ...defaults.curves, ...(l.adjust?.curves ?? {}) },
        reduce: { ...defaults.reduce, ...(l.adjust?.reduce ?? {}) },
      },
    }))
  return { version: 1, source: v.source, layers }
}

/** The uploads a saved edit refers to: its source picture and masks. */
export function editFiles(edit: ImageEdit): string[] {
  return [edit.source, ...edit.layers.filter((l) => l.selection === 'mask' && l.mask).map((l) => l.mask)]
}

/** A 32-bit FNV-1a hash of some bytes (cache keys). */
export function hashBytes(bytes: Uint8Array | Uint8ClampedArray, seed = 0x811c9dc5): number {
  let h = seed >>> 0
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h
}

/** Colour reduction's merge trees, by what they were built from: the
 * expensive part, so moving the colour count only cuts a cached tree. */
export class ReduceCache {
  private entries = new Map<string, Prepared>()
  constructor(private limit = 6) {}
  get(key: string): Prepared | undefined {
    const hit = this.entries.get(key)
    if (hit) {
      this.entries.delete(key)
      this.entries.set(key, hit)
    }
    return hit
  }
  set(key: string, value: Prepared): void {
    this.entries.set(key, value)
    while (this.entries.size > this.limit) this.entries.delete(this.entries.keys().next().value as string)
  }
}

export interface RenderLayersInput {
  rgba: Uint8ClampedArray
  w: number
  h: number
  layers: EditLayer[]
  /** Each mask layer's selection at w × h. */
  masks: Map<string, Uint8Array>
  /** Palettes to use instead of computing them (the full picture uses the
   * preview's, so it comes out the way it looked). */
  palettes?: Map<string, Palette>
  cache?: ReduceCache
}

/** The layers applied to a copy of the picture; returns it with each
 * colour-reducing layer's palette. */
export function renderLayers(input: RenderLayersInput): { rgba: Uint8ClampedArray; palettes: Map<string, Palette> } {
  const { w, h, layers, masks } = input
  const scale = sizeScale(w, h)
  let current = input.rgba.slice()
  const palettes = new Map<string, Palette>()
  for (const layer of layers) {
    if (!layer.visible || isIdentity(layer.adjust)) continue
    let weights: Uint8Array | null = null
    if (layer.selection === 'mask') {
      const mask = masks.get(layer.id)
      if (!mask || mask.length !== w * h) continue
      weights = featherMask(mask, w, h, (layer.feather / 100) * 15 * scale)
      if (!weights.some((v) => v > 0)) continue
    }
    const adjusted = applyAdjustments(current, w, h, layer.adjust, new Uint8ClampedArray(current.length), scale)
    if (layer.adjust.reduce.enabled) {
      const lab = analysisLab(adjusted, w, h, scale)
      let palette = input.palettes?.get(layer.id)
      if (!palette) {
        const key = `${w}x${h}:${hashBytes(adjusted)}:${weights ? hashBytes(weights) : 'all'}`
        let prepared = input.cache?.get(key)
        if (!prepared) {
          prepared = prepareReduce(lab, weights, w * h)
          input.cache?.set(key, prepared)
        }
        palette = cutPalette(prepared, layer.adjust.reduce.colors)
      }
      palettes.set(layer.id, palette)
      if (palette.count > 0) applyPalette(adjusted, w, h, palette, weights, adjusted, scale, lab)
    }
    if (weights) {
      for (let i = 0; i < w * h; i++) {
        const m = weights[i] / 255
        if (m === 0) continue
        const p = i * 4
        current[p] = current[p] + (adjusted[p] - current[p]) * m
        current[p + 1] = current[p + 1] + (adjusted[p + 1] - current[p + 1]) * m
        current[p + 2] = current[p + 2] + (adjusted[p + 2] - current[p + 2]) * m
      }
    } else {
      current = adjusted
    }
  }
  return { rgba: current, palettes }
}
