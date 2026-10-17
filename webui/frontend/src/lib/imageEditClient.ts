import { OUTPUT_MAX_SIDE, WORK_PIXELS, fitPixels, fitSide, renderLayers, ReduceCache, type EditLayer } from './imageEdit.ts'
import { resizeChannel } from './imageMath.ts'
import type { Palette } from './colorReduce.ts'
import { LatestOnly } from './latestOnly.ts'

/** The image editor's renderer: the picture at a working size for the live
 * preview and at full size for the result, rendered in a worker so the
 * editor stays responsive while a slider moves. One render at a time; of
 * the ones asked for meanwhile only the newest runs, the rest resolve to
 * null (see LatestOnly). */

export interface Picture {
  rgba: Uint8ClampedArray
  w: number
  h: number
}

export interface Rendered {
  rgba: Uint8ClampedArray
  palettes: Map<string, Palette>
}

/** The picture at `url` decoded at the editor's working size and at its
 * full size (up to OUTPUT_MAX_SIDE). */
export async function loadPicture(url: string): Promise<{ work: Picture; full: Picture }> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Couldn't load the picture (HTTP ${response.status})`)
  const bitmap = await createImageBitmap(await response.blob())
  try {
    const fullSize = fitSide(bitmap.width, bitmap.height, OUTPUT_MAX_SIDE)
    const workSize = fitPixels(fullSize.w, fullSize.h, WORK_PIXELS)
    return { work: draw(bitmap, workSize.w, workSize.h), full: draw(bitmap, fullSize.w, fullSize.h) }
  } finally {
    bitmap.close()
  }
}

function draw(bitmap: ImageBitmap, w: number, h: number): Picture {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, w, h)
  return { rgba: ctx.getImageData(0, 0, w, h).data, w, h }
}

/** A grey mask PNG (uploads/) read back as one byte per pixel at w × h. */
export async function loadMask(url: string, w: number, h: number): Promise<Uint8Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Couldn't load a selection (HTTP ${response.status})`)
  const bitmap = await createImageBitmap(await response.blob())
  try {
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(bitmap, 0, 0)
    const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data
    const mask = new Uint8Array(bitmap.width * bitmap.height)
    for (let i = 0; i < mask.length; i++) mask[i] = data[i * 4]
    return resizeChannel(mask, bitmap.width, bitmap.height, w, h)
  } finally {
    bitmap.close()
  }
}

/** A mask as a grey PNG. */
export function maskToPng(mask: Uint8Array, w: number, h: number): Promise<Blob> {
  const rgba = new Uint8ClampedArray(w * h * 4)
  for (let i = 0; i < mask.length; i++) {
    rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = mask[i]
    rgba[i * 4 + 3] = 255
  }
  return pictureToPng({ rgba, w, h })
}

export function pictureToPng(picture: Picture): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = picture.w
  canvas.height = picture.h
  canvas.getContext('2d')!.putImageData(new ImageData(picture.rgba, picture.w, picture.h), 0, 0)
  return new Promise((resolve, reject) => canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed'))), 'image/png'))
}

type Pending = { resolve: (value: unknown) => void; reject: (err: Error) => void }

export class ImageEditEngine {
  private worker: Worker | null = null
  private nextId = 1
  private pending = new Map<number, Pending>()
  private renders = new LatestOnly<Rendered>()
  private work: Picture | null = null
  private full: Picture | null = null
  // Without a worker (old browsers, tests): the same work, inline.
  private cache = new ReduceCache()

  constructor() {
    if (typeof Worker === 'undefined') return
    try {
      this.worker = new Worker(new URL('./imageEdit.worker.ts', import.meta.url), { type: 'module' })
      this.worker.onmessage = (event: MessageEvent) => {
        const { id, type, error } = event.data ?? {}
        const entry = this.pending.get(id)
        if (!entry) return
        this.pending.delete(id)
        if (type === 'error') entry.reject(new Error(error))
        else entry.resolve(event.data)
      }
      this.worker.onerror = () => {
        for (const entry of this.pending.values()) entry.reject(new Error('The image editor stopped working. Close it and open it again.'))
        this.pending.clear()
      }
    } catch {
      this.worker = null
    }
  }

  private call(message: Record<string, unknown>, transfer: Transferable[] = []): Promise<Record<string, unknown>> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      this.worker!.postMessage({ ...message, id }, transfer)
    })
  }

  async init(work: Picture, full: Picture): Promise<void> {
    this.work = work
    this.full = full
    if (this.worker) await this.call({ type: 'init', work, full })
  }

  /** The layers over the working-size picture; null if a newer render was
   * asked for before this one got its turn. */
  render(layers: EditLayer[], masks: Map<string, Uint8Array>): Promise<Rendered | null> {
    // The masks as they are now: a brush stroke keeps changing its array.
    const snapshot = new Map([...masks].map(([id, m]) => [id, m.slice()]))
    return this.renders.run(async () => {
      if (!this.worker) {
        if (!this.work) throw new Error('No picture loaded')
        await new Promise((r) => setTimeout(r, 0))
        return renderLayers({ rgba: this.work.rgba, w: this.work.w, h: this.work.h, layers, masks: snapshot, cache: this.cache })
      }
      const reply = await this.call({ type: 'render', layers, masks: [...snapshot] }, [...snapshot.values()].map((m) => m.buffer))
      return { rgba: reply.rgba as Uint8ClampedArray, palettes: new Map(reply.palettes as [string, Palette][]) }
    })
  }

  /** The layers over the full picture. */
  async apply(layers: EditLayer[], masks: Map<string, Uint8Array>): Promise<Picture> {
    if (!this.worker) {
      if (!this.work || !this.full) throw new Error('No picture loaded')
      const palettes = renderLayers({ rgba: this.work.rgba, w: this.work.w, h: this.work.h, layers, masks, cache: this.cache }).palettes
      const full = new Map<string, Uint8Array>()
      for (const [id, m] of masks) full.set(id, resizeChannel(m, this.work.w, this.work.h, this.full.w, this.full.h))
      const { rgba } = renderLayers({ rgba: this.full.rgba, w: this.full.w, h: this.full.h, layers, masks: full, palettes })
      return { rgba, w: this.full.w, h: this.full.h }
    }
    const reply = await this.call({ type: 'apply', layers, masks: [...masks].map(([id, m]) => [id, m.slice()]) })
    return { rgba: reply.rgba as Uint8ClampedArray, w: reply.w as number, h: reply.h as number }
  }

  dispose(): void {
    this.worker?.terminate()
    this.worker = null
    for (const entry of this.pending.values()) entry.reject(new Error('closed'))
    this.pending.clear()
  }
}
