import { ReduceCache, renderLayers, type EditLayer } from './imageEdit.ts'
import { resizeChannel } from './imageMath.ts'

/** Worker half of ImageEditEngine (imageEditClient.ts): holds the picture
 * at the editor's working size and at full size, renders the layers over
 * it, and keeps the colour reduction's merge trees between renders. */

interface Picture {
  rgba: Uint8ClampedArray
  w: number
  h: number
}

let work: Picture | null = null
let full: Picture | null = null
const cache = new ReduceCache()

const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer)

self.onmessage = (event: MessageEvent) => {
  const msg = event.data
  try {
    if (msg.type === 'init') {
      work = msg.work
      full = msg.full
      post({ type: 'ready', id: msg.id })
    } else if (msg.type === 'render') {
      if (!work) throw new Error('No picture loaded')
      const masks = new Map<string, Uint8Array>(msg.masks)
      const { rgba, palettes } = renderLayers({ rgba: work.rgba, w: work.w, h: work.h, layers: msg.layers as EditLayer[], masks, cache })
      post({ type: 'rendered', id: msg.id, rgba, palettes: [...palettes] }, [rgba.buffer])
    } else if (msg.type === 'apply') {
      if (!work || !full) throw new Error('No picture loaded')
      const layers = msg.layers as EditLayer[]
      const workMasks = new Map<string, Uint8Array>(msg.masks)
      // The palettes of exactly these layers at the working size - what the
      // preview shows - so the full picture comes out the way it looked
      // (the merge trees are cached, so this is quick).
      const palettes = renderLayers({ rgba: work.rgba, w: work.w, h: work.h, layers, masks: workMasks, cache }).palettes
      const masks = new Map<string, Uint8Array>()
      for (const [id, mask] of workMasks) masks.set(id, resizeChannel(mask, work.w, work.h, full.w, full.h))
      const { rgba } = renderLayers({ rgba: full.rgba, w: full.w, h: full.h, layers, masks, palettes })
      post({ type: 'applied', id: msg.id, rgba, w: full.w, h: full.h }, [rgba.buffer])
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, error: err instanceof Error ? err.message : String(err) })
  }
}
