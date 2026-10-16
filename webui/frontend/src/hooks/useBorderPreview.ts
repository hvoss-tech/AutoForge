import { useEffect, useRef, useState } from 'react'
import { borderActive, borderPreviewKey, decodeBorderMesh, type BorderMeshData, type BorderSettings } from '../lib/border'
import { describeApiError } from '../lib/apiError'

export interface BorderPreview {
  mesh: BorderMeshData | null
  /** The print with / without the border, mm. */
  sizeMm: [number, number] | null
  modelSizeMm: [number, number] | null
  loading: boolean
  error: string | null
}

const EMPTY: BorderPreview = { mesh: null, sizeMm: null, modelSizeMm: null, loading: false, error: null }

/** The border mesh for the model the 3D view shows (POST
 * /api/outputs/border-preview/{source}), refetched when the border, the
 * source or its base height changes. `source` is a job id or '__init__'
 * (the auto-preview), null with nothing to border. Only asked for while
 * `enabled` — the first request per model reads its whole STL. */
export function useBorderPreview(
  source: string | null,
  border: BorderSettings,
  backgroundHeight: number,
  enabled: boolean,
): BorderPreview {
  const [state, setState] = useState<BorderPreview>(EMPTY)
  const lastKey = useRef<string | null>(null)
  const lastSource = useRef<string | null>(null)
  const key = source && enabled ? borderPreviewKey(source, border, backgroundHeight) : null

  useEffect(() => {
    if (!key || !source) {
      lastKey.current = null
      setState(EMPTY)
      return
    }
    if (key === lastKey.current) {
      // Back to the border on screen before a newer request finished.
      setState((s) => (s.loading ? { ...s, loading: false } : s))
      return
    }
    const abort = new AbortController()
    // Another model: its predecessor's border must not stay drawn around it
    // until the new one arrives.
    const sameModel = source === lastSource.current
    setState((s) => (sameModel ? { ...s, loading: true } : { ...EMPTY, loading: true }))
    // Dragging a slider asks for many borders; only the one it stops at.
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/outputs/border-preview/${encodeURIComponent(source)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ border, background_height: backgroundHeight }),
          signal: abort.signal,
        })
        const data = await res.json().catch(() => null)
        if (abort.signal.aborted) return
        lastSource.current = source
        if (!res.ok) {
          lastKey.current = key
          setState({ ...EMPTY, error: describeApiError(data, res.status) })
          return
        }
        lastKey.current = key
        setState({
          mesh: borderActive(border) ? decodeBorderMesh(data) : null,
          sizeMm: data?.size_mm ?? null,
          modelSizeMm: data?.model_size_mm ?? null,
          loading: false,
          error: null,
        })
      } catch {
        if (!abort.signal.aborted) setState((s) => ({ ...s, loading: false, error: 'Could not build the border preview' }))
      }
    }, 120)
    return () => {
      clearTimeout(timer)
      abort.abort()
    }
    // `key` covers source, border and height.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return state
}
