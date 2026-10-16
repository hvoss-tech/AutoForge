import { useEffect } from 'react'
import { useAppStore } from '../store/appStore'

/** Keeps the color core's layer range (sliderLayerRange) on the result
 * that is on screen. A run or a prune sends it over the websocket while it
 * works, but a page reload, a history step or a finished prune only change
 * which result is shown — the range stayed at its placeholder (75, the
 * max_layers default) and the color core ran far past the result's top
 * layer. Asked again whenever the completed result changes. Mounted once,
 * in App. */
export function useResultLayerRange() {
  const jobId = useAppStore((s) => (s.currentJob?.status === 'completed' ? s.currentJob.job_id : null))
  const setSliderLayerRange = useAppStore((s) => s.setSliderLayerRange)

  useEffect(() => {
    if (!jobId) return
    const abort = new AbortController()
    fetch(`/api/sliders/range?job_id=${encodeURIComponent(jobId)}`, { signal: abort.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data || abort.signal.aborted) return
        if (useAppStore.getState().currentJob?.job_id !== jobId) return
        if (!Number.isFinite(data.min_layer) || !Number.isFinite(data.max_layer)) return
        const current = useAppStore.getState().sliderLayerRange
        if (current.min !== data.min_layer || current.max !== data.max_layer) {
          setSliderLayerRange({ min: data.min_layer, max: data.max_layer })
        }
      })
      .catch(() => {
        // Keeps the range it has; the next result change asks again.
      })
    return () => abort.abort()
  }, [jobId, setSliderLayerRange])
}
