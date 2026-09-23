import { useEffect, useRef, useState } from 'react'
import { useDebouncedCallback } from './useDebouncedCallback'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { slidersNeedRender } from '../lib/sliderDiff'
import { describeApiError } from '../lib/apiError'
import { sliderRenderPaused } from '../lib/history'

// Mirrors api/preview.py's INIT_JOB_SENTINEL — tells the backend "there's
// no completed job yet, render against the post-upload auto-preview state
// instead" when POSTing a slider edit.
export const INIT_JOB_SENTINEL = '__init__'

/** Re-renders the preview mesh/image whenever the color layers change.
 * Mounted once (in the bottom panel), independent of which tab is shown.
 * Returns whether a render is in flight. */
export function useSliderPreviewRender(): boolean {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const currentJob = useAppStore((s) => s.currentJob)
  const initState = useAppStore((s) => s.initState)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const pruningJob = useAppStore((s) => s.pruningJob)
  const lastRenderedSlidersRef = useRef(JSON.stringify(colorSliders))
  const lastRenderedJobRef = useRef<string>(INIT_JOB_SENTINEL)
  const [isRendering, setIsRendering] = useState(false)
  const lastRenderErrorRef = useRef<string | null>(null)

  // Slider edits only make sense once there's a discretized solution to
  // recolor — either a completed job, or (now that api/init.py actually
  // runs the heightmap-init algorithm instead of a no-op stub) the
  // pre-run auto-preview: its height map is real, so a slider edit can be
  // rendered against it exactly like a completed job's, letting the user
  // assign colors manually before ever running the real optimizer.
  // Not while a job is running: its live broadcasts replace the stack every
  // few iterations, and each one used to trigger a pointless auto-preview
  // re-render competing with the optimizer for the GPU.
  // Pruning counts too: it mutates the result's optimizer in place and
  // broadcasts each pass's stack, and every broadcast used to fire a render
  // that read the optimizer (on the GPU) while pruning was rewriting it.
  const jobActive = sliderRenderPaused(currentJob, pruningJob)
  const hasResult = !jobActive && (currentJob?.status === 'completed' || initState.status === 'ready')
  const jobId = currentJob?.status === 'completed' ? currentJob.job_id : INIT_JOB_SENTINEL

  const triggerPreviewRender = useDebouncedCallback(async (sliders: typeof colorSliders, filaments: Filament[], jobId: string | undefined) => {
    setIsRendering(true)
    try {
      const response = await fetch('/api/preview/render-with-sliders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sliders, active_filaments: filaments, job_id: jobId }),
      })
      if (!response.ok) {
        // e.g. the result was computed before a server restart and can't be
        // re-colored — previously the edit just silently did nothing.
        const message = `Preview not updated: ${describeApiError(await response.json().catch(() => null), response.status)}`
        if (message !== lastRenderErrorRef.current) useAppStore.getState().pushToast(message, 'warning')
        lastRenderErrorRef.current = message
      } else {
        lastRenderErrorRef.current = null
      }
    } catch {
      // Ignore — the WS-driven preview simply won't update this round.
    } finally {
      setIsRendering(false)
    }
  }, [], 150)

  useEffect(() => {
    // While a job runs, its broadcasts *are* the rendered state — track them
    // so completion doesn't look like an edit and re-render the result.
    if (jobActive) {
      lastRenderedSlidersRef.current = JSON.stringify(colorSliders)
      lastRenderedJobRef.current = currentJob?.job_id ?? jobId
      return
    }
    if (!hasResult) return

    const currentKey = JSON.stringify(colorSliders)
    // A different target (undo/redo to another result, or back to the
    // pre-run preview) has its own previously-rendered colors on disk, so
    // it needs a render even when the slider stacks happen to match.
    const targetChanged = jobId !== lastRenderedJobRef.current
    if (!targetChanged && currentKey === lastRenderedSlidersRef.current) return

    // ...except the pre-run preview with nothing assigned yet: rendering that
    // would replace the photo-draped mesh with an all-gray one.
    const nothingAssigned = jobId === INIT_JOB_SENTINEL && !colorSliders.some((s) => s.enabled && s.filament_uuid)
    const needsRender = targetChanged
      ? !nothingAssigned
      : slidersNeedRender(JSON.parse(lastRenderedSlidersRef.current), colorSliders)
    if (needsRender) {
      triggerPreviewRender(colorSliders, activeFilaments, jobId)
    }

    lastRenderedSlidersRef.current = currentKey
    lastRenderedJobRef.current = jobId
  }, [colorSliders, hasResult, jobActive, activeFilaments, jobId, triggerPreviewRender])


  return isRendering
}
