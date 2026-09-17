import React, { useEffect, useMemo, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { Box, Loader2, AlertTriangle } from 'lucide-react'
import { ThreeDView } from './ThreeDView'
import { ResultsHistory } from './ResultsHistory'
import { getStackHandles, getStackSegments, filterActiveHandles } from '../lib/colorStack'

export const Preview3DPanel: React.FC = () => {
  const previewImage = useAppStore((s) => s.previewImage)
  const stlFile = useAppStore((s) => s.stlFile)
  const initState = useAppStore((s) => s.initState)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const inputImage = useAppStore((s) => s.inputImage)
  const setPreviewImage = useAppStore((s) => s.setPreviewImage)
  const setInitState = useAppStore((s) => s.setInitState)
  const setSliderLayerRange = useAppStore((s) => s.setSliderLayerRange)
  const currentJob = useAppStore((s) => s.currentJob)
  const previewVersion = useAppStore((s) => s.previewVersion)
  const bumpPreviewVersion = useAppStore((s) => s.bumpPreviewVersion)
  const wsRef = useRef<WebSocket | null>(null)

  const setSliders = useAppStore((s) => s.setSliders)
  const applySliders = useAppStore((s) => s.applySliders)
  const updateSlider = useAppStore((s) => s.updateSlider)
  const colorSliders = useAppStore((s) => s.colorSliders)
  const filaments = useAppStore((s) => s.filaments)
  const colorSlidersRef = useRef(colorSliders)

  // Before any optimization result exists, show what the currently-assigned
  // slider colors would look like as a simple stacked-layer preview instead
  // of an empty/placeholder panel — but only once a real filament has been
  // assigned to at least one slider (the four default columns start enabled
  // with no filament, which would otherwise render an uninformative gray box).
  const hasAssignedSliderColor = colorSliders.some((s) => s.enabled && s.layer > 0 && s.filament_uuid)
  const stackSegments = useMemo(() => {
    if (!hasAssignedSliderColor) return []
    const handles = filterActiveHandles(getStackHandles(colorSliders, filaments))
    return getStackSegments(handles)
  }, [colorSliders, filaments, hasAssignedSliderColor])

  const optimizationStarted = currentJob && ['running', 'paused', 'pending'].includes(currentJob.status)
  const jobFailed = currentJob && currentJob.status === 'failed'
  const jobHasResult = currentJob?.status === 'completed'
  const jobError = currentJob?.error

  useEffect(() => {
    colorSlidersRef.current = colorSliders
  }, [colorSliders])

  // Connect to preview WebSocket for live updates
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/preview`

    // Set on unmount so the socket closed by cleanup doesn't schedule a
    // reconnect — otherwise every remount (StrictMode in dev, HMR) left an
    // orphaned socket reconnecting forever and handling every update twice.
    let disposed = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      if (disposed) return
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        ws.send('connected')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'preview_update' && data.image) {
            // This channel is shared by every connected browser tab, with no
            // per-connection subscription. Only drop the message when this
            // tab is actively tracking a *different* job — that's the
            // harmful case (another job's colors overwrite the one you're
            // watching). A tab with no job in focus still accepts updates:
            // jobs started outside this tab's own Run button (via the API,
            // or picked up from history) have no other way to be noticed.
            const activeJobId = useAppStore.getState().currentJob?.job_id
            if (activeJobId && data.job_id !== activeJobId) return

            setPreviewImage(`data:image/png;base64,${data.image}`)
            // A completed job's colored PLY may have been regenerated
            // in-place (e.g. from a slider edit) — the URL is otherwise
            // unchanged, so ThreeDView would never refetch it without this.
            bumpPreviewVersion()

            // Only apply slider updates when the backend actually sends a
            // non-empty stack.  Empty payloads must NOT clear the current
            // sliders (that previously made the color core disappear).
            if (data.sliders && Array.isArray(data.sliders) && data.sliders.length > 0) {
              const range = Number.isFinite(data.min_layer) && Number.isFinite(data.max_layer)
                ? { min: data.min_layer, max: data.max_layer }
                : undefined
              applySliders(data.sliders, range)
            }
          }
        } catch {
          // ignore non-JSON messages
        }
      }

      ws.onclose = () => {
        if (!disposed) reconnectTimer = setTimeout(connect, 2000)
      }

      ws.onerror = () => {
        // ignore
      }
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [setPreviewImage, setSliders, bumpPreviewVersion])

  // Poll init status when we have an image, but only up through the
  // pre-run "init preview" phase. Once a real optimization result exists,
  // this must stop entirely: `/api/sliders/from-optimizer` always returns
  // the stack derived from the stored (unedited) solution, never the
  // user's live slider edits, so leaving this interval running after
  // completion re-applied that stale stack over the user's edits every
  // second — the color sliders / color core "snapping back" a moment
  // after every drag, independent of (and in addition to) any network
  // race on the edit itself.
  useEffect(() => {
    if (!inputImage) return
    if (optimizationStarted) return
    if (jobFailed) return
    if (jobHasResult) return

    let cancelled = false
    let lastStatus: string | null = null

    const poll = async () => {
      try {
        const res = await fetch('/api/init/status')
        const data = await res.json()
        if (cancelled) return

        if (data.status !== 'idle') {
          setInitState({ status: data.status })
        }

        // Only on the transition into "ready" — this used to refetch the
        // preview image and re-apply /api/sliders/from-optimizer every
        // second. That endpoint answers for the latest *completed job*, not
        // this preview, so after undoing back to the pre-run phase it kept
        // overwriting the restored sliders with the finished job's stack
        // (and each overwrite became a new history entry, wiping redo).
        if (data.status === 'ready' && lastStatus !== 'ready') {
          if (Number.isFinite(data.min_layer) && Number.isFinite(data.max_layer)) {
            setSliderLayerRange({ min: data.min_layer, max: data.max_layer })
          }
          const previewRes = await fetch('/api/init/preview')
          const previewData = await previewRes.json()
          if (!cancelled && previewData.image) {
            setPreviewImage(`data:image/png;base64,${previewData.image}`)
          }
        }
        lastStatus = data.status
      } catch {
        // ignore
      }
    }

    poll()
    const interval = setInterval(poll, 1000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [inputImage, setPreviewImage, setInitState, setSliderLayerRange, optimizationStarted, jobFailed, jobHasResult])

  const coloredPlyUrl = stlFile && currentJob?.job_id
    ? `/api/outputs/colored-ply/${currentJob.job_id}?v=${previewVersion}`
    : null
  // The auto-preview mesh (api/init.py) — the real heightmap-init result,
  // draped with the original photo, so the 3D view shows actual geometry
  // (not a flat 2D image or the fake uniform-block stackSegments preview)
  // the moment an image + filament are present, before the user has ever
  // clicked Run. `previewVersion` also bumps on every slider-triggered
  // re-render (ColorSliders posts job_id: "__init__" during this phase),
  // so manually assigning colors updates this mesh live.
  const initMeshUrl = !stlFile && initState.status === 'ready'
    ? `/api/init/mesh?v=${previewVersion}`
    : null

  const showNoFilamentsWarning = inputImage && activeFilaments.length === 0 && initState.status !== 'initializing' && !previewImage

  return (
    <div className="flex flex-col h-full bg-gray-800 rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-700">
        <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
          <Box className="w-3.5 h-3.5" />
          3D Preview
        </h3>
      </div>
      <div className="flex-1 relative overflow-hidden">
        {stlFile ? (
          <div data-testid="three-d-view" className="w-full h-full">
            <ThreeDView coloredPlyUrl={coloredPlyUrl} className="w-full h-full" />
          </div>
        ) : optimizationStarted && previewImage ? (
          <img
            src={previewImage}
            alt="Preview"
            className="absolute inset-0 w-full h-full object-contain"
            data-testid="preview-image"
          />
        ) : jobFailed ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-red-400 p-4 overflow-y-auto" data-testid="optimization-error">
            <AlertTriangle className="w-8 h-8 mb-2 opacity-70 flex-shrink-0" />
            <p className="text-xs text-center mb-1">Optimization failed</p>
            {jobError && (() => {
              // friendly_error_message() (backend) leads with a plain-
              // language summary, then a blank line, then the raw
              // exception text (allocator stats for an OOM, a traceback
              // fragment otherwise) — split them so the summary reads as
              // a normal sentence and the technical part is opt-in rather
              // than filling the panel with a wall of text by default.
              const [summary, ...rest] = jobError.split('\n\n')
              const detail = rest.join('\n\n')
              return (
                <div className="max-w-64 text-center">
                  <p className="text-xs text-red-500/80 whitespace-pre-wrap">{summary}</p>
                  {detail && (
                    <details className="mt-2 text-left">
                      <summary className="text-[10px] text-red-500/60 cursor-pointer">Show details</summary>
                      <pre className="mt-1 text-[10px] text-red-500/60 whitespace-pre-wrap break-words max-h-32 overflow-y-auto">{detail}</pre>
                    </details>
                  )}
                </div>
              )
            })()}
          </div>
        ) : initMeshUrl ? (
          <div data-testid="init-three-d-view" className="w-full h-full">
            <ThreeDView coloredPlyUrl={initMeshUrl} className="w-full h-full" />
          </div>
        ) : stackSegments.length > 0 ? (
          <div data-testid="color-stack-preview" className="w-full h-full">
            <ThreeDView stackSegments={stackSegments} className="w-full h-full" />
          </div>
        ) : previewImage ? (
          <img
            src={previewImage}
            alt="Preview"
            className="absolute inset-0 w-full h-full object-contain"
            data-testid="preview-image"
          />
        ) : initState.status === 'initializing' ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-400" data-testid="init-loading">
            <Loader2 className="w-6 h-6 animate-spin mb-2" />
            <span className="text-xs">Initializing heightmap...</span>
          </div>
        ) : showNoFilamentsWarning ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-yellow-400 p-4" data-testid="no-filaments-warning">
            <AlertTriangle className="w-8 h-8 mb-2 opacity-70" />
            <p className="text-xs text-center">Add filaments to the Active Filaments list to generate a 3D preview</p>
          </div>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-500" data-testid="preview-placeholder">
            <Box className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-xs">Upload an image to generate 3D preview</p>
          </div>
        )}
        <div style={{ position: 'absolute', bottom: 8, right: 8, zIndex: 10 }} data-testid="results-history-anchor">
          <ResultsHistory />
        </div>
      </div>
    </div>
  )
}
