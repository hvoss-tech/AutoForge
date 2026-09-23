import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { describeApiError } from '../lib/apiError'
import { IDENTITY_ZOOM, zoomAt, type ZoomTransform } from '../lib/layout'
import { onUiCommand } from '../lib/uiEvents'
import { useElementWidth, useFlash, usePersistentState } from '../hooks/usePersistentState'
import { type ImageView as View } from '../store/appStore'
import {
  AlertTriangle,
  Brush,
  Check,
  Columns2,
  Crosshair,
  Eraser,
  Flame,
  Image as ImageIcon,
  Layers,
  Loader2,
  RefreshCw,
  SplitSquareHorizontal,
  Trash2,
  Upload,
  X,
  ZoomIn,
} from 'lucide-react'
import { FocusMaskOverlay, useFocusMask, type FocusTool } from './FocusMask'
import { DifferenceImage, DifferenceLegend, useDifferenceMap } from './DifferenceView'
import { DEFAULT_FOCUS_STRENGTH, sliderFromStrength, strengthFromSlider } from '../lib/focusMask'

const VIEWS: View[] = ['original', 'result', 'split', 'compare', 'difference']
const VIEW_LABELS: Record<View, string> = { original: 'Original', result: 'Result', split: 'Side by side', compare: 'Compare', difference: 'Differences' }
const VIEW_HINTS: Record<View, string> = {
  original: 'The picture you uploaded',
  result: 'What the print will look like',
  split: 'Picture and result next to each other',
  compare: 'One image with a divider you can drag',
  difference: 'Where the print differs from the picture — bright areas differ most',
}
const VIEW_ICONS: Record<View, React.FC<{ className?: string }>> = {
  original: ImageIcon,
  result: Layers,
  split: Columns2,
  compare: SplitSquareHorizontal,
  difference: Flame,
}
/** Header width from which every view button shows its label (below it,
 * icons only, except the selected one). */
const WIDE_HEADER = 640

/** Wheel to zoom (about the cursor), drag to pan, double-click to reset.
 * Children position themselves with `transformStyle(zoom)`; viewports given
 * the same zoom stay in lockstep, so both images of a comparison always show
 * the same spot. */
const ZoomViewport: React.FC<{
  zoom: ZoomTransform
  setZoom: React.Dispatch<React.SetStateAction<ZoomTransform>>
  children: React.ReactNode
  testId?: string
  /** Left button paints (the focus-area brush): pan with the middle or
   * right button instead, and don't reset on double-click. */
  paintMode?: boolean
}> = ({ zoom, setZoom, children, testId, paintMode = false }) => {
  const ref = useRef<HTMLDivElement>(null)
  const pan = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    // React's onWheel is passive, so it can't stop the page from scrolling.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      setZoom((t) => zoomAt(t, e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - rect.left, e.clientY - rect.top))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [setZoom])

  return (
    <div
      ref={ref}
      className={`relative w-full h-full overflow-hidden ${paintMode ? '' : zoom.scale > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-in'}`}
      onContextMenu={paintMode ? (e) => e.preventDefault() : undefined}
      onPointerDown={(e) => {
        if (paintMode ? e.button !== 1 && e.button !== 2 : e.button !== 0) return
        if (zoom.scale === 1) return
        e.currentTarget.setPointerCapture(e.pointerId)
        pan.current = { x: e.clientX, y: e.clientY, tx: zoom.x, ty: zoom.y }
      }}
      onPointerMove={(e) => {
        const p = pan.current
        if (p) setZoom((t) => ({ ...t, x: p.tx + e.clientX - p.x, y: p.ty + e.clientY - p.y }))
      }}
      onPointerUp={() => {
        pan.current = null
      }}
      onDoubleClick={paintMode ? undefined : () => setZoom(IDENTITY_ZOOM)}
      data-testid={testId}
    >
      {children}
    </div>
  )
}

const transformStyle = (t: ZoomTransform): React.CSSProperties => ({
  transform: `translate(${t.x}px, ${t.y}px) scale(${t.scale})`,
  transformOrigin: '0 0',
})

export const InputImagePanel: React.FC = () => {
  const inputImage = useAppStore((s) => s.inputImage)
  const applyUploadedImage = useAppStore((s) => s.applyUploadedImage)
  const currentJob = useAppStore((s) => s.currentJob)
  const previewImage = useAppStore((s) => s.previewImage)
  const previewVersion = useAppStore((s) => s.previewVersion)
  const view = useAppStore((s) => s.imageView)
  const setView = useAppStore((s) => s.setImageView)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [zoom, setZoom] = useState<ZoomTransform>(IDENTITY_ZOOM)
  const [divider, setDivider] = useState(50)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [flashing, flash] = useFlash()
  const headerRef = useRef<HTMLDivElement>(null)
  const headerWidth = useElementWidth(headerRef)
  // Narrow panels (laptop screens) keep every control on one row by
  // dropping labels, the least important ones first.
  const viewLabels = headerWidth === 0 || headerWidth >= 560
  const focusLabel = headerWidth === 0 || headerWidth >= 480
  const changeLabel = headerWidth === 0 || headerWidth >= 700

  // Focus areas: painting mode, tool and brush size.
  const focusMask = useFocusMask()
  const hasFocusAreas = useAppStore((s) => !!s.settings.priority_mask)
  const focusStrength = useAppStore((s) => s.settings.priority_mask_strength || DEFAULT_FOCUS_STRENGTH)
  const setFocusStrength = (strength: number) => {
    const settings = useAppStore.getState().settings
    if (settings.priority_mask_strength !== strength) useAppStore.getState().setSettings({ ...settings, priority_mask_strength: strength })
  }
  const [painting, setPainting] = useState(false)
  const [tool, setTool] = useState<FocusTool>('paint')
  const [brushPercent, setBrushPercent] = usePersistentState<number>('autoforge-focus-brush-size', 6)

  const jobActive = !!currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)
  // The printed result, next to the original: the optimizer's image (or the
  // slider-edited one) once done, the live preview while running.
  const resultSrc = currentJob?.status === 'completed'
    ? `/api/outputs/current-preview/${currentJob.job_id}?v=${previewVersion}`
    : jobActive && previewImage
      ? previewImage
      : null
  const effectiveView: View = painting ? 'original' : resultSrc ? view : 'original'
  // Only compared while the Differences view is on screen.
  const diff = useDifferenceMap(effectiveView === 'difference' ? inputImage : null, effectiveView === 'difference' ? resultSrc : null)

  const startPainting = useCallback(() => {
    setTool('paint')
    setPainting(true)
  }, [])

  // A new picture ends painting (its mask starts empty).
  useEffect(() => setPainting(false), [inputImage])

  // Keys while painting: Esc or Enter finishes, B/E switch brush and
  // eraser, [ and ] change the brush size.
  useEffect(() => {
    if (!painting) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' && (target as HTMLInputElement).type !== 'range' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const key = e.key.toLowerCase()
      if (key === 'escape' || key === 'enter') setPainting(false)
      else if (key === 'b') setTool('paint')
      else if (key === 'e') setTool('erase')
      else if (key === '[') setBrushPercent(Math.max(1, brushPercent - 1))
      else if (key === ']') setBrushPercent(Math.min(25, brushPercent + 1))
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [painting, brushPercent, setBrushPercent])

  // Switch to side by side when a run finishes, so the result isn't hidden.
  const prevStatus = useRef(currentJob?.status)
  useEffect(() => {
    if (prevStatus.current && prevStatus.current !== 'completed' && currentJob?.status === 'completed') setView('split')
    prevStatus.current = currentJob?.status
  }, [currentJob?.status, setView])

  // A new picture starts unzoomed.
  useEffect(() => setZoom(IDENTITY_ZOOM), [inputImage])

  // The "Image" workflow step: pick a file, or point at the image already there.
  useEffect(
    () =>
      onUiCommand('focus-image', () => {
        if (useAppStore.getState().inputImage) flash()
        else fileInputRef.current?.click()
      }),
    [flash],
  )

  // Two uploads can be in flight at once (a fast double-drop, or a drop
  // right after picking a file) with no guarantee the network resolves them
  // in the order they started — without a sequence guard, an older upload
  // finishing last would silently win and show its (stale) image under
  // the newer request. Only the most recently *started* upload's result is
  // ever applied; every other one is dropped, success or failure alike.
  const uploadSeqRef = useRef(0)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      // Used to be ignored without any feedback.
      setUploadError(`"${file.name}" isn't an image. Use a PNG or JPEG file.`)
      return
    }
    setUploadError(null)
    const seq = ++uploadSeqRef.current

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/images/upload', { method: 'POST', body: formData })
      if (!response.ok) {
        throw new Error(`Upload failed: ${describeApiError(await response.json().catch(() => null), response.status)}`)
      }
      const data = await response.json()
      if (!data.filename) throw new Error('Upload failed: server did not return a filename')
      if (uploadSeqRef.current !== seq) return // superseded by a later upload

      // One action, because everything derived from the old image has to go
      // at the same moment the new one arrives — the previous result, its
      // mesh, the live preview and the server's auto-preview state. Setting
      // just the filename and the URL left the old picture in the 3D panel
      // (and the old heightmap being served as the new image's preview).
      await applyUploadedImage(data.filename, URL.createObjectURL(file))
      // Init is triggered reactively by useAutoPreviewInit.
    } catch (e) {
      if (uploadSeqRef.current !== seq) return // superseded; not worth surfacing
      // No local blob preview on failure: Run must not look ready for an
      // image the server never received.
      console.error('Failed to upload image:', e)
      setUploadError(e instanceof Error ? e.message : 'Failed to upload image')
    }
  }, [applyUploadedImage])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes('Files')) return
    e.preventDefault()
    setIsDragging(true)
  }, [])

  // Bringing another file over means the old error no longer applies.
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) setUploadError(null)
  }, [])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0])
    e.target.value = ''
  }, [handleFile])

  const fileInput = (
    <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileInput} data-testid="image-file-input" />
  )

  const errorBox = uploadError && (
    <div className="flex items-start gap-2 px-3 py-2 rounded bg-red-600/15 border border-red-600/40 text-xs text-red-600" role="alert" data-testid="image-upload-error">
      <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <div>{uploadError}</div>
        <div className="text-gray-400 mt-0.5">Try another file — PNG and JPEG work best.</div>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation()
          setUploadError(null)
        }}
        className="text-red-500 hover:text-red-400"
        aria-label="Dismiss"
        data-testid="image-upload-error-dismiss"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )

  const zoomed = zoom.scale > 1

  const viewport = (content: React.ReactNode) => (
    <ZoomViewport zoom={zoom} setZoom={setZoom}>
      <div className="w-full h-full" style={transformStyle(zoom)}>
        {content}
      </div>
    </ZoomViewport>
  )

  const originalImg = inputImage && (
    <img src={inputImage} alt="Original" draggable={false} className="w-full h-full object-contain select-none" data-testid="input-image" />
  )
  const resultImg = resultSrc && (
    <img src={resultSrc} alt="Result" draggable={false} className="w-full h-full object-contain select-none [image-rendering:pixelated]" data-testid="result-image" />
  )

  const startDividerDrag = (e: React.PointerEvent) => {
    e.stopPropagation()
    const container = (e.currentTarget as HTMLElement).parentElement
    if (!container) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    const move = (ev: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      setDivider(Math.min(100, Math.max(0, ((ev.clientX - rect.left) / rect.width) * 100)))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const compareView = (
    <ZoomViewport zoom={zoom} setZoom={setZoom} testId="image-compare">
      {/* Both layers get the same transform; only the result's clip is in
          screen space, so the divider always lines up with the cut. */}
      <div className="absolute inset-0" style={transformStyle(zoom)}>{originalImg}</div>
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${divider}%)` }}>
        <div className="w-full h-full" style={transformStyle(zoom)}>{resultImg}</div>
      </div>
      <span className="absolute top-1 left-2 px-1.5 rounded bg-black/50 text-[11px] text-white pointer-events-none">Original</span>
      <span className="absolute top-1 right-2 px-1.5 rounded bg-black/50 text-[11px] text-white pointer-events-none">Result</span>
      <div className="absolute top-0 bottom-0 w-0.5 bg-white/80 pointer-events-none" style={{ left: `${divider}%` }} />
      <div
        role="slider"
        tabIndex={0}
        aria-label="Compare divider"
        aria-valuenow={Math.round(divider)}
        aria-valuemin={0}
        aria-valuemax={100}
        onPointerDown={startDividerDrag}
        onDoubleClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') setDivider((d) => Math.max(0, d - 5))
          else if (e.key === 'ArrowRight') setDivider((d) => Math.min(100, d + 5))
        }}
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-5 h-8 rounded bg-white shadow cursor-ew-resize flex items-center justify-center text-[10px] text-gray-900 select-none"
        style={{ left: `${divider}%` }}
        data-testid="compare-divider"
      >
        ⇔
      </div>
    </ZoomViewport>
  )

  const paintView = (
    <div className="w-full h-full flex flex-col min-h-0" data-testid="focus-paint-view">
      <div className="flex-1 min-h-0">
        <ZoomViewport zoom={zoom} setZoom={setZoom} paintMode testId="focus-paint-viewport">
          <div className="absolute inset-0" style={transformStyle(zoom)}>
            {originalImg}
            <FocusMaskOverlay mask={focusMask} painting tool={tool} brushPercent={brushPercent} />
          </div>
        </ZoomViewport>
      </div>
      <div className="flex flex-col gap-1 px-3 py-1.5 border-t border-gray-700 text-[11px] text-gray-400" data-testid="focus-hint">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="text-gray-300">Paint over what matters most — faces, eyes, lettering.</span>
          <label
            className="flex items-center gap-1.5 text-gray-300"
            title="How many times more a painted area counts than the rest when the result is compared with your picture (2× to 100×, default 10×). The rest still counts, just less. Applies to the whole picture and is used from the next run."
          >
            Painted areas count
            <input
              type="range"
              min={0}
              max={100}
              value={sliderFromStrength(focusStrength)}
              onChange={(e) => setFocusStrength(strengthFromSlider(Number(e.target.value)))}
              className="w-20 xl:w-24 accent-cyan-500"
              aria-label="Focus strength"
              aria-valuetext={`${focusStrength} times`}
              data-testid="focus-strength"
            />
            <span className="w-8 text-right font-medium text-cyan-300 tabular-nums" data-testid="focus-strength-value">{focusStrength}×</span>
            more
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-x-3">
          <span>Scroll to zoom · right-drag to pan · B / E brush and eraser · [ ] size</span>
          <span className="ml-auto tabular-nums" data-testid="focus-share">
            {focusMask.saving ? 'Saving…' : focusMask.share > 0 ? `${Math.max(1, Math.round(focusMask.share * 100))}% marked · used from the next run` : 'Nothing marked yet'}
          </span>
        </div>
      </div>
    </div>
  )

  const segButton = (on: boolean) =>
    `flex items-center gap-1 px-2 py-0.5 whitespace-nowrap ${on ? 'bg-gray-600 text-gray-100' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/60'} disabled:opacity-40 disabled:hover:text-gray-400 disabled:hover:bg-transparent`
  const outlineButton = 'flex items-center gap-1 px-2 py-0.5 rounded text-xs text-gray-300 hover:bg-gray-700 border border-gray-600 whitespace-nowrap'

  const paintToolbar = (
    <div className="flex items-center gap-2 min-w-0" data-testid="focus-toolbar">
      <div className="flex rounded border border-gray-600 overflow-hidden text-xs" role="group" aria-label="Focus brush">
        <button onClick={() => setTool('paint')} aria-pressed={tool === 'paint'} className={segButton(tool === 'paint')} title="Paint focus areas (B)" data-testid="focus-tool-paint">
          <Brush className="w-3.5 h-3.5" />
          {focusLabel && 'Paint'}
        </button>
        <button onClick={() => setTool('erase')} aria-pressed={tool === 'erase'} className={segButton(tool === 'erase')} title="Erase focus areas (E)" data-testid="focus-tool-erase">
          <Eraser className="w-3.5 h-3.5" />
          {focusLabel && 'Erase'}
        </button>
      </div>
      <label className="flex items-center gap-1.5 text-xs text-gray-400" title="Brush size ([ and ] keys)">
        Size
        <input
          type="range"
          min={1}
          max={25}
          value={brushPercent}
          onChange={(e) => setBrushPercent(Number(e.target.value))}
          className="w-16 xl:w-20 accent-cyan-500"
          aria-label="Brush size"
          data-testid="focus-brush-size"
        />
      </label>
      <button
        onClick={focusMask.clear}
        disabled={focusMask.share === 0 && !hasFocusAreas}
        className={`${outlineButton} disabled:opacity-40 disabled:hover:bg-transparent`}
        title="Remove all focus areas"
        data-testid="focus-clear-btn"
      >
        <Trash2 className="w-3 h-3" />
        {changeLabel && 'Clear'}
      </button>
      <button
        onClick={() => setPainting(false)}
        className="flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white whitespace-nowrap"
        title="Finish painting (Esc)"
        data-testid="focus-done-btn"
      >
        <Check className="w-3.5 h-3.5" />
        Done
      </button>
    </div>
  )

  const viewControls = (
    <div className="flex items-center gap-2 min-w-0">
      {zoomed && (
        <button
          onClick={() => setZoom(IDENTITY_ZOOM)}
          className="flex items-center gap-1 px-1.5 py-0.5 rounded text-xs text-gray-300 hover:bg-gray-700 tabular-nums"
          title="Reset zoom (or double-click the image)"
          data-testid="image-zoom-reset"
        >
          <ZoomIn className="w-3 h-3" />
          {zoom.scale.toFixed(1)}×
        </button>
      )}
      <div className="flex rounded border border-gray-600 overflow-hidden text-xs" role="group" aria-label="Image view" data-testid="image-view-toggle">
        {VIEWS.map((v) => {
          const Icon = VIEW_ICONS[v]
          const selected = effectiveView === v
          return (
            <button
              key={v}
              onClick={() => setView(v)}
              disabled={v !== 'original' && !resultSrc}
              aria-pressed={selected}
              aria-label={VIEW_LABELS[v]}
              title={v !== 'original' && !resultSrc ? 'Run the optimizer to see the result' : `${VIEW_LABELS[v]} — ${VIEW_HINTS[v]}`}
              className={segButton(selected)}
              data-testid={`image-view-${v}`}
            >
              {!viewLabels && <Icon className="w-3.5 h-3.5" />}
              {(viewLabels || selected) && VIEW_LABELS[v]}
            </button>
          )
        })}
      </div>
      <button
        onClick={startPainting}
        className={`relative ${outlineButton} ${hasFocusAreas ? 'border-cyan-600 text-cyan-300' : ''}`}
        title={
          hasFocusAreas
            ? 'Focus areas are set — the optimizer works hardest there. Click to change them.'
            : 'Mark the parts of the picture that matter most (faces, eyes, text)'
        }
        aria-label="Focus areas"
        data-testid="focus-areas-btn"
        data-active={hasFocusAreas || undefined}
      >
        <Crosshair className="w-3 h-3" />
        {focusLabel && 'Focus'}
        {hasFocusAreas && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" aria-hidden data-testid="focus-areas-dot" />}
      </button>
      <button onClick={() => fileInputRef.current?.click()} className={outlineButton} title="Upload a different image" aria-label="Change image" data-testid="change-image-btn">
        <RefreshCw className="w-3 h-3" />
        {changeLabel && 'Change'}
      </button>
    </div>
  )

  return (
    <div className={`flex flex-col h-full bg-gray-800 rounded-lg overflow-hidden ${flashing ? 'ring-2 ring-cyan-500' : ''} ${painting ? 'ring-1 ring-cyan-600' : ''}`} data-testid="image-panel">
      <div ref={headerRef} className="flex items-center justify-between gap-2 px-3 py-1.5 border-b border-gray-700 min-h-9">
        <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1 flex-shrink-0">
          {painting ? <Crosshair className="w-3.5 h-3.5 text-cyan-400" /> : <ImageIcon className="w-3.5 h-3.5" />}
          {painting ? 'Focus areas' : 'Image'}
        </h3>
        {inputImage && (painting ? paintToolbar : viewControls)}
      </div>

      <div
        className={`flex-1 relative p-2 flex items-center justify-center overflow-hidden min-h-0 ${isDragging && inputImage ? 'ring-2 ring-inset ring-blue-400' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={() => setIsDragging(false)}
        title={inputImage && !painting ? 'Scroll to zoom, drag to pan, double-click to reset' : undefined}
      >
        {fileInput}
        {inputImage ? (
          <>
            {effectiveView === 'original' && (painting ? paintView : viewport(originalImg))}
            {effectiveView === 'result' && viewport(resultImg)}
            {effectiveView === 'split' && (
              <div className="w-full h-full grid grid-cols-2 gap-2">
                <figure className="relative min-h-0 flex flex-col">
                  <div className="flex-1 min-h-0">{viewport(originalImg)}</div>
                  <figcaption className="text-center text-[11px] text-gray-400 pt-1">Original</figcaption>
                </figure>
                <figure className="relative min-h-0 flex flex-col">
                  <div className="flex-1 min-h-0">{viewport(resultImg)}</div>
                  <figcaption className="text-center text-[11px] text-gray-400 pt-1">Result</figcaption>
                </figure>
              </div>
            )}
            {effectiveView === 'compare' && compareView}
            {effectiveView === 'difference' && (
              <div className="w-full h-full flex flex-col min-h-0" data-testid="image-difference">
                <div className="flex-1 min-h-0 relative">
                  {viewport(<DifferenceImage canvas={diff.canvas} version={diff.version} />)}
                  {diff.status === 'working' && !diff.canvas && (
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-400 gap-1.5">
                      <Loader2 className="w-4 h-4 animate-spin" /> Comparing…
                    </div>
                  )}
                </div>
                <DifferenceLegend status={diff.status} stats={diff.stats} onPaintFocus={startPainting} />
              </div>
            )}
            {uploadError && <div className="absolute left-2 right-2 bottom-2">{errorBox}</div>}
          </>
        ) : (
          <div
            className={`w-full h-full flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg cursor-pointer transition-colors p-4 ${
              isDragging ? 'border-blue-400 bg-blue-500/10' : uploadError ? 'border-red-500/60' : 'border-gray-600 hover:border-gray-500'
            }`}
            onClick={() => fileInputRef.current?.click()}
            data-testid="image-drop-zone"
          >
            <Upload className="w-8 h-8 text-gray-500" />
            <p className="text-sm text-gray-300">Drop an image here or click to upload</p>
            <p className="text-xs text-gray-400">PNG or JPEG. Pictures with clear shapes and contrast work best.</p>
            {errorBox}
          </div>
        )}
      </div>
    </div>
  )
}
