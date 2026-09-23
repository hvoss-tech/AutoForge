import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { describeApiError } from '../lib/apiError'
import { IDENTITY_ZOOM, zoomAt, type ZoomTransform } from '../lib/layout'
import { onUiCommand } from '../lib/uiEvents'
import { useFlash } from '../hooks/usePersistentState'
import { AlertTriangle, Image as ImageIcon, RefreshCw, Upload, X, ZoomIn } from 'lucide-react'

type View = 'original' | 'result' | 'split' | 'compare'

const VIEW_LABELS: Record<View, string> = { original: 'Original', result: 'Result', split: 'Side by side', compare: 'Compare' }
const VIEW_HINTS: Partial<Record<View, string>> = { compare: 'One image with a divider you can drag' }

/** Wheel to zoom (about the cursor), drag to pan, double-click to reset.
 * Children position themselves with `transformStyle(zoom)`; viewports given
 * the same zoom stay in lockstep, so both images of a comparison always show
 * the same spot. */
const ZoomViewport: React.FC<{
  zoom: ZoomTransform
  setZoom: React.Dispatch<React.SetStateAction<ZoomTransform>>
  children: React.ReactNode
  testId?: string
}> = ({ zoom, setZoom, children, testId }) => {
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
      className={`relative w-full h-full overflow-hidden ${zoom.scale > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-in'}`}
      onPointerDown={(e) => {
        if (e.button !== 0 || zoom.scale === 1) return
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
      onDoubleClick={() => setZoom(IDENTITY_ZOOM)}
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

  const jobActive = !!currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)
  // The printed result, next to the original: the optimizer's image (or the
  // slider-edited one) once done, the live preview while running.
  const resultSrc = currentJob?.status === 'completed'
    ? `/api/outputs/current-preview/${currentJob.job_id}?v=${previewVersion}`
    : jobActive && previewImage
      ? previewImage
      : null
  const effectiveView: View = resultSrc ? view : 'original'

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

  return (
    <div className={`flex flex-col h-full bg-gray-800 rounded-lg overflow-hidden ${flashing ? 'ring-2 ring-cyan-500' : ''}`} data-testid="image-panel">
      <div className="flex items-center justify-between gap-2 px-3 py-1.5 border-b border-gray-700 min-h-9">
        <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
          <ImageIcon className="w-3.5 h-3.5" />
          Image
        </h3>
        {inputImage && (
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
              {(['original', 'result', 'split', 'compare'] as View[]).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  disabled={v !== 'original' && !resultSrc}
                  aria-pressed={effectiveView === v}
                  title={v !== 'original' && !resultSrc ? 'Run the optimizer to see the result' : VIEW_HINTS[v]}
                  className={`px-2 py-0.5 whitespace-nowrap ${effectiveView === v ? 'bg-gray-600 text-gray-100' : 'text-gray-400 hover:text-gray-200'} disabled:opacity-40 disabled:hover:text-gray-400`}
                  data-testid={`image-view-${v}`}
                >
                  {VIEW_LABELS[v]}
                </button>
              ))}
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-gray-300 hover:bg-gray-700 border border-gray-600"
              title="Upload a different image"
              data-testid="change-image-btn"
            >
              <RefreshCw className="w-3 h-3" />
              Change
            </button>
          </div>
        )}
      </div>

      <div
        className={`flex-1 relative p-2 flex items-center justify-center overflow-hidden min-h-0 ${isDragging && inputImage ? 'ring-2 ring-inset ring-blue-400' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={() => setIsDragging(false)}
        title={inputImage ? 'Scroll to zoom, drag to pan, double-click to reset' : undefined}
      >
        {fileInput}
        {inputImage ? (
          <>
            {effectiveView === 'original' && viewport(originalImg)}
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
