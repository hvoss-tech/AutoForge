import React, { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowLeftRight, Check, ChevronLeft, ChevronRight, Columns2, ImageOff, Loader2, Minimize2, SplitSquareHorizontal } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { useAppStore } from '../store/appStore'
import { IDENTITY_ZOOM, type ZoomTransform } from '../lib/layout'
import { ZoomViewport, transformStyle } from './ZoomViewport'
import {
  bestOfImageUrl,
  describeLossGap,
  galleryGrid,
  isChoosingBestOf,
  lowestLossRun,
  stepRun,
  toggleCompareRun,
} from '../lib/bestOf'

/** A side of the A/B comparison: a run (0-based), or the uploaded picture. */
type Side = number | 'original'
type View = { kind: 'grid' } | { kind: 'inspect'; run: number } | { kind: 'compare'; a: Side; b: Side; mode: 'slider' | 'split' }

const buttonBase = 'flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium disabled:cursor-not-allowed whitespace-nowrap'
const primaryButton = `${buttonBase} bg-blue-600 hover:bg-blue-500 text-white disabled:bg-gray-700 disabled:text-gray-400`
const outlineButton = `${buttonBase} border border-gray-600 text-gray-200 hover:bg-gray-800 disabled:text-gray-500 disabled:hover:bg-transparent`
const iconButton = 'p-1.5 rounded text-gray-300 hover:text-gray-100 hover:bg-gray-700'
const label = 'absolute px-1.5 rounded bg-black/60 text-[11px] text-white pointer-events-none'

/** Every run of a best-of series, once all of them are done: look at them
 * side by side, zoom into one, compare two A/B, and keep one. Keeping it
 * finishes the job like a single run. Opens by itself when the series is
 * done; hidden, the top bar's "Choose a result" brings it back. */
export const BestOfGallery: React.FC = () => {
  const job = useAppStore((s) => s.currentJob)
  const open = useAppStore((s) => s.bestOfGalleryOpen)
  const setOpen = useAppStore((s) => s.setBestOfGalleryOpen)
  const choosing = isChoosingBestOf(job)
  const jobId = choosing && job ? job.job_id : null
  // Opened once per series; hiding it must not reopen it on the next update.
  const openedFor = useRef<string | null>(null)

  useEffect(() => {
    if (jobId && openedFor.current !== jobId) {
      openedFor.current = jobId
      setOpen(true)
    } else if (!jobId) {
      setOpen(false)
    }
  }, [jobId, setOpen])

  if (!jobId || !job?.best_of_losses) return null
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Gallery key={jobId} jobId={jobId} losses={job.best_of_losses} onHide={() => setOpen(false)} />
    </Dialog>
  )
}

const Gallery: React.FC<{ jobId: string; losses: number[]; onHide: () => void }> = ({ jobId, losses, onHide }) => {
  const inputImage = useAppStore((s) => s.inputImage)
  const pickBestOfRun = useAppStore((s) => s.pickBestOfRun)
  const pushToast = useAppStore((s) => s.pushToast)
  const [view, setView] = useState<View>({ kind: 'grid' })
  const [marked, setMarked] = useState<number[]>([])
  const [zoom, setZoom] = useState<ZoomTransform>(IDENTITY_ZOOM)
  const [picking, setPicking] = useState<number | null>(null)
  const best = lowestLossRun(losses)
  const count = losses.length

  const show = (next: View) => {
    // Flipping between runs keeps the zoom (the same detail in each);
    // going back to all runs starts the next look unzoomed.
    if (next.kind === 'grid') setZoom(IDENTITY_ZOOM)
    setView(next)
  }

  const keep = async (run: number) => {
    setPicking(run)
    try {
      await pickBestOfRun(jobId, run)
      onHide()
    } catch (e) {
      pushToast(`Couldn't keep run ${run + 1}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setPicking(null)
    }
  }

  const compareFrom = (run: number) => {
    const other = marked.find((r) => r !== run) ?? (run !== best ? best : stepRun(run, 1, count))
    show({ kind: 'compare', a: run, b: other, mode: 'slider' })
  }

  // Arrow keys flip through the runs while one is inspected.
  useEffect(() => {
    if (view.kind !== 'inspect') return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if ((e.target as HTMLElement | null)?.closest('input, select, textarea')) return
      e.preventDefault()
      setView({ kind: 'inspect', run: stepRun(view.run, e.key === 'ArrowLeft' ? -1 : 1, count) })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view, count])

  const sideSrc = (side: Side) => (side === 'original' ? inputImage : bestOfImageUrl(jobId, side))
  const sideName = (side: Side) => (side === 'original' ? 'Original' : `Run ${side + 1}`)
  const busy = picking !== null

  return (
    <DialogContent
      className="max-w-[96vw] w-[96vw] h-[92vh] flex flex-col p-0 gap-0 bg-gray-900 border-gray-700"
      data-testid="best-of-gallery"
      data-view={view.kind}
      // Esc goes back to all runs first; from there it hides the gallery.
      onEscapeKeyDown={(e) => {
        if (view.kind !== 'grid') {
          e.preventDefault()
          show({ kind: 'grid' })
        }
      }}
    >
      <div className="flex items-start justify-between gap-4 px-4 py-3 border-b border-gray-700">
        <div className="min-w-0">
          <DialogTitle className="text-sm font-semibold text-gray-100 leading-none tracking-normal">
            Choose a result · {count} runs
          </DialogTitle>
          <DialogDescription className="mt-1.5 text-xs text-gray-400">
            A lower loss is closer to your picture on average, but it can't judge everything: look at faces, edges and fine
            details, and keep the run you like best.
          </DialogDescription>
        </div>
        <button onClick={onHide} className={outlineButton} title="Hide the gallery; the top bar brings it back" data-testid="best-of-hide">
          <Minimize2 className="w-3.5 h-3.5" />
          Hide
        </button>
      </div>

      {view.kind === 'grid' && (
        <GridView
          jobId={jobId}
          losses={losses}
          best={best}
          marked={marked}
          onInspect={(run) => show({ kind: 'inspect', run })}
          onMark={(run) => setMarked((m) => toggleCompareRun(m, run))}
          onCompare={() => marked.length === 2 && show({ kind: 'compare', a: marked[0], b: marked[1], mode: 'slider' })}
        />
      )}

      {view.kind === 'inspect' && (
        <>
          <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-gray-700">
            <button onClick={() => show({ kind: 'grid' })} className={outlineButton} data-testid="best-of-back">
              <ArrowLeft className="w-3.5 h-3.5" />
              All runs
            </button>
            <button onClick={() => setView({ kind: 'inspect', run: stepRun(view.run, -1, count) })} className={iconButton} aria-label="Previous run" data-testid="best-of-prev">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-sm font-medium text-gray-100 tabular-nums" data-testid="best-of-inspect-run">
              Run {view.run + 1} <span className="text-gray-500 font-normal">of {count}</span>
            </span>
            <button onClick={() => setView({ kind: 'inspect', run: stepRun(view.run, 1, count) })} className={iconButton} aria-label="Next run" data-testid="best-of-next">
              <ChevronRight className="w-4 h-4" />
            </button>
            <LossText losses={losses} run={view.run} />
            <div className="ml-auto flex items-center gap-2">
              <button onClick={() => compareFrom(view.run)} className={outlineButton} data-testid="best-of-compare-from">
                <SplitSquareHorizontal className="w-3.5 h-3.5" />
                Compare A/B
              </button>
              <KeepButton run={view.run} picking={picking} disabled={busy} onKeep={keep} />
            </div>
          </div>
          <div className="relative flex-1 min-h-0 bg-gray-950">
            <ZoomViewport zoom={zoom} setZoom={setZoom} testId="best-of-inspect-viewport">
              <div className="w-full h-full" style={transformStyle(zoom)}>
                <RunImage src={bestOfImageUrl(jobId, view.run)} alt={`Run ${view.run + 1}`} testId="best-of-inspect-image" />
              </div>
            </ZoomViewport>
          </div>
          <Hint text="Scroll to zoom · drag to pan · double-click to reset · ← → other runs · Esc back to all runs" zoom={zoom} />
        </>
      )}

      {view.kind === 'compare' && (
        <>
          <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-gray-700">
            <button onClick={() => show({ kind: 'grid' })} className={outlineButton} data-testid="best-of-back">
              <ArrowLeft className="w-3.5 h-3.5" />
              All runs
            </button>
            <SidePicker name="A" value={view.a} losses={losses} hasOriginal={!!inputImage} onChange={(a) => setView({ ...view, a })} />
            <button
              onClick={() => setView({ ...view, a: view.b, b: view.a })}
              className={iconButton}
              title="Swap A and B"
              aria-label="Swap A and B"
              data-testid="best-of-swap"
            >
              <ArrowLeftRight className="w-4 h-4" />
            </button>
            <SidePicker name="B" value={view.b} losses={losses} hasOriginal={!!inputImage} onChange={(b) => setView({ ...view, b })} />
            <div className="flex rounded border border-gray-600 overflow-hidden ml-2" role="group" aria-label="Comparison layout">
              {(['slider', 'split'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setView({ ...view, mode })}
                  aria-pressed={view.mode === mode}
                  className={`flex items-center gap-1 px-2.5 py-1.5 text-xs ${view.mode === mode ? 'bg-gray-700 text-gray-100' : 'text-gray-300 hover:bg-gray-800'}`}
                  data-testid={`best-of-mode-${mode}`}
                >
                  {mode === 'slider' ? <SplitSquareHorizontal className="w-3.5 h-3.5" /> : <Columns2 className="w-3.5 h-3.5" />}
                  {mode === 'slider' ? 'Slider' : 'Side by side'}
                </button>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-2">
              {[view.a, view.b].map((side, i) =>
                side === 'original' ? null : (
                  <KeepButton key={i} run={side} picking={picking} disabled={busy} onKeep={keep} prefix={i === 0 ? 'A' : 'B'} />
                ),
              )}
            </div>
          </div>
          <div className="relative flex-1 min-h-0 bg-gray-950" data-testid="best-of-compare">
            {view.mode === 'slider' ? (
              <SliderCompare
                zoom={zoom}
                setZoom={setZoom}
                a={{ src: sideSrc(view.a), name: `A · ${sideName(view.a)}` }}
                b={{ src: sideSrc(view.b), name: `B · ${sideName(view.b)}` }}
              />
            ) : (
              <div className="flex h-full gap-1">
                {[
                  { side: view.a, tag: 'A' },
                  { side: view.b, tag: 'B' },
                ].map(({ side, tag }) => (
                  <div key={tag} className="relative flex-1 min-w-0">
                    <ZoomViewport zoom={zoom} setZoom={setZoom} testId={`best-of-split-${tag.toLowerCase()}`}>
                      <div className="w-full h-full" style={transformStyle(zoom)}>
                        <RunImage src={sideSrc(side)} alt={sideName(side)} />
                      </div>
                    </ZoomViewport>
                    <span className={`${label} top-1 left-2`}>{tag} · {sideName(side)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <Hint
            text={`Scroll to zoom · drag to pan · double-click to reset${view.mode === 'slider' ? ' · drag the divider' : ' · both sides move together'} · Esc back to all runs`}
            zoom={zoom}
          />
        </>
      )}
    </DialogContent>
  )
}

const GridView: React.FC<{
  jobId: string
  losses: number[]
  best: number
  marked: number[]
  onInspect: (run: number) => void
  onMark: (run: number) => void
  onCompare: () => void
}> = ({ jobId, losses, best, marked, onInspect, onMark, onCompare }) => {
  const { columns, rows } = galleryGrid(losses.length)
  return (
    <>
      <div className="flex flex-wrap items-center gap-3 px-4 py-2 border-b border-gray-700 text-xs text-gray-400">
        <span>Click a run to look at it closely.</span>
        <div className="ml-auto flex items-center gap-2">
          <span data-testid="best-of-compare-hint">
            {marked.length === 2 ? `Run ${marked[0] + 1} and run ${marked[1] + 1} marked` : 'Mark two runs to compare them A/B'}
          </span>
          <button onClick={onCompare} disabled={marked.length !== 2} className={primaryButton} data-testid="best-of-compare-btn">
            <SplitSquareHorizontal className="w-3.5 h-3.5" />
            Compare A/B
          </button>
        </div>
      </div>
      <div
        className="flex-1 min-h-0 overflow-auto p-3 grid gap-3"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${rows}, minmax(160px, 1fr))` }}
        data-testid="best-of-grid"
        data-columns={columns}
        data-rows={rows}
      >
        {losses.map((_loss, run) => {
          const mark = marked.indexOf(run)
          return (
            <div
              key={run}
              role="button"
              tabIndex={0}
              onClick={() => onInspect(run)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onInspect(run)
                }
              }}
              className={`group flex flex-col min-h-0 rounded-lg border overflow-hidden cursor-zoom-in bg-gray-950 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                mark >= 0 ? 'border-amber-400 ring-1 ring-amber-400' : 'border-gray-700 hover:border-blue-500'
              }`}
              aria-label={`Run ${run + 1}, loss ${losses[run].toFixed(4)}`}
              data-testid={`best-of-card-${run}`}
              data-marked={mark >= 0}
            >
              <div className="relative flex-1 min-h-0">
                <RunImage src={bestOfImageUrl(jobId, run)} alt={`Run ${run + 1}`} />
                {run === best && (
                  <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-emerald-600/90 text-[11px] font-medium text-white pointer-events-none" data-testid="best-of-lowest">
                    Lowest loss
                  </span>
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onMark(run)
                  }}
                  onKeyDown={(e) => e.stopPropagation()}
                  aria-pressed={mark >= 0}
                  className={`absolute top-1.5 right-1.5 flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                    mark >= 0 ? 'bg-amber-400 text-gray-900' : 'bg-black/60 text-gray-100 opacity-80 group-hover:opacity-100 hover:bg-black/80'
                  }`}
                  title={mark >= 0 ? 'Unmark' : 'Mark for an A/B comparison'}
                  data-testid={`best-of-mark-${run}`}
                >
                  {mark >= 0 ? (mark === 0 ? 'A' : 'B') : 'Compare'}
                </button>
              </div>
              <div className="flex items-center gap-2 px-2.5 py-1.5 border-t border-gray-800 text-xs">
                <span className="font-medium text-gray-100">Run {run + 1}</span>
                <LossText losses={losses} run={run} />
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}

/** "Loss 1.2345 · +3.2%" — how close a run is, and how far behind the best. */
const LossText: React.FC<{ losses: number[]; run: number }> = ({ losses, run }) => {
  const gap = describeLossGap(losses, run)
  return (
    <span className="text-xs text-gray-400 tabular-nums" title="Lower is closer to the input image" data-testid="best-of-loss">
      Loss {losses[run].toFixed(4)}
      <span className={gap === 'Lowest loss' ? 'text-emerald-500' : 'text-gray-500'}> · {gap}</span>
    </span>
  )
}

const KeepButton: React.FC<{ run: number; picking: number | null; disabled: boolean; onKeep: (run: number) => void; prefix?: string }> = ({
  run,
  picking,
  disabled,
  onKeep,
  prefix,
}) => (
  <button onClick={() => onKeep(run)} disabled={disabled} className={primaryButton} data-testid={`best-of-keep-${run}`}>
    {picking === run ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
    {prefix ? `Keep ${prefix} (run ${run + 1})` : `Keep run ${run + 1}`}
  </button>
)

const SidePicker: React.FC<{ name: string; value: Side; losses: number[]; hasOriginal: boolean; onChange: (side: Side) => void }> = ({
  name,
  value,
  losses,
  hasOriginal,
  onChange,
}) => (
  <label className="flex items-center gap-1.5 text-xs text-gray-400">
    <span className="font-semibold text-gray-200">{name}</span>
    <select
      value={String(value)}
      onChange={(e) => onChange(e.target.value === 'original' ? 'original' : Number(e.target.value))}
      className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-gray-100"
      data-testid={`best-of-side-${name.toLowerCase()}`}
    >
      {losses.map((loss, run) => (
        <option key={run} value={run}>
          Run {run + 1} · {loss.toFixed(4)}
        </option>
      ))}
      {hasOriginal && <option value="original">Original picture</option>}
    </select>
  </label>
)

/** A run's image, letterboxed; says so when it couldn't be loaded. */
const RunImage: React.FC<{ src: string | null; alt: string; testId?: string }> = ({ src, alt, testId }) => {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [src])
  if (!src || failed) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-xs text-gray-500">
        <ImageOff className="w-5 h-5" />
        No image
      </div>
    )
  }
  return (
    <img
      src={src}
      alt={alt}
      draggable={false}
      onError={() => setFailed(true)}
      className="absolute inset-0 w-full h-full object-contain select-none [image-rendering:pixelated]"
      data-testid={testId}
    />
  )
}

/** A over B with a draggable divider; both move together when zoomed. */
const SliderCompare: React.FC<{
  zoom: ZoomTransform
  setZoom: React.Dispatch<React.SetStateAction<ZoomTransform>>
  a: { src: string | null; name: string }
  b: { src: string | null; name: string }
}> = ({ zoom, setZoom, a, b }) => {
  const [divider, setDivider] = useState(50)
  const dragCleanup = useRef<(() => void) | null>(null)
  useEffect(() => () => dragCleanup.current?.(), [])

  const startDrag = (e: React.PointerEvent) => {
    e.stopPropagation()
    const container = (e.currentTarget as HTMLElement).parentElement
    if (!container) return
    const move = (ev: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      setDivider(Math.min(100, Math.max(0, ((ev.clientX - rect.left) / rect.width) * 100)))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
      dragCleanup.current = null
    }
    dragCleanup.current?.()
    dragCleanup.current = up
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
  }

  return (
    <ZoomViewport zoom={zoom} setZoom={setZoom} testId="best-of-slider">
      <div className="absolute inset-0" style={transformStyle(zoom)}>
        <RunImage src={a.src} alt={a.name} />
      </div>
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${divider}%)` }}>
        <div className="absolute inset-0" style={transformStyle(zoom)}>
          <RunImage src={b.src} alt={b.name} />
        </div>
      </div>
      <span className={`${label} top-1 left-2`}>{a.name}</span>
      <span className={`${label} top-1 right-2`}>{b.name}</span>
      <div className="absolute top-0 bottom-0 w-0.5 bg-white/80 pointer-events-none" style={{ left: `${divider}%` }} />
      <div
        role="slider"
        tabIndex={0}
        aria-label="Comparison divider"
        aria-valuenow={Math.round(divider)}
        aria-valuemin={0}
        aria-valuemax={100}
        onPointerDown={startDrag}
        onDoubleClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault()
            e.stopPropagation()
            setDivider((d) => Math.min(100, Math.max(0, d + (e.key === 'ArrowLeft' ? -5 : 5))))
          }
        }}
        className="absolute top-0 bottom-0 w-6 -ml-3 cursor-ew-resize flex items-center justify-center focus:outline-none group"
        style={{ left: `${divider}%` }}
        data-testid="best-of-divider"
      >
        <span className="w-4 h-8 rounded bg-white/90 shadow group-focus-visible:ring-2 group-focus-visible:ring-blue-500" />
      </div>
    </ZoomViewport>
  )
}

const Hint: React.FC<{ text: string; zoom: ZoomTransform }> = ({ text, zoom }) => (
  <div className="flex items-center gap-3 px-4 py-1.5 border-t border-gray-700 text-[11px] text-gray-500">
    <span>{text}</span>
    {zoom.scale > 1 && <span className="ml-auto tabular-nums text-gray-400" data-testid="best-of-zoom">{zoom.scale.toFixed(1)}×</span>}
  </div>
)
