import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Brush,
  ChevronDown,
  ChevronUp,
  CircleDashed,
  Eraser,
  Eye,
  EyeOff,
  Hand,
  Expand,
  FlipHorizontal2,
  Lasso,
  Loader2,
  Plus,
  Redo2,
  Shrink,
  SlidersHorizontal,
  SquareDashed,
  Trash2,
  Undo2,
  WandSparkles,
  X,
} from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { useAppStore } from '../store/appStore'
import { IDENTITY_ZOOM, type ZoomTransform } from '../lib/layout'
import { ZoomViewport, transformStyle } from './ZoomViewport'
import { AdjustPanel } from './imageEditor/AdjustPanel'
import { Histogram } from './imageEditor/Histogram'
import { histogram as computeHistogram, isIdentity } from '../lib/imageAdjust'
import { rgbaToLab } from '../lib/imageMath'
import type { Palette } from '../lib/colorReduce'
import { editChangesNothing, newLayer, nextLayerName, parseEdit, savedEdit, type EditLayer } from '../lib/imageEdit'
import {
  brushStroke,
  combine,
  ellipsePoints,
  growMask,
  invertMask,
  magicWand,
  maskOutline,
  maskShare,
  polygonMask,
  rectPoints,
  type Point,
  type SelectMode,
} from '../lib/selection'
import { ImageEditEngine, loadMask, loadPicture, maskToPng, pictureToPng, type Picture } from '../lib/imageEditClient'

type Tool = 'pan' | 'brush' | 'eraser' | 'lasso' | 'rect' | 'ellipse' | 'wand'

interface Doc {
  layers: EditLayer[]
  /** Mask layers' selections at the working size. Never changed in place
   * once committed (undo keeps the old arrays). */
  masks: Map<string, Uint8Array>
}

const TOOLS: { id: Tool; label: string; key: string; icon: React.FC<{ className?: string }>; hint: string }[] = [
  { id: 'pan', label: 'Hand', key: 'H', icon: Hand, hint: 'Drag to move around the picture' },
  { id: 'brush', label: 'Brush', key: 'B', icon: Brush, hint: 'Paint the part this layer should change' },
  { id: 'eraser', label: 'Eraser', key: 'E', icon: Eraser, hint: 'Paint away from the selection' },
  { id: 'lasso', label: 'Lasso', key: 'L', icon: Lasso, hint: 'Draw around an area freehand' },
  { id: 'rect', label: 'Rectangle', key: 'M', icon: SquareDashed, hint: 'Drag a rectangle' },
  { id: 'ellipse', label: 'Ellipse', key: 'O', icon: CircleDashed, hint: 'Drag an ellipse' },
  { id: 'wand', label: 'Magic wand', key: 'W', icon: WandSparkles, hint: 'Click a colour, then move Tolerance to grow or shrink that selection' },
]
const SHAPE_TOOLS: Tool[] = ['lasso', 'rect', 'ellipse', 'wand']
const HISTORY_LIMIT = 40

const segButton = (on: boolean) =>
  `flex items-center gap-1 px-2 py-1 text-xs whitespace-nowrap ${on ? 'bg-gray-600 text-gray-100' : 'text-gray-300 hover:bg-gray-700/70'}`
const outlineButton =
  'flex items-center gap-1 px-2.5 py-1 rounded text-xs text-gray-200 border border-gray-600 hover:bg-gray-800 disabled:opacity-40 disabled:hover:bg-transparent whitespace-nowrap'
const primaryButton =
  'flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white disabled:bg-gray-700 disabled:text-gray-400 whitespace-nowrap'

/** Opens from the image panel's Edit button; see ImageEditorDialog. */
export const ImageEditor: React.FC = () => {
  const open = useAppStore((s) => s.imageEditorOpen)
  if (!open) return null
  return <ImageEditorDialog />
}

const ImageEditorDialog: React.FC = () => {
  const setOpen = useAppStore((s) => s.setImageEditorOpen)
  const applyImageEdit = useAppStore((s) => s.applyImageEdit)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const pushToast = useAppStore((s) => s.pushToast)

  // What was edited before (settings.image_edit), read once on opening.
  const initial = useMemo(() => {
    const settings = useAppStore.getState().settings
    const edit = parseEdit(settings.image_edit)
    return { edit, source: edit?.source ?? settings.input_image }
  }, [])

  const engineRef = useRef<ImageEditEngine | null>(null)
  const [work, setWork] = useState<Picture | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [doc, setDocState] = useState<Doc>({ layers: [], masks: new Map() })
  const docRef = useRef(doc)
  docRef.current = doc
  const history = useRef<{ past: Doc[]; future: Doc[] }>({ past: [], future: [] })
  const [historyVersion, setHistoryVersion] = useState(0)
  // Masks as last saved (uploads/ file names), to upload only changed ones.
  const savedMasks = useRef(new Map<string, { mask: Uint8Array; file: string }>())
  const [selectedId, setSelectedId] = useState<string>('')
  const [dirty, setDirty] = useState(false)
  const [tool, setTool] = useState<Tool>('pan')
  const [mode, setMode] = useState<SelectMode>('replace')
  const [brushSize, setBrushSize] = useState(40)
  const [brushHardness, setBrushHardness] = useState(0.5)
  const [tolerance, setTolerance] = useState(24)
  const [contiguous, setContiguous] = useState(true)
  const [showSelection, setShowSelection] = useState(true)
  const [before, setBefore] = useState(false)
  const [zoom, setZoom] = useState<ZoomTransform>(IDENTITY_ZOOM)
  const [rendering, setRendering] = useState(false)
  const [applying, setApplying] = useState(false)
  const [palettes, setPalettes] = useState<Map<string, Palette>>(new Map())
  const [bins, setBins] = useState<ReturnType<typeof computeHistogram> | null>(null)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  // Bumped on every selection change: a brush stroke changes its mask in place.
  const [maskVersion, setMaskVersion] = useState(0)

  // ---- loading ----
  useEffect(() => {
    let cancelled = false
    const engine = new ImageEditEngine()
    engineRef.current = engine
    ;(async () => {
      try {
        if (!initial.source) throw new Error('There is no picture to edit yet.')
        const { work, full } = await loadPicture(`/uploads/${encodeURIComponent(initial.source)}`)
        await engine.init(work, full)
        const layers = initial.edit?.layers.map((l) => ({ ...l, adjust: JSON.parse(JSON.stringify(l.adjust)) })) ?? [newLayer('Layer 1')]
        const masks = new Map<string, Uint8Array>()
        for (const layer of layers) {
          if (layer.selection !== 'mask') continue
          const mask = layer.mask ? await loadMask(`/uploads/${encodeURIComponent(layer.mask)}`, work.w, work.h) : new Uint8Array(work.w * work.h)
          masks.set(layer.id, mask)
          if (layer.mask) savedMasks.current.set(layer.id, { mask, file: layer.mask })
        }
        if (cancelled) return
        setDocState({ layers, masks })
        setSelectedId(layers[layers.length - 1].id)
        setWork(work)
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      cancelled = true
      engine.dispose()
    }
  }, [initial])

  // ---- history ----
  /** Before a change: the current state becomes an undo step. */
  const commit = useCallback(() => {
    const h = history.current
    h.past.push(docRef.current)
    if (h.past.length > HISTORY_LIMIT) h.past.shift()
    h.future = []
    setHistoryVersion((v) => v + 1)
    setDirty(true)
  }, [])
  const setDoc = useCallback((next: Doc) => {
    docRef.current = next
    setDocState(next)
  }, [])
  const undo = useCallback(() => {
    const h = history.current
    const prev = h.past.pop()
    if (!prev) return
    h.future.push(docRef.current)
    setDoc(prev)
    if (!prev.layers.some((l) => l.id === selectedId)) setSelectedId(prev.layers[prev.layers.length - 1]?.id ?? '')
    setHistoryVersion((v) => v + 1)
  }, [selectedId, setDoc])
  const redo = useCallback(() => {
    const h = history.current
    const next = h.future.pop()
    if (!next) return
    h.past.push(docRef.current)
    setDoc(next)
    if (!next.layers.some((l) => l.id === selectedId)) setSelectedId(next.layers[next.layers.length - 1]?.id ?? '')
    setHistoryVersion((v) => v + 1)
  }, [selectedId, setDoc])

  const layer = doc.layers.find((l) => l.id === selectedId) ?? doc.layers[doc.layers.length - 1]
  const updateLayer = (id: string, patch: Partial<EditLayer>) => {
    const d = docRef.current
    setDoc({ ...d, layers: d.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)) })
  }

  // ---- rendering ----
  const previewRef = useRef<HTMLCanvasElement>(null)
  const lastRendered = useRef<Uint8ClampedArray | null>(null)
  const renderQueued = useRef(false)

  const draw = useCallback((rgba: Uint8ClampedArray | null) => {
    const canvas = previewRef.current
    if (!canvas || !work || !rgba) return
    canvas.width = work.w
    canvas.height = work.h
    canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(rgba), work.w, work.h), 0, 0)
  }, [work])

  useEffect(() => {
    const engine = engineRef.current
    if (!work || !engine || renderQueued.current) return
    renderQueued.current = true
    // One render per frame at most; the newest state wins.
    requestAnimationFrame(async () => {
      renderQueued.current = false
      const d = docRef.current
      const masks = new Map([...d.masks].filter(([id]) => d.layers.some((l) => l.id === id && l.selection === 'mask')))
      setRendering(true)
      try {
        const result = await engine.render(d.layers, masks)
        if (!result) return
        lastRendered.current = result.rgba
        setPalettes(result.palettes)
        setBins(computeHistogram(result.rgba))
        if (!beforeRef.current) draw(result.rgba)
        setRendering(false)
      } catch (e) {
        setRendering(false)
        if (e instanceof Error && e.message !== 'closed') pushToast(`The preview failed: ${e.message}`)
      }
    })
  }, [doc, work, draw, pushToast])

  const beforeRef = useRef(before)
  beforeRef.current = before
  useEffect(() => {
    if (!work) return
    draw(before ? work.rgba : lastRendered.current ?? work.rgba)
  }, [before, work, draw])

  // The selection as an outline ("marching ants"), so the picture under it
  // keeps its true colours while it's being adjusted.
  const currentMask = layer && layer.selection === 'mask' ? doc.masks.get(layer.id) ?? null : null
  const outline = useMemo(
    () => (currentMask && work ? maskOutline(currentMask, work.w, work.h) : null),
    // maskVersion: a brush stroke changes its mask in place.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentMask, work, maskVersion],
  )

  // ---- fitting the picture into the viewport ----
  const stageRef = useRef<HTMLDivElement>(null)
  const [stage, setStage] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setStage({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [work])
  const fit = useMemo(() => {
    if (!work || !stage.w || !stage.h) return null
    const s = Math.min(stage.w / work.w, stage.h / work.h)
    const w = work.w * s
    const h = work.h * s
    return { w, h, left: (stage.w - w) / 2, top: (stage.h - h) / 2, ratio: work.w / w }
  }, [work, stage])

  // ---- selection tools ----
  const labRef = useRef<Float32Array | null>(null)
  const sourceLab = () => {
    if (!labRef.current && work) labRef.current = rgbaToLab(work.rgba, work.w * work.h)
    return labRef.current!
  }
  const toImage = (e: { clientX: number; clientY: number }): Point => {
    const rect = previewRef.current!.getBoundingClientRect()
    return [((e.clientX - rect.left) / rect.width) * work!.w, ((e.clientY - rect.top) / rect.height) * work!.h]
  }
  const effectiveMode = (e: { shiftKey: boolean; altKey: boolean }): SelectMode => (e.shiftKey ? 'add' : e.altKey ? 'subtract' : mode)

  /** The current layer's selection to change, as a fresh copy: an empty
   * one for a layer that changed the whole picture (full when taking away). */
  const editableMask = (m: SelectMode | 'erase'): Uint8Array => {
    const n = work!.w * work!.h
    const existing = layer.selection === 'mask' ? docRef.current.masks.get(layer.id) : undefined
    if (existing) return existing.slice()
    return m === 'subtract' || m === 'erase' ? new Uint8Array(n).fill(255) : new Uint8Array(n)
  }
  const setMask = (mask: Uint8Array) => {
    const d = docRef.current
    const masks = new Map(d.masks)
    masks.set(layer.id, mask)
    setDoc({ layers: d.layers.map((l) => (l.id === layer.id ? { ...l, selection: 'mask' } : l)), masks })
    setMaskVersion((v) => v + 1)
  }

  const gesture = useRef<
    | { kind: 'brush'; mask: Uint8Array; last: Point; erase: boolean }
    | { kind: 'lasso'; points: Point[]; mode: SelectMode }
    | { kind: 'box'; start: Point; end: Point; mode: SelectMode }
    | null
  >(null)
  const [shape, setShape] = useState<{ kind: 'lasso'; points: Point[] } | { kind: 'rect' | 'ellipse'; a: Point; b: Point } | null>(null)
  const [cursor, setCursor] = useState<Point | null>(null)

  const onPointerDown = (e: React.PointerEvent) => {
    if (!work || !layer || e.button !== 0 || tool === 'pan') return
    const p = toImage(e)
    ;(e.currentTarget as Element).setPointerCapture(e.pointerId)
    if (tool === 'brush' || tool === 'eraser') {
      const erase = tool === 'eraser' || e.altKey
      commit()
      const mask = editableMask(erase ? 'erase' : 'add')
      brushStroke(mask, work.w, work.h, p[0], p[1], p[0], p[1], brushSize / 2, brushHardness, erase)
      gesture.current = { kind: 'brush', mask, last: p, erase }
      setMask(mask)
    } else if (tool === 'lasso') {
      gesture.current = { kind: 'lasso', points: [p], mode: effectiveMode(e) }
      setShape({ kind: 'lasso', points: [p] })
    } else if (tool === 'rect' || tool === 'ellipse') {
      gesture.current = { kind: 'box', start: p, end: p, mode: effectiveMode(e) }
      setShape({ kind: tool, a: p, b: p })
    } else if (tool === 'wand') {
      if (p[0] < 0 || p[1] < 0 || p[0] >= work.w || p[1] >= work.h) return
      const m = effectiveMode(e)
      commit()
      const base = editableMask(m)
      const mask = base.slice()
      combine(mask, magicWand(sourceLab(), work.w, work.h, p[0], p[1], tolerance, contiguous), m)
      setMask(mask)
      lastWand.current = { layerId: layer.id, point: p, mode: m, base, result: mask }
    }
  }

  // The last magic-wand click, while it's still the newest change to its
  // layer's selection: changing the tolerance (or "connected only") redoes
  // that click, so the selection can be grown or shrunk until it fits.
  const lastWand = useRef<{ layerId: string; point: Point; mode: SelectMode; base: Uint8Array; result: Uint8Array } | null>(null)
  useEffect(() => {
    const lw = lastWand.current
    if (!lw || !work || docRef.current.masks.get(lw.layerId) !== lw.result) return
    const mask = lw.base.slice()
    combine(mask, magicWand(sourceLab(), work.w, work.h, lw.point[0], lw.point[1], tolerance, contiguous), lw.mode)
    lw.result = mask
    const d = docRef.current
    setDoc({ ...d, masks: new Map(d.masks).set(lw.layerId, mask) })
    setMaskVersion((v) => v + 1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tolerance, contiguous])

  const onPointerMove = (e: React.PointerEvent) => {
    if (!work) return
    const p = toImage(e)
    if (tool === 'brush' || tool === 'eraser') setCursor(p)
    const g = gesture.current
    if (!g) return
    if (g.kind === 'brush') {
      brushStroke(g.mask, work.w, work.h, g.last[0], g.last[1], p[0], p[1], brushSize / 2, brushHardness, g.erase)
      g.last = p
      // Same array, new map: the preview and overlay pick it up.
      setMask(g.mask)
    } else if (g.kind === 'lasso') {
      const last = g.points[g.points.length - 1]
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) * (fit ? 1 / fit.ratio : 1) * zoom.scale < 2) return
      g.points.push(p)
      setShape({ kind: 'lasso', points: [...g.points] })
    } else {
      g.end = p
      setShape({ kind: tool as 'rect' | 'ellipse', a: g.start, b: p })
    }
  }

  const onPointerUp = () => {
    const g = gesture.current
    gesture.current = null
    if (!g || !work) return
    // A stroke painted into its own copy (see editableMask): undo still
    // has the mask from before it.
    if (g.kind === 'brush') return
    setShape(null)
    let points: Point[]
    if (g.kind === 'lasso') points = g.points
    else {
      const [x0, y0] = g.start
      const [x1, y1] = g.end
      if (Math.abs(x1 - x0) < 2 || Math.abs(y1 - y0) < 2) return
      points = tool === 'ellipse' ? ellipsePoints(x0, y0, x1, y1) : rectPoints(x0, y0, x1, y1)
    }
    if (points.length < 3) return
    commit()
    const mask = editableMask(g.mode)
    combine(mask, polygonMask(work.w, work.h, points), g.mode)
    setMask(mask)
  }

  const selectionAction = (fn: (mask: Uint8Array) => Uint8Array) => {
    if (!work || !layer) return
    commit()
    setMask(fn(editableMask('add')))
  }

  // ---- layers ----
  const addLayer = () => {
    commit()
    const d = docRef.current
    const created = newLayer(nextLayerName(d.layers, 'Layer'))
    setDoc({ ...d, layers: [...d.layers, created] })
    setSelectedId(created.id)
  }
  const removeLayer = (id: string) => {
    const d = docRef.current
    commit()
    if (d.layers.length === 1) {
      // Always one layer: the last one is reset instead.
      const fresh = newLayer('Layer 1')
      setDoc({ layers: [fresh], masks: new Map() })
      setSelectedId(fresh.id)
      return
    }
    const index = d.layers.findIndex((l) => l.id === id)
    const layers = d.layers.filter((l) => l.id !== id)
    const masks = new Map(d.masks)
    masks.delete(id)
    setDoc({ layers, masks })
    if (selectedId === id) setSelectedId(layers[Math.max(0, index - 1)].id)
  }
  const moveLayer = (id: string, delta: number) => {
    const d = docRef.current
    const index = d.layers.findIndex((l) => l.id === id)
    const to = index + delta
    if (to < 0 || to >= d.layers.length) return
    commit()
    const layers = [...d.layers]
    const [moved] = layers.splice(index, 1)
    layers.splice(to, 0, moved)
    setDoc({ ...d, layers })
  }

  // ---- closing ----
  const close = useCallback(async () => {
    if (applying) return
    if (dirty) {
      const ok = await requestConfirm({
        title: 'Discard your changes?',
        message: 'The picture stays as it was before you opened the editor.',
        confirmLabel: 'Discard',
      })
      if (!ok) return
    }
    setOpen(false)
  }, [applying, dirty, requestConfirm, setOpen])

  const apply = async () => {
    const engine = engineRef.current
    if (!engine || !work) return
    const d = docRef.current
    setApplying(true)
    try {
      if (editChangesNothing(d.layers)) {
        // Nothing left to change: back to the picture as uploaded.
        if (initial.edit) await applyImageEdit(null, null)
        setOpen(false)
        return
      }
      // Changed selections go up as files; unchanged ones keep theirs.
      const layers: EditLayer[] = []
      for (const l of d.layers) {
        if (l.selection !== 'mask') {
          layers.push({ ...l, mask: '' })
          continue
        }
        const mask = d.masks.get(l.id)!
        const saved = savedMasks.current.get(l.id)
        let file = saved && saved.mask === mask ? saved.file : ''
        if (!file) {
          const form = new FormData()
          form.append('file', await maskToPng(mask, work.w, work.h), 'selection.png')
          const response = await fetch('/api/images/upload', { method: 'POST', body: form })
          if (!response.ok) throw new Error(`Couldn't save a selection (HTTP ${response.status})`)
          file = (await response.json()).filename
          savedMasks.current.set(l.id, { mask, file })
        }
        layers.push({ ...l, mask: file })
      }
      const masks = new Map([...d.masks].filter(([id]) => d.layers.some((l) => l.id === id && l.selection === 'mask')))
      const picture = await engine.apply(d.layers, masks)
      const blob = await pictureToPng(picture)
      await applyImageEdit(savedEdit(initial.source, layers), blob)
      setOpen(false)
    } catch (e) {
      pushToast(`Couldn't apply the edit: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setApplying(false)
    }
  }

  // ---- keys ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' && (target as HTMLInputElement).type !== 'range' && (target as HTMLInputElement).type !== 'checkbox' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) return
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        e.stopPropagation()
        if (e.shiftKey) redo()
        else undo()
        return
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        e.stopPropagation()
        redo()
        return
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const t = TOOLS.find((x) => x.key.toLowerCase() === e.key.toLowerCase())
      if (t) {
        setTool(t.id)
        e.preventDefault()
      } else if (e.key === '[') setBrushSize((s) => Math.max(2, s - 4))
      else if (e.key === ']') setBrushSize((s) => Math.min(400, s + 4))
      else if (e.key === '\\') setBefore(true)
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === '\\') setBefore(false)
    }
    window.addEventListener('keydown', onKey, true)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKey, true)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [undo, redo])

  void historyVersion
  const canUndo = history.current.past.length > 0
  const canRedo = history.current.future.length > 0
  const share = currentMask ? maskShare(currentMask) : 1
  const toolInfo = TOOLS.find((t) => t.id === tool)!
  const viewportPaint = tool !== 'pan'

  return (
    <Dialog open onOpenChange={(o) => !o && close()}>
      <DialogContent
        className="max-w-[98vw] w-[98vw] h-[95vh] flex flex-col p-0 gap-0 bg-gray-900 border-gray-700"
        onEscapeKeyDown={(e) => {
          // Esc first drops a shape being drawn.
          if (gesture.current || shape) {
            e.preventDefault()
            gesture.current = null
            setShape(null)
            return
          }
          e.preventDefault()
          close()
        }}
        onInteractOutside={(e) => e.preventDefault()}
        data-testid="image-editor"
        data-ready={!!work}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-700">
          <SlidersHorizontal className="w-4 h-4 text-cyan-400 flex-shrink-0" />
          <div className="min-w-0">
            <DialogTitle className="text-sm font-semibold text-gray-100 leading-none tracking-normal">Edit image</DialogTitle>
            <DialogDescription className="mt-1 text-xs text-gray-400 truncate">
              Adjust the whole picture, or select a part and adjust only that. The optimizer then works from the edited picture; the original stays untouched.
            </DialogDescription>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button onClick={undo} disabled={!canUndo} className={outlineButton} title="Undo (Ctrl+Z)" aria-label="Undo" data-testid="editor-undo">
              <Undo2 className="w-3.5 h-3.5" />
            </button>
            <button onClick={redo} disabled={!canRedo} className={outlineButton} title="Redo (Ctrl+Shift+Z)" aria-label="Redo" data-testid="editor-redo">
              <Redo2 className="w-3.5 h-3.5" />
            </button>
            <button
              onPointerDown={() => setBefore(true)}
              onPointerUp={() => setBefore(false)}
              onPointerLeave={() => setBefore(false)}
              className={`${outlineButton} ${before ? 'bg-gray-700' : ''}`}
              title="Hold to see the picture before your edits (or hold \)"
              data-testid="editor-before"
            >
              <FlipHorizontal2 className="w-3.5 h-3.5" />
              Hold for before
            </button>
            <button onClick={close} disabled={applying} className={outlineButton} data-testid="editor-cancel">
              Cancel
            </button>
            <button onClick={apply} disabled={!work || applying} className={primaryButton} data-testid="editor-apply">
              {applying ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              {applying ? 'Applying…' : 'Apply'}
            </button>
          </div>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Tools */}
          <div className="flex flex-col gap-1 p-1.5 border-r border-gray-700" role="toolbar" aria-label="Tools" aria-orientation="vertical">
            {TOOLS.map((t) => {
              const Icon = t.icon
              return (
                <button
                  key={t.id}
                  onClick={() => setTool(t.id)}
                  aria-pressed={tool === t.id}
                  className={`p-2 rounded ${tool === t.id ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                  title={`${t.label} (${t.key}) — ${t.hint}`}
                  aria-label={t.label}
                  data-testid={`editor-tool-${t.id}`}
                >
                  <Icon className="w-4 h-4" />
                </button>
              )
            })}
          </div>

          {/* Canvas */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 border-b border-gray-700 text-xs text-gray-300 min-h-[38px]" data-testid="editor-tool-options">
              <span className="font-medium text-gray-100">{toolInfo.label}</span>
              <span className="text-gray-500 hidden xl:inline">{toolInfo.hint}</span>
              {SHAPE_TOOLS.includes(tool) && (
                <div className="flex rounded border border-gray-600 overflow-hidden" role="group" aria-label="Selection mode">
                  {(['replace', 'add', 'subtract'] as SelectMode[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => setMode(m)}
                      aria-pressed={mode === m}
                      className={segButton(mode === m)}
                      title={m === 'replace' ? 'Start a new selection' : m === 'add' ? 'Add to the selection (or hold Shift)' : 'Take away from the selection (or hold Alt)'}
                      data-testid={`editor-mode-${m}`}
                    >
                      {m === 'replace' ? 'New' : m === 'add' ? 'Add' : 'Subtract'}
                    </button>
                  ))}
                </div>
              )}
              {(tool === 'brush' || tool === 'eraser') && (
                <>
                  <label className="flex items-center gap-1.5" title="Brush size ([ and ])">
                    Size
                    <input type="range" min={2} max={400} value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} className="w-24 accent-cyan-500" data-testid="editor-brush-size" />
                  </label>
                  <label className="flex items-center gap-1.5" title="How hard the brush edge is">
                    Hardness
                    <input type="range" min={0} max={100} value={Math.round(brushHardness * 100)} onChange={(e) => setBrushHardness(Number(e.target.value) / 100)} className="w-20 accent-cyan-500" />
                  </label>
                </>
              )}
              {tool === 'wand' && (
                <>
                  <label className="flex items-center gap-1.5" title="How different a colour may be and still count as similar. Changing it updates your last click.">
                    Tolerance
                    <input type="range" min={1} max={60} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} className="w-24 accent-cyan-500" data-testid="editor-wand-tolerance" />
                    <span className="w-5 tabular-nums text-gray-400">{tolerance}</span>
                  </label>
                  <label className="flex items-center gap-1.5" title="Only colours joined to the one you click; off selects that colour everywhere">
                    <input type="checkbox" checked={contiguous} onChange={(e) => setContiguous(e.target.checked)} className="accent-cyan-500" data-testid="editor-wand-contiguous" />
                    Connected only
                  </label>
                </>
              )}
              <div className="ml-auto flex items-center gap-1.5">
                <button onClick={() => selectionAction((m) => (invertMask(m), m))} disabled={!work} className={outlineButton} title="Swap selected and not selected" data-testid="editor-invert">
                  Invert
                </button>
                <button onClick={() => selectionAction((m) => growMask(m, work!.w, work!.h, Math.max(2, work!.w / 200)))} disabled={!currentMask} className={outlineButton} title="Make the selection a little bigger" data-testid="editor-grow">
                  <Expand className="w-3 h-3" />
                </button>
                <button onClick={() => selectionAction((m) => growMask(m, work!.w, work!.h, -Math.max(2, work!.w / 200)))} disabled={!currentMask} className={outlineButton} title="Make the selection a little smaller" data-testid="editor-shrink">
                  <Shrink className="w-3 h-3" />
                </button>
                <button
                  onClick={() => setShowSelection((v) => !v)}
                  disabled={!currentMask}
                  className={outlineButton}
                  title={showSelection ? 'Hide the selection outline' : 'Show the selection outline'}
                  data-testid="editor-show-selection"
                >
                  {showSelection ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                </button>
              </div>
            </div>

            <div ref={stageRef} className="relative flex-1 min-h-0 bg-gray-950 overflow-hidden">
              {loadError ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-sm text-red-400" role="alert" data-testid="editor-error">
                  {loadError}
                  <button onClick={() => setOpen(false)} className={outlineButton}>
                    <X className="w-3 h-3" /> Close
                  </button>
                </div>
              ) : !work || !fit ? (
                <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-gray-400">
                  <Loader2 className="w-4 h-4 animate-spin" /> Loading the picture…
                </div>
              ) : (
                <ZoomViewport zoom={zoom} setZoom={setZoom} paintMode={viewportPaint} testId="editor-viewport">
                  {/* The tools listen on the whole stage: a shape may start
                      outside the picture's edge (it's clipped to it). */}
                  <div
                    className="absolute inset-0"
                    style={{ ...transformStyle(zoom), cursor: tool === 'pan' ? undefined : tool === 'brush' || tool === 'eraser' ? 'none' : 'crosshair' }}
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerCancel={onPointerUp}
                    onPointerLeave={() => setCursor(null)}
                  >
                    <div className="absolute" style={{ left: fit.left, top: fit.top, width: fit.w, height: fit.h }} data-testid="editor-canvas">
                      <canvas ref={previewRef} className="absolute inset-0 w-full h-full [image-rendering:auto]" data-testid="editor-preview" />
                      <svg viewBox={`0 0 ${work.w} ${work.h}`} className="absolute inset-0 w-full h-full pointer-events-none" preserveAspectRatio="none">
                        {outline && outline.path && showSelection && !before && (
                          <g data-testid="editor-selection-outline" data-segments={outline.segments}>
                            {/* White under black dashes: visible on any colour. Stays
                                1 px at every zoom; moves unless it's huge. */}
                            <path d={outline.path} fill="none" stroke="#fff" strokeWidth={1} vectorEffect="non-scaling-stroke" shapeRendering="crispEdges" />
                            <path
                              d={outline.path}
                              fill="none"
                              stroke="#000"
                              strokeWidth={1}
                              strokeDasharray="4 4"
                              vectorEffect="non-scaling-stroke"
                              shapeRendering="crispEdges"
                              className={outline.segments < 40000 ? 'marching-ants' : undefined}
                            />
                          </g>
                        )}
                        {shape?.kind === 'lasso' && (
                          <polyline points={shape.points.map((p) => p.join(',')).join(' ')} fill="rgba(34,211,238,0.15)" stroke="#22d3ee" strokeWidth={1.5 * fit.ratio / zoom.scale} strokeDasharray={`${4 * fit.ratio / zoom.scale}`} />
                        )}
                        {shape && shape.kind !== 'lasso' && (
                          shape.kind === 'rect' ? (
                            <rect x={Math.min(shape.a[0], shape.b[0])} y={Math.min(shape.a[1], shape.b[1])} width={Math.abs(shape.b[0] - shape.a[0])} height={Math.abs(shape.b[1] - shape.a[1])} fill="rgba(34,211,238,0.15)" stroke="#22d3ee" strokeWidth={1.5 * fit.ratio / zoom.scale} strokeDasharray={`${4 * fit.ratio / zoom.scale}`} />
                          ) : (
                            <ellipse cx={(shape.a[0] + shape.b[0]) / 2} cy={(shape.a[1] + shape.b[1]) / 2} rx={Math.abs(shape.b[0] - shape.a[0]) / 2} ry={Math.abs(shape.b[1] - shape.a[1]) / 2} fill="rgba(34,211,238,0.15)" stroke="#22d3ee" strokeWidth={1.5 * fit.ratio / zoom.scale} strokeDasharray={`${4 * fit.ratio / zoom.scale}`} />
                          )
                        )}
                        {cursor && (tool === 'brush' || tool === 'eraser') && (
                          <circle cx={cursor[0]} cy={cursor[1]} r={brushSize / 2} fill="none" stroke={tool === 'eraser' ? '#f87171' : '#22d3ee'} strokeWidth={1.5 * fit.ratio / zoom.scale} />
                        )}
                      </svg>
                    </div>
                  </div>
                </ZoomViewport>
              )}
              {before && <span className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-xs text-white pointer-events-none" data-testid="editor-before-badge">Before</span>}
              {rendering && work && (
                <span className="absolute top-2 right-2 flex items-center gap-1 px-2 py-0.5 rounded bg-black/60 text-[11px] text-gray-200 pointer-events-none">
                  <Loader2 className="w-3 h-3 animate-spin" /> Updating
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 px-3 py-1 border-t border-gray-700 text-[11px] text-gray-500">
              <span>Scroll to zoom · {viewportPaint ? 'right-drag' : 'drag'} to pan · double-click to reset · hold \ for before</span>
              {zoom.scale > 1 && <span className="ml-auto tabular-nums text-gray-400">{zoom.scale.toFixed(1)}×</span>}
            </div>
          </div>

          {/* Sidebar */}
          <aside className="w-80 flex-shrink-0 border-l border-gray-700 flex flex-col min-h-0" data-testid="editor-sidebar">
            <div className="px-3 pt-2.5 pb-2 border-b border-gray-800">
              <Histogram bins={bins} className="rounded bg-gray-950" />
            </div>
            <div className="px-3 py-2 border-b border-gray-800">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-gray-200">Layers</span>
                <button onClick={addLayer} disabled={!work} className="flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300 disabled:opacity-40" title="Another set of adjustments, for another part of the picture" data-testid="editor-add-layer">
                  <Plus className="w-3 h-3" /> Add layer
                </button>
              </div>
              <ul className="space-y-1 max-h-44 overflow-y-auto" data-testid="editor-layers">
                {[...doc.layers].reverse().map((l) => {
                  const mask = l.selection === 'mask' ? doc.masks.get(l.id) : undefined
                  const selected = layer?.id === l.id
                  return (
                    <li
                      key={l.id}
                      onClick={() => setSelectedId(l.id)}
                      className={`group flex items-center gap-1.5 px-1.5 py-1 rounded cursor-pointer ${selected ? 'bg-cyan-600/20 ring-1 ring-cyan-600/60' : 'hover:bg-gray-800'}`}
                      data-testid={`editor-layer-${doc.layers.indexOf(l)}`}
                      data-selected={selected}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          commit()
                          updateLayer(l.id, { visible: !l.visible })
                        }}
                        className="p-0.5 text-gray-400 hover:text-gray-100"
                        aria-label={l.visible ? 'Hide layer' : 'Show layer'}
                        data-testid="editor-layer-visible"
                      >
                        {l.visible ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>
                      {renamingId === l.id ? (
                        <input
                          autoFocus
                          defaultValue={l.name}
                          onClick={(e) => e.stopPropagation()}
                          onBlur={(e) => {
                            const name = e.target.value.trim()
                            if (name && name !== l.name) {
                              commit()
                              updateLayer(l.id, { name })
                            }
                            setRenamingId(null)
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
                            if (e.key === 'Escape') {
                              e.stopPropagation()
                              setRenamingId(null)
                            }
                          }}
                          className="flex-1 min-w-0 bg-gray-800 border border-gray-600 rounded px-1 text-xs text-gray-100"
                        />
                      ) : (
                        <span
                          className={`flex-1 min-w-0 truncate text-xs ${l.visible ? 'text-gray-100' : 'text-gray-500 line-through'}`}
                          onDoubleClick={() => setRenamingId(l.id)}
                          title="Double-click to rename"
                        >
                          {l.name}
                        </span>
                      )}
                      <span className="text-[10px] text-gray-400 tabular-nums" data-testid="editor-layer-scope">
                        {mask ? `${Math.round(maskShare(mask) * 100)}%` : 'All'}
                      </span>
                      {isIdentity(l.adjust) && <span className="text-[10px] text-gray-600" title="This layer doesn't change anything yet">·</span>}
                      <div className="flex opacity-0 group-hover:opacity-100 focus-within:opacity-100">
                        <button onClick={(e) => (e.stopPropagation(), moveLayer(l.id, 1))} className="p-0.5 text-gray-400 hover:text-gray-100" aria-label="Move up" title="Move up (applies later)">
                          <ChevronUp className="w-3.5 h-3.5" />
                        </button>
                        <button onClick={(e) => (e.stopPropagation(), moveLayer(l.id, -1))} className="p-0.5 text-gray-400 hover:text-gray-100" aria-label="Move down" title="Move down (applies earlier)">
                          <ChevronDown className="w-3.5 h-3.5" />
                        </button>
                        <button onClick={(e) => (e.stopPropagation(), removeLayer(l.id))} className="p-0.5 text-gray-400 hover:text-red-400" aria-label="Delete layer" data-testid="editor-layer-delete">
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
            {layer && (
              <div className="flex-1 min-h-0 overflow-y-auto px-3 pb-4">
                <div className="py-2 space-y-2">
                  <div className="text-[11px] text-gray-400">Applies to</div>
                  <div className="flex rounded border border-gray-600 overflow-hidden" role="group" aria-label="What this layer changes">
                    <button
                      onClick={() => {
                        if (layer.selection === 'all') return
                        commit()
                        updateLayer(layer.id, { selection: 'all' })
                      }}
                      aria-pressed={layer.selection === 'all'}
                      className={`flex-1 justify-center ${segButton(layer.selection === 'all')}`}
                      data-testid="editor-scope-all"
                    >
                      Whole picture
                    </button>
                    <button
                      onClick={() => {
                        if (layer.selection === 'mask') return
                        commit()
                        const d = docRef.current
                        const masks = new Map(d.masks)
                        if (!masks.has(layer.id)) masks.set(layer.id, new Uint8Array(work!.w * work!.h))
                        setDoc({ layers: d.layers.map((l) => (l.id === layer.id ? { ...l, selection: 'mask' } : l)), masks })
                        if (tool === 'pan') setTool('lasso')
                      }}
                      aria-pressed={layer.selection === 'mask'}
                      className={`flex-1 justify-center ${segButton(layer.selection === 'mask')}`}
                      data-testid="editor-scope-selection"
                    >
                      Selection{currentMask ? ` · ${Math.round(share * 100)}%` : ''}
                    </button>
                  </div>
                  {layer.selection === 'mask' && (
                    <>
                      {share === 0 && (
                        <p className="text-[11px] text-amber-400" data-testid="editor-empty-selection">
                          Nothing selected yet: use the brush, lasso, shapes or magic wand on the left.
                        </p>
                      )}
                      <label className="block" title="Blends the layer's edges into the rest of the picture">
                        <div className="flex justify-between text-[11px] text-gray-400">
                          <span>Soft edges</span>
                          <span className="tabular-nums">{layer.feather}</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={layer.feather}
                          onPointerDown={commit}
                          onChange={(e) => updateLayer(layer.id, { feather: Number(e.target.value) })}
                          className="w-full accent-cyan-500 h-4"
                          data-testid="editor-feather"
                        />
                      </label>
                    </>
                  )}
                </div>
                <AdjustPanel
                  key={layer.id}
                  adjust={layer.adjust}
                  histogram={bins}
                  palette={palettes.get(layer.id) ?? null}
                  onBegin={commit}
                  onChange={(adjust) => updateLayer(layer.id, { adjust })}
                />
              </div>
            )}
          </aside>
        </div>
      </DialogContent>
    </Dialog>
  )
}
