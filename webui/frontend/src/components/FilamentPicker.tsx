import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Search } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'

const UNASSIGNED_COLOR = '#555555'
const POPUP_WIDTH = 288
const POPUP_MAX_HEIGHT = 320

interface FilamentPickerProps {
  value: string
  onChange: (filament: Filament) => void
  label: string
  testId: string
  swatchTestId?: string
  /** Shown when `value` names no known filament — the base row's color can
   * be a plain hex (a background chosen by hand, or auto-selection failing)
   * that belongs to no library entry, and "Unassigned" grey would then hide
   * the color the print actually uses. */
  fallbackColor?: string
}

const Swatch: React.FC<{ color: string; testId?: string }> = ({ color, testId }) => (
  <span className="w-3.5 h-3.5 rounded-full border border-gray-600 flex-shrink-0" style={{ backgroundColor: color }} data-testid={testId} />
)

/** Filament choice with color swatches and search. Active filaments come
 * first; the rest of the library is one scroll away (picking one also makes
 * it active). The native <select> showed names only, and active ones only. */
export const FilamentPicker: React.FC<FilamentPickerProps> = ({ value, onChange, label, testId, swatchTestId, fallbackColor }) => {
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const libraryFilaments = useAppStore((s) => s.filaments)
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [library, setLibrary] = useState<Filament[] | null>(null)
  const [highlight, setHighlight] = useState(0)
  const [position, setPosition] = useState<{ left: number; top: number; maxHeight: number } | null>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const known = useMemo(() => {
    const byUuid = new Map<string, Filament>()
    for (const f of [...libraryFilaments, ...(library ?? []), ...activeFilaments]) byUuid.set(f.uuid, f)
    return byUuid
  }, [libraryFilaments, library, activeFilaments])
  const selected = value ? known.get(value) : undefined

  // The whole library (every type), fetched when the picker opens.
  useEffect(() => {
    if (!open) return
    let cancelled = false
    fetch('/api/filaments')
      .then((r) => (r.ok ? r.json() : []))
      .then((all: Filament[]) => { if (!cancelled) setLibrary(all) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [open])

  const { active, others } = useMemo(() => {
    const q = query.trim().toLowerCase()
    const matches = (f: Filament) => !q || `${f.brand} ${f.name} ${f.filament_type}`.toLowerCase().includes(q)
    const activeUuids = new Set(activeFilaments.map((f) => f.uuid))
    return {
      active: activeFilaments.filter(matches),
      others: (library ?? []).filter((f) => !activeUuids.has(f.uuid) && matches(f)).sort((a, b) => a.brand.localeCompare(b.brand) || a.name.localeCompare(b.name)),
    }
  }, [activeFilaments, library, query])
  const options = useMemo(() => [...active, ...others], [active, others])

  const place = () => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    const below = window.innerHeight - rect.bottom - 8
    const above = rect.top - 8
    const openUp = below < 200 && above > below
    const maxHeight = Math.min(POPUP_MAX_HEIGHT, openUp ? above : below)
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPUP_WIDTH - 8))
    setPosition({ left, top: openUp ? rect.top - maxHeight - 4 : rect.bottom + 4, maxHeight })
  }

  useLayoutEffect(() => {
    if (open) place()
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      if (!popupRef.current?.contains(target) && !triggerRef.current?.contains(target)) setOpen(false)
    }
    const onScroll = (e: Event) => {
      if (!popupRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      document.removeEventListener('mousedown', onDown)
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [open])

  useEffect(() => setHighlight(0), [query, open])
  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${highlight}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [highlight])

  const choose = (f: Filament) => {
    setOpen(false)
    setQuery('')
    onChange(f)
    triggerRef.current?.focus()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlight((h) => Math.min(options.length - 1, h + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlight((h) => Math.max(0, h - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (options[highlight]) choose(options[highlight])
    } else if (e.key === 'Escape') {
      e.preventDefault()
      setOpen(false)
      triggerRef.current?.focus()
    }
  }

  const option = (f: Filament, index: number) => (
    <button
      key={f.uuid}
      role="option"
      aria-selected={f.uuid === value}
      data-index={index}
      onMouseEnter={() => setHighlight(index)}
      onClick={() => choose(f)}
      className={`w-full flex items-center gap-2 px-2.5 py-1.5 text-left text-xs ${index === highlight ? 'bg-gray-700' : ''} ${f.uuid === value ? 'text-cyan-500' : 'text-gray-100'}`}
      data-testid={`filament-option-${f.uuid}`}
    >
      <Swatch color={f.color} />
      <span className="flex-1 min-w-0 truncate">
        {f.name}
        <span className="text-gray-400"> · {f.brand}{f.filament_type ? ` ${f.filament_type}` : ''}</span>
      </span>
      <span className="text-[11px] text-gray-400 tabular-nums">TD {f.td}</span>
    </button>
  )

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (!open && (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault()
            setOpen(true)
          }
        }}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        className="min-w-0 flex-1 flex items-center gap-1.5 text-left text-xs border border-transparent hover:border-gray-700 rounded px-1 py-0.5 text-gray-100"
        data-testid={testId}
        data-value={value}
      >
        <Swatch color={selected?.color ?? fallbackColor ?? UNASSIGNED_COLOR} testId={swatchTestId} />
        <span className={`flex-1 min-w-0 truncate ${selected ? '' : 'text-gray-400'}`}>
          {selected ? selected.name : fallbackColor ?? 'Unassigned'}
        </span>
        <ChevronDown className="w-3 h-3 text-gray-400 flex-shrink-0" />
      </button>

      {open && position &&
        createPortal(
          <div
            ref={popupRef}
            className="fixed z-[60] flex flex-col rounded border border-gray-700 bg-gray-800 shadow-xl"
            style={{ left: position.left, top: position.top, width: POPUP_WIDTH, maxHeight: position.maxHeight }}
            onKeyDown={onKeyDown}
            data-testid="filament-picker"
          >
            <div className="relative p-1.5 border-b border-gray-700">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search filaments…"
                aria-label="Search filaments"
                className="w-full pl-7 pr-2 py-1 rounded text-xs bg-gray-900 border border-gray-600 text-gray-100 placeholder:text-gray-500 outline-none focus:border-cyan-600"
                data-testid="filament-picker-search"
              />
            </div>
            <div ref={listRef} role="listbox" aria-label={label} className="flex-1 overflow-y-auto py-1">
              <div className="px-2.5 pt-1 pb-0.5 text-[11px] uppercase tracking-wide text-gray-500">Active</div>
              {active.length ? active.map((f, i) => option(f, i)) : <p className="px-2.5 py-1 text-xs text-gray-500">{query ? 'No match' : 'No active filaments'}</p>}
              <div className="px-2.5 pt-2 pb-0.5 text-[11px] uppercase tracking-wide text-gray-500">More from the library</div>
              {library === null ? (
                <p className="px-2.5 py-1 text-xs text-gray-500">Loading…</p>
              ) : others.length ? (
                others.map((f, i) => option(f, active.length + i))
              ) : (
                <p className="px-2.5 py-1 text-xs text-gray-500">{query ? 'No match' : 'Nothing else in the library'}</p>
              )}
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}
