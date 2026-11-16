import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, ChevronUp, Hand, Layers, Printer, Search } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import {
  bedSize,
  defaultManualSwaps,
  initialChoice,
  modelsOf,
  nozzleLabel,
  popularPrinters,
  searchPrinters,
  shortPrinterName,
  variantsOf,
  vendorsOf,
  type SlicerPrinter,
} from '../lib/slicerPrinters'
import { preparePrinter } from '../lib/slicerExport'

const selectClass =
  'w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50'

interface PrinterList {
  orca_version: string
  printers: SlicerPrinter[]
}

let printerListCache: Promise<PrinterList> | null = null

export function loadPrinters(): Promise<PrinterList> {
  if (!printerListCache) {
    printerListCache = fetch('/api/outputs/3mf/printers').then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json() as Promise<PrinterList>
    })
    printerListCache.catch(() => {
      printerListCache = null
    })
  }
  return printerListCache
}

interface Props {
  open: boolean
  onClose: () => void
  /** 'export': pick the printer, then export. 'choose': only pick it (Settings). */
  mode: 'export' | 'choose'
  /** The printer remembered in the project settings, and its swap choice. */
  rememberedPrinter?: string
  rememberedManualSwaps?: boolean | null
  onConfirm: (printer: string, manualSwaps: boolean) => void
}

/** The printer picker for slicer projects (OrcaSlicer's printer presets)
 * plus how colour changes happen on it. */
export const SlicerProjectDialog: React.FC<Props> = ({ open, onClose, mode, rememberedPrinter, rememberedManualSwaps, onConfirm }) => {
  const [list, setList] = useState<PrinterList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const [query, setQuery] = useState('')
  const [highlight, setHighlight] = useState(0)
  const [name, setName] = useState('')
  const [browsing, setBrowsing] = useState(true)
  const [showBrands, setShowBrands] = useState(false)
  const [manual, setManual] = useState(true)
  const searchRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setError(null)
    loadPrinters()
      .then((data) => {
        if (cancelled) return
        setList(data)
        const start = initialChoice(data.printers, rememberedPrinter)
        setName(start?.name ?? '')
        // A known printer starts collapsed to its card; otherwise straight to search.
        setBrowsing(!start)
        setShowBrands(false)
        setQuery('')
        setHighlight(0)
      })
      .catch(() => {
        if (!cancelled) setError('Could not load the printer list.')
      })
    return () => {
      cancelled = true
    }
  }, [open, rememberedPrinter, reload])

  // Focus the search as it appears (also after "Change", whose button just
  // vanished). Done at mount: a later focus() lost the race against the
  // dialog's focus trap, which moves focus off a removed element.
  const searchMounted = (el: HTMLInputElement | null) => {
    searchRef.current = el
    if (el && document.activeElement !== el) el.focus()
  }

  const printers = list?.printers ?? []
  const selected = printers.find((p) => p.name === name)
  const popular = useMemo(() => popularPrinters(printers), [printers])
  const results = useMemo(() => (query.trim() ? searchPrinters(printers, query, 40) : popular), [printers, query, popular])
  const vendors = useMemo(() => vendorsOf(printers), [printers])
  const vendor = selected?.vendor ?? ''
  const models = useMemo(() => (vendor ? modelsOf(printers, vendor) : []), [printers, vendor])
  const variants = useMemo(() => (selected ? variantsOf(printers, selected.vendor, selected.model) : []), [printers, selected])

  // The swap mode follows the printer unless the user chose one for it.
  useEffect(() => {
    setManual(defaultManualSwaps(selected, rememberedManualSwaps, rememberedPrinter))
  }, [selected, rememberedManualSwaps, rememberedPrinter])

  // Fetch the chosen printer's profiles now, so the export doesn't wait.
  useEffect(() => {
    if (open && selected) preparePrinter(selected.name)
  }, [open, selected])

  const pick = (p: SlicerPrinter) => {
    setName(p.name)
    setBrowsing(false)
    setQuery('')
  }

  const chooseVendor = (v: string) => {
    const model = v ? modelsOf(printers, v)[0] : ''
    const first = model ? variantsOf(printers, v, model)[0] : undefined
    if (first) setName(first.name)
  }

  const chooseModel = (m: string) => {
    const first = selected && m ? variantsOf(printers, selected.vendor, m)[0] : undefined
    if (first) setName(first.name)
  }

  const onSearchKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlight((h) => Math.min(h + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlight((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter' && results[highlight]) {
      e.preventDefault()
      pick(results[highlight])
    }
  }

  const recommendedManual = selected ? !selected.auto_swaps : true
  const remembered = printers.find((p) => p.name === rememberedPrinter)
  const confirmLabel = mode === 'export' ? 'Export .3mf' : 'Use this printer'

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-lg bg-gray-900 border-gray-700 p-0 gap-0" data-testid="slicer-project-dialog">
        <div className="px-5 pt-4 pb-3 border-b border-gray-800">
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold text-gray-100">
            <Printer className="w-4 h-4 text-cyan-400" />
            {mode === 'export' ? 'Slicer project (.3mf)' : 'Your printer'}
          </DialogTitle>
          <DialogDescription className="text-xs text-gray-400 mt-1">
            {mode === 'export' && !rememberedPrinter
              ? 'Which printer is this for? The project opens in OrcaSlicer set up for it, with every colour change in place. You only choose once.'
              : 'Slicer projects are set up for this printer, with every colour change in place.'}
          </DialogDescription>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[65vh] overflow-y-auto">
          {error && (
            <div className="flex items-center justify-between rounded border border-red-800 bg-red-950 px-3 py-2 text-xs text-red-200" role="alert">
              {error}
              <button onClick={() => setReload((n) => n + 1)} className="underline hover:text-white">Try again</button>
            </div>
          )}
          {!list && !error && <p className="text-xs text-gray-400">Loading printers…</p>}

          {list && selected && !browsing && (
            <div className="flex items-center gap-3 rounded-lg border border-cyan-800/60 bg-cyan-950/30 px-3 py-2.5" data-testid="slicer-selected-printer">
              <Printer className="w-5 h-5 text-cyan-300 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="text-sm text-gray-100 truncate">{shortPrinterName(selected.name)}</div>
                <div className="text-[11px] text-gray-400">
                  {nozzleLabel(selected)} nozzle · bed {bedSize(selected)}
                  {selected.auto_swaps ? ' · changes filament itself' : ''}
                </div>
              </div>
              <button
                onClick={() => setBrowsing(true)}
                className="px-2 py-1 text-xs rounded border border-gray-600 text-gray-200 hover:bg-gray-800"
                data-testid="slicer-change-printer"
              >
                Change
              </button>
            </div>
          )}

          {list && browsing && (
            <div className="space-y-2">
              <div className="relative">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-gray-500" />
                <input
                  ref={searchMounted}
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value)
                    setHighlight(0)
                  }}
                  onKeyDown={onSearchKey}
                  placeholder="Search your printer, e.g. “a1 mini” or “neptune 4”"
                  aria-label="Search printers"
                  className="w-full bg-gray-800 border border-gray-600 rounded pl-8 pr-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
                  data-testid="slicer-printer-search"
                />
              </div>
              <div className="text-[11px] uppercase tracking-wide text-gray-500">{query.trim() ? `${results.length === 40 ? '40+' : results.length} matches` : 'Popular printers'}</div>
              <ul className="max-h-52 overflow-y-auto rounded border border-gray-800 divide-y divide-gray-800" role="listbox" aria-label="Printers" data-testid="slicer-printer-results">
                {results.length === 0 && <li className="px-3 py-2 text-xs text-gray-400">No printer matches. Try fewer words, or browse by brand below.</li>}
                {results.map((p, i) => (
                  <li key={p.name}>
                    <button
                      onClick={() => pick(p)}
                      onMouseEnter={() => setHighlight(i)}
                      role="option"
                      aria-selected={p.name === name}
                      className={`w-full flex items-center gap-2 px-3 py-1.5 text-left text-xs ${i === highlight ? 'bg-gray-800' : ''}`}
                      data-testid="slicer-printer-result"
                    >
                      <span className="flex-1 truncate text-gray-200">{shortPrinterName(p.name)}</span>
                      <span className="text-[11px] text-gray-500">{nozzleLabel(p)}</span>
                      {p.name === name && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                    </button>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => setShowBrands((v) => !v)}
                className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-gray-200"
                aria-expanded={showBrands}
                data-testid="slicer-browse-brands"
              >
                {showBrands ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />} Browse by brand ({vendors.length} brands, {printers.length} printers)
              </button>
              {showBrands && (
                <div className="grid grid-cols-3 gap-2">
                  <select value={vendor} onChange={(e) => chooseVendor(e.target.value)} className={selectClass} aria-label="Brand" data-testid="slicer-vendor">
                    <option value="">Brand…</option>
                    {vendors.map((v) => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <select value={selected?.model ?? ''} onChange={(e) => chooseModel(e.target.value)} disabled={!vendor} className={selectClass} aria-label="Printer" data-testid="slicer-model">
                    {models.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <select value={name} onChange={(e) => setName(e.target.value)} disabled={!selected} className={selectClass} aria-label="Nozzle" data-testid="slicer-nozzle">
                    {variants.map((p) => <option key={p.name} value={p.name}>{nozzleLabel(p)}</option>)}
                  </select>
                </div>
              )}
              {selected && (
                <p className="text-[11px] text-gray-400" data-testid="slicer-printer-bed">
                  Selected: <span className="text-gray-200">{selected.name}</span> · bed {bedSize(selected)}
                </p>
              )}
              {remembered && (
                <button
                  onClick={() => {
                    setName(remembered.name)
                    setBrowsing(false)
                    setQuery('')
                  }}
                  className="text-[11px] text-cyan-300 hover:text-cyan-200 underline"
                  data-testid="slicer-keep-printer"
                >
                  Keep {shortPrinterName(remembered.name)} ({nozzleLabel(remembered)})
                </button>
              )}
            </div>
          )}

          {list && selected && (
            <fieldset className="space-y-1.5" data-testid="slicer-swaps">
              <legend className="text-xs font-semibold text-gray-200 mb-1.5">Colour changes</legend>
              <div className="grid grid-cols-2 gap-2" role="radiogroup">
                {[
                  { value: false, icon: Layers, title: 'Automatic', text: 'The printer swaps filament itself (AMS, MMU or tool changer).', testid: 'slicer-swaps-auto' },
                  { value: true, icon: Hand, title: 'Pause & swap by hand', text: 'The print pauses at each colour; you change the spool and resume.', testid: 'slicer-swaps-manual' },
                ].map((o) => {
                  const active = manual === o.value
                  return (
                    <button
                      key={o.title}
                      role="radio"
                      aria-checked={active}
                      onClick={() => setManual(o.value)}
                      className={`text-left rounded-lg border px-3 py-2 transition-colors ${active ? 'border-cyan-500 bg-cyan-950/40' : 'border-gray-700 hover:border-gray-500'}`}
                      data-testid={o.testid}
                    >
                      <div className="flex items-center gap-1.5 text-xs font-medium text-gray-100">
                        <o.icon className="w-3.5 h-3.5" /> {o.title}
                        {recommendedManual === o.value && <span className="ml-auto text-[10px] font-normal text-cyan-300">for this printer</span>}
                      </div>
                      <p className="mt-0.5 text-[11px] leading-snug text-gray-400">{o.text}</p>
                    </button>
                  )
                })}
              </div>
            </fieldset>
          )}

          {list && (
            <p className="text-[11px] leading-snug text-gray-500">
              Uses OrcaSlicer {list.orca_version}'s own printer, print and filament settings, with your layer heights and 100 % infill.
              The first time, a printer's settings are downloaded from OrcaSlicer's GitHub.
            </p>
          )}
        </div>

        <div className="flex justify-end gap-2 px-5 py-3 border-t border-gray-800">
          <button onClick={onClose} className="px-3 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="slicer-cancel">
            Cancel
          </button>
          <button
            onClick={() => selected && onConfirm(selected.name, manual)}
            disabled={!selected}
            className="px-3 py-1.5 text-xs rounded text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
            data-testid="slicer-export"
          >
            {confirmLabel}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
