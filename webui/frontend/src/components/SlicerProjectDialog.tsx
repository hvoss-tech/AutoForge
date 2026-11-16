import React, { useEffect, useMemo, useState } from 'react'
import { Printer, Search } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import {
  defaultManualSwaps,
  initialChoice,
  modelsOf,
  searchPrinters,
  variantsOf,
  vendorsOf,
  type SlicerPrinter,
} from '../lib/slicerPrinters'

const selectClass =
  'w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-cyan-500'

interface PrinterList {
  orca_version: string
  printers: SlicerPrinter[]
}

let printerListCache: Promise<PrinterList> | null = null

function loadPrinters(): Promise<PrinterList> {
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
  /** The printer remembered in the project settings, and its swap choice. */
  rememberedPrinter?: string
  rememberedManualSwaps?: boolean | null
  exporting: boolean
  onExport: (printer: string, manualSwaps: boolean) => void
}

/** File › Slicer project: pick the printer (OrcaSlicer's presets) and how
 * colour changes happen, then export. */
export const SlicerProjectDialog: React.FC<Props> = ({ open, onClose, rememberedPrinter, rememberedManualSwaps, exporting, onExport }) => {
  const [list, setList] = useState<PrinterList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [vendor, setVendor] = useState('')
  const [model, setModel] = useState('')
  const [name, setName] = useState('')
  const [manual, setManual] = useState(true)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setError(null)
    loadPrinters()
      .then((data) => {
        if (cancelled) return
        setList(data)
        const start = initialChoice(data.printers, rememberedPrinter)
        setVendor(start?.vendor ?? '')
        setModel(start?.model ?? '')
        setName(start?.name ?? '')
        setQuery('')
      })
      .catch(() => {
        if (!cancelled) setError('Could not load the printer list')
      })
    return () => {
      cancelled = true
    }
  }, [open, rememberedPrinter])

  const printers = list?.printers ?? []
  const vendors = useMemo(() => vendorsOf(printers), [printers])
  const models = useMemo(() => (vendor ? modelsOf(printers, vendor) : []), [printers, vendor])
  const variants = useMemo(() => (vendor && model ? variantsOf(printers, vendor, model) : []), [printers, vendor, model])
  const results = useMemo(() => searchPrinters(printers, query), [printers, query])
  const selected = printers.find((p) => p.name === name)

  // The swap mode follows the printer unless the user chose one for it.
  useEffect(() => {
    setManual(defaultManualSwaps(selected, rememberedManualSwaps, rememberedPrinter))
  }, [selected, rememberedManualSwaps, rememberedPrinter])

  const pick = (p: SlicerPrinter) => {
    setVendor(p.vendor)
    setModel(p.model)
    setName(p.name)
    setQuery('')
  }

  const chooseVendor = (v: string) => {
    setVendor(v)
    const first = v ? modelsOf(printers, v)[0] ?? '' : ''
    setModel(first)
    setName(first ? variantsOf(printers, v, first)[0]?.name ?? '' : '')
  }

  const chooseModel = (m: string) => {
    setModel(m)
    setName(m ? variantsOf(printers, vendor, m)[0]?.name ?? '' : '')
  }

  const bed = selected ? `${Math.round(selected.bed[2] - selected.bed[0])} × ${Math.round(selected.bed[3] - selected.bed[1])} mm` : null

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-lg bg-gray-900 border-gray-700 p-5" data-testid="slicer-project-dialog">
        <DialogTitle className="flex items-center gap-2 text-base text-gray-100">
          <Printer className="w-4 h-4" /> Slicer project (.3mf)
        </DialogTitle>
        <DialogDescription className="text-xs text-gray-400">
          An OrcaSlicer project of the model with every colour change already placed, set up for your printer.
        </DialogDescription>

        {error && <p className="text-xs text-red-300" role="alert">{error}</p>}
        {!list && !error && <p className="text-xs text-gray-400">Loading printers…</p>}

        {list && (
          <div className="space-y-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2 top-1.5 text-gray-500" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search printers, e.g. “x1 carbon” or “neptune 4”"
                className="w-full bg-gray-800 border border-gray-600 rounded pl-7 pr-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
                data-testid="slicer-printer-search"
              />
              {query && (
                <div className="absolute z-10 mt-1 w-full max-h-56 overflow-y-auto bg-gray-800 border border-gray-700 rounded shadow-lg" data-testid="slicer-printer-results">
                  {results.length === 0 && <p className="px-2 py-1.5 text-xs text-gray-400">No printer matches</p>}
                  {results.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => pick(p)}
                      className="block w-full text-left px-2 py-1 text-xs text-gray-200 hover:bg-gray-700"
                      data-testid="slicer-printer-result"
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-3 gap-2">
              <label className="text-[11px] text-gray-400 space-y-1">
                <span>Brand</span>
                <select value={vendor} onChange={(e) => chooseVendor(e.target.value)} className={selectClass} data-testid="slicer-vendor">
                  <option value="">Choose…</option>
                  {vendors.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
              <label className="text-[11px] text-gray-400 space-y-1">
                <span>Printer</span>
                <select value={model} onChange={(e) => chooseModel(e.target.value)} disabled={!vendor} className={selectClass} data-testid="slicer-model">
                  {models.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              <label className="text-[11px] text-gray-400 space-y-1">
                <span>Nozzle</span>
                <select value={name} onChange={(e) => setName(e.target.value)} disabled={!model} className={selectClass} data-testid="slicer-nozzle">
                  {variants.map((p) => <option key={p.name} value={p.name}>{p.name === `${p.model} ${p.nozzle} nozzle` ? `${p.nozzle} mm` : p.name.replace(p.model, '').trim() || p.name}</option>)}
                </select>
              </label>
            </div>
            {selected && <p className="text-[11px] text-gray-500" data-testid="slicer-printer-bed">{selected.name} · bed {bed}</p>}

            <fieldset className="space-y-1" disabled={!selected}>
              <legend className="text-[11px] text-gray-400 mb-1">Colour changes</legend>
              <label className="flex items-start gap-2 text-xs text-gray-200">
                <input type="radio" checked={!manual} onChange={() => setManual(false)} data-testid="slicer-swaps-auto" />
                <span>The printer changes filament itself <span className="text-gray-500">(AMS, MMU or tool changer)</span></span>
              </label>
              <label className="flex items-start gap-2 text-xs text-gray-200">
                <input type="radio" checked={manual} onChange={() => setManual(true)} data-testid="slicer-swaps-manual" />
                <span>Pause the print at each change, I swap by hand <span className="text-gray-500">(the printer's own pause command)</span></span>
              </label>
            </fieldset>

            <p className="text-[11px] text-gray-500">
              Printer, print and filament settings are OrcaSlicer {list.orca_version}'s own presets, with 100 % infill and
              your layer heights. The first export for a printer downloads its presets from OrcaSlicer's GitHub.
            </p>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-2">
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="slicer-cancel">
            Cancel
          </button>
          <button
            onClick={() => selected && onExport(selected.name, manual)}
            disabled={!selected || exporting}
            className="px-3 py-1.5 text-sm rounded text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
            data-testid="slicer-export"
          >
            {exporting ? 'Preparing…' : 'Export .3mf'}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
