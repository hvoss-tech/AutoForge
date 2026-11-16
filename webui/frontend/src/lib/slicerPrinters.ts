/** Printers for File › Slicer project (.3mf): OrcaSlicer's printer presets
 * as served by GET /api/outputs/3mf/printers. */

export interface SlicerPrinter {
  /** OrcaSlicer's preset name, e.g. "Bambu Lab X1 Carbon 0.4 nozzle". */
  name: string
  vendor: string
  model: string
  nozzle: string
  /** Printable area: [x0, y0, x1, y1] in mm. */
  bed: [number, number, number, number]
  /** The printer changes filament on its own (AMS, MMU, tool changer). */
  auto_swaps: boolean
}

export interface PrinterChoice {
  vendor: string
  model: string
  name: string
}

/** Vendors in display order. */
export function vendorsOf(printers: SlicerPrinter[]): string[] {
  return [...new Set(printers.map((p) => p.vendor))].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
}

export function modelsOf(printers: SlicerPrinter[], vendor: string): string[] {
  return [...new Set(printers.filter((p) => p.vendor === vendor).map((p) => p.model))].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }),
  )
}

/** The presets of one model (its nozzle sizes), the 0.4 mm one first. */
export function variantsOf(printers: SlicerPrinter[], vendor: string, model: string): SlicerPrinter[] {
  return printers
    .filter((p) => p.vendor === vendor && p.model === model)
    .sort((a, b) => Number(b.nozzle === '0.4') - Number(a.nozzle === '0.4') || a.name.localeCompare(b.name, undefined, { numeric: true }))
}

/** Printers whose name contains every word of ``query`` (any order, any case). */
export function searchPrinters(printers: SlicerPrinter[], query: string, limit = 50): SlicerPrinter[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return []
  const out: SlicerPrinter[] = []
  for (const p of printers) {
    const hay = `${p.vendor} ${p.name}`.toLowerCase()
    if (words.every((w) => hay.includes(w))) {
      out.push(p)
      if (out.length >= limit) break
    }
  }
  return out
}

/** What the dialog starts on: the remembered printer, else nothing picked. */
export function initialChoice(printers: SlicerPrinter[], remembered: string | undefined | null): PrinterChoice | null {
  const p = remembered ? printers.find((x) => x.name === remembered) : undefined
  return p ? { vendor: p.vendor, model: p.model, name: p.name } : null
}

/** Whether the export pauses for hand swaps: the remembered choice when the
 * user made one for this printer, else whatever the printer can do. */
export function defaultManualSwaps(printer: SlicerPrinter | undefined, remembered: boolean | null | undefined, rememberedFor: string | undefined): boolean {
  if (printer && remembered !== null && remembered !== undefined && rememberedFor === printer.name) return remembered
  return printer ? !printer.auto_swaps : true
}
