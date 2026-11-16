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

/** Printers whose name contains every word of ``query`` (any order, any
 * case). Each model's presets stay together with its 0.4 mm one first, so
 * Enter on the top hit picks the usual nozzle. */
export function searchPrinters(printers: SlicerPrinter[], query: string, limit = 50): SlicerPrinter[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return []
  const matches = printers.filter((p) => {
    const hay = `${p.vendor} ${p.name}`.toLowerCase()
    return words.every((w) => hay.includes(w))
  })
  const modelOrder = new Map<string, number>()
  for (const p of matches) if (!modelOrder.has(p.model)) modelOrder.set(p.model, modelOrder.size)
  return matches
    .map((p, i) => ({ p, i }))
    .sort((a, b) =>
      modelOrder.get(a.p.model)! - modelOrder.get(b.p.model)! ||
      Number(b.p.nozzle === '0.4') - Number(a.p.nozzle === '0.4') ||
      a.i - b.i,
    )
    .slice(0, limit)
    .map(({ p }) => p)
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

/** Quick picks shown before anything is searched: common printers for
 * colour-layer prints. Only the ones the server actually lists are shown. */
export const POPULAR_PRINTERS = [
  'Bambu Lab A1 mini 0.4 nozzle',
  'Bambu Lab A1 0.4 nozzle',
  'Bambu Lab P1S 0.4 nozzle',
  'Bambu Lab X1 Carbon 0.4 nozzle',
  'Bambu Lab H2D 0.4 nozzle',
  'Prusa MK4S 0.4 nozzle',
  'Prusa CORE One 0.4 nozzle',
  'Creality K1C 0.4 nozzle',
  'Creality Ender-3 V3 SE 0.4 nozzle',
  'Elegoo Centauri Carbon 0.4 nozzle',
  'Elegoo Neptune 4 Pro 0.4 nozzle',
  'Anycubic Kobra 3 0.4 nozzle',
  'Qidi Q1 Pro 0.4 nozzle',
  'Snapmaker U1 (0.4 nozzle)',
]

export function popularPrinters(printers: SlicerPrinter[]): SlicerPrinter[] {
  const byName = new Map(printers.map((p) => [p.name, p]))
  return POPULAR_PRINTERS.map((n) => byName.get(n)).filter((p): p is SlicerPrinter => !!p)
}

/** "Bambu Lab X1 Carbon 0.4 nozzle" -> "Bambu Lab X1 Carbon"; names that
 * don't follow the pattern are kept whole. */
export function shortPrinterName(name: string): string {
  const short = name.replace(/\s*\(?\s*[\d.+]+\s*(mm)?\s*nozzle\)?\s*$/i, '').trim()
  return short || name
}

/** The nozzle part of a preset for the nozzle picker: "0.4 mm", or the
 * remainder of the name for variants like "HF0.4 nozzle". */
export function nozzleLabel(p: SlicerPrinter): string {
  if (p.name === `${p.model} ${p.nozzle} nozzle` || p.name === `${p.model} (${p.nozzle} nozzle)`) return `${p.nozzle} mm`
  return p.name.replace(p.model, '').replace(/[()]/g, '').trim() || p.name
}

export function bedSize(p: SlicerPrinter): string {
  return `${Math.round(p.bed[2] - p.bed[0])} × ${Math.round(p.bed[3] - p.bed[1])} mm`
}
