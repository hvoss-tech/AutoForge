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

export const DEFAULT_NOZZLE = 0.4

const sameNozzle = (a: string | number, b: number) => Math.abs(Number(a) - b) < 1e-3

/** Order presets by how well their nozzle fits ``nozzle``: the same one,
 * else the closest, a tie going to the common 0.4 mm. */
function byNozzleFit(nozzle: number) {
  const fit = (p: SlicerPrinter) => {
    const n = Number(p.nozzle)
    // Rounded: |0.6 - 0.4| is 0.19999999999999996 in floating point, which
    // broke the tie with |0.2 - 0.4|.
    const off = (x: number) => Math.round(Math.abs(n - x) * 1e6)
    return Number.isFinite(n) ? [off(nozzle), off(DEFAULT_NOZZLE)] : [Infinity, Infinity]
  }
  return (a: SlicerPrinter, b: SlicerPrinter) => {
    const [da, sa] = fit(a)
    const [db, sb] = fit(b)
    return da - db || sa - sb
  }
}

/** The presets of one model (its nozzle sizes), the one for ``nozzle`` first. */
export function variantsOf(printers: SlicerPrinter[], vendor: string, model: string, nozzle = DEFAULT_NOZZLE): SlicerPrinter[] {
  return printers
    .filter((p) => p.vendor === vendor && p.model === model)
    .sort((a, b) => byNozzleFit(nozzle)(a, b) || a.name.localeCompare(b.name, undefined, { numeric: true }))
}

/** Printers whose name contains every word of ``query`` (any order, any
 * case). Each model's presets stay together, the one that fits ``nozzle``
 * (the configured nozzle, 0.4 mm by default) best first, so Enter on the
 * top hit picks the right one. */
export function searchPrinters(printers: SlicerPrinter[], query: string, limit = 50, nozzle = DEFAULT_NOZZLE): SlicerPrinter[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return []
  const matches = printers.filter((p) => {
    const hay = `${p.vendor} ${p.name}`.toLowerCase()
    return words.every((w) => hay.includes(w))
  })
  const fit = byNozzleFit(nozzle)
  const modelOrder = new Map<string, number>()
  for (const p of matches) if (!modelOrder.has(p.model)) modelOrder.set(p.model, modelOrder.size)
  return matches
    .map((p, i) => ({ p, i }))
    .sort((a, b) =>
      modelOrder.get(a.p.model)! - modelOrder.get(b.p.model)! ||
      fit(a.p, b.p) ||
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

/** The popular printers, each as its preset for ``nozzle`` when it has one. */
export function popularPrinters(printers: SlicerPrinter[], nozzle = DEFAULT_NOZZLE): SlicerPrinter[] {
  const byName = new Map(printers.map((p) => [p.name, p]))
  return POPULAR_PRINTERS.map((n) => byName.get(n))
    .filter((p): p is SlicerPrinter => !!p)
    .map((p) => nozzleMatch(printers, p, nozzle).printer)
}

export interface NozzleMatch {
  /** 'same': the preset has the nozzle; 'switched': the printer's preset
   * for the nozzle is used instead; 'missing': the printer has no preset for
   * it, so the picked one is used and the user has to set the nozzle. */
  kind: 'same' | 'switched' | 'missing'
  printer: SlicerPrinter
  /** The printer's nozzle sizes (mm), for the warning. */
  available: number[]
}

/** Which preset a slicer project for ``picked`` uses with a ``nozzle``
 * (mm) result: the same rule as the server (OrcaProfiles.printer_for_nozzle). */
export function nozzleMatch(printers: SlicerPrinter[], picked: SlicerPrinter, nozzle: number | null | undefined): NozzleMatch {
  const siblings = printers.filter((p) => p.vendor === picked.vendor && p.model === picked.model)
  const available = [...new Set(siblings.map((p) => Number(p.nozzle)).filter((n) => Number.isFinite(n)))].sort((a, b) => a - b)
  if (!nozzle || nozzle <= 0 || sameNozzle(picked.nozzle, nozzle)) return { kind: 'same', printer: picked, available }
  const matches = siblings.filter((p) => sameNozzle(p.nozzle, nozzle))
  if (matches.length === 0) return { kind: 'missing', printer: picked, available }
  const styled = picked.name.replace(picked.nozzle, '{n}')
  const sameStyle = matches.find((p) => styled.replace('{n}', p.nozzle) === p.name)
  const plainest = [...matches].sort((a, b) => a.name.length - b.name.length || a.name.localeCompare(b.name))[0]
  return { kind: 'switched', printer: sameStyle ?? plainest, available }
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
