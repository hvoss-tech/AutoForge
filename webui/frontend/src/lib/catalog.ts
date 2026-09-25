/** Search, filtering and sorting for the filamentcolors.xyz catalog dialog.
 * Kept import-free so the node unit tests can load it directly. */

export interface CatalogEntry {
  id: number
  brand: string
  name: string
  color_name: string
  filament_type: string
  type_detail: string
  color: string
  td: number
  color_family: string
  available: boolean
  published: string
  url: string
}

export interface CatalogResponse {
  count: number
  db_last_modified: number | null
  newest_published: string
  generated_at: string | null
  last_checked: number | null
  added_since_release: number
  updating: boolean
  error: string | null
  filaments: CatalogEntry[]
  /** swatch id -> uuid of the library filament that already is that entry */
  library_uuids: Record<string, string>
}

export type CatalogSort = 'brand' | 'name' | 'td' | 'newest' | 'closest'

export interface CatalogFilter {
  query: string
  filamentType: string
  brand: string
  family: string
  hideInLibrary: boolean
  /** '#rrggbb' to rank by closeness to, or '' */
  matchColor: string
}

export const EMPTY_FILTER: CatalogFilter = {
  query: '',
  filamentType: '',
  brand: '',
  family: '',
  hideInLibrary: false,
  matchColor: '',
}

/** filamentcolors.xyz's color families, with a representative dot color. */
export const COLOR_FAMILIES: { name: string; dot: string }[] = [
  { name: 'Red', dot: '#d32f2f' },
  { name: 'Orange', dot: '#f57c00' },
  { name: 'Yellow', dot: '#fbc02d' },
  { name: 'Green', dot: '#388e3c' },
  { name: 'Blue', dot: '#1976d2' },
  { name: 'Purple', dot: '#7b1fa2' },
  { name: 'Pink', dot: '#ec6fa6' },
  { name: 'Brown', dot: '#795548' },
  { name: 'Black', dot: '#111111' },
  { name: 'Gray', dot: '#8a8a8a' },
  { name: 'White', dot: '#f5f5f5' },
  { name: 'Transparent', dot: 'transparent' },
]

export function tokenize(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.replace(/^#/, ''))
    .filter(Boolean)
}

function haystack(e: CatalogEntry): string {
  return `${e.brand} ${e.name} ${e.type_detail} ${e.filament_type} ${e.color_family} ${e.color.replace('#', '')}`.toLowerCase()
}

/** Every word of the query must appear somewhere (brand, name, type,
 * color family or hex), in any order: "bambu jade", "petg blue", "ff0000". */
export function matchesQuery(e: CatalogEntry, tokens: string[]): boolean {
  if (tokens.length === 0) return true
  const text = haystack(e)
  return tokens.every((t) => text.includes(t))
}

export function isInLibrary(e: CatalogEntry, libraryUuids: Record<string, string>): boolean {
  return Object.prototype.hasOwnProperty.call(libraryUuids, String(e.id))
}

export function filterCatalog(
  entries: CatalogEntry[],
  filter: CatalogFilter,
  libraryUuids: Record<string, string> = {},
): CatalogEntry[] {
  const tokens = tokenize(filter.query)
  return entries.filter(
    (e) =>
      (!filter.filamentType || e.filament_type === filter.filamentType) &&
      (!filter.brand || e.brand === filter.brand) &&
      (!filter.family || e.color_family === filter.family) &&
      (!filter.hideInLibrary || !isInLibrary(e, libraryUuids)) &&
      matchesQuery(e, tokens),
  )
}

// ---- color distance (CIE76 in CIELAB — plenty to rank "closest") ----------

export function hexToLab(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  const r = lin((n >> 16) & 255), g = lin((n >> 8) & 255), b = lin(n & 255)
  // sRGB -> XYZ (D65), normalised by the white point
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116)
  const fx = f(x), fy = f(y), fz = f(z)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

export function deltaE(a: string, b: string): number {
  const la = hexToLab(a)
  const lb = hexToLab(b)
  if (!la || !lb) return Infinity
  return Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2])
}

/** How close two colors look, in words — ΔE alone means nothing to most people. */
export function describeDeltaE(d: number): string {
  if (d < 2.3) return 'Near identical'
  if (d < 6) return 'Very close'
  if (d < 12) return 'Close'
  if (d < 25) return 'Similar'
  return 'Different'
}

export function sortCatalog(entries: CatalogEntry[], sort: CatalogSort, matchColor = ''): CatalogEntry[] {
  const byBrand = (a: CatalogEntry, b: CatalogEntry) =>
    a.brand.localeCompare(b.brand) || a.name.localeCompare(b.name) || a.id - b.id
  const copy = [...entries]
  switch (sort) {
    case 'name':
      return copy.sort((a, b) => a.name.localeCompare(b.name) || byBrand(a, b))
    case 'td':
      return copy.sort((a, b) => a.td - b.td || byBrand(a, b))
    case 'newest':
      return copy.sort((a, b) => b.published.localeCompare(a.published) || b.id - a.id)
    case 'closest': {
      if (!hexToLab(matchColor)) return copy.sort(byBrand)
      const dist = new Map(copy.map((e) => [e.id, deltaE(e.color, matchColor)]))
      return copy.sort((a, b) => dist.get(a.id)! - dist.get(b.id)! || byBrand(a, b))
    }
    default:
      return copy.sort(byBrand)
  }
}

/** Values of a field with how many entries have each, most common first —
 * for the type/brand dropdowns. */
export function facetCounts(entries: CatalogEntry[], key: 'filament_type' | 'brand'): { value: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const e of entries) counts.set(e[key], (counts.get(e[key]) ?? 0) + 1)
  return [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))
}

/** How many search filters are narrowing the results ("hide ones in my
 * library" is a remembered preference and doesn't count). */
export function activeFilterCount(filter: CatalogFilter): number {
  return [filter.query.trim(), filter.filamentType, filter.brand, filter.family, filter.matchColor].filter(Boolean).length
}

/** The library filament a catalog entry becomes (mirrors the server's
 * FilamentService.add_catalog_entries). */
export function catalogUuid(id: number): string {
  return `filamentcolors-${id}`
}
