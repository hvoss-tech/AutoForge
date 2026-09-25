import React from 'react'
import { useAppStore } from '../store/appStore'
import { refreshLibrary } from '../services/filamentService'
import { describeApiError } from '../lib/apiError'
import { contrastTextColor } from '../lib/color'
import {
  COLOR_FAMILIES,
  EMPTY_FILTER,
  activeFilterCount,
  deltaE,
  describeDeltaE,
  facetCounts,
  filterCatalog,
  isInLibrary,
  sortCatalog,
  type CatalogEntry,
  type CatalogFilter,
  type CatalogResponse,
  type CatalogSort,
} from '../lib/catalog'
import type { Filament } from '../types'
import { Check, ExternalLink, Globe, Loader2, Pipette, Plus, RefreshCw, Search, X } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'

// Cards rendered per batch; more load as the list is scrolled.
const PAGE = 60

function readPref(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}

function writePref(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // not remembered — fine
  }
}

const SORTS: CatalogSort[] = ['brand', 'name', 'td', 'newest', 'closest']

// The last catalog fetched, so reopening the dialog shows results at once
// while the (cheap, local) refetch picks up library changes.
let cachedCatalog: CatalogResponse | null = null

const selectClass = 'bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-cyan-500 max-w-[11rem]'

export const CatalogModal: React.FC = () => {
  const open = useAppStore((s) => s.catalogModalOpen)
  const setOpen = useAppStore((s) => s.setCatalogModalOpen)
  const setNewFilamentModalOpen = useAppStore((s) => s.setNewFilamentModalOpen)
  const setCustomLibraryLoaded = useAppStore((s) => s.setCustomLibraryLoaded)
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const pushToast = useAppStore((s) => s.pushToast)

  const [catalog, setCatalog] = React.useState<CatalogResponse | null>(cachedCatalog)
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [filter, setFilter] = React.useState<CatalogFilter>(() => ({
    ...EMPTY_FILTER,
    hideInLibrary: readPref('autoforge-catalog-hide-in-library', '0') === '1',
  }))
  const [sort, setSort] = React.useState<CatalogSort>(() => {
    const s = readPref('autoforge-catalog-sort', 'brand') as CatalogSort
    return SORTS.includes(s) && s !== 'closest' ? s : 'brand'
  })
  const [owned, setOwned] = React.useState(() => readPref('autoforge-catalog-owned', '1') === '1')
  const [activate, setActivate] = React.useState(() => readPref('autoforge-catalog-activate', '0') === '1')
  const [visible, setVisible] = React.useState(PAGE)
  const [pending, setPending] = React.useState<Set<number>>(new Set())
  const searchRef = React.useRef<HTMLInputElement>(null)
  const listRef = React.useRef<HTMLDivElement>(null)
  const sentinelRef = React.useRef<HTMLDivElement>(null)

  const load = React.useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const res = await fetch('/api/filaments/catalog')
      if (!res.ok) throw new Error(describeApiError(await res.json().catch(() => null), res.status))
      const data: CatalogResponse = await res.json()
      cachedCatalog = data
      setCatalog(data)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    if (open) load()
  }, [open, load])

  const update = (patch: Partial<CatalogFilter>) => {
    setFilter((f) => ({ ...f, ...patch }))
    setVisible(PAGE)
    listRef.current?.scrollTo({ top: 0 })
  }

  const chooseSort = (s: CatalogSort) => {
    setSort(s)
    setVisible(PAGE)
    if (s !== 'closest') writePref('autoforge-catalog-sort', s)
  }

  const setMatchColor = (hex: string) => {
    update({ matchColor: hex })
    if (hex) setSort('closest')
    else if (sort === 'closest') setSort('brand')
  }

  // "Hide ones in my library" is a remembered preference, not a search filter.
  const resetFilters = () => {
    update({ ...EMPTY_FILTER, hideInLibrary: filter.hideInLibrary })
    if (sort === 'closest') setSort('brand')
  }

  const entries = catalog?.filaments ?? []
  const libraryUuids = catalog?.library_uuids ?? {}
  const results = React.useMemo(
    () => sortCatalog(filterCatalog(entries, filter, libraryUuids), sort, filter.matchColor),
    [entries, filter, libraryUuids, sort],
  )
  // Dropdown counts follow the other filters, so they never offer an empty choice.
  const typeFacets = React.useMemo(() => facetCounts(filterCatalog(entries, { ...filter, filamentType: '' }, libraryUuids), 'filament_type'), [entries, filter, libraryUuids])
  const brandFacets = React.useMemo(
    () => facetCounts(filterCatalog(entries, { ...filter, brand: '' }, libraryUuids), 'brand').sort((a, b) => a.value.localeCompare(b.value)),
    [entries, filter, libraryUuids],
  )
  const familiesPresent = React.useMemo(() => new Set(entries.map((e) => e.color_family)), [entries])
  const shown = results.slice(0, visible)

  // Load the next batch as the end of the list scrolls into view.
  React.useEffect(() => {
    const el = sentinelRef.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const obs = new IntersectionObserver(
      (items) => {
        if (items.some((i) => i.isIntersecting)) setVisible((v) => v + PAGE)
      },
      { root: listRef.current, rootMargin: '300px' },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [shown.length, results.length])

  const add = async (entry: CatalogEntry) => {
    setPending((p) => new Set(p).add(entry.id))
    try {
      const res = await fetch('/api/filaments/catalog/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: [entry.id], owned, activate: false }),
      })
      if (!res.ok) throw new Error(describeApiError(await res.json().catch(() => null), res.status))
      const data: { added: Filament[]; existing: Filament[] } = await res.json()
      const filament = data.added[0] ?? data.existing[0]
      setCatalog((c) => {
        if (!c || !filament) return c
        const next = { ...c, library_uuids: { ...c.library_uuids, [String(entry.id)]: filament.uuid } }
        cachedCatalog = next
        return next
      })
      setCustomLibraryLoaded(true)
      // Activate through the store, so it lands in undo history like a click on "+".
      if (activate && filament) addActiveFilament(filament)
      await refreshLibrary(filament?.filament_type)
      pushToast(
        data.added.length
          ? `Added ${entry.brand} ${entry.name} to your library${activate ? ' and activated it' : ''}.`
          : `${entry.brand} ${entry.name} is already in your library.`,
        'info',
      )
    } catch (e) {
      pushToast(`Couldn't add ${entry.brand} ${entry.name}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setPending((p) => {
        const next = new Set(p)
        next.delete(entry.id)
        return next
      })
    }
  }

  const filtersOn = activeFilterCount(filter)
  const inLibraryCount = Object.keys(libraryUuids).length

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="max-w-5xl w-[96vw] h-[88vh] flex flex-col p-0 gap-0 bg-gray-900 border-gray-700 overflow-hidden"
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          searchRef.current?.focus()
        }}
        data-testid="catalog-modal"
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3 border-b border-gray-700">
          <div className="min-w-0">
            <DialogTitle className="text-base font-semibold text-gray-100 flex items-center gap-2 leading-none tracking-normal">
              <Globe className="w-4 h-4 text-cyan-500" />
              Filament Catalog
            </DialogTitle>
            <DialogDescription className="mt-1.5 text-xs text-gray-400" data-testid="catalog-summary">
              {catalog ? (
                <>
                  <span className="text-gray-200 font-medium tabular-nums">{catalog.count}</span> filaments with a measured TD from{' '}
                  <a href="https://filamentcolors.xyz" target="_blank" rel="noopener noreferrer" className="text-cyan-500 hover:underline">
                    filamentcolors.xyz
                  </a>
                  {inLibraryCount > 0 && <> · {inLibraryCount} already in your library</>}
                  {catalog.added_since_release > 0 && <> · {catalog.added_since_release} new since this release</>}
                </>
              ) : (
                'Search community-measured filaments and add yours to the library.'
              )}
              {catalog?.updating && (
                <span className="inline-flex items-center gap-1 ml-2 text-gray-500" data-testid="catalog-updating">
                  <Loader2 className="w-3 h-3 animate-spin" /> checking for new filaments…
                </span>
              )}
            </DialogDescription>
          </div>
          <button onClick={() => setOpen(false)} className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-800" aria-label="Close" data-testid="catalog-close">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Search and filters */}
        <div className="px-5 py-3 space-y-2.5 border-b border-gray-700 bg-gray-900">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                ref={searchRef}
                value={filter.query}
                onChange={(e) => update({ query: e.target.value })}
                placeholder="Search brand, color, type or #hex — e.g. “bambu jade white”"
                className="w-full pl-9 pr-8 py-2 rounded-md text-sm bg-gray-800 border border-gray-600 text-gray-100 placeholder:text-gray-500 focus:outline-none focus:border-cyan-500"
                aria-label="Search the catalog"
                data-testid="catalog-search"
              />
              {filter.query && (
                <button onClick={() => update({ query: '' })} className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-gray-400 hover:text-gray-100" aria-label="Clear search" data-testid="catalog-search-clear">
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            {/* Rank by closeness to a color: for "which filament is this?" */}
            <div className={`flex items-center rounded-md border ${filter.matchColor ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-600'} `}>
              <label className="flex items-center gap-1.5 pl-2.5 pr-2 py-2 text-xs text-gray-300 cursor-pointer hover:text-gray-100" title="Show the filaments closest to a color first">
                {filter.matchColor ? (
                  <span className="w-4 h-4 rounded-full border border-gray-500" style={{ backgroundColor: filter.matchColor }} />
                ) : (
                  <Pipette className="w-4 h-4" />
                )}
                <span className="whitespace-nowrap">{filter.matchColor ? filter.matchColor.toUpperCase() : 'Match a color'}</span>
                <input
                  type="color"
                  value={filter.matchColor || '#808080'}
                  onChange={(e) => setMatchColor(e.target.value)}
                  className="sr-only"
                  data-testid="catalog-match-color"
                />
              </label>
              {filter.matchColor && (
                <button onClick={() => setMatchColor('')} className="pr-2 text-gray-400 hover:text-gray-100" aria-label="Stop matching a color" data-testid="catalog-match-clear">
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap" role="group" aria-label="Color family">
            <button
              onClick={() => update({ family: '' })}
              aria-pressed={!filter.family}
              className={`px-2 h-6 rounded-full text-[11px] border ${!filter.family ? 'border-cyan-500 text-cyan-500 bg-cyan-500/10' : 'border-gray-600 text-gray-300 hover:bg-gray-800'}`}
              data-testid="catalog-family-all"
            >
              All colors
            </button>
            {COLOR_FAMILIES.filter((f) => familiesPresent.has(f.name)).map((f) => {
              const selected = filter.family === f.name
              return (
                <button
                  key={f.name}
                  onClick={() => update({ family: selected ? '' : f.name })}
                  aria-pressed={selected}
                  aria-label={f.name}
                  title={f.name}
                  className={`w-6 h-6 rounded-full border transition-transform hover:scale-110 ${selected ? 'ring-2 ring-cyan-500 ring-offset-2 ring-offset-gray-900 border-transparent' : 'border-gray-600'}`}
                  style={
                    f.name === 'Transparent'
                      ? { backgroundImage: 'conic-gradient(#bbb 25%, #fff 0 50%, #bbb 0 75%, #fff 0)', backgroundSize: '8px 8px' }
                      : { backgroundColor: f.dot }
                  }
                  data-testid={`catalog-family-${f.name}`}
                />
              )
            })}
          </div>

          <div className="flex items-center gap-x-3 gap-y-2 flex-wrap text-xs text-gray-400">
            <label className="flex items-center gap-1.5">
              Type
              <select value={filter.filamentType} onChange={(e) => update({ filamentType: e.target.value })} className={selectClass} data-testid="catalog-type-filter">
                <option value="">All types</option>
                {typeFacets.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.value} ({t.count})
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5">
              Brand
              <select value={filter.brand} onChange={(e) => update({ brand: e.target.value })} className={selectClass} data-testid="catalog-brand-filter">
                <option value="">All brands</option>
                {brandFacets.map((b) => (
                  <option key={b.value} value={b.value}>
                    {b.value} ({b.count})
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5">
              Sort
              <select value={sort} onChange={(e) => chooseSort(e.target.value as CatalogSort)} className={selectClass} data-testid="catalog-sort">
                <option value="brand">Brand</option>
                <option value="name">Color name</option>
                <option value="td">TD (opaque first)</option>
                <option value="newest">Newest</option>
                <option value="closest" disabled={!filter.matchColor}>
                  Closest to color
                </option>
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={filter.hideInLibrary}
                onChange={(e) => {
                  update({ hideInLibrary: e.target.checked })
                  writePref('autoforge-catalog-hide-in-library', e.target.checked ? '1' : '0')
                }}
                data-testid="catalog-hide-in-library"
              />
              Hide ones in my library
            </label>
            <span className="ml-auto flex items-center gap-2">
              {filtersOn > 0 && (
                <button onClick={resetFilters} className="text-cyan-500 hover:underline" data-testid="catalog-reset-filters">
                  Reset filters
                </button>
              )}
              <span className="tabular-nums text-gray-300" data-testid="catalog-result-count" data-count={results.length}>
                {results.length} {results.length === 1 ? 'result' : 'results'}
              </span>
            </span>
          </div>
        </div>

        {/* Results */}
        <div ref={listRef} className="flex-1 overflow-y-auto min-h-0 px-5 py-4" data-testid="catalog-results">
          {!catalog && loading && (
            <div className="h-full flex items-center justify-center gap-2 text-sm text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading the catalog…
            </div>
          )}
          {!catalog && loadError && (
            <div className="h-full flex flex-col items-center justify-center gap-3 text-sm text-gray-400" data-testid="catalog-error">
              <span>The catalog couldn't be loaded: {loadError}</span>
              <button onClick={load} className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs">
                <RefreshCw className="w-3.5 h-3.5" /> Try again
              </button>
            </div>
          )}
          {catalog && results.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center gap-3 text-center text-sm text-gray-400" data-testid="catalog-empty">
              <Search className="w-8 h-8 text-gray-600" />
              <div>
                <div className="text-gray-200 font-medium">No filaments match</div>
                <div className="text-xs mt-1">Try fewer words, another spelling, or clear the filters.</div>
              </div>
              <div className="flex items-center gap-2">
                {filtersOn > 0 && (
                  <button onClick={resetFilters} className="px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs" data-testid="catalog-clear-filters">
                    Clear filters
                  </button>
                )}
                {filtersOn === 0 && filter.hideInLibrary && (
                  <button
                    onClick={() => {
                      update({ hideInLibrary: false })
                      writePref('autoforge-catalog-hide-in-library', '0')
                    }}
                    className="px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs"
                    data-testid="catalog-show-in-library"
                  >
                    Show ones in my library
                  </button>
                )}
                <button
                  onClick={() => {
                    setOpen(false)
                    setNewFilamentModalOpen(true)
                  }}
                  className="px-3 py-1.5 rounded border border-gray-600 hover:bg-gray-800 text-gray-200 text-xs"
                  data-testid="catalog-create-own"
                >
                  Create it yourself
                </button>
              </div>
            </div>
          )}
          {results.length > 0 && (
            <>
              <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))' }}>
                {shown.map((entry) => (
                  <CatalogCard
                    key={entry.id}
                    entry={entry}
                    inLibrary={isInLibrary(entry, libraryUuids)}
                    pending={pending.has(entry.id)}
                    matchColor={filter.matchColor}
                    onAdd={() => add(entry)}
                  />
                ))}
              </div>
              {shown.length < results.length && (
                <div ref={sentinelRef} className="flex justify-center pt-4">
                  <button onClick={() => setVisible((v) => v + PAGE)} className="px-3 py-1.5 rounded border border-gray-600 text-xs text-gray-300 hover:bg-gray-800" data-testid="catalog-show-more">
                    Show more ({results.length - shown.length} left)
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* How added filaments land in the library */}
        <div className="flex items-center gap-4 flex-wrap px-5 py-2.5 border-t border-gray-700 text-xs text-gray-300">
          <label className="flex items-center gap-1.5 cursor-pointer" title="Owned filaments show up under “Owned only” in the library">
            <input
              type="checkbox"
              checked={owned}
              onChange={(e) => {
                setOwned(e.target.checked)
                writePref('autoforge-catalog-owned', e.target.checked ? '1' : '0')
              }}
              data-testid="catalog-owned-toggle"
            />
            Mark added filaments as owned
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer" title="Active filaments are the ones the optimizer can use">
            <input
              type="checkbox"
              checked={activate}
              onChange={(e) => {
                setActivate(e.target.checked)
                writePref('autoforge-catalog-activate', e.target.checked ? '1' : '0')
              }}
              data-testid="catalog-activate-toggle"
            />
            Also make them active for the optimizer
          </label>
          <span className="ml-auto text-gray-500">
            Data by the filamentcolors.xyz community — TDs are measured, your spool may differ slightly.
          </span>
        </div>
      </DialogContent>
    </Dialog>
  )
}

const CatalogCard: React.FC<{
  entry: CatalogEntry
  inLibrary: boolean
  pending: boolean
  matchColor: string
  onAdd: () => void
}> = ({ entry, inLibrary, pending, matchColor, onAdd }) => {
  const text = contrastTextColor(entry.color)
  const distance = matchColor ? deltaE(entry.color, matchColor) : null
  // Fixed black/white: the theme's gray-* classes invert in light mode, but a
  // badge sits on the filament's own color, not on the page.
  const badge = text === '#000000' ? 'bg-white/70 text-black' : 'bg-black/45 text-white'
  const showDetail = entry.type_detail && entry.type_detail !== entry.filament_type

  return (
    <div
      className={`group flex flex-col rounded-lg border overflow-hidden transition-colors ${inLibrary ? 'border-emerald-600/60' : 'border-gray-700 hover:border-gray-500'} bg-gray-800/60`}
      data-testid={`catalog-card-${entry.id}`}
      data-in-library={inLibrary}
    >
      <div className="relative h-20 flex-shrink-0" style={{ backgroundColor: entry.color }}>
        <span
          className={`absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded text-[10px] font-semibold tabular-nums ${badge}`}
          title="Transmission distance: how see-through the filament is (higher = more translucent)"
          data-testid={`catalog-td-${entry.id}`}
        >
          TD {entry.td}
        </span>
        {inLibrary && (
          <span className="absolute top-1.5 left-1.5 flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-600 text-white">
            <Check className="w-3 h-3" /> In library
          </span>
        )}
        {distance !== null && Number.isFinite(distance) && (
          <span className={`absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${badge}`} title={`Color difference ΔE ${distance.toFixed(1)}`} data-testid={`catalog-distance-${entry.id}`}>
            {describeDeltaE(distance)} · ΔE {distance.toFixed(1)}
          </span>
        )}
        <span className={`absolute bottom-1.5 right-1.5 text-[10px] font-mono opacity-0 group-hover:opacity-80 transition-opacity`} style={{ color: text }}>
          {entry.color.toUpperCase()}
        </span>
      </div>
      <div className="flex-1 flex flex-col gap-1 p-2.5 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-gray-400 truncate" title={entry.brand}>
          {entry.brand}
        </div>
        <div className="text-sm font-medium text-gray-100 leading-snug line-clamp-2" title={entry.name}>
          {entry.color_name}
        </div>
        <div className="flex flex-wrap gap-1 text-[10px]">
          <span className="px-1.5 py-px rounded bg-gray-700 text-gray-300">{entry.filament_type}</span>
          {showDetail && <span className="px-1.5 py-px rounded bg-gray-700/60 text-gray-400 truncate max-w-full" title={entry.type_detail}>{entry.type_detail}</span>}
          {!entry.available && (
            <span className="px-1.5 py-px rounded border border-gray-600 text-gray-400" title="filamentcolors.xyz lists this filament as no longer available to buy">
              Unavailable
            </span>
          )}
        </div>
        <div className="mt-auto pt-2 flex items-center gap-1.5">
          {inLibrary ? (
            <span className="flex-1 flex items-center justify-center gap-1 py-1 rounded text-xs text-emerald-500 border border-emerald-600/40" data-testid={`catalog-in-library-${entry.id}`}>
              <Check className="w-3.5 h-3.5" /> In your library
            </span>
          ) : (
            <button
              onClick={onAdd}
              disabled={pending}
              className="flex-1 flex items-center justify-center gap-1 py-1 rounded text-xs font-medium bg-cyan-600 hover:bg-cyan-500 disabled:opacity-60 text-white"
              aria-label={`Add ${entry.brand} ${entry.name} to your library`}
              data-testid={`catalog-add-${entry.id}`}
            >
              {pending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
              Add
            </button>
          )}
          <a
            href={entry.url}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 rounded border border-gray-600 text-gray-400 hover:text-gray-100 hover:bg-gray-700"
            title="View on filamentcolors.xyz"
            aria-label={`View ${entry.brand} ${entry.name} on filamentcolors.xyz`}
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </div>
  )
}
