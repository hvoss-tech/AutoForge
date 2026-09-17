import React from 'react'
import { useAppStore } from '../store/appStore'
import { useFilamentLoader } from '../services/filamentService'
import { AlertTriangle, Check, ChevronDown, ChevronRight, ChevronsLeft, Hand, Pencil, Plus, Save, Search, Tag, Upload, X } from 'lucide-react'
import type { Filament } from '../types'
import { FilamentSwatch } from './FilamentSwatch'
import { EditFilamentModal } from './EditFilamentModal'
import { sortFilaments, type FilamentSort } from '../lib/library'
import { onUiCommand } from '../lib/uiEvents'
import { useFlash } from '../hooks/usePersistentState'

interface BrandGroup {
  name: string
  filaments: Filament[]
}

function readPref<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return (allowed as readonly string[]).includes(v ?? '') ? (v as T) : fallback
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

export const FilamentLibrary: React.FC<{ onCollapse?: () => void }> = ({ onCollapse }) => {
  const filaments = useAppStore((s) => s.filaments)
  const filamentTypes = useAppStore((s) => s.filamentTypes)
  const activeTab = useAppStore((s) => s.activeTab)
  const filterQuery = useAppStore((s) => s.filterQuery)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const setFilterQuery = useAppStore((s) => s.setFilterQuery)
  const setNewFilamentModalOpen = useAppStore((s) => s.setNewFilamentModalOpen)
  const setImportModalOpen = useAppStore((s) => s.setImportModalOpen)
  const customLibraryLoaded = useAppStore((s) => s.customLibraryLoaded)
  const setCustomLibraryLoaded = useAppStore((s) => s.setCustomLibraryLoaded)
  const pushToast = useAppStore((s) => s.pushToast)
  const autoSaveLibrary = useAppStore((s) => s.autoSaveLibrary)
  const setAutoSaveLibrary = useAppStore((s) => s.setAutoSaveLibrary)
  const [ownedOnly, setOwnedOnly] = React.useState(() => readPref('autoforge-owned-only', ['1', '0'] as const, '0') === '1')
  const [sort, setSort] = React.useState<FilamentSort>(() => readPref('autoforge-filament-sort', ['name', 'color', 'td'] as const, 'name'))
  // Replaces the separate Active Filaments list, which repeated the library
  // (active rows are already checked and tinted) and pushed it down.
  const [activeOnly, setActiveOnly] = React.useState(() => readPref('autoforge-active-only', ['1', '0'] as const, '0') === '1')
  const [hintDismissed, setHintDismissed] = React.useState(() => readPref('autoforge-drag-hint-dismissed', ['1', '0'] as const, '0') === '1')
  const searchRef = React.useRef<HTMLInputElement>(null)
  const [flashing, flash] = useFlash()

  const chooseActiveOnly = (on: boolean) => {
    setActiveOnly(on)
    writePref('autoforge-active-only', on ? '1' : '0')
  }

  useFilamentLoader()

  // The "Filaments" workflow step: show the whole library and put the cursor in search.
  React.useEffect(
    () =>
      onUiCommand('focus-library', () => {
        chooseActiveOnly(false)
        flash()
        requestAnimationFrame(() => searchRef.current?.focus())
      }),
    [flash],
  )

  React.useEffect(() => {
    fetch('/api/filaments/has-custom-library')
      .then((r) => r.json())
      .then((data) => {
        if (data.exists) setCustomLibraryLoaded(true)
      })
      .catch(() => {})
  }, [setCustomLibraryLoaded])

  const activeUuids = React.useMemo(() => new Set(activeFilaments.map((f) => f.uuid)), [activeFilaments])

  const displayFilaments = React.useMemo(
    () => (customLibraryLoaded ? filaments : filaments.filter((f) => f.source === 'user')),
    [filaments, customLibraryLoaded],
  )

  const filteredFilaments = React.useMemo(() => {
    const q = filterQuery.toLowerCase()
    // Active filaments are shown from every type, not just the open tab.
    const source = activeOnly ? activeFilaments : displayFilaments
    return source.filter(
      (f) => (!q || f.brand.toLowerCase().includes(q) || f.name.toLowerCase().includes(q)) && (!ownedOnly || f.owned),
    )
  }, [displayFilaments, activeFilaments, activeOnly, filterQuery, ownedOnly])

  const handleSaveLibrary = React.useCallback(async () => {
    try {
      const res = await fetch('/api/filaments')
      const all = await res.json()
      const blob = new Blob([JSON.stringify(all, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'filament_library.json'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      pushToast(`Failed to export filament library: ${e instanceof Error ? e.message : String(e)}`)
    }
  }, [pushToast])

  const brandGroups: BrandGroup[] = React.useMemo(() => {
    const groups: Record<string, Filament[]> = {}
    for (const f of filteredFilaments) {
      if (!groups[f.brand]) groups[f.brand] = []
      groups[f.brand].push(f)
    }
    return Object.entries(groups)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, items]) => ({ name, filaments: sortFilaments(items, sort) }))
  }, [filteredFilaments, sort])

  const secondaryButton = 'flex items-center gap-1 px-2 py-1 rounded text-xs border border-gray-600 text-gray-300 hover:bg-gray-800'

  const activeCount = activeFilaments.length

  return (
    <div className={`h-full flex flex-col ${flashing ? 'ring-2 ring-inset ring-cyan-500' : ''}`} style={{ backgroundColor: 'var(--bg-sidebar)' }} data-testid="filament-library">
      <div className="px-3 pt-2.5 pb-2 border-b border-gray-700 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-gray-200">Filament Library</h2>
          <div className="flex items-center gap-1 min-w-0">
            <span className="text-xs text-gray-400 truncate" data-testid="tab-label">
              {activeOnly ? 'Active' : activeTab} • {activeOnly ? activeCount : displayFilaments.length} filaments
            </span>
            {onCollapse && (
              <button onClick={onCollapse} className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700" title="Hide the library" aria-label="Hide the library" data-testid="sidebar-collapse-btn">
                <ChevronsLeft className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* One line instead of a second list: how many the optimizer uses, and a filter to see just those. */}
        <div className="flex items-center justify-between gap-2 text-xs" data-testid="active-filaments-summary" data-count={activeCount}>
          {activeCount === 0 ? (
            <span className="flex items-center gap-1.5 text-yellow-500" title="The optimizer only uses active filaments">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
              No active filaments — click + on the ones you own
            </span>
          ) : (
            <>
              <span className="text-gray-300" title="The optimizer only uses active filaments">
                <span className="font-semibold text-emerald-500 tabular-nums">{activeCount}</span> active, used by the optimizer
              </span>
              <button
                onClick={() => chooseActiveOnly(!activeOnly)}
                aria-pressed={activeOnly}
                className={`px-2 py-0.5 rounded border ${activeOnly ? 'border-emerald-600 bg-emerald-600/20 text-emerald-500' : 'border-gray-600 text-gray-300 hover:bg-gray-800'}`}
                data-testid="active-only-toggle"
              >
                Show only active
              </button>
            </>
          )}
        </div>

        {/* Tabs: only when there's a choice to make (a lone "PLA" tab was a wasted row). */}
        {!activeOnly && filamentTypes.length > 1 && (
        <div className="flex gap-1 overflow-x-auto pb-0.5" role="tablist">
          {filamentTypes.map((type) => (
            <button
              key={type}
              role="tab"
              aria-selected={activeTab === type}
              onClick={() => setActiveTab(type)}
              className={`px-2 py-1 text-xs rounded whitespace-nowrap ${activeTab === type ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
              data-testid={`tab-${type}`}
            >
              {type}
            </button>
          ))}
        </div>
        )}

        {/* Search + filters: next to the list they filter (they used to sit at
            the very bottom of the sidebar, and were cut off on laptop screens). */}
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
          <input
            ref={searchRef}
            placeholder="Search by brand or name…"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="w-full pl-7 pr-7 py-1 rounded text-xs bg-gray-900 border border-gray-600 text-gray-100 placeholder:text-gray-500"
            aria-label="Search filaments"
            data-testid="filter-input"
          />
          {filterQuery && (
            <button onClick={() => setFilterQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 p-0.5 text-gray-400 hover:text-gray-200" aria-label="Clear search" data-testid="filter-clear-btn">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 text-xs">
          <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={ownedOnly}
              onChange={(e) => {
                setOwnedOnly(e.target.checked)
                writePref('autoforge-owned-only', e.target.checked ? '1' : '0')
              }}
              data-testid="owned-only-toggle"
            />
            Owned only
          </label>
          <label className="flex items-center gap-1 text-gray-400">
            Sort
            <select
              value={sort}
              onChange={(e) => {
                setSort(e.target.value as FilamentSort)
                writePref('autoforge-filament-sort', e.target.value)
              }}
              className="bg-gray-900 border border-gray-600 rounded px-1 py-0.5 text-gray-200"
              data-testid="filament-sort"
            >
              <option value="name">Name</option>
              <option value="color">Color</option>
              <option value="td">TD</option>
            </select>
          </label>
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={() => setNewFilamentModalOpen(true)} className="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white" data-testid="new-filament-btn">
            <Plus className="w-3.5 h-3.5" /> New
          </button>
          <button onClick={() => setImportModalOpen(true)} className={secondaryButton} data-testid="import-btn">
            <Upload className="w-3.5 h-3.5" /> Import
          </button>
          <button onClick={handleSaveLibrary} className={secondaryButton} title="Download the whole library as a JSON file" data-testid="save-library-btn">
            <Save className="w-3.5 h-3.5" /> Export
          </button>
        </div>

        {/* On by default: the library is the list of filaments you own, not a
            document you compose, so an edit shouldn't need confirming. */}
        <label className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer" title="Write every filament edit to the library as you make it. With this off, changes are only kept when you press Save in the edit dialog.">
          <input
            type="checkbox"
            checked={autoSaveLibrary}
            onChange={(e) => setAutoSaveLibrary(e.target.checked)}
            data-testid="library-auto-save-toggle"
          />
          Auto save
          <span className="text-gray-500" data-testid="library-auto-save-state">{autoSaveLibrary ? '— changes are saved as you make them' : '— off'}</span>
        </label>
      </div>

      {!hintDismissed && brandGroups.length > 0 && (
        <div className="flex items-start gap-2 mx-2 mt-2 px-2 py-1.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-[11px] text-gray-300" data-testid="drag-hint">
          <Hand className="w-3.5 h-3.5 flex-shrink-0 mt-px text-cyan-500" />
          <span className="flex-1">Drag a filament onto a color band or the color column to use it. Hover a filament to edit it or mark it as owned.</span>
          <button
            onClick={() => {
              setHintDismissed(true)
              writePref('autoforge-drag-hint-dismissed', '1')
            }}
            className="text-gray-400 hover:text-gray-100"
            aria-label="Dismiss hint"
            data-testid="drag-hint-dismiss"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto min-h-0" data-testid="filament-list">
        {brandGroups.map((group) => (
          <BrandFolder key={group.name} group={group} activeUuids={activeUuids} />
        ))}
        {brandGroups.length === 0 && (
          <div className="p-4 text-center text-xs text-gray-400">
            {activeOnly
              ? activeCount === 0
                ? 'No active filaments yet.'
                : 'No active filaments match.'
              : !customLibraryLoaded
                ? 'Import a library or create filaments to get started'
                : ownedOnly
                  ? 'No owned filaments here. Hover a filament and click the tag to mark it as owned.'
                  : 'No filaments found'}
          </div>
        )}
      </div>

      <EditFilamentModal />
    </div>
  )
}

const BrandFolder: React.FC<{ group: BrandGroup; activeUuids: Set<string> }> = ({ group, activeUuids }) => {
  const [expanded, setExpanded] = React.useState(true)

  return (
    <div className="border-b border-gray-800">
      <button
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="w-full flex items-center gap-1 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
        data-testid={`brand-${group.name}`}
      >
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span className="font-medium">{group.name}</span>
        <span className="text-gray-500 ml-auto">({group.filaments.length})</span>
      </button>
      {expanded && (
        <div className="pl-3">
          {group.filaments.map((f) => (
            <FilamentItem key={f.uuid} filament={f} isActive={activeUuids.has(f.uuid)} />
          ))}
        </div>
      )}
    </div>
  )
}

const FilamentItem: React.FC<{ filament: Filament; isActive: boolean }> = ({ filament, isActive }) => {
  const addActiveFilament = useAppStore((s) => s.addActiveFilament)
  const removeActiveFilament = useAppStore((s) => s.removeActiveFilament)
  const setEditingFilament = useAppStore((s) => s.setEditingFilament)
  const setEditFilamentModalOpen = useAppStore((s) => s.setEditFilamentModalOpen)

  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/json', JSON.stringify(filament))
    e.dataTransfer.effectAllowed = 'copy'
  }

  const pushToast = useAppStore((s) => s.pushToast)

  const handleToggleActive = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (isActive) removeActiveFilament(filament.uuid)
    else addActiveFilament(filament)
  }

  const openEditor = () => {
    setEditingFilament(filament)
    setEditFilamentModalOpen(true)
  }

  const toggleOwned = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const response = await fetch(`/api/filaments/${filament.uuid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...filament, owned: !filament.owned }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const saved: Filament = await response.json()
      // Library data, not an undo step.
      useAppStore.setState((state) => ({
        filaments: state.filaments.map((f) => (f.uuid === saved.uuid ? saved : f)),
        activeFilaments: state.activeFilaments.map((f) => (f.uuid === saved.uuid ? saved : f)),
      }))
      if (isActive) {
        await fetch('/api/filaments/active', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(saved) }).catch(() => {})
      }
    } catch (err) {
      pushToast(`Failed to update "${filament.name}": ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDoubleClick={openEditor}
      className={`group flex items-center gap-1.5 px-2 py-1 text-xs cursor-grab active:cursor-grabbing ${isActive ? 'bg-emerald-500/10' : 'hover:bg-gray-800'}`}
      data-testid={`filament-${filament.uuid}`}
      data-active={isActive}
      title="Double-click to edit · drag onto a color band"
    >
      {/* Was a "⌄" chevron for active filaments, which read as "expand". */}
      <button
        onClick={handleToggleActive}
        className={`flex-shrink-0 w-5 h-5 flex items-center justify-center rounded ${
          isActive ? 'bg-emerald-600 text-white hover:bg-red-600' : 'text-emerald-500 hover:bg-emerald-500/20 border border-gray-600'
        }`}
        aria-label={isActive ? `Remove ${filament.name} from active filaments` : `Add ${filament.name} to active filaments`}
        title={isActive ? 'Active — click to remove' : 'Add to active filaments'}
        data-testid={`toggle-filament-${filament.uuid}`}
      >
        {isActive ? <Check className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
      </button>

      <FilamentSwatch filament={filament} size="sm" showLabel={false} />
      <span className="text-gray-200 truncate flex-1">{filament.name}</span>
      {/* Editing and "owned" used to be reachable only by double-clicking. */}
      <button
        onClick={(e) => {
          e.stopPropagation()
          openEditor()
        }}
        className="p-0.5 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700 opacity-0 group-hover:opacity-100 focus:opacity-100"
        title="Edit filament"
        aria-label={`Edit ${filament.name}`}
        data-testid={`edit-filament-${filament.uuid}`}
      >
        <Pencil className="w-3 h-3" />
      </button>
      <button
        onClick={toggleOwned}
        aria-pressed={filament.owned}
        className={
          filament.owned
            ? 'px-1 rounded text-[11px] bg-gray-700 text-gray-300 hover:bg-gray-600'
            : 'p-0.5 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700 opacity-0 group-hover:opacity-100 focus:opacity-100'
        }
        title={filament.owned ? 'Owned — click to unmark' : 'Mark as owned'}
        aria-label={filament.owned ? `Unmark ${filament.name} as owned` : `Mark ${filament.name} as owned`}
        data-testid={`toggle-owned-${filament.uuid}`}
      >
        {filament.owned ? <span data-testid={`owned-badge-${filament.uuid}`}>owned</span> : <Tag className="w-3 h-3" />}
      </button>
      <span className="text-[11px] text-gray-400 tabular-nums w-10 text-right" title="Transmission distance">TD {filament.td}</span>
    </div>
  )
}
