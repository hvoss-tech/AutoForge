import React from 'react'
import { useAppStore } from '../store/appStore'
import { useFilamentLoader } from '../services/filamentService'
import { ChevronDown, ChevronRight, Plus, Save, Upload } from 'lucide-react'
import type { Filament } from '../types'
import { ActiveFilamentsPanel } from './ActiveFilamentsPanel'
import { FilamentSwatch } from './FilamentSwatch'
import { EditFilamentModal } from './EditFilamentModal'

interface BrandGroup {
  name: string
  filaments: Filament[]
}

export const FilamentLibrary: React.FC = () => {
  const filaments = useAppStore((s) => s.filaments)
  const filamentTypes = useAppStore((s) => s.filamentTypes)
  const activeTab = useAppStore((s) => s.activeTab)
  const filterQuery = useAppStore((s) => s.filterQuery)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const setFilterQuery = useAppStore((s) => s.setFilterQuery)
  const setSettingsModalOpen = useAppStore((s) => s.setSettingsModalOpen)
  const setNewFilamentModalOpen = useAppStore((s) => s.setNewFilamentModalOpen)
  const setImportModalOpen = useAppStore((s) => s.setImportModalOpen)
  const customLibraryLoaded = useAppStore((s) => s.customLibraryLoaded)
  const setCustomLibraryLoaded = useAppStore((s) => s.setCustomLibraryLoaded)
  const pushToast = useAppStore((s) => s.pushToast)

  useFilamentLoader()

  React.useEffect(() => {
    fetch('/api/filaments/has-custom-library')
      .then((r) => r.json())
      .then((data) => {
        if (data.exists) {
          setCustomLibraryLoaded(true)
        }
      })
      .catch(() => {})
  }, [setCustomLibraryLoaded])

  const activeUuids = React.useMemo(() => new Set(activeFilaments.map((f) => f.uuid)), [activeFilaments])

  const userFilaments = React.useMemo(() => filaments.filter((f) => f.source === 'user'), [filaments])

  const displayFilaments = React.useMemo(() => {
    if (!customLibraryLoaded) return userFilaments
    return filaments
  }, [filaments, userFilaments, customLibraryLoaded])

  const filteredFilaments = React.useMemo(() => {
    if (!filterQuery) return displayFilaments
    const q = filterQuery.toLowerCase()
    return displayFilaments.filter(
      (f) => f.brand.toLowerCase().includes(q) || f.name.toLowerCase().includes(q)
    )
  }, [displayFilaments, filterQuery])

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
      console.error('Failed to export filament library:', e)
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
      .map(([name, items]) => ({ name, filaments: items }))
  }, [filteredFilaments])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-sidebar)' }}>
      {/* Active Filaments */}
      <ActiveFilamentsPanel />

      {/* Library Header */}
      <div className="p-3 border-b border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-gray-200">Filament Library</h2>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 overflow-x-auto pb-1">
          {filamentTypes.map((type) => (
            <button
              key={type}
              onClick={() => setActiveTab(type)}
              className={`px-2 py-1 text-xs rounded whitespace-nowrap ${
                activeTab === type
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              data-testid={`tab-${type}`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Tree view */}
      <div className="flex-1 overflow-y-auto" data-testid="filament-list">
        {brandGroups.map((group) => (
          <BrandFolder key={group.name} group={group} activeUuids={activeUuids} />
        ))}
        {brandGroups.length === 0 && (
          <div className="p-4 text-center text-xs text-gray-500">
            {!customLibraryLoaded
              ? 'Import a custom library or create filaments to get started'
              : 'No filaments found'}
          </div>
        )}
      </div>

      <EditFilamentModal />

      {/* Bottom controls */}
      <div style={{ padding: 8, borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
        <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginBottom: 4 }}>Filter</div>
        <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
          <input
            placeholder="Search filaments..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="w-full"
            style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 4, padding: '2px 6px', fontSize: 11, color: 'var(--text-primary)' }}
            data-testid="filter-input"
          />
          <button
            onClick={() => setFilterQuery('')}
            style={{ backgroundColor: 'transparent', color: 'var(--text-secondary)', border: 'none', cursor: 'pointer', fontSize: 11 }}
          >
            X
          </button>
        </div>
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          <button
            onClick={() => setNewFilamentModalOpen(true)}
            style={{ backgroundColor: 'var(--cyan-accent)', color: '#000', borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 600, border: 'none', cursor: 'pointer' }}
            data-testid="new-filament-btn"
          >
            <Plus className="w-3 h-3" />
            New Filament
          </button>
          <button
            onClick={() => setImportModalOpen(true)}
            style={{ backgroundColor: 'transparent', color: 'var(--text-secondary)', borderRadius: 4, padding: '2px 8px', fontSize: 10, border: '1px solid var(--border)', cursor: 'pointer' }}
            data-testid="import-btn"
          >
            <Upload className="w-3 h-3" />
            Import
          </button>
          <button
            onClick={handleSaveLibrary}
            style={{ backgroundColor: 'transparent', color: 'var(--text-secondary)', borderRadius: 4, padding: '2px 8px', fontSize: 10, border: '1px solid var(--border)', cursor: 'pointer' }}
            data-testid="save-library-btn"
          >
            <Save className="w-3 h-3" />
            Save Library
          </button>
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: 10, marginTop: 4 }} data-testid="tab-label">
          {activeTab} • {displayFilaments.length} filaments
        </div>
      </div>
    </div>
  )
}

const BrandFolder: React.FC<{ group: BrandGroup; activeUuids: Set<string> }> = ({ group, activeUuids }) => {
  const [expanded, setExpanded] = React.useState(true)

  return (
    <div className="border-b border-gray-800">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
        data-testid={`brand-${group.name}`}
      >
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span className="font-medium">{group.name}</span>
        <span className="text-gray-500 ml-auto">({group.filaments.length})</span>
      </button>
      {expanded && (
        <div className="pl-4">
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

  const handleToggleActive = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (isActive) {
      removeActiveFilament(filament.uuid)
    } else {
      addActiveFilament(filament)
    }
  }

  const handleDoubleClick = () => {
    setEditingFilament(filament)
    setEditFilamentModalOpen(true)
  }

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDoubleClick={handleDoubleClick}
      className="flex items-center gap-1.5 px-2 py-1 text-xs hover:bg-gray-800 cursor-grab active:cursor-grabbing"
      data-testid={`filament-${filament.uuid}`}
      title="Double-click to edit"
    >
      {/* +/- button on the left */}
      <button
        onClick={handleToggleActive}
        className={`flex-shrink-0 w-4 h-4 flex items-center justify-center rounded transition-colors ${
          isActive
            ? 'text-red-400 hover:text-red-300 hover:bg-red-900/30'
            : 'text-green-400 hover:text-green-300 hover:bg-green-900/30'
        }`}
        data-testid={`toggle-filament-${filament.uuid}`}
      >
        {isActive ? <ChevronDown className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
      </button>

      <FilamentSwatch filament={filament} size="sm" showLabel={false} />
      <span className="text-gray-300 truncate flex-1">{filament.name}</span>
    </div>
  )
}
