import React from 'react'
import { useAppStore } from '../store/appStore'
import type { Filament } from '../types'
import { NumberInput } from './ui/number-input'
import { refreshLibrary } from '../services/filamentService'
import { Check, X, Pencil, Trash2 } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from './ui/dialog'

const DEFAULT_TYPES = ['PLA', 'PETG', 'ABS', 'TPU', 'ASA', 'Nylon', 'PC', 'PVB']
const NEW_TYPE_SENTINEL = '__new_type__'

export const EditFilamentModal: React.FC = () => {
  const editFilamentModalOpen = useAppStore((s) => s.editFilamentModalOpen)
  const setEditFilamentModalOpen = useAppStore((s) => s.setEditFilamentModalOpen)
  const editingFilament = useAppStore((s) => s.editingFilament)
  const setEditingFilament = useAppStore((s) => s.setEditingFilament)
  const filamentTypes = useAppStore((s) => s.filamentTypes)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const setActiveFilaments = useAppStore((s) => s.setActiveFilaments)
  const removeActiveFilament = useAppStore((s) => s.removeActiveFilament)

  const [brand, setBrand] = React.useState('')
  const [name, setName] = React.useState('')
  const [filamentType, setFilamentType] = React.useState('PLA')
  const [customType, setCustomType] = React.useState('')
  const [td, setTd] = React.useState(5.0)
  const [owned, setOwned] = React.useState(false)
  const [colorR, setColorR] = React.useState(128)
  const [colorG, setColorG] = React.useState(128)
  const [colorB, setColorB] = React.useState(128)
  const [confirmingDelete, setConfirmingDelete] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [savedAt, setSavedAt] = React.useState<number | null>(null)
  const [baseline, setBaseline] = React.useState<Filament | null>(null)
  const autoSaveLibrary = useAppStore((s) => s.autoSaveLibrary)

  React.useEffect(() => {
    if (!editingFilament) return
    setBrand(editingFilament.brand)
    setName(editingFilament.name)
    setFilamentType(editingFilament.filament_type || 'PLA')
    setTd(editingFilament.td)
    setOwned(editingFilament.owned)
    const hex = editingFilament.color.replace('#', '')
    setColorR(parseInt(hex.slice(0, 2), 16) || 0)
    setColorG(parseInt(hex.slice(2, 4), 16) || 0)
    setColorB(parseInt(hex.slice(4, 6), 16) || 0)
    setConfirmingDelete(false)
    setCustomType('')
    setError(null)
    setSavedAt(null)
    setBaseline(editingFilament)
  }, [editingFilament])

  const isAddingCustomType = filamentType === NEW_TYPE_SENTINEL
  const effectiveType = isAddingCustomType ? customType.trim() : filamentType

  const hexColor = `#${colorR.toString(16).padStart(2, '0')}${colorG.toString(16).padStart(2, '0')}${colorB.toString(16).padStart(2, '0')}`

  // "Different from what the library holds" — auto save only fires on a real
  // change, so opening the dialog writes nothing back, and a save that just
  // succeeded doesn't immediately queue another one. `baseline` deliberately
  // is not `editingFilament`: rewriting that after each save would reset
  // every field from the response and yank the cursor out of whatever the
  // user was still typing.
  const dirty = !!baseline && (
    brand !== baseline.brand ||
    name !== baseline.name ||
    hexColor.toLowerCase() !== (baseline.color || '').toLowerCase() ||
    td !== baseline.td ||
    owned !== baseline.owned ||
    effectiveType !== (baseline.filament_type || 'PLA')
  )

  const close = () => {
    // A still-pending debounced save (closed within the 400ms window) must
    // be flushed, not discarded — the effect's own cleanup would otherwise
    // just clearTimeout it away, dropping whatever edit was in flight.
    flushPendingSave()
    setEditFilamentModalOpen(false)
    setEditingFilament(null)
  }


  /** Writes the form to the library. Returns false if it couldn't.
   *
   * `reloadLibrary` is off for auto saves: refreshLibrary() refetches types,
   * brands and the list *and* switches the visible tab to the filament's
   * type, which is right when the user presses Done but jarring (and three
   * extra requests) every time a debounced keystroke lands. The saved record
   * is patched into the list instead, which is all the list needs. */
  const persist = React.useCallback(async ({ reloadLibrary = true } = {}): Promise<boolean> => {
    // Any call to persist() (debounce firing, a manual flush, or Done)
    // makes the current auto-save timer moot — clearing it here (rather
    // than only in the effect's own cleanup) is what lets a manual flush
    // and the debounce race safely without double-POSTing.
    if (pendingSaveTimerRef.current) {
      clearTimeout(pendingSaveTimerRef.current)
      pendingSaveTimerRef.current = null
    }
    if (!editingFilament || !brand || !name) return false
    if (isAddingCustomType && !effectiveType) return false
    setError(null)

    const updated = {
      brand,
      name,
      color: hexColor,
      td,
      owned,
      uuid: editingFilament.uuid,
      filament_type: effectiveType,
      source: editingFilament.source || 'user',
    }

    try {
      const response = await fetch(`/api/filaments/${editingFilament.uuid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated),
      })
      if (!response.ok) {
        setError('Failed to save changes')
        return false
      }
      const saved = await response.json()

      // If this filament is active, refresh that list too — the sliders
      // and color core read filament color/TD by uuid lookup, so simply
      // syncing this list is enough for them to pick up the new values.
      const active = useAppStore.getState().activeFilaments
      if (active.some((f) => f.uuid === editingFilament.uuid)) {
        setActiveFilaments(active.map((f) => (f.uuid === editingFilament.uuid ? saved : f)))
        await fetch('/api/filaments/active', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saved),
        }).catch(() => {})
      }
      // Sliders keep their own TD copy (it's what the render uses); carry a
      // changed library TD over to the sliders using this filament, or the
      // edit would have no visible effect.
      if (saved.td !== editingFilament.td) {
        const { colorSliders, setSliders } = useAppStore.getState()
        if (colorSliders.some((s) => s.filament_uuid === saved.uuid)) {
          // Not a hand edit of the sliders themselves — a library TD change
          // following through to the sliders that reference it. Marking it
          // hand-edited spuriously triggered the "Replace your color
          // layers?" confirmation on the next Run, for a change the user
          // never made to the layers directly.
          setSliders(
            colorSliders.map((s) => (s.filament_uuid === saved.uuid ? { ...s, td: saved.td } : s)),
            `Updated TD to ${saved.td}`,
            { handEdited: false },
          )
        }
      }
      if (reloadLibrary) {
        await refreshLibrary(effectiveType)
      } else {
        useAppStore.setState((state) => ({
          filaments: state.filaments.map((f) => (f.uuid === saved.uuid ? saved : f)),
        }))
      }
      setBaseline(saved)
      setSavedAt(Date.now())
      return true
    } catch (err) {
      console.error('Failed to update filament:', err)
      setError('Failed to save changes')
      return false
    }
  }, [editingFilament, brand, name, hexColor, td, owned, effectiveType, isAddingCustomType, setActiveFilaments])

  // Auto save: every edit lands in the library shortly after it's made, so
  // closing the dialog (or the tab) can't quietly discard it. Debounced,
  // because dragging an R/G/B slider would otherwise fire a request per pixel.
  //
  // The timer must re-arm on every field change, not just on the dirty
  // false->true transition: `dirty` is a boolean that stays `true` across a
  // whole drag sequence, so keying the effect on `dirty` alone armed one
  // 400ms timer at the *first* change and never rescheduled it — a drag
  // that ran longer than 400ms saved whatever value was live at that one
  // moment and then never saved again, so the final value only ever reached
  // the library via the "Done" button.
  const persistRef = React.useRef(persist)
  persistRef.current = persist
  const pendingSaveTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  React.useEffect(() => {
    if (!autoSaveLibrary || !editFilamentModalOpen || !editingFilament) return
    if (!dirty) return
    const timer = setTimeout(() => {
      pendingSaveTimerRef.current = null
      persistRef.current({ reloadLibrary: false })
    }, 400)
    pendingSaveTimerRef.current = timer
    return () => {
      clearTimeout(timer)
      if (pendingSaveTimerRef.current === timer) pendingSaveTimerRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSaveLibrary, editFilamentModalOpen, editingFilament, dirty, brand, name, filamentType, customType, td, owned, colorR, colorG, colorB])

  // Flushes (rather than discards) a still-pending debounced save — used
  // both when the dialog is closed without waiting out the debounce and
  // when the tab itself is closing (the `pagehide` listener below).
  const flushPendingSave = React.useCallback(() => {
    if (!pendingSaveTimerRef.current) return
    clearTimeout(pendingSaveTimerRef.current)
    pendingSaveTimerRef.current = null
    persistRef.current({ reloadLibrary: false })
  }, [])

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    window.addEventListener('pagehide', flushPendingSave)
    return () => window.removeEventListener('pagehide', flushPendingSave)
  }, [flushPendingSave])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (await persist()) close()
  }

  const handleDelete = async () => {
    if (!editingFilament) return
    if (!confirmingDelete) {
      setConfirmingDelete(true)
      return
    }
    setError(null)
    try {
      const response = await fetch(`/api/filaments/${editingFilament.uuid}`, { method: 'DELETE' })
      if (!response.ok) {
        setError('Failed to delete filament')
        return
      }
      if (activeFilaments.some((f) => f.uuid === editingFilament.uuid)) {
        await removeActiveFilament(editingFilament.uuid)
      }
      await refreshLibrary()
      close()
    } catch (err) {
      console.error('Failed to delete filament:', err)
      setError('Failed to delete filament')
    }
  }

  if (!editingFilament) return null

  const allTypes = [...new Set([...filamentTypes, ...DEFAULT_TYPES])]

  return (
    <Dialog open={editFilamentModalOpen} onOpenChange={(open) => { if (!open) close() }}>
      <DialogContent className="p-0 gap-0 bg-gray-900 border-gray-700" data-testid="edit-filament-modal">
        <div className="bg-gray-900 rounded-lg w-full flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="text-sm font-semibold text-gray-200 flex items-center gap-2 leading-none tracking-normal">
            <Pencil className="w-4 h-4" />
            Edit Filament
          </DialogTitle>
          <DialogDescription className="sr-only">
            Edit or delete this filament's brand, name, type, transmission distance, and color.
          </DialogDescription>
          <button onClick={close} className="text-gray-400 hover:text-gray-200" aria-label="Close" data-testid="close-edit-filament">
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Brand</label>
              <input
                type="text"
                value={brand}
                onChange={(e) => setBrand(e.target.value)}
                required
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
                placeholder="Brand"
                data-testid="edit-filament-brand"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
                placeholder="Filament name"
                data-testid="edit-filament-name"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Type</label>
              <select
                value={filamentType}
                onChange={(e) => setFilamentType(e.target.value)}
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
                data-testid="edit-filament-type"
              >
                {allTypes.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
                <option value={NEW_TYPE_SENTINEL}>+ Add new category…</option>
              </select>
              {isAddingCustomType && (
                <input
                  type="text"
                  value={customType}
                  onChange={(e) => setCustomType(e.target.value)}
                  required
                  autoFocus
                  placeholder="New category name"
                  className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500 mt-1"
                  data-testid="edit-filament-custom-type"
                />
              )}
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">Transmission Distance (TD)</label>
              <NumberInput
                value={td}
                onValueChange={setTd}
                step={0.1}
                min={0}
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
                data-testid="edit-filament-td"
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-400">Color</label>
            <div className="flex items-center gap-3">
              <div className="w-16 h-16 rounded border border-gray-600 flex-shrink-0" style={{ backgroundColor: hexColor }} />
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-red-400 w-6">R</label>
                  <input type="range" min={0} max={255} value={colorR} onChange={(e) => setColorR(parseInt(e.target.value))} className="flex-1 accent-red-500" />
                  <input type="number" min={0} max={255} value={colorR} onChange={(e) => setColorR(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))} className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-green-400 w-6">G</label>
                  <input type="range" min={0} max={255} value={colorG} onChange={(e) => setColorG(parseInt(e.target.value))} className="flex-1 accent-green-500" />
                  <input type="number" min={0} max={255} value={colorG} onChange={(e) => setColorG(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))} className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-blue-400 w-6">B</label>
                  <input type="range" min={0} max={255} value={colorB} onChange={(e) => setColorB(parseInt(e.target.value))} className="flex-1 accent-blue-500" />
                  <input type="number" min={0} max={255} value={colorB} onChange={(e) => setColorB(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))} className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500" />
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400">Hex</label>
              <input type="text" value={hexColor} readOnly className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-200 font-mono w-24" />
              <input
                type="color"
                value={hexColor}
                onChange={(e) => {
                  const hex = e.target.value
                  setColorR(parseInt(hex.slice(1, 3), 16))
                  setColorG(parseInt(hex.slice(3, 5), 16))
                  setColorB(parseInt(hex.slice(5, 7), 16))
                }}
                className="w-8 h-6 rounded cursor-pointer"
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setOwned(!owned)}
              className={`w-10 h-5 rounded-full transition-colors ${owned ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              <div className={`w-4 h-4 rounded-full bg-white transition-transform ${owned ? 'translate-x-5' : 'translate-x-0.5'}`} />
            </button>
            <label className="text-xs text-gray-300">Owned</label>
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}

          <div className="flex justify-between items-center gap-2 pt-2 border-t border-gray-700">
            <button
              type="button"
              onClick={handleDelete}
              onBlur={() => setConfirmingDelete(false)}
              className={`px-3 py-1.5 rounded text-xs text-white flex items-center gap-1 ${confirmingDelete ? 'bg-red-700 hover:bg-red-600' : 'bg-gray-700 hover:bg-red-900/50 text-red-400'}`}
              data-testid="delete-filament-btn"
            >
              <Trash2 className="w-3 h-3" />
              {confirmingDelete ? 'Confirm delete?' : 'Delete'}
            </button>
            <div className="flex items-center gap-2">
              {autoSaveLibrary && (
                <span className="flex items-center gap-1 text-[11px] text-gray-400" data-testid="edit-filament-autosave-state" data-dirty={dirty || undefined}>
                  {dirty ? (
                    'Saving…'
                  ) : savedAt ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-500" /> Saved
                    </>
                  ) : (
                    'Auto save is on'
                  )}
                </span>
              )}
              <button type="button" onClick={close} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200" data-testid="close-edit-filament-btn">
                {autoSaveLibrary ? 'Close' : 'Cancel'}
              </button>
              <button type="submit" className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs text-white" data-testid="save-filament-btn">
                {autoSaveLibrary ? 'Done' : 'Save Changes'}
              </button>
            </div>
          </div>
        </form>
      </div>
      </DialogContent>
    </Dialog>
  )
}
