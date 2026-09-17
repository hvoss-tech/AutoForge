import React from 'react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import { refreshLibrary } from '../services/filamentService'
import { describeApiError } from '../lib/apiError'
import { X, Plus } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from './ui/dialog'

const DEFAULT_TYPES = ['PLA', 'PETG', 'ABS', 'TPU', 'ASA', 'Nylon', 'PC', 'PVB']
const NEW_TYPE_SENTINEL = '__new_type__'

export const NewFilamentModal: React.FC = () => {
  const newFilamentModalOpen = useAppStore((s) => s.newFilamentModalOpen)
  const setNewFilamentModalOpen = useAppStore((s) => s.setNewFilamentModalOpen)
  const filamentTypes = useAppStore((s) => s.filamentTypes)
  const setCustomLibraryLoaded = useAppStore((s) => s.setCustomLibraryLoaded)
  const pushToast = useAppStore((s) => s.pushToast)

  const [brand, setBrand] = React.useState('')
  const [name, setName] = React.useState('')
  const [filamentType, setFilamentType] = React.useState(filamentTypes[0] || 'PLA')
  const [customType, setCustomType] = React.useState('')
  const [td, setTd] = React.useState(5.0)
  const [owned, setOwned] = React.useState(false)
  const [colorR, setColorR] = React.useState(128)
  const [colorG, setColorG] = React.useState(128)
  const [colorB, setColorB] = React.useState(128)

  const isAddingCustomType = filamentType === NEW_TYPE_SENTINEL
  const effectiveType = isAddingCustomType ? customType.trim() : filamentType

  const hexColor = `#${colorR.toString(16).padStart(2, '0')}${colorG.toString(16).padStart(2, '0')}${colorB.toString(16).padStart(2, '0')}`

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!brand || !name) return
    if (isAddingCustomType && !effectiveType) return

    const newFilament = {
      brand,
      name,
      color: hexColor,
      td,
      owned,
      uuid: '',
      filament_type: effectiveType,
      source: 'user',
    }

    try {
      const response = await fetch('/api/filaments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newFilament),
      })
      if (!response.ok) {
        throw new Error(describeApiError(await response.json().catch(() => null), response.status))
      }
      await response.json()

      setCustomLibraryLoaded(true)
      await refreshLibrary(effectiveType)

      setNewFilamentModalOpen(false)
      setBrand('')
      setName('')
      setFilamentType(effectiveType)
      setCustomType('')
      setTd(5.0)
      setOwned(false)
      setColorR(128)
      setColorG(128)
      setColorB(128)
    } catch (err) {
      console.error('Failed to create filament:', err)
      pushToast(`Failed to create filament: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const allTypes = [...new Set([...filamentTypes, ...DEFAULT_TYPES])]

  return (
    <Dialog open={newFilamentModalOpen} onOpenChange={setNewFilamentModalOpen}>
      <DialogContent className="p-0 gap-0 bg-gray-900 border-gray-700" data-testid="new-filament-modal">
        <div className="bg-gray-900 rounded-lg w-full flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="text-sm font-semibold text-gray-200 flex items-center gap-2 leading-none tracking-normal">
            <Plus className="w-4 h-4" />
            New Filament
          </DialogTitle>
          <DialogDescription className="sr-only">
            Create a new filament with a brand, name, type, transmission distance, and color.
          </DialogDescription>
          <button
            onClick={() => setNewFilamentModalOpen(false)}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close" data-testid="close-new-filament"
          >
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
                data-testid="new-filament-brand"
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
                data-testid="new-filament-name"
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
                data-testid="new-filament-type"
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
                  className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
                  data-testid="new-filament-custom-type"
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
                data-testid="new-filament-td"
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-400">Color</label>
            <div className="flex items-center gap-3">
              <div
                className="w-16 h-16 rounded border border-gray-600 flex-shrink-0"
                style={{ backgroundColor: hexColor }}
              />
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-red-400 w-6">R</label>
                  <input
                    type="range"
                    min={0}
                    max={255}
                    value={colorR}
                    onChange={(e) => setColorR(parseInt(e.target.value))}
                    className="flex-1 accent-red-500"
                  />
                  <input
                    type="number"
                    min={0}
                    max={255}
                    value={colorR}
                    onChange={(e) => setColorR(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))}
                    className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-green-400 w-6">G</label>
                  <input
                    type="range"
                    min={0}
                    max={255}
                    value={colorG}
                    onChange={(e) => setColorG(parseInt(e.target.value))}
                    className="flex-1 accent-green-500"
                  />
                  <input
                    type="number"
                    min={0}
                    max={255}
                    value={colorG}
                    onChange={(e) => setColorG(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))}
                    className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-blue-400 w-6">B</label>
                  <input
                    type="range"
                    min={0}
                    max={255}
                    value={colorB}
                    onChange={(e) => setColorB(parseInt(e.target.value))}
                    className="flex-1 accent-blue-500"
                  />
                  <input
                    type="number"
                    min={0}
                    max={255}
                    value={colorB}
                    onChange={(e) => setColorB(Math.min(255, Math.max(0, parseInt(e.target.value) || 0)))}
                    className="w-14 text-xs bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 text-center focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400">Hex</label>
              <input
                type="text"
                value={hexColor}
                readOnly
                className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-200 font-mono w-24"
              />
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

          <div className="flex justify-end gap-2 pt-2 border-t border-gray-700">
            <button
              type="button"
              onClick={() => setNewFilamentModalOpen(false)}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs text-white"
              data-testid="create-filament-submit"
            >
              Create Filament
            </button>
          </div>
        </form>
      </div>
      </DialogContent>
    </Dialog>
  )
}
