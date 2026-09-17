import React from 'react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'

export const StatusBar: React.FC = () => {
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  const undo = useAppStore((s) => s.undo)
  const redo = useAppStore((s) => s.redo)
  const historyIndex = useAppStore((s) => s.historyIndex)
  const historyLength = useAppStore((s) => s.historyLength)

  // These bind directly to the same `settings` object the Settings modal
  // edits, so a change here or there is reflected in both immediately —
  // there is no separate "global params" copy to fall out of sync.
  const updateSetting = (key: keyof typeof settings, value: number) => {
    setSettings({ ...settings, [key]: value })
  }

  const layerHeight = settings.layer_height || 0.04
  const baseLayers = Math.round((settings.background_height || 0) / layerHeight)

  return (
    <div
      className="flex items-center gap-2 px-3 text-xs"
      style={{ height: 40, backgroundColor: 'var(--bg-toolbar)', borderTop: '1px solid var(--border)' }}
      data-testid="global-params"
    >
      <label style={{ color: 'var(--text-secondary)' }}>Layer Height</label>
      <NumberInput value={settings.layer_height} step={0.01} min={0.01} onValueChange={(v) => updateSetting('layer_height', v)} className="w-16 px-1 py-0.5 rounded text-xs text-center" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)' }} data-testid="global-layer-height" />

      <label style={{ color: 'var(--text-secondary)' }}>Background Height</label>
      <NumberInput value={settings.background_height} step={0.01} min={0} onValueChange={(v) => updateSetting('background_height', v)} className="w-16 px-1 py-0.5 rounded text-xs text-center" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)' }} data-testid="global-background-height" />

      <label style={{ color: 'var(--text-secondary)' }}>Base Layers</label>
      <NumberInput
        value={baseLayers}
        step={1}
        min={0}
        integer
        onValueChange={(layers) => updateSetting('background_height', parseFloat((layers * layerHeight).toFixed(4)))}
        className="w-16 px-1 py-0.5 rounded text-xs text-center"
        style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)' }}
        data-testid="global-base-layers"
      />

      <label style={{ color: 'var(--text-secondary)' }}>Dimension (mm)</label>
      <NumberInput value={settings.stl_output_size} step={1} min={10} onValueChange={(v) => updateSetting('stl_output_size', v)} className="w-16 px-1 py-0.5 rounded text-xs text-center" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border)' }} data-testid="global-stl-size" />

      <div className="flex-1" />
      <div className="flex items-center gap-2 text-xs">
        <button
          onClick={() => undo()}
          disabled={historyIndex <= 0}
          className="px-2 py-0.5 rounded disabled:opacity-30"
          style={{ color: 'var(--green-accent)' }}
        >
          ↶ Undo
        </button>
        <button
          onClick={() => redo()}
          disabled={historyIndex >= historyLength - 1}
          className="px-2 py-0.5 rounded disabled:opacity-30"
          style={{ color: 'var(--border-light)' }}
        >
          ↷ Redo
        </button>
      </div>
    </div>
  )
}
