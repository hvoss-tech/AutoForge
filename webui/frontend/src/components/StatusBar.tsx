import React, { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import { STL_SIZE_HINT, largeStlSizeWarning } from '../lib/stlSize'

const inputClass = 'w-16 px-1 py-0.5 rounded text-xs text-center bg-gray-900 border border-gray-600 text-gray-100'

/** Base plus the highest color layer currently set, and the most the print
 * can reach — from layer × layer height, not the stored depth_mm, which goes
 * stale when the layer height changes. */
const HeightReadout: React.FC = () => {
  const currentJob = useAppStore((s) => s.currentJob)
  const colorSliders = useAppStore((s) => s.colorSliders)
  const sliderLayerRange = useAppStore((s) => s.sliderLayerRange)
  const settings = useAppStore((s) => s.settings)

  const layerHeight = settings.layer_height || 0.04
  const enabledDepths = colorSliders.filter((s) => s.enabled && s.layer > 0).map((s) => s.layer * layerHeight)
  // Both measure from the build plate (Z=0), matching the mesh's top surface
  // (background_height + print layers, see helpers/colored_mesh.py's top_z).
  const currentMeshHeight = (settings.background_height || 0) + (enabledDepths.length > 0 ? Math.max(...enabledDepths) : 0)
  // sliderLayerRange.max is the print-layer count of the last real result; before
  // one exists (or it's out of date) the next run's limit is Max layers.
  const jobHasResult = currentJob?.status === 'completed'
  const totalMeshLayers = jobHasResult ? sliderLayerRange.max : (settings.max_layers || sliderLayerRange.max)
  const totalMeshHeight = (settings.background_height || 0) + totalMeshLayers * layerHeight

  return (
    <div className="ml-auto flex items-center gap-3 text-gray-400" data-testid="mesh-height-label">
      <span title="Base plus the highest color layer currently set">
        Top color layer: <span className="text-gray-100 tabular-nums" data-testid="mesh-height-current">{currentMeshHeight.toFixed(2)} mm</span>
      </span>
      <span aria-hidden>·</span>
      <span title={jobHasResult ? 'Height of the optimized result' : 'Highest the print can get with the current Max layers setting'}>
        Max print height: <span className="text-gray-100 tabular-nums" data-testid="mesh-height-max">{totalMeshHeight.toFixed(2)} mm</span>
      </span>
    </div>
  )
}

export const StatusBar: React.FC = () => {
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  // Background height used to appear twice (in mm and as "Base Layers") —
  // one value, shown in whichever unit the user thinks in.
  const [baseUnit, setBaseUnit] = useState<'mm' | 'layers'>(() => {
    try {
      return localStorage.getItem('autoforge-base-unit') === 'layers' ? 'layers' : 'mm'
    } catch {
      return 'mm'
    }
  })

  // Same `settings` object the Settings dialog edits — never out of sync.
  const updateSetting = (key: keyof typeof settings, value: number) => {
    setSettings({ ...settings, [key]: value })
  }

  const layerHeight = settings.layer_height || 0.04
  const baseLayers = Math.round((settings.background_height || 0) / layerHeight)
  const stlSizeWarning = largeStlSizeWarning(settings.stl_output_size)

  const chooseUnit = (unit: 'mm' | 'layers') => {
    setBaseUnit(unit)
    try {
      localStorage.setItem('autoforge-base-unit', unit)
    } catch {
      // not persisted — fine
    }
  }

  return (
    <div
      className="flex items-center gap-4 px-3 text-xs bg-gray-800 border-t border-gray-700 text-gray-300 flex-shrink-0"
      style={{ height: 'var(--statusbar-height)' }}
      role="toolbar"
      aria-label="Print settings"
      data-testid="global-params"
    >
      <span className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Print</span>
      <label className="flex items-center gap-1.5" title="Height of every color layer">
        Layer height
        <NumberInput value={settings.layer_height} step={0.01} min={0.01} onValueChange={(v) => updateSetting('layer_height', v)} className={inputClass} data-testid="global-layer-height" />
        <span className="text-gray-400">mm</span>
      </label>

      <div className="flex items-center gap-1.5" title="Solid base printed in the background color before the color layers">
        <span>Base</span>
        {baseUnit === 'mm' ? (
          <NumberInput value={settings.background_height} step={0.01} min={0} onValueChange={(v) => updateSetting('background_height', v)} className={inputClass} aria-label="Base height in mm" data-testid="global-background-height" />
        ) : (
          <NumberInput
            value={baseLayers}
            step={1}
            min={0}
            integer
            onValueChange={(layers) => updateSetting('background_height', parseFloat((layers * layerHeight).toFixed(4)))}
            className={inputClass}
            aria-label="Base height in layers"
            data-testid="global-base-layers"
          />
        )}
        <div className="flex rounded border border-gray-600 overflow-hidden" role="group" aria-label="Base height unit">
          {(['mm', 'layers'] as const).map((unit) => (
            <button
              key={unit}
              onClick={() => chooseUnit(unit)}
              aria-pressed={baseUnit === unit}
              className={`px-1.5 py-0.5 ${baseUnit === unit ? 'bg-gray-600 text-gray-100' : 'text-gray-400 hover:text-gray-200'}`}
              data-testid={`base-unit-${unit}`}
            >
              {unit}
            </button>
          ))}
        </div>
        <span className="text-gray-400 tabular-nums" data-testid="base-converted">
          {baseUnit === 'mm' ? `= ${baseLayers} layers` : `= ${(settings.background_height || 0).toFixed(2)} mm`}
        </span>
      </div>

      <label className="flex items-center gap-1.5" title={STL_SIZE_HINT}>
        STL size
        <NumberInput
          value={settings.stl_output_size}
          step={1}
          min={10}
          onValueChange={(v) => updateSetting('stl_output_size', v)}
          className={`${inputClass} ${stlSizeWarning ? 'border-amber-500' : ''}`}
          aria-describedby={stlSizeWarning ? 'stl-size-warning' : undefined}
          data-testid="global-stl-size"
        />
        <span className="text-gray-400">mm</span>
        {stlSizeWarning && (
          <span
            id="stl-size-warning"
            role="status"
            className="flex items-center gap-1 text-amber-500 cursor-help"
            title={`${stlSizeWarning}\n\n${STL_SIZE_HINT}`}
            data-testid="stl-size-warning"
          >
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="hidden xl:inline">Large — uses a lot of VRAM</span>
          </span>
        )}
      </label>

      <label className="flex items-center gap-1.5" title="Upper limit of color layers above the base for the next run">
        Max layers
        <NumberInput value={settings.max_layers} step={1} min={1} max={200} integer onValueChange={(v) => updateSetting('max_layers', v)} className={inputClass} data-testid="global-max-layers" />
      </label>

      <HeightReadout />
    </div>
  )
}
