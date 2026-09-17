import React from 'react'
import { ChevronDown, ChevronRight, Info, RotateCcw, Settings, X } from 'lucide-react'
import { defaultSettings, useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import type { OptimizationSettings as SettingsType } from '../types'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { QUALITY_PRESETS, presetForIterations } from '../lib/settingsPresets'

type FieldType = 'number' | 'boolean' | 'select' | 'color'

interface Field {
  key: keyof SettingsType
  label: string
  help: string
  type: FieldType
  options?: { value: string; label: string }[]
  step?: number
  min?: number
  max?: number
  integer?: boolean
  unit?: string
}

// Settings a user is likely to change. CLI-only knobs (file paths, FlatForge,
// TensorBoard, random seed, …) aren't exposed in the web UI at all.
const BASIC_FIELDS: Field[] = [
  { key: 'max_layers', label: 'Max layers', help: 'Upper limit of color layers above the base. More layers allow finer shading but a taller print.', type: 'number', step: 1, min: 1, max: 200, integer: true },
  { key: 'layer_height', label: 'Layer height', unit: 'mm', help: 'Should match the layer height you print with.', type: 'number', step: 0.01, min: 0.01 },
  { key: 'background_height', label: 'Base height', unit: 'mm', help: 'Solid base printed in the background color before any color layer.', type: 'number', step: 0.01, min: 0 },
  { key: 'stl_output_size', label: 'Size', unit: 'mm', help: 'Length of the longest side of the print.', type: 'number', step: 1, min: 10, integer: true },
  { key: 'auto_background_color', label: 'Pick base color automatically', help: 'Let the optimizer choose the base color from your filaments. Turn off to set it yourself.', type: 'boolean' },
  { key: 'background_color', label: 'Base color', help: 'Used when the base color is not picked automatically.', type: 'color' },
  {
    key: 'init_heightmap_method',
    label: 'Height estimate',
    help: 'How the starting heights are guessed. Color clustering works for most pictures; depth estimation can suit photos with a clear foreground.',
    type: 'select',
    options: [
      { value: 'kmeans', label: 'Color clustering' },
      { value: 'depth', label: 'Depth estimation (photos)' },
    ],
  },
]

const ADVANCED_FIELDS: Field[] = [
  { key: 'learning_rate', label: 'Learning rate', help: 'Step size of the optimizer. Higher learns faster but can become unstable.', type: 'number', step: 0.001, min: 0.0001 },
  { key: 'early_stopping', label: 'Early stopping', help: 'Stop when the result has not improved for this many iterations.', type: 'number', step: 100, min: 100, integer: true },
  { key: 'discrete_check', label: 'Result check interval', help: 'How often (in iterations) the current best printable result is evaluated.', type: 'number', step: 10, min: 10, integer: true },
  { key: 'warmup_fraction', label: 'Temperature warmup', help: 'Fraction of the run over which the temperature goes from start to end.', type: 'number', step: 0.05, min: 0, max: 1 },
  { key: 'learning_rate_warmup_fraction', label: 'Learning rate warmup', help: 'Fraction of the run over which the learning rate ramps up.', type: 'number', step: 0.01, min: 0, max: 1 },
  { key: 'init_tau', label: 'Start temperature', help: 'How freely colors may mix early on (Gumbel-softmax tau). Higher explores more.', type: 'number', step: 0.1, min: 0.01 },
  { key: 'final_tau', label: 'End temperature', help: 'How strictly each layer commits to one filament by the end (Gumbel-softmax tau).', type: 'number', step: 0.001, min: 0.001 },
  { key: 'min_layers', label: 'Min layers', help: 'Lower limit of color layers.', type: 'number', step: 1, min: 0, integer: true },
  { key: 'processing_reduction_factor', label: 'Processing reduction', help: 'Works on a downscaled image during optimization. Higher is faster and uses less memory, with less detail.', type: 'number', step: 1, min: 1, integer: true },
  { key: 'nozzle_diameter', label: 'Nozzle diameter', unit: 'mm', help: 'Details finer than the nozzle cannot be printed and are smoothed away.', type: 'number', step: 0.05, min: 0.1 },
  { key: 'num_init_rounds', label: 'Init rounds', help: 'Number of attempts when estimating the starting heights; the best is kept.', type: 'number', step: 1, min: 1, integer: true },
  { key: 'num_init_cluster_layers', label: 'Cluster layers', help: 'Number of height levels used by color clustering (-1 = automatic).', type: 'number', step: 1, min: -1, integer: true },
]

const ADVANCED_KEY = 'autoforge-settings-advanced-open'

export const SettingsModal: React.FC = () => {
  const settingsModalOpen = useAppStore((s) => s.settingsModalOpen)
  const setSettingsModalOpen = useAppStore((s) => s.setSettingsModalOpen)
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  const currentJob = useAppStore((s) => s.currentJob)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const [advancedOpen, setAdvancedOpen] = React.useState(() => {
    try {
      return localStorage.getItem(ADVANCED_KEY) === '1'
    } catch {
      return false
    }
  })

  const toggleAdvanced = () => {
    setAdvancedOpen((open) => {
      try {
        localStorage.setItem(ADVANCED_KEY, open ? '0' : '1')
      } catch {
        // not remembered — fine
      }
      return !open
    })
  }

  const updateSetting = (key: keyof SettingsType, value: string | number | boolean) => {
    setSettings({ ...useAppStore.getState().settings, [key]: value })
  }

  const handleReset = async () => {
    const ok = await requestConfirm({
      title: 'Reset settings?',
      message: 'All settings go back to their defaults. Your image, filaments and color layers are not affected.',
      confirmLabel: 'Reset',
      danger: true,
    })
    if (ok) setSettings({ ...defaultSettings, input_image: useAppStore.getState().settings.input_image })
  }

  const jobActive = !!currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)
  const activePreset = presetForIterations(settings.iterations)

  const renderField = (field: Field, showHelp: boolean) => (
    <div key={field.key} className="flex flex-col gap-1">
      <label className="flex items-center gap-1 text-xs text-gray-300" htmlFor={`setting-${field.key}`}>
        {field.label}
        {field.unit && <span className="text-gray-500">({field.unit})</span>}
        {!showHelp && (
          <span title={field.help} className="text-gray-500 cursor-help" aria-label={field.help}>
            <Info className="w-3 h-3" />
          </span>
        )}
      </label>
      {field.type === 'boolean' ? (
        <button
          id={`setting-${field.key}`}
          role="switch"
          aria-checked={!!settings[field.key]}
          onClick={() => updateSetting(field.key, !settings[field.key])}
          className={`w-10 h-5 rounded-full transition-colors ${settings[field.key] ? 'bg-blue-600' : 'bg-gray-600'}`}
          data-testid={`setting-${field.key}`}
        >
          <div className={`w-4 h-4 rounded-full bg-white transition-transform ${settings[field.key] ? 'translate-x-5' : 'translate-x-0.5'}`} />
        </button>
      ) : field.type === 'select' ? (
        <select
          id={`setting-${field.key}`}
          value={settings[field.key] as string}
          onChange={(e) => updateSetting(field.key, e.target.value)}
          className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-100 focus:outline-none focus:border-blue-500"
          data-testid={`setting-${field.key}`}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      ) : field.type === 'color' ? (
        <input
          id={`setting-${field.key}`}
          type="color"
          value={settings[field.key] as string}
          onChange={(e) => updateSetting(field.key, e.target.value)}
          disabled={field.key === 'background_color' && settings.auto_background_color}
          className="w-12 h-7 rounded cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          data-testid={`setting-${field.key}`}
        />
      ) : (
        <NumberInput
          id={`setting-${field.key}`}
          value={settings[field.key] as number}
          onValueChange={(v) => updateSetting(field.key, v)}
          integer={field.integer}
          className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-100 focus:outline-none focus:border-blue-500"
          step={field.step}
          min={field.min}
          max={field.max}
          data-testid={`setting-${field.key}`}
        />
      )}
      {showHelp && <p className="text-[11px] leading-snug text-gray-400">{field.help}</p>}
    </div>
  )

  return (
    <Dialog open={settingsModalOpen} onOpenChange={setSettingsModalOpen}>
      <DialogContent className="max-w-3xl max-h-[90vh] flex flex-col p-0 gap-0 bg-gray-900 border-gray-700" data-testid="settings-modal">
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700">
          <div>
            <DialogTitle className="text-sm font-semibold text-gray-100 flex items-center gap-2 leading-none tracking-normal">
              <Settings className="w-4 h-4" />
              Settings
            </DialogTitle>
            <DialogDescription className="text-xs text-gray-400 mt-1">
              Changes are saved automatically and used for the next run{jobActive ? ' (not the one currently running)' : ''}.
            </DialogDescription>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={handleReset} className="flex items-center gap-1 px-2 py-1 text-xs rounded text-gray-300 hover:bg-gray-800" data-testid="settings-reset-btn">
              <RotateCcw className="w-3.5 h-3.5" /> Reset to defaults
            </button>
            <button onClick={() => setSettingsModalOpen(false)} className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-800" aria-label="Close settings" data-testid="close-settings">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          <section>
            <h3 className="text-xs font-semibold text-gray-200 mb-2">Quality</h3>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex rounded border border-gray-600 overflow-hidden" role="group" aria-label="Quality preset">
                {QUALITY_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => updateSetting('iterations', p.iterations)}
                    aria-pressed={activePreset === p.id}
                    className={`px-3 py-1.5 text-xs ${activePreset === p.id ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
                    title={p.description}
                    data-testid={`settings-preset-${p.id}`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-300" htmlFor="setting-iterations">
                Iterations
                <NumberInput
                  id="setting-iterations"
                  value={settings.iterations}
                  onValueChange={(v) => updateSetting('iterations', v)}
                  integer
                  min={100}
                  step={100}
                  className="w-24 text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-100"
                  data-testid="setting-iterations"
                />
              </label>
              {activePreset === 'custom' && <span className="text-xs text-gray-400">Custom</span>}
            </div>
            <p className="text-[11px] text-gray-400 mt-1.5">
              {QUALITY_PRESETS.find((p) => p.id === activePreset)?.description ?? 'More iterations take longer but usually match the picture more closely.'}
            </p>
          </section>

          <section>
            <h3 className="text-xs font-semibold text-gray-200 mb-2">Print</h3>
            <div className="grid grid-cols-2 gap-x-6 gap-y-4">{BASIC_FIELDS.map((f) => renderField(f, true))}</div>
          </section>

          <section className="border border-gray-700 rounded">
            <button
              onClick={toggleAdvanced}
              aria-expanded={advancedOpen}
              className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-200 hover:bg-gray-800 rounded"
              data-testid="settings-advanced-toggle"
            >
              <span>Advanced optimizer settings</span>
              <span className="flex items-center gap-1 font-normal text-gray-400">
                Only needed for fine-tuning
                {advancedOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
              </span>
            </button>
            {advancedOpen && (
              <div className="grid grid-cols-3 gap-x-4 gap-y-3 px-3 pb-3 pt-1" data-testid="settings-advanced">
                {ADVANCED_FIELDS.map((f) => renderField(f, false))}
              </div>
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  )
}
