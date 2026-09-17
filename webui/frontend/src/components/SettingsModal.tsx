import React from 'react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import { Settings, X, Play, Square, Download, FileText, Box, Image as ImageIcon, ChevronDown, ChevronRight } from 'lucide-react'
import type { OptimizationSettings as SettingsType } from '../types'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from './ui/dialog'
import { Button } from './ui/button'

interface SettingsGroup {
  title: string
  fields: { key: keyof SettingsType; label: string; type: 'number' | 'text' | 'boolean' | 'select' | 'color'; options?: string[]; step?: number; min?: number; max?: number; integer?: boolean }[]
}

// Deliberately scoped down to the settings a webUI user is actually likely
// to want to touch. Removed entirely: I/O Settings (input image is managed
// by the Input Image panel; csv/json/output-folder/priority-mask are
// CLI-only concepts with no file picker here), Pruning (the dedicated
// Pruning button/modal already covers max colors/swaps/layer — this group
// only duplicated that plus auto-prune-during-optimize toggles nobody here
// uses), FlatForge (a CLI export mode, not exposed elsewhere in the webUI),
// and Other (random seed / MPS backend / TensorBoard / visualize are
// CLI/debugging knobs). "Best Of" and "STL Output Size" were also dropped —
// the latter is edited as "Dimension (mm)" in the bottom bar instead.
const SETTINGS_GROUPS: SettingsGroup[] = [
  {
    title: 'Optimization',
    fields: [
      { key: 'iterations', label: 'Iterations', type: 'number', step: 100, min: 100, integer: true },
      { key: 'learning_rate', label: 'Learning Rate', type: 'number', step: 0.001, min: 0.0001 },
      { key: 'warmup_fraction', label: 'Warmup Fraction', type: 'number', step: 0.05, min: 0, max: 1 },
      { key: 'learning_rate_warmup_fraction', label: 'LR Warmup Fraction', type: 'number', step: 0.01, min: 0, max: 1 },
      { key: 'early_stopping', label: 'Early Stopping', type: 'number', step: 100, min: 100, integer: true },
      { key: 'discrete_check', label: 'Discrete Check', type: 'number', step: 10, min: 10, integer: true },
    ],
  },
  {
    title: 'Gumbel-Softmax',
    fields: [
      { key: 'init_tau', label: 'Init Tau', type: 'number', step: 0.1, min: 0.01 },
      { key: 'final_tau', label: 'Final Tau', type: 'number', step: 0.001, min: 0.001 },
    ],
  },
  {
    title: 'Layers',
    fields: [
      { key: 'layer_height', label: 'Layer Height (mm)', type: 'number', step: 0.01, min: 0.01 },
      { key: 'max_layers', label: 'Max Layers', type: 'number', step: 1, min: 1, max: 200, integer: true },
      { key: 'min_layers', label: 'Min Layers', type: 'number', step: 1, min: 0, integer: true },
      { key: 'background_height', label: 'Background Height (mm)', type: 'number', step: 0.01, min: 0 },
      { key: 'background_color', label: 'Background Color', type: 'color' },
      { key: 'auto_background_color', label: 'Auto Background Color', type: 'boolean' },
    ],
  },
  {
    title: 'Output',
    fields: [
      { key: 'processing_reduction_factor', label: 'Processing Reduction', type: 'number', step: 1, min: 1, integer: true },
      { key: 'nozzle_diameter', label: 'Nozzle Diameter (mm)', type: 'number', step: 0.05, min: 0.1 },
    ],
  },
  {
    title: 'Initialization',
    fields: [
      { key: 'num_init_rounds', label: 'Init Rounds', type: 'number', step: 1, min: 1, integer: true },
      { key: 'num_init_cluster_layers', label: 'Cluster Layers', type: 'number', step: 1, min: -1, integer: true },
      { key: 'init_heightmap_method', label: 'Heightmap Method', type: 'select', options: ['kmeans', 'depth'] },
    ],
  },
]

export const SettingsModal: React.FC = () => {
  const settingsModalOpen = useAppStore((s) => s.settingsModalOpen)
  const setSettingsModalOpen = useAppStore((s) => s.setSettingsModalOpen)
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  const currentJob = useAppStore((s) => s.currentJob)
  const startOptimization = useAppStore((s) => s.startOptimization)
  const cancelOptimization = useAppStore((s) => s.cancelOptimization)
  const resumeOptimization = useAppStore((s) => s.resumeOptimization)
  const pushToast = useAppStore((s) => s.pushToast)

  const [expandedGroups, setExpandedGroups] = React.useState<Record<string, boolean>>(
    Object.fromEntries(SETTINGS_GROUPS.map((g) => [g.title, true]))
  )

  const toggleGroup = (title: string) => {
    setExpandedGroups((prev) => ({ ...prev, [title]: !prev[title] }))
  }

  const updateSetting = (key: keyof SettingsType, value: string | number | boolean) => {
    setSettings({ ...settings, [key]: value })
  }

  const handleStart = async () => {
    try {
      await startOptimization()
    } catch (e) {
      // Unlike TopBar's own Run button (which shows an inline error), this
      // one had no visible failure path at all — a rejection (e.g. "add an
      // active filament first") just did nothing, with the reason only in
      // the console.
      console.error('Failed to start optimization:', e)
      pushToast(`Failed to start optimization: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleCancel = async () => {
    if (!currentJob) return
    try {
      await cancelOptimization(currentJob.job_id)
    } catch (e) {
      pushToast(`Failed to cancel: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleDownload = async (type: 'stl' | 'preview' | 'instructions' | 'project') => {
    if (!currentJob) return
    const response = await fetch(`/api/outputs/${type}/${currentJob.job_id}`)
    if (!response.ok) {
      const err = await response.json().catch(() => ({}))
      pushToast(`Download failed: ${err.detail ?? `HTTP ${response.status}`}`)
      return
    }
    {
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const extensions: Record<string, string> = { stl: 'stl', preview: 'png', instructions: 'txt', project: 'hfp' }
      a.download = `final_model.${extensions[type]}`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <Dialog open={settingsModalOpen} onOpenChange={setSettingsModalOpen}>
      <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col" data-testid="settings-modal">
        <div className="bg-gray-900 rounded-lg w-full flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="text-sm font-semibold text-gray-200 flex items-center gap-2 leading-none tracking-normal">
            <Settings className="w-4 h-4" />
            Settings
          </DialogTitle>
          <DialogDescription className="sr-only">
            Configure optimization, layer, output, and pruning parameters, then start or cancel the current job.
          </DialogDescription>
          <button
            onClick={() => setSettingsModalOpen(false)}
            className="text-gray-400 hover:text-gray-200"
            data-testid="close-settings"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Settings content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {SETTINGS_GROUPS.map((group) => (
            <div key={group.title} className="border border-gray-700 rounded">
              <button
                onClick={() => toggleGroup(group.title)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-300 hover:bg-gray-800 rounded-t"
                data-testid={`settings-group-${group.title}`}
              >
                {group.title}
                {expandedGroups[group.title] ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>
              {expandedGroups[group.title] && (
                <div className="px-3 pb-3 grid grid-cols-2 gap-2">
                  {group.fields.map((field) => (
                    <div key={field.key} className="flex flex-col gap-0.5">
                      <label className="text-xs text-gray-400">{field.label}</label>
                      {field.type === 'boolean' ? (
                        <button
                          onClick={() => updateSetting(field.key, !settings[field.key])}
                          className={`w-10 h-5 rounded-full transition-colors ${
                            settings[field.key] ? 'bg-blue-600' : 'bg-gray-600'
                          }`}
                          data-testid={`setting-${field.key}`}
                        >
                          <div
                            className={`w-4 h-4 rounded-full bg-white transition-transform ${
                              settings[field.key] ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      ) : field.type === 'select' ? (
                        <select
                          value={settings[field.key] as string}
                          onChange={(e) => updateSetting(field.key, e.target.value)}
                          className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-200 focus:outline-none focus:border-blue-500"
                          data-testid={`setting-${field.key}`}
                        >
                          {field.options?.map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : field.type === 'color' ? (
                        <input
                          type="color"
                          value={settings[field.key] as string}
                          onChange={(e) => updateSetting(field.key, e.target.value)}
                          className="w-10 h-6 rounded cursor-pointer"
                          data-testid={`setting-${field.key}`}
                        />
                      ) : (
                        <NumberInput
                          value={settings[field.key] as number}
                          onValueChange={(v) => updateSetting(field.key, v)}
                          integer={field.integer}
                          className="text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-200 focus:outline-none focus:border-blue-500"
                          step={field.step}
                          min={field.min}
                          max={field.max}
                          data-testid={`setting-${field.key}`}
                        />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer with actions */}
        <div className="px-4 py-3 border-t border-gray-700 flex items-center gap-2">
          {currentJob?.status === 'paused' && (
            <button
              onClick={() => resumeOptimization(currentJob.job_id).catch((e) => pushToast(`Failed to resume: ${e instanceof Error ? e.message : String(e)}`))}
              className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs text-white"
              data-testid="resume-btn"
            >
              <Play className="w-3 h-3" />
              Resume
            </button>
          )}
          {currentJob && ['pending', 'running', 'paused'].includes(currentJob.status) ? (
            <button
              onClick={handleCancel}
              className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-xs text-white"
              data-testid="cancel-btn"
            >
              <Square className="w-3 h-3" />
              Cancel
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs text-white"
              data-testid="start-btn"
            >
              <Play className="w-3 h-3" />
              Start Optimization
            </button>
          )}

          {/* Download buttons */}
          {currentJob?.status === 'completed' && (
            <>
              <button
                onClick={() => handleDownload('stl')}
                className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
                data-testid="download-stl"
              >
                <Box className="w-3 h-3" />
                STL
              </button>
              <button
                onClick={() => handleDownload('preview')}
                className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
                data-testid="download-preview"
              >
                <ImageIcon className="w-3 h-3" />
                Preview
              </button>
              <button
                onClick={() => handleDownload('instructions')}
                className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
                data-testid="download-instructions"
              >
                <FileText className="w-3 h-3" />
                Instructions
              </button>
              <button
                onClick={() => handleDownload('project')}
                className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
                data-testid="download-project"
              >
                <Download className="w-3 h-3" />
                Project
              </button>
            </>
          )}

          {/* Job status */}
          {currentJob && (
            <div className="ml-auto flex items-center gap-2">
              {currentJob.status === 'running' && (
                <>
                  <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  <span className="text-xs text-gray-400">
                    Iter {currentJob.iteration}/{currentJob.total_iterations}
                    {currentJob.loss !== null && ` | Loss: ${currentJob.loss.toFixed(4)}`}
                  </span>
                </>
              )}
              {currentJob.status === 'completed' && (
                <span className="text-xs text-green-400">Completed</span>
              )}
              {currentJob.status === 'failed' && (
                <span className="text-xs text-red-400">Failed: {currentJob.error}</span>
              )}
              {currentJob.status === 'cancelled' && (
                <span className="text-xs text-yellow-400">Cancelled</span>
              )}
            </div>
          )}
        </div>
      </div>
      </DialogContent>
    </Dialog>
  )
}
