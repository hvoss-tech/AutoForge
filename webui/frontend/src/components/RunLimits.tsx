import React from 'react'
import { SlidersHorizontal } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import {
  RUN_LIMIT_DEFAULT,
  RUN_LIMIT_MIN,
  colorLimitNote,
  describeRunLimits,
  isLimited,
  normalizeLimit,
  type RunLimitKey,
} from '../lib/runLimits'

const ROWS: { key: RunLimitKey; label: string; unit: string; help: string }[] = [
  {
    key: 'max_colors',
    label: 'Colors',
    unit: 'colors, base included',
    help: 'How many different filaments the print may use, counting the base. Fewer colors means fewer spools to have on hand.',
  },
  {
    key: 'max_swaps',
    label: 'Swaps',
    unit: 'filament changes',
    help: 'How often the filament may be changed during the print. Each swap is a pause at the printer.',
  },
]

/** "No limit" / "At most N" for the colors and swaps of the next run. The
 * optimizer keeps to these while it runs (instead of pruning afterwards). */
export const RunLimitsEditor: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  const activeCount = useAppStore((s) => s.activeFilaments.length)

  const update = (key: RunLimitKey, value: number | null) => {
    setSettings({ ...useAppStore.getState().settings, [key]: normalizeLimit(key, value) })
  }

  return (
    <div className="space-y-3" data-testid="run-limits-editor">
      {ROWS.map((row) => {
        const value = settings[row.key]
        const limited = isLimited(value)
        const note = row.key === 'max_colors' ? colorLimitNote(value, activeCount) : null
        return (
          <div key={row.key} className="flex flex-col gap-1" data-testid={`run-limit-${row.key}`} data-limited={limited}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="w-12 text-xs text-gray-300" id={`run-limit-${row.key}-label`}>{row.label}</span>
              <div className="flex rounded border border-gray-600 overflow-hidden" role="radiogroup" aria-labelledby={`run-limit-${row.key}-label`}>
                <button
                  role="radio"
                  aria-checked={!limited}
                  onClick={() => update(row.key, null)}
                  className={`px-2.5 py-1 text-xs ${!limited ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
                  data-testid={`run-limit-${row.key}-unlimited`}
                >
                  No limit
                </button>
                <button
                  role="radio"
                  aria-checked={limited}
                  onClick={() => !limited && update(row.key, RUN_LIMIT_DEFAULT[row.key])}
                  className={`px-2.5 py-1 text-xs ${limited ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
                  data-testid={`run-limit-${row.key}-limited`}
                >
                  At most
                </button>
              </div>
              {limited && (
                <>
                  <NumberInput
                    value={value}
                    onValueChange={(v) => update(row.key, v)}
                    integer
                    min={RUN_LIMIT_MIN[row.key]}
                    step={1}
                    aria-label={`Maximum ${row.unit}`}
                    className="w-16 text-xs bg-gray-800 border border-gray-600 rounded px-2 py-1 text-gray-100 focus:outline-none focus:border-blue-500"
                    data-testid={`run-limit-${row.key}-value`}
                  />
                  <span className="text-xs text-gray-400">{row.unit}</span>
                </>
              )}
            </div>
            {!compact && <p className="text-[11px] leading-snug text-gray-400">{row.help}</p>}
            {note && <p className="text-[11px] leading-snug text-amber-500" data-testid={`run-limit-${row.key}-note`}>{note}</p>}
          </div>
        )
      })}
    </div>
  )
}

/** Next to Run: what the next run is limited to, and a popover to change it. */
export const RunLimitsButton: React.FC<{ disabled?: boolean }> = ({ disabled = false }) => {
  const settings = useAppStore((s) => s.settings)
  const [open, setOpen] = React.useState(false)
  const rootRef = React.useRef<HTMLDivElement>(null)
  const summary = describeRunLimits(settings)

  React.useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="relative" ref={rootRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Limit how many colors and filament swaps the next run may use"
        className={`flex items-center gap-1.5 px-2 py-1.5 text-xs rounded border disabled:opacity-40 disabled:cursor-not-allowed ${
          summary ? 'border-blue-600 text-blue-400 hover:bg-blue-600/15' : 'border-gray-600 text-gray-300 hover:bg-gray-800'
        }`}
        data-testid="run-limits-btn"
        data-limited={!!summary}
      >
        <SlidersHorizontal className="w-3.5 h-3.5" />
        <span data-testid="run-limits-summary">{summary ? summary.replace(/^at most/, '≤') : 'No limits'}</span>
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Limits for the next run"
          className="absolute right-0 top-full mt-1 w-[26rem] z-50 p-3 rounded border border-gray-700 bg-gray-800 shadow-lg"
          data-testid="run-limits-popover"
        >
          <p className="text-xs font-semibold text-gray-100">Limits for the next run</p>
          <p className="text-[11px] text-gray-400 mt-0.5 mb-3">
            The optimizer keeps to these while it runs, so the result never needs more colors or swaps than you allow.
          </p>
          <RunLimitsEditor compact />
        </div>
      )}
    </div>
  )
}
