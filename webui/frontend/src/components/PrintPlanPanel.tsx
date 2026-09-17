import React, { useMemo } from 'react'
import { Copy, Download } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { withEffectiveBaseColor } from '../lib/baseColor'
import { buildPrintPlan, printPlanText } from '../lib/printPlan'

const Stat: React.FC<{ label: string; value: string; hint?: string; testId: string }> = ({ label, value, hint, testId }) => (
  <div className="px-3 py-2 rounded bg-gray-800 min-w-24" title={hint} data-testid={testId}>
    <div className="text-[11px] uppercase tracking-wide text-gray-400">{label}</div>
    <div className="text-lg font-semibold text-gray-100 tabular-nums">{value}</div>
  </div>
)

/** What to actually do at the printer: the filament swaps for the color
 * layers as they are right now (including hand edits), plus the numbers
 * that matter (colors, swaps, height). */
export const PrintPlanPanel: React.FC = () => {
  const colorSliders = useAppStore((s) => s.colorSliders)
  const filaments = useAppStore((s) => s.filaments)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const rawSettings = useAppStore((s) => s.settings)
  const resolvedBase = useAppStore((s) => s.resolvedBase)
  const pushToast = useAppStore((s) => s.pushToast)
  // The base color the print really uses: with auto-selection on, that is
  // the filament the pipeline chose, not what settings still holds.
  const settings = useMemo(() => withEffectiveBaseColor(rawSettings, resolvedBase), [rawSettings, resolvedBase])
  const known = useMemo(() => [...activeFilaments, ...filaments], [activeFilaments, filaments])
  const plan = useMemo(() => buildPrintPlan(colorSliders, known, settings), [colorSliders, known, settings])
  const text = useMemo(() => printPlanText(plan, settings), [plan, settings])

  if (plan.bands.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-4 text-center" data-testid="print-plan">
        <p className="text-sm text-gray-400">The print plan appears once there are color layers — run the optimizer or add bands by hand.</p>
      </div>
    )
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      pushToast('Swap instructions copied', 'info')
    } catch {
      pushToast('Copying failed — use Download instead', 'warning')
    }
  }

  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
    const a = document.createElement('a')
    a.href = url
    a.download = 'swap_instructions.txt'
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="h-full flex gap-4 p-3 min-h-0" data-testid="print-plan">
      <div className="flex flex-col gap-2 flex-shrink-0">
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Filaments" value={String(plan.colors)} hint="Different filaments used above the base" testId="plan-colors" />
          <Stat label="Swaps" value={String(plan.swaps)} hint="Filament changes after the first color" testId="plan-swaps" />
          <Stat label="Layers" value={String(plan.topLayer)} hint="Color layers above the base" testId="plan-layers" />
          <Stat label="Height" value={`${plan.totalHeightMm.toFixed(2)} mm`} hint="Total print height including the base" testId="plan-height" />
        </div>
        <div className="flex gap-2">
          <button onClick={copy} className="flex items-center gap-1 px-2.5 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="plan-copy-btn">
            <Copy className="w-3.5 h-3.5" /> Copy
          </button>
          <button onClick={download} className="flex items-center gap-1 px-2.5 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="plan-download-btn">
            <Download className="w-3.5 h-3.5" /> Download .txt
          </button>
        </div>
        <p className="text-[11px] text-gray-400 max-w-56">Follows the color layers as currently edited.</p>
      </div>

      <div className="flex-1 min-w-0 overflow-y-auto rounded border border-gray-800" data-testid="plan-swaps-list">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-900 text-gray-400 text-[11px] uppercase tracking-wide">
            <tr>
              <th className="text-left font-normal px-3 py-1.5">At layer</th>
              <th className="text-left font-normal px-3 py-1.5">Height</th>
              <th className="text-left font-normal px-3 py-1.5">Swap to</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-gray-800 text-gray-300">
              <td className="px-3 py-1.5 tabular-nums">1</td>
              <td className="px-3 py-1.5 tabular-nums">{(settings.background_height || 0).toFixed(2)} mm</td>
              <td className="px-3 py-1.5">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full border border-gray-600" style={{ backgroundColor: settings.background_color }} />
                  Base / background color
                </span>
              </td>
            </tr>
            {plan.swapsList.map((swap) => (
              <tr key={swap.layerNumber} className="border-t border-gray-800 text-gray-100" data-testid="plan-swap-row">
                <td className="px-3 py-1.5 tabular-nums">{swap.layerNumber}</td>
                <td className="px-3 py-1.5 tabular-nums">{swap.heightMm.toFixed(2)} mm</td>
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full border border-gray-600" style={{ backgroundColor: swap.filament?.color ?? '#555555' }} />
                    {swap.filament ? `${swap.filament.brand} ${swap.filament.name}` : 'Unassigned'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
