import React from 'react'
import { Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { ColorSliders } from './ColorSliders'
import { PrintPlanPanel } from './PrintPlanPanel'
import { useSliderPreviewRender } from '../hooks/useSliderPreviewRender'
import { buildPrintPlan } from '../lib/printPlan'
import { onUiCommand } from '../lib/uiEvents'
import { useFlash } from '../hooks/usePersistentState'

export const BottomPanel: React.FC = () => {
  const tab = useAppStore((s) => s.bottomTab)
  const setTab = useAppStore((s) => s.setBottomTab)
  const bandCount = useAppStore((s) => s.colorSliders.length)
  const colorSliders = useAppStore((s) => s.colorSliders)
  const settings = useAppStore((s) => s.settings)
  const isRendering = useSliderPreviewRender()
  const swaps = React.useMemo(() => buildPrintPlan(colorSliders, [], settings).swaps, [colorSliders, settings])
  const [flashing, flash] = useFlash()

  // The "Adjust" workflow step.
  React.useEffect(
    () =>
      onUiCommand('focus-layers', () => {
        setTab('layers')
        flash()
      }),
    [setTab, flash],
  )

  const tabClass = (active: boolean) =>
    `px-3 py-1.5 text-xs font-medium border-b-2 ${active ? 'border-cyan-500 text-gray-100' : 'border-transparent text-gray-400 hover:text-gray-200'}`

  return (
    <div className={`h-full flex flex-col bg-gray-900 ${flashing ? 'ring-2 ring-inset ring-cyan-500' : ''}`} data-testid="bottom-panel">
      <div className="flex items-center justify-between px-2 border-b border-gray-800" role="tablist">
        <div className="flex">
          <button role="tab" aria-selected={tab === 'layers'} onClick={() => setTab('layers')} className={tabClass(tab === 'layers')} data-testid="tab-color-layers">
            Color layers <span className="text-gray-500">({bandCount})</span>
          </button>
          <button role="tab" aria-selected={tab === 'plan'} onClick={() => setTab('plan')} className={tabClass(tab === 'plan')} data-testid="tab-print-plan">
            Print plan <span className="text-gray-500">({swaps} swaps)</span>
          </button>
        </div>
        <span className="flex items-center gap-1 text-xs text-gray-400" data-testid="slider-render-status">
          {isRendering && (
            <>
              <Loader2 className="w-3 h-3 animate-spin" /> Updating preview…
            </>
          )}
        </span>
      </div>
      {/* Both tabs stay mounted (hidden via display:none rather than
          conditionally rendered) so switching tabs doesn't lose
          ColorSliders' row filter, scroll position, or an uncommitted
          number-field draft. */}
      <div className="flex-1 min-h-0 relative">
        <div className={`absolute inset-0 ${tab === 'layers' ? '' : 'hidden'}`}>
          <ColorSliders />
        </div>
        <div className={`absolute inset-0 ${tab === 'plan' ? '' : 'hidden'}`}>
          <PrintPlanPanel />
        </div>
      </div>
    </div>
  )
}
