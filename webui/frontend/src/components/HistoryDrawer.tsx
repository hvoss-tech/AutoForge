import React from 'react'
import { History, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/** Undo history as a side drawer (it used to be a popover inside the 3D
 * preview, covering the model). Newest step at the bottom, like the order
 * Undo walks back through. */
export const HistoryDrawer: React.FC = () => {
  const open = useAppStore((s) => s.historyOpen)
  const setOpen = useAppStore((s) => s.setHistoryOpen)
  const historyEntries = useAppStore((s) => s.historyEntries)
  const historyIndex = useAppStore((s) => s.historyIndex)
  const restoreToIndex = useAppStore((s) => s.restoreToIndex)
  const clearHistory = useAppStore((s) => s.clearHistory)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const listRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (open) listRef.current?.querySelector('[data-current="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [open, historyIndex])

  React.useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  if (!open) return null

  const handleClear = async () => {
    const ok = await requestConfirm({
      title: 'Clear history?',
      message: 'Undo and redo steps are forgotten. Your current project stays exactly as it is.',
      confirmLabel: 'Clear history',
      danger: true,
    })
    if (ok) clearHistory()
  }

  return (
    <aside
      className="fixed right-0 w-80 z-40 flex flex-col bg-gray-900 border-l border-gray-700 shadow-2xl"
      style={{ top: 'var(--topbar-height)', bottom: 'var(--statusbar-height)' }}
      data-testid="history-drawer"
      aria-label="History"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-gray-100">
          <History className="w-4 h-4" /> History
        </h2>
        <div className="flex items-center gap-1">
          <button
            onClick={handleClear}
            disabled={historyEntries.length <= 1}
            className="px-2 py-1 text-xs rounded text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:hover:bg-transparent"
            data-testid="history-clear-btn"
          >
            Clear
          </button>
          <button onClick={() => setOpen(false)} className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700" aria-label="Close history" data-testid="history-close-btn">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
      <p className="px-3 py-2 text-xs text-gray-400 border-b border-gray-800">Click a step to go back to it. Newer steps stay available until you make a new change.</p>
      <div ref={listRef} className="flex-1 overflow-y-auto" data-testid="history-list">
        {historyEntries.length === 0 && <div className="p-3 text-xs text-gray-400">No changes yet</div>}
        {historyEntries.map((entry, i) => {
          const current = i === historyIndex
          const future = i > historyIndex
          return (
            <button
              key={`${entry.timestamp}-${i}`}
              onClick={() => restoreToIndex(i)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800 hover:bg-gray-800 ${current ? 'bg-blue-600/15 border-l-2 border-l-blue-500' : ''}`}
              data-testid={`history-entry-${i}`}
              data-current={current}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className={`text-xs ${future ? 'text-gray-500' : 'text-gray-100'} ${current ? 'font-semibold' : ''}`}>{entry.label}</span>
                <span className="text-[11px] text-gray-500 tabular-nums flex-shrink-0">{formatTime(entry.timestamp)}</span>
              </div>
              {entry.jobStatus === 'completed' && entry.jobId && (
                <span className="text-[11px] text-emerald-500">with result {entry.jobId.slice(0, 8)}</span>
              )}
            </button>
          )
        })}
      </div>
    </aside>
  )
}
