import React from 'react'
import { useAppStore } from '../store/appStore'

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const ResultsHistory: React.FC = () => {
  const [open, setOpen] = React.useState(false)
  const historyEntries = useAppStore((s) => s.historyEntries)
  const historyIndex = useAppStore((s) => s.historyIndex)
  const restoreToIndex = useAppStore((s) => s.restoreToIndex)
  const clearHistory = useAppStore((s) => s.clearHistory)
  const listRef = React.useRef<HTMLDivElement>(null)

  // Open scrolled to the current step — with 50 entries the "▶" marker was
  // usually far below the fold.
  React.useEffect(() => {
    if (open) listRef.current?.querySelector('[data-current="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [open, historyIndex])

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        style={{ color: 'var(--cyan-accent)', fontSize: 10, border: '1px solid var(--border)', borderRadius: 4, padding: '2px 8px', backgroundColor: 'var(--bg-panel)', cursor: 'pointer' }}
        data-testid="history-open-btn"
      >
        History
      </button>
    )
  }

  return (
    <div style={{ position: 'absolute', bottom: 44, right: 8, width: 320, backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 4, zIndex: 50, boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid var(--border)', fontSize: 10, color: 'var(--text-secondary)' }}>
        <span>History — click a step to jump to it (same as Undo/Redo)</span>
        <span style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => { if (window.confirm('Clear the undo history? Your current project stays as it is.')) clearHistory() }}
            disabled={historyEntries.length <= 1}
            style={{ cursor: 'pointer', background: 'none', border: 'none', color: 'var(--text-secondary)', opacity: historyEntries.length <= 1 ? 0.4 : 1 }}
            data-testid="history-clear-btn"
          >
            Clear
          </button>
          <button onClick={() => setOpen(false)} style={{ cursor: 'pointer', background: 'none', border: 'none', color: 'var(--text-secondary)' }} data-testid="history-close-btn" aria-label="Close history">X</button>
        </span>
      </div>
      <div ref={listRef} style={{ maxHeight: 240, overflowY: 'auto' }} data-testid="history-list">
        {historyEntries.length === 0 && (
          <div style={{ padding: 8, fontSize: 10, color: 'var(--text-secondary)' }}>No history yet</div>
        )}
        {historyEntries.map((entry, i) => (
          <div
            key={entry.timestamp}
            onClick={() => { restoreToIndex(i); setOpen(false) }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              padding: '4px 8px',
              cursor: 'pointer',
              borderBottom: '1px solid var(--border)',
              backgroundColor: i === historyIndex ? 'var(--bg-input)' : 'transparent',
            }}
            data-testid={`history-entry-${i}`}
            data-current={i === historyIndex}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-primary)' }}>
              <span>{i === historyIndex ? '▶ ' : ''}{entry.label}</span>
              <span style={{ color: 'var(--text-secondary)' }}>{formatTime(entry.timestamp)}</span>
            </div>
            {entry.jobId && (
              <span style={{ fontSize: 8, color: entry.jobStatus === 'completed' ? 'var(--green-accent)' : 'var(--text-secondary)' }}>
                {entry.jobId.slice(0, 8)}… {entry.jobStatus ?? ''}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
