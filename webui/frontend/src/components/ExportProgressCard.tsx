import React, { useEffect } from 'react'
import { AlertTriangle, CheckCircle2, Loader2, X } from 'lucide-react'

export interface ExportState {
  status: 'running' | 'done' | 'error'
  title: string
  fraction: number
  stage: string
  error?: string
  /** Shown once done: notes (informational) and warnings (to fix by hand). */
  notes?: string[]
  warnings?: string[]
}

interface Props {
  state: ExportState | null
  onCancel: () => void
  onRetry: () => void
  onDismiss: () => void
  /** Offered next to "Try again" when an export fails: the printer may be the problem. */
  onChangePrinter?: () => void
}

/** A small card in the corner that shows an export's progress, so a long
 * export neither blocks the app nor leaves the user guessing. */
export const ExportProgressCard: React.FC<Props> = ({ state, onCancel, onRetry, onDismiss, onChangePrinter }) => {
  // A finished export says so briefly, then gets out of the way - unless it
  // has a warning, which stays until read and closed.
  const warned = !!state?.warnings?.length
  const noted = !!state?.notes?.length
  useEffect(() => {
    if (state?.status !== 'done' || warned) return
    const t = setTimeout(onDismiss, noted ? 7000 : 2500)
    return () => clearTimeout(t)
  }, [state?.status, warned, noted, onDismiss])

  if (!state) return null
  const percent = Math.round(state.fraction * 100)
  return (
    <div
      className="fixed bottom-4 right-4 z-[60] w-80 rounded-lg border border-gray-700 bg-gray-900/95 shadow-xl px-4 py-3"
      role="status"
      aria-live="polite"
      data-testid="export-progress"
    >
      <div className="flex items-center gap-2">
        {state.status === 'running' && <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />}
        {state.status === 'done' && !warned && <CheckCircle2 className="w-4 h-4 text-green-400" />}
        {state.status === 'done' && warned && <AlertTriangle className="w-4 h-4 text-amber-400" />}
        {state.status === 'error' && <AlertTriangle className="w-4 h-4 text-amber-400" />}
        <span className="flex-1 text-xs font-medium text-gray-100 truncate">{state.title}</span>
        {state.status !== 'running' && (
          <button onClick={onDismiss} className="text-gray-400 hover:text-gray-100" aria-label="Close" data-testid="export-progress-close">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {state.status === 'error' ? (
        <>
          <p className="mt-2 text-xs text-red-200" data-testid="export-progress-error">{state.error}</p>
          <div className="mt-2 flex justify-end gap-2">
            {onChangePrinter && (
              <button onClick={onChangePrinter} className="px-2.5 py-1 text-xs rounded text-gray-300 hover:bg-gray-800" data-testid="export-progress-change-printer">
                Change printer
              </button>
            )}
            <button onClick={onRetry} className="px-2.5 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="export-progress-retry">
              Try again
            </button>
          </div>
        </>
      ) : (
        <>
          <div
            className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-700"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
          >
            <div
              className={`h-full rounded-full transition-[width] duration-200 ease-out ${state.status === 'done' ? 'bg-green-500' : 'bg-cyan-500'}`}
              style={{ width: `${Math.max(percent, 2)}%` }}
            />
          </div>
          <div className="mt-1.5 flex items-center justify-between text-[11px] text-gray-400">
            <span data-testid="export-progress-stage">{state.status === 'done' ? 'Downloaded' : state.stage}</span>
            {state.status === 'running' ? (
              <span className="flex items-center gap-2">
                {percent}%
                <button onClick={onCancel} className="text-gray-400 hover:text-gray-100 underline" data-testid="export-progress-cancel">
                  Cancel
                </button>
              </span>
            ) : (
              <span>100%</span>
            )}
          </div>
          {state.status === 'done' && warned && (
            <div className="mt-2 rounded border border-amber-700/70 bg-amber-950/40 px-2.5 py-2 text-[11px] leading-snug text-amber-200" data-testid="export-progress-warning">
              {state.warnings!.map((w) => <p key={w}>{w}</p>)}
            </div>
          )}
          {state.status === 'done' && noted && (
            <div className="mt-1.5 text-[11px] leading-snug text-gray-400" data-testid="export-progress-note">
              {state.notes!.map((n) => <p key={n}>{n}</p>)}
            </div>
          )}
        </>
      )}
    </div>
  )
}
