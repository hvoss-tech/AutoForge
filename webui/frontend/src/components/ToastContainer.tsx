import React from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, Info, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { ToastLevel } from '../store/appStore'

const LEVEL_STYLE: Record<ToastLevel, { bg: string; border: string; icon: React.ReactNode }> = {
  error: { bg: 'bg-red-950/95', border: 'border-red-700', icon: <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" /> },
  warning: { bg: 'bg-yellow-950/95', border: 'border-yellow-700', icon: <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" /> },
  info: { bg: 'bg-gray-800/95', border: 'border-gray-600', icon: <Info className="w-4 h-4 text-blue-400 flex-shrink-0" /> },
}

/**
 * App-wide, always-on-top error/warning/info feed. Before this existed, most
 * failed actions (a rejected fetch, a background init/optimization error,
 * an OOM) only ever reached `console.error` — nothing in the page itself
 * told the user something went wrong, especially from inside a modal that
 * covers whatever inline error state a panel underneath it might show.
 */
export const ToastContainer: React.FC = () => {
  const toasts = useAppStore((s) => s.toasts)
  const dismissToast = useAppStore((s) => s.dismissToast)

  if (toasts.length === 0) return null

  // Portaled to <body> and given a z-index well above Radix Dialog's z-50
  // (components/ui/dialog.tsx) — a toast raised while a modal is open (the
  // exact moment a background job failure is most likely to be missed
  // otherwise) must stay above and clickable, not hidden behind the modal.
  // `pointerEvents: 'auto'` is required, not cosmetic: Radix sets
  // `document.body.style.pointerEvents = 'none'` while any Dialog is open
  // (to keep focus trapped inside it for accessibility) and only re-enables
  // it on DialogContent itself — this portal is a sibling of that content,
  // not a descendant, so without overriding it here the toast is visible
  // but inert the entire time a modal is open.
  return createPortal(
    <div
      style={{ position: 'fixed', top: 12, right: 12, zIndex: 10000, display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 380, pointerEvents: 'auto' }}
      data-testid="toast-container"
    >
      {toasts.map((toast) => {
        const style = LEVEL_STYLE[toast.level]
        return (
          <div
            key={toast.id}
            className={`flex items-start gap-2 rounded-lg border px-3 py-2 shadow-lg ${style.bg} ${style.border}`}
            data-testid={`toast-${toast.level}`}
          >
            {style.icon}
            <span className="text-xs text-gray-100 flex-1 break-words whitespace-pre-wrap">{toast.message}</span>
            <button
              onClick={() => dismissToast(toast.id)}
              className="text-gray-400 hover:text-gray-200 flex-shrink-0"
              aria-label="Dismiss"
              data-testid={`toast-dismiss-${toast.id}`}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )
      })}
    </div>,
    document.body
  )
}
