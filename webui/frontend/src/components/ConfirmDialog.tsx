import React from 'react'
import { AlertTriangle } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'

/** App-wide confirmation dialog, driven by `requestConfirm()` in the store —
 * replaces the browser's unstyled window.confirm(). */
export const ConfirmDialog: React.FC = () => {
  const request = useAppStore((s) => s.confirmRequest)
  const resolveConfirm = useAppStore((s) => s.resolveConfirm)

  return (
    <Dialog open={!!request} onOpenChange={(open) => { if (!open) resolveConfirm(false) }}>
      <DialogContent className="max-w-md bg-gray-900 border-gray-700 p-5" data-testid="confirm-dialog">
        {request && (
          <>
            <DialogTitle className="flex items-center gap-2 text-base text-gray-100">
              {request.danger && <AlertTriangle className="w-4 h-4 text-amber-500" />}
              {request.title}
            </DialogTitle>
            <DialogDescription className="text-sm text-gray-300 whitespace-pre-line">{request.message}</DialogDescription>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => resolveConfirm(false)}
                className="px-3 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 text-gray-100"
                data-testid="confirm-cancel"
              >
                Cancel
              </button>
              <button
                onClick={() => resolveConfirm(true)}
                autoFocus
                className={`px-3 py-1.5 text-sm rounded text-white ${request.danger ? 'bg-red-600 hover:bg-red-500' : 'bg-blue-600 hover:bg-blue-500'}`}
                data-testid="confirm-ok"
              >
                {request.confirmLabel}
              </button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
