import React from 'react'
import { useAppStore } from '../store/appStore'
import { refreshLibrary } from '../services/filamentService'
import { describeApiError } from '../lib/apiError'
import { X, Upload, FileJson, FileText, AlertTriangle } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from './ui/dialog'

type PendingFile = { ext: 'json' | 'csv'; text: string }

// A page's first second fires a dozen-plus requests (every panel's own
// on-mount fetch, project/filament/history loads, the preview WebSocket
// handshake); a browser caps HTTP/1.1 at ~6 concurrent connections per
// origin, so a fetch issued into that window can end up queued behind it
// for seconds rather than actually failing. Retrying a genuine failure a
// couple of times with a short backoff turns that transient stall into an
// invisible delay instead of a hard "Import failed" error.
async function fetchWithRetry(url: string, init: RequestInit, retries = 3): Promise<Response> {
  let lastErr: unknown
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fetch(url, init)
    } catch (err) {
      lastErr = err
      if (attempt < retries - 1) await new Promise((r) => setTimeout(r, 500 * (attempt + 1)))
    }
  }
  throw lastErr
}

export const ImportModal: React.FC = () => {
  const importModalOpen = useAppStore((s) => s.importModalOpen)
  const setImportModalOpen = useAppStore((s) => s.setImportModalOpen)
  const setCustomLibraryLoaded = useAppStore((s) => s.setCustomLibraryLoaded)
  // The store's `filaments` is only the current tab's list; the replace
  // warning must state how many filaments the whole library really loses.
  const [existingCount, setExistingCount] = React.useState<number | null>(null)

  const [dragOver, setDragOver] = React.useState(false)
  const [importStatus, setImportStatus] = React.useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [pending, setPending] = React.useState<PendingFile | null>(null)
  const [confirmingReplace, setConfirmingReplace] = React.useState(false)
  const [importing, setImporting] = React.useState(false)
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  const readFile = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext !== 'json' && ext !== 'csv') {
      setImportStatus({ type: 'error', message: 'Only JSON and CSV files are supported' })
      return
    }
    setImportStatus(null)
    const reader = new FileReader()
    reader.onerror = () => setImportStatus({ type: 'error', message: 'Could not read the file' })
    reader.onload = () => {
      setPending({ ext, text: reader.result as string })
      setConfirmingReplace(false)
      fetch('/api/filaments')
        .then((r) => r.json())
        .then((all) => setExistingCount(Array.isArray(all) ? all.length : null))
        .catch(() => setExistingCount(null))
    }
    reader.readAsText(file)
  }

  const runImport = async (mode: 'merge' | 'replace') => {
    if (!pending) return
    setImporting(true)
    setImportStatus(null)
    try {
      let response: Response
      if (pending.ext === 'csv') {
        response = await fetchWithRetry('/api/filaments/import-csv', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ contents: pending.text, mode }),
        })
      } else {
        const data = JSON.parse(pending.text)
        const arr = Array.isArray(data) ? data : [data]
        response = await fetchWithRetry(`/api/filaments/import-json?mode=${mode}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(arr),
        })
      }

      const data = await response.json()

      if (response.ok && data.status === 'ok') {
        setImportStatus({ type: 'success', message: data.message })
        setCustomLibraryLoaded(true)
        setPending(null)
        setConfirmingReplace(false)

        await refreshLibrary()
      } else {
        setImportStatus({ type: 'error', message: data?.message && !data?.detail ? data.message : describeApiError(data, response.status) })
      }
    } catch (err) {
      setImportStatus({ type: 'error', message: `Import failed: ${err}` })
    } finally {
      setImporting(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) readFile(file)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) readFile(file)
  }

  const handleClose = () => {
    setImportModalOpen(false)
    setImportStatus(null)
    setPending(null)
    setConfirmingReplace(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <Dialog open={importModalOpen} onOpenChange={setImportModalOpen}>
      <DialogContent data-testid="import-modal">
        <div className="bg-gray-900 rounded-lg w-full flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="text-sm font-semibold text-gray-200 flex items-center gap-2 leading-none tracking-normal">
            <Upload className="w-4 h-4" />
            Import Filaments
          </DialogTitle>
          <DialogDescription className="sr-only">
            Import filaments from a JSON or CSV file by dropping or browsing for it.
          </DialogDescription>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-200"
            data-testid="close-import"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'
            }`}
          >
            <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
            <p className="text-sm text-gray-300 mb-1">Drop a file here or click to browse</p>
            <p className="text-xs text-gray-500">Supports JSON and CSV formats</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.csv"
              onChange={handleFileInput}
              className="hidden"
              data-testid="file-input"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="mt-3 px-4 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
            >
              Browse Files
            </button>
          </div>

          {!pending && (
            <div className="flex gap-3">
              <div className="flex-1 flex items-center gap-2 p-3 bg-gray-800 rounded">
                <FileJson className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="text-xs text-gray-200 font-medium">JSON</p>
                  <p className="text-xs text-gray-500">AutoForge format</p>
                </div>
              </div>
              <div className="flex-1 flex items-center gap-2 p-3 bg-gray-800 rounded">
                <FileText className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-xs text-gray-200 font-medium">CSV</p>
                  <p className="text-xs text-gray-500">Spreadsheet format</p>
                </div>
              </div>
            </div>
          )}

          {pending && !confirmingReplace && (
            <div className="p-3 bg-gray-800 rounded space-y-3" data-testid="import-mode-choice">
              <p className="text-xs text-gray-300">
                File ready ({pending.ext.toUpperCase()}). Add these filaments to your existing library, or replace the entire library with just this file?
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => runImport('merge')}
                  disabled={importing}
                  className="flex-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-xs text-white"
                  data-testid="import-mode-merge"
                >
                  Add to Library
                  <span className="block text-[10px] font-normal opacity-80">Updates any same-named filament, keeps the rest</span>
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmingReplace(true)}
                  disabled={importing}
                  className="flex-1 px-3 py-1.5 bg-red-900/40 hover:bg-red-900/60 border border-red-700 disabled:opacity-50 rounded text-xs text-red-300"
                  data-testid="import-mode-replace"
                >
                  Replace Entire Library
                  <span className="block text-[10px] font-normal opacity-80">{existingCount === null ? 'Removes all current filaments' : `Removes all ${existingCount} current filaments`}</span>
                </button>
              </div>
            </div>
          )}

          {pending && confirmingReplace && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded space-y-3" data-testid="import-replace-confirm">
              <p className="text-xs text-red-200 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                This deletes {existingCount === null ? 'all the' : `all ${existingCount}`} filaments currently in your library and replaces them with the ones from this file. This can't be undone. Are you sure?
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmingReplace(false)}
                  disabled={importing}
                  className="flex-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-xs text-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => runImport('replace')}
                  disabled={importing}
                  className="flex-1 px-3 py-1.5 bg-red-700 hover:bg-red-600 disabled:opacity-50 rounded text-xs text-white"
                  data-testid="import-replace-confirm-btn"
                >
                  {importing ? 'Replacing…' : 'Yes, Replace Library'}
                </button>
              </div>
            </div>
          )}

          {importStatus && (
            <div className={`p-3 rounded text-xs ${
              importStatus.type === 'success'
                ? 'bg-green-900/30 text-green-300 border border-green-700'
                : 'bg-red-900/30 text-red-300 border border-red-700'
            }`}>
              {importStatus.message}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2 border-t border-gray-700">
            <button
              onClick={handleClose}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
            >
              Close
            </button>
          </div>
        </div>
      </div>
      </DialogContent>
    </Dialog>
  )
}
