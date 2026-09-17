import React, { useEffect, useRef, useState } from 'react'
import { durableInputImageUrl, useAppStore } from '../store/appStore'
import { ChevronDown, Save, FolderOpen, PackageOpen } from 'lucide-react'

export const FileMenu: React.FC = () => {
  const [open, setOpen] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const colorSliders = useAppStore((s) => s.colorSliders)
  const settings = useAppStore((s) => s.settings)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const inputImage = useAppStore((s) => s.inputImage)
  const currentJob = useAppStore((s) => s.currentJob)
  const loadProjectFromFile = useAppStore((s) => s.loadProjectFromFile)

  const canExport = currentJob?.status === 'completed'

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const handleSave = () => {
    const data = {
      version: 1,
      savedAt: new Date().toISOString(),
      colorSliders,
      settings,
      activeFilaments,
      // A raw `blob:` object URL only resolves in this browser tab — saving
      // it verbatim means reopening the file later (or on another
      // machine/profile) shows a permanently broken image even though the
      // server still has the real upload. Persist the durable server path
      // instead.
      inputImage: durableInputImageUrl(inputImage, settings),
    }
    downloadBlob(
      new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }),
      `autoforge-project-${Date.now()}.json`,
    )
    setOpen(false)
  }

  const handleLoadClick = () => {
    setOpen(false)
    fileInputRef.current?.click()
  }

  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setLoadError(null)
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      await loadProjectFromFile(data)
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to load project file')
    }
  }

  const handleExport = async () => {
    setOpen(false)
    if (!canExport || !currentJob) return
    try {
      const response = await fetch(`/api/outputs/export/${currentJob.job_id}`)
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Export failed' }))
        setLoadError(err.detail ?? 'Export failed')
        return
      }
      downloadBlob(await response.blob(), `${currentJob.job_id}_export.zip`)
    } catch {
      setLoadError('Export failed')
    }
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800 rounded"
        data-testid="file-menu-btn"
      >
        File
        <ChevronDown className="w-3 h-3" />
      </button>

      {open && (
        <div
          className="absolute left-0 top-full mt-1 w-56 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 py-1"
          data-testid="file-menu-dropdown"
        >
          <button
            onClick={handleSave}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-700 text-left"
            data-testid="file-menu-save"
          >
            <Save className="w-3.5 h-3.5" />
            Save Project…
          </button>
          <button
            onClick={handleLoadClick}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-700 text-left"
            data-testid="file-menu-load"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            Load Project…
          </button>
          <button
            onClick={handleExport}
            disabled={!canExport}
            title={canExport ? undefined : 'Run an optimization first'}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-700 text-left disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent"
            data-testid="file-menu-export"
          >
            <PackageOpen className="w-3.5 h-3.5" />
            Export Project (.zip)
          </button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={handleFileSelected}
        data-testid="file-menu-load-input"
      />

      {loadError && (
        <div
          className="absolute left-0 top-full mt-1 w-64 bg-red-900/80 border border-red-700 rounded px-2 py-1 text-xs text-red-200 z-50"
          data-testid="file-menu-error"
        >
          {loadError}
        </div>
      )}
    </div>
  )
}
