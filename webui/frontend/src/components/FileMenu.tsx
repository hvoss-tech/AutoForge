import React, { useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'
import { Box, ChevronDown, FilePlus, FileText, FolderOpen, Image as ImageIcon, PackageOpen, Save, Layers } from 'lucide-react'
import { buildPrintPlan, printPlanText } from '../lib/printPlan'
import { withEffectiveBaseColor } from '../lib/baseColor'
import { describeApiError } from '../lib/apiError'
import { onUiCommand } from '../lib/uiEvents'

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

const itemClass =
  'w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-700 text-left disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent'

export const FileMenu: React.FC = () => {
  const [open, setOpen] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [flatforgeStlFiles, setFlatforgeStlFiles] = useState<string[]>([])
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const colorSliders = useAppStore((s) => s.colorSliders)
  const settings = useAppStore((s) => s.settings)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const filaments = useAppStore((s) => s.filaments)
  const currentJob = useAppStore((s) => s.currentJob)
  const previewImage = useAppStore((s) => s.previewImage)
  const loadProjectFromFile = useAppStore((s) => s.loadProjectFromFile)
  const saveProjectToFile = useAppStore((s) => s.saveProjectToFile)
  const startNewProject = useAppStore((s) => s.startNewProject)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const resolvedBase = useAppStore((s) => s.resolvedBase)

  // The .zip, STL and .hfp are written by an optimizer run; the auto-preview
  // has no printable model. The swap instructions and the image are made from
  // the color layers as they are, so they work as soon as there are bands.
  const canExport = currentJob?.status === 'completed'
  const exportHint = canExport ? undefined : 'Needs an optimizer run — the preview before a run has no printable model yet'
  const hasBands = buildPrintPlan(colorSliders, [], settings).bands.length > 0
  const canExportImage = canExport || !!previewImage
  const jobActive = !!currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)

  useEffect(() => onUiCommand('open-file-menu', () => setOpen(true)), [])

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const handleSave = () => {
    saveProjectToFile()
    setOpen(false)
  }

  const handleNew = async () => {
    setOpen(false)
    const ok = await requestConfirm({
      title: 'Start a new project?',
      message: 'The image, color layers and result are cleared. Settings and active filaments stay.\nYou can get everything back with Undo.',
      confirmLabel: 'Start new project',
    })
    if (ok) startNewProject()
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
      const data = JSON.parse(await file.text())
      await loadProjectFromFile(data, file.name)
    } catch (err) {
      setLoadError(err instanceof Error ? `Could not load "${file.name}": ${err.message}` : 'Failed to load project file')
    }
  }

  const downloadFromServer = async (url: string, filename: string) => {
    setOpen(false)
    try {
      const response = await fetch(url)
      if (!response.ok) {
        setLoadError(`Export failed: ${describeApiError(await response.json().catch(() => null), response.status)}`)
        return
      }
      downloadBlob(await response.blob(), filename)
    } catch {
      setLoadError('Export failed')
    }
  }

  const handleInstructions = () => {
    setOpen(false)
    // The instructions name the base color to start with, so they have to use
    // the one the pipeline resolved rather than the stale setting.
    const planSettings = withEffectiveBaseColor(settings, resolvedBase)
    const plan = buildPrintPlan(colorSliders, [...activeFilaments, ...filaments], planSettings)
    downloadBlob(new Blob([printPlanText(plan, planSettings)], { type: 'text/plain' }), 'swap_instructions.txt')
  }

  const handleImage = () => {
    if (canExport && currentJob) {
      downloadFromServer(`/api/outputs/current-preview/${currentJob.job_id}`, 'final_model.png')
      return
    }
    setOpen(false)
    if (!previewImage) return
    fetch(previewImage)
      .then((r) => r.blob())
      .then((blob) => downloadBlob(blob, 'preview.png'))
      .catch(() => setLoadError('Export failed'))
  }

  const jobId = currentJob?.job_id
  const isFlatforge = !!currentJob?.flatforge

  // FlatForge runs write one STL per material (no final_model.stl), so the
  // single-file `/api/outputs/stl/{jobId}` endpoint has nothing to serve —
  // without this, the STL entry just 404'd and the "Everything (.zip)"
  // bundle (which does contain them) was the only way to get the print
  // files out of the UI at all. Fetched lazily, only while the menu holding
  // it is actually open.
  useEffect(() => {
    if (!open || !canExport || !isFlatforge || !jobId) {
      setFlatforgeStlFiles([])
      return
    }
    let cancelled = false
    fetch(`/api/outputs/stl-list/${jobId}`)
      .then((r) => (r.ok ? r.json() : { files: [] }))
      .then((data) => {
        if (!cancelled) setFlatforgeStlFiles(Array.isArray(data.files) ? data.files : [])
      })
      .catch(() => {
        if (!cancelled) setFlatforgeStlFiles([])
      })
    return () => {
      cancelled = true
    }
  }, [open, canExport, isFlatforge, jobId])

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => {
          setOpen((o) => !o)
          // An old error has nothing to do with whatever the user does next.
          setLoadError(null)
        }}
        className="flex items-center gap-1 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800 rounded"
        aria-haspopup="menu"
        aria-expanded={open}
        data-testid="file-menu-btn"
      >
        File
        <ChevronDown className="w-3 h-3" />
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 w-64 bg-gray-800 border border-gray-700 rounded shadow-lg z-50 py-1" role="menu" data-testid="file-menu-dropdown">
          <div className="px-3 pt-1 pb-0.5 text-[11px] uppercase tracking-wide text-gray-500">Project</div>
          <button onClick={handleNew} disabled={jobActive} title={jobActive ? 'Wait for the run to finish or cancel it' : undefined} className={itemClass} role="menuitem" data-testid="file-menu-new">
            <FilePlus className="w-3.5 h-3.5" /> New project
          </button>
          <button onClick={handleSave} className={itemClass} role="menuitem" data-testid="file-menu-save">
            <Save className="w-3.5 h-3.5" /> Save project…
            <span className="ml-auto text-[11px] text-gray-500">Ctrl+S</span>
          </button>
          <button onClick={handleLoadClick} className={itemClass} role="menuitem" data-testid="file-menu-load">
            <FolderOpen className="w-3.5 h-3.5" /> Load project…
          </button>

          <div className="my-1 border-t border-gray-700" />
          <div className="px-3 pt-1 pb-0.5 text-[11px] uppercase tracking-wide text-gray-500">Export result</div>
          <button onClick={() => jobId && downloadFromServer(`/api/outputs/export/${jobId}`, `${jobId}_export.zip`)} disabled={!canExport} title={exportHint} className={itemClass} role="menuitem" data-testid="file-menu-export">
            <PackageOpen className="w-3.5 h-3.5" /> Everything (.zip)
          </button>
          {isFlatforge ? (
            flatforgeStlFiles.length > 0 ? (
              flatforgeStlFiles.map((name) => (
                <button
                  key={name}
                  onClick={() => jobId && downloadFromServer(`/api/outputs/file/${jobId}/${encodeURIComponent(name)}`, name)}
                  className={itemClass}
                  role="menuitem"
                  data-testid="download-stl-flatforge"
                >
                  <Box className="w-3.5 h-3.5" /> {name}
                </button>
              ))
            ) : (
              <button disabled title="FlatForge writes one STL per material — use “Everything (.zip)” above" className={itemClass} role="menuitem" data-testid="download-stl">
                <Box className="w-3.5 h-3.5" /> 3D model (.stl)
              </button>
            )
          ) : (
            <button onClick={() => jobId && downloadFromServer(`/api/outputs/stl/${jobId}`, 'final_model.stl')} disabled={!canExport} title={exportHint} className={itemClass} role="menuitem" data-testid="download-stl">
              <Box className="w-3.5 h-3.5" /> 3D model (.stl)
            </button>
          )}
          <button onClick={handleInstructions} disabled={!hasBands} title={hasBands ? undefined : 'Add color layers first'} className={itemClass} role="menuitem" data-testid="download-instructions">
            <FileText className="w-3.5 h-3.5" /> Swap instructions (.txt)
          </button>
          <button onClick={handleImage} disabled={!canExportImage} title={canExportImage ? undefined : 'Upload an image first'} className={itemClass} role="menuitem" data-testid="download-preview">
            <ImageIcon className="w-3.5 h-3.5" /> {canExport ? 'Result image (.png)' : 'Preview image (.png)'}
          </button>
          <button onClick={() => jobId && downloadFromServer(`/api/outputs/project/${jobId}`, 'final_model.hfp')} disabled={!canExport} title={exportHint} className={itemClass} role="menuitem" data-testid="download-project">
            <Layers className="w-3.5 h-3.5" /> HueForge project (.hfp)
          </button>
          <p className="px-3 pt-1 pb-1 text-[11px] text-gray-400">
            Swap instructions and image follow your edits; the .zip, STL and .hfp are the optimizer's output.
          </p>
        </div>
      )}

      <input ref={fileInputRef} type="file" accept="application/json,.json" className="hidden" onChange={handleFileSelected} data-testid="file-menu-load-input" />

      {loadError && (
        <div className="absolute left-0 top-full mt-1 w-72 flex items-start gap-2 bg-red-950 border border-red-700 rounded px-3 py-2 text-xs text-red-200 z-50" role="alert" data-testid="file-menu-error">
          <span className="flex-1">{loadError}</span>
          <button onClick={() => setLoadError(null)} className="text-red-300 hover:text-white" aria-label="Dismiss" data-testid="file-menu-error-dismiss">
            ×
          </button>
        </div>
      )}
    </div>
  )
}
