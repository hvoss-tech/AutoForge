import React, { useCallback, useEffect, useRef, useState } from 'react'
import { projectFileData, useAppStore } from '../store/appStore'
import { Box, ChevronDown, FilePlus, FileText, FolderOpen, Image as ImageIcon, PackageOpen, Printer, Save, Layers } from 'lucide-react'
import { buildPrintPlan, printPlanText } from '../lib/printPlan'
import { withEffectiveBaseColor } from '../lib/baseColor'
import { describeApiError } from '../lib/apiError'
import { onUiCommand } from '../lib/uiEvents'
import { exportFileName } from '../lib/project'
import { downloadBlob } from '../lib/download'
import { describeBorder, normalizeBorder } from '../lib/border'
import { SlicerProjectDialog } from './SlicerProjectDialog'
import { ExportProgressCard, type ExportState } from './ExportProgressCard'
import { preparePrinter, runSlicerExport } from '../lib/slicerExport'
import { shortPrinterName } from '../lib/slicerPrinters'


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
  const projectName = useAppStore((s) => s.projectName)
  const pushToast = useAppStore((s) => s.pushToast)
  const [exporting, setExporting] = useState(false)
  const [slicerDialogOpen, setSlicerDialogOpen] = useState(false)
  const [slicerExport, setSlicerExport] = useState<ExportState | null>(null)
  const slicerAbort = useRef<AbortController | null>(null)
  const lastSlicerExport = useRef<{ printer: string; manualSwaps: boolean | null; printSettings: boolean } | null>(null)
  const exporting3mf = slicerExport?.status === 'running'
  const runInputsByJob = useAppStore((s) => s.runInputsByJob)
  const setSettings = useAppStore((s) => s.setSettings)
  const named = (suffix: string, fallback: string) => exportFileName(projectName, suffix, fallback)
  // The border picked in the 3D view goes into every STL download.
  const border = normalizeBorder(settings.border)
  const borderSummary = describeBorder(border)
  const withBorder: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ border }),
  }

  // The .zip, STL and .hfp are written by an optimizer run; the auto-preview
  // has no printable model. The swap instructions and the image are made from
  // the color layers as they are, so they work as soon as there are bands.
  const canExport = currentJob?.status === 'completed'
  // The nozzle the result was made for (the server uses the same): the
  // run's, else the current setting.
  const runNozzle = currentJob ? Number(runInputsByJob[currentJob.job_id]?.settings?.nozzle_diameter) : NaN
  const resultNozzle = Number.isFinite(runNozzle) && runNozzle > 0 ? runNozzle : settings.nozzle_diameter
  const exportHint = canExport ? undefined : 'Needs an optimizer run — the preview before a run has no printable model yet'
  const hasBands = buildPrintPlan(colorSliders, [], settings).bands.length > 0
  const canExportImage = canExport || !!previewImage
  const jobActive = !!currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)

  useEffect(() => onUiCommand('open-file-menu', () => setOpen(true)), [])

  // Opening the menu with a printer picked: load its profiles now, so a
  // click on "Slicer project" doesn't wait for a download.
  useEffect(() => {
    if (open && canExport && settings.orca_printer) preparePrinter(settings.orca_printer, resultNozzle)
  }, [open, canExport, settings.orca_printer, resultNozzle])

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

  const downloadFromServer = async (url: string, filename: string, init?: RequestInit) => {
    setOpen(false)
    try {
      const response = await fetch(url, init)
      if (!response.ok) {
        setLoadError(`Export failed: ${describeApiError(await response.json().catch(() => null), response.status)}`)
        return
      }
      downloadBlob(await response.blob(), filename)
    } catch {
      setLoadError('Export failed')
    }
  }

  // The bundle holds the full-size STL and mesh (hundreds of MB) and is
  // fetched whole before the browser's download starts, so say it's coming.
  const handleExportZip = async () => {
    if (!jobId || exporting) return
    setExporting(true)
    pushToast('Preparing the export zip — the download starts when it is ready', 'info')
    try {
      await downloadFromServer(`/api/outputs/export/${jobId}`, named('project.zip', `${jobId}_export.zip`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: projectName,
          project: projectFileData(useAppStore.getState()),
          border,
          base: useAppStore.getState().resolvedBase,
        }),
      })
    } finally {
      setExporting(false)
    }
  }

  // The slicer project carries the colour layers as they are now and the
  // base the pipeline resolved. It runs in the background with a progress
  // card; the printer is remembered in the project, so the next export (and
  // the .zip) use it without asking again.
  const exportSlicerProject = async (printer: string, manualSwaps: boolean | null, printSettings: boolean) => {
    if (!jobId || exporting3mf) return
    lastSlicerExport.current = { printer, manualSwaps, printSettings }
    const state = useAppStore.getState()
    if (
      manualSwaps !== null &&
      (state.settings.orca_printer !== printer ||
        state.settings.orca_manual_swaps !== manualSwaps ||
        (state.settings.orca_autoforge_settings !== false) !== printSettings)
    ) {
      setSettings({ ...state.settings, orca_printer: printer, orca_manual_swaps: manualSwaps, orca_autoforge_settings: printSettings })
    }
    const title = `Slicer project · ${shortPrinterName(printer)}`
    const abort = new AbortController()
    slicerAbort.current = abort
    setSlicerExport({ status: 'running', title, fraction: 0, stage: 'Starting' })
    try {
      const current = useAppStore.getState()
      const result = await runSlicerExport(
        jobId,
        {
          name: projectName,
          project: { ...projectFileData(current), settings: withEffectiveBaseColor(current.settings, current.resolvedBase) },
          border,
          base: current.resolvedBase,
          printer,
          manual_swaps: manualSwaps,
          autoforge_settings: printSettings,
        },
        (p) => setSlicerExport({ status: 'running', title, fraction: p.fraction, stage: p.stage }),
        abort.signal,
      )
      downloadBlob(result.blob, named('model.3mf', `${jobId}.3mf`))
      setSlicerExport({
        status: 'done',
        title: result.printer ? `Slicer project · ${shortPrinterName(result.printer)}` : title,
        fraction: 1,
        stage: 'Done',
        notes: result.notes,
        warnings: result.warnings,
      })
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setSlicerExport(null)
      } else {
        setSlicerExport({ status: 'error', title, fraction: 0, stage: '', error: err instanceof Error ? err.message : 'Export failed' })
      }
    } finally {
      slicerAbort.current = null
    }
  }

  // One click when a printer is known; otherwise ask for it first.
  const handleSlicerProject = () => {
    setOpen(false)
    if (settings.orca_printer) exportSlicerProject(settings.orca_printer, settings.orca_manual_swaps ?? null, settings.orca_autoforge_settings !== false)
    else setSlicerDialogOpen(true)
  }

  const dismissSlicerExport = useCallback(() => setSlicerExport(null), [])

  const handleInstructions = () => {
    setOpen(false)
    // The instructions name the base color to start with, so they have to use
    // the one the pipeline resolved rather than the stale setting.
    const planSettings = withEffectiveBaseColor(settings, resolvedBase)
    const known = [...activeFilaments, ...filaments]
    const plan = buildPrintPlan(colorSliders, known, planSettings)
    const baseFilament = known.find((f) => f.uuid === planSettings.base_filament_uuid) ?? null
    downloadBlob(new Blob([printPlanText(plan, planSettings, baseFilament)], { type: 'text/plain' }), named('swap_instructions.txt', 'swap_instructions.txt'))
  }

  const handleImage = () => {
    if (canExport && currentJob) {
      downloadFromServer(`/api/outputs/current-preview/${currentJob.job_id}`, named('result.png', 'final_model.png'))
      return
    }
    setOpen(false)
    if (!previewImage) return
    fetch(previewImage)
      .then((r) => r.blob())
      .then((blob) => downloadBlob(blob, named('preview.png', 'preview.png')))
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
    <div className="relative flex-shrink-0" ref={menuRef}>
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
          <button onClick={handleExportZip} disabled={!canExport || exporting} title={exportHint} className={itemClass} role="menuitem" data-testid="file-menu-export">
            <PackageOpen className="w-3.5 h-3.5" /> {exporting ? 'Preparing zip…' : 'Everything (.zip)'}
          </button>
          {isFlatforge ? (
            flatforgeStlFiles.length > 0 ? (
              flatforgeStlFiles.map((name) => (
                <button
                  key={name}
                  onClick={() => jobId && downloadFromServer(`/api/outputs/file/${jobId}/${encodeURIComponent(name)}`, named(name, name), withBorder)}
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
            <button onClick={() => jobId && downloadFromServer(`/api/outputs/stl/${jobId}`, named('model.stl', 'final_model.stl'), withBorder)} disabled={!canExport} title={exportHint} className={itemClass} role="menuitem" data-testid="download-stl">
              <Box className="w-3.5 h-3.5" /> 3D model (.stl)
            </button>
          )}
          <button
            onClick={handleSlicerProject}
            disabled={!canExport || isFlatforge || exporting3mf}
            title={isFlatforge ? 'FlatForge prints one STL per colour; the slicer project is for layer-swap prints' : exportHint}
            className={itemClass}
            role="menuitem"
            data-testid="download-3mf"
          >
            <Printer className="w-3.5 h-3.5" />
            <span className="flex-1 min-w-0">
              <span className="block">{exporting3mf ? 'Exporting slicer project…' : 'Slicer project (.3mf)'}</span>
              <span className="block text-[11px] text-gray-400 truncate" data-testid="download-3mf-printer">
                {settings.orca_printer ? `for ${shortPrinterName(settings.orca_printer)}` : 'Choose your printer…'}
              </span>
            </span>
          </button>
          {settings.orca_printer && (
            <button
              onClick={() => {
                setOpen(false)
                setSlicerDialogOpen(true)
              }}
              disabled={!canExport || isFlatforge || exporting3mf}
              className={`${itemClass} pl-8 text-gray-400`}
              role="menuitem"
              data-testid="download-3mf-change-printer"
            >
              Other printer or settings…
            </button>
          )}
          <button onClick={handleInstructions} disabled={!hasBands} title={hasBands ? undefined : 'Add color layers first'} className={itemClass} role="menuitem" data-testid="download-instructions">
            <FileText className="w-3.5 h-3.5" /> Swap instructions (.txt)
          </button>
          <button onClick={handleImage} disabled={!canExportImage} title={canExportImage ? undefined : 'Upload an image first'} className={itemClass} role="menuitem" data-testid="download-preview">
            <ImageIcon className="w-3.5 h-3.5" /> {canExport ? 'Result image (.png)' : 'Preview image (.png)'}
          </button>
          <button onClick={() => jobId && downloadFromServer(`/api/outputs/project/${jobId}`, named('hueforge.hfp', 'final_model.hfp'))} disabled={!canExport} title={exportHint} className={itemClass} role="menuitem" data-testid="download-project">
            <Layers className="w-3.5 h-3.5" /> HueForge project (.hfp)
          </button>
          {borderSummary && (
            <p className="px-3 pt-1 text-[11px] text-cyan-300" data-testid="file-menu-border-note">
              Border in the STL: {borderSummary}
            </p>
          )}
          <p className="px-3 pt-1 pb-1 text-[11px] text-gray-400">
            Swap instructions, image and the .zip follow your edits; the separate .hfp is the optimizer's output.
            {settings.orca_printer ? ' The .zip includes the slicer project.' : ''}
          </p>
        </div>
      )}

      <SlicerProjectDialog
        open={slicerDialogOpen}
        mode="export"
        onClose={() => setSlicerDialogOpen(false)}
        rememberedPrinter={settings.orca_printer}
        rememberedManualSwaps={settings.orca_manual_swaps}
        rememberedAutoforgeSettings={settings.orca_autoforge_settings}
        nozzle={resultNozzle}
        onConfirm={(printer, manualSwaps, printSettings) => {
          setSlicerDialogOpen(false)
          exportSlicerProject(printer, manualSwaps, printSettings)
        }}
      />
      <ExportProgressCard
        state={slicerExport}
        onCancel={() => slicerAbort.current?.abort()}
        onRetry={() => {
          const last = lastSlicerExport.current
          if (last) exportSlicerProject(last.printer, last.manualSwaps, last.printSettings)
        }}
        onDismiss={dismissSlicerExport}
        onChangePrinter={() => {
          setSlicerExport(null)
          setSlicerDialogOpen(true)
        }}
      />

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
