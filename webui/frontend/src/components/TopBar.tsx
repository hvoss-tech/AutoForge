import React, { useEffect, useMemo, useRef, useState } from 'react'
import { currentProjectFingerprint, useAppStore } from '../store/appStore'
import {
  AlertTriangle,
  ArrowUpCircle,
  Check,
  HelpCircle,
  History,
  Loader2,
  Moon,
  Pause,
  Play,
  Redo2,
  Scissors,
  Settings,
  Square,
  Sun,
  Undo2,
  WifiOff,
} from 'lucide-react'
import { PruningModal } from './PruningModal'
import { FileMenu } from './FileMenu'
import { createProgressTracker, formatDuration, type ProgressTracker } from '../lib/progress'
import { getWorkflowSteps, type WorkflowStep } from '../lib/workflow'
import { staleReasons } from '../lib/staleResult'
import { settingName } from '../lib/history'
import { sparklinePath } from '../lib/lossHistory'
import { hasUnsavedChanges } from '../lib/project'
import { buildPrintPlan } from '../lib/printPlan'
import { describePruningChange, resultCounts } from '../lib/pruning'
import { onUiCommand, sendUiCommand, type UiCommand } from '../lib/uiEvents'
import { useFlash } from '../hooks/usePersistentState'

interface UpdateInfo {
  current_version: string
  latest_version: string | null
  update_available: boolean
  release_url: string
}

const iconButton =
  'p-1.5 rounded text-gray-300 hover:text-gray-100 hover:bg-gray-700 disabled:opacity-35 disabled:hover:bg-transparent disabled:cursor-not-allowed'

// One filled button (the next thing to do); everything else is outlined.
const buttonBase = 'flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium disabled:cursor-not-allowed'
const primaryButton = `${buttonBase} bg-blue-600 hover:bg-blue-500 text-white disabled:bg-gray-700 disabled:text-gray-400`
const outlineButton = (tone: string) =>
  `${buttonBase} border ${tone} disabled:border-gray-700 disabled:text-gray-500 disabled:hover:bg-transparent`

const STEP_COMMANDS: Record<WorkflowStep['id'], UiCommand> = {
  image: 'focus-image',
  filaments: 'focus-library',
  run: 'focus-run',
  adjust: 'focus-layers',
  export: 'open-file-menu',
}

const WorkflowSteps: React.FC = () => {
  const inputImage = useAppStore((s) => s.inputImage)
  const activeFilamentCount = useAppStore((s) => s.activeFilaments.length)
  const jobStatus = useAppStore((s) => s.currentJob?.status ?? null)
  const steps = useMemo(
    () => getWorkflowSteps({ hasImage: !!inputImage, activeFilamentCount, jobStatus }),
    [inputImage, activeFilamentCount, jobStatus],
  )
  const current = steps.find((s) => s.state === 'current')
  // Once there's a result the first three steps are just history: keep their
  // checkmarks but drop the labels to make room. While a run is going the
  // progress readout needs the room too, so only the current step keeps its
  // label — without this the bar overflowed and the project name was drawn
  // on top of the undo buttons.
  const running = jobStatus === 'running' || jobStatus === 'paused' || jobStatus === 'pending'
  const compact = jobStatus === 'completed'

  return (
    <ol className="flex items-center gap-1 text-xs flex-shrink-0" data-testid="workflow-steps" title={current?.hint}>
      {steps.map((step, i) => {
        const hideLabel = running ? step.state !== 'current' : compact && step.state === 'done'
        return (
          <li key={step.id} className="flex items-center gap-1" data-testid={`workflow-step-${step.id}`} data-state={step.state}>
            {i > 0 && <span className="w-3 h-px bg-gray-600" aria-hidden />}
            <button
              onClick={() => sendUiCommand(STEP_COMMANDS[step.id])}
              className="flex items-center gap-1 rounded px-0.5 hover:bg-gray-800"
              title={`${step.label}: ${step.hint}`}
              aria-label={`${step.label} (${step.state === 'done' ? 'done' : step.state === 'current' ? 'current step' : 'upcoming'})`}
              data-testid={`workflow-step-btn-${step.id}`}
            >
              <span
                className={`flex items-center justify-center w-5 h-5 rounded-full text-[11px] font-semibold ${
                  step.state === 'done'
                    ? 'bg-emerald-600 text-white'
                    : step.state === 'current'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                }`}
              >
                {step.state === 'done' ? <Check className="w-3 h-3" /> : i + 1}
              </span>
              <span className={`${hideLabel ? 'sr-only' : ''} ${step.state === 'upcoming' ? 'text-gray-500' : 'text-gray-200'}`}>{step.label}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

const HistoryControls: React.FC = () => {
  const undo = useAppStore((s) => s.undo)
  const redo = useAppStore((s) => s.redo)
  const historyIndex = useAppStore((s) => s.historyIndex)
  const historyLength = useAppStore((s) => s.historyLength)
  const historyEntries = useAppStore((s) => s.historyEntries)
  const historyOpen = useAppStore((s) => s.historyOpen)
  const setHistoryOpen = useAppStore((s) => s.setHistoryOpen)
  const canUndo = historyIndex > 0
  const canRedo = historyIndex < historyLength - 1

  return (
    <div className="flex items-center gap-0.5 px-1 border-x border-gray-700 flex-shrink-0" role="group" aria-label="Undo history">
      <button
        onClick={() => undo()}
        disabled={!canUndo}
        className={iconButton}
        title={canUndo ? `Undo: ${historyEntries[historyIndex]?.label} (Ctrl+Z)` : 'Nothing to undo'}
        aria-label="Undo"
        data-testid="undo-btn"
      >
        <Undo2 className="w-4 h-4" />
      </button>
      <button
        onClick={() => setHistoryOpen(!historyOpen)}
        className={`${iconButton} ${historyOpen ? 'bg-gray-700 text-gray-100' : ''}`}
        title="History"
        aria-label="History"
        aria-expanded={historyOpen}
        data-testid="history-open-btn"
      >
        <History className="w-4 h-4" />
      </button>
      <button
        onClick={() => redo()}
        disabled={!canRedo}
        className={iconButton}
        title={canRedo ? `Redo: ${historyEntries[historyIndex + 1]?.label} (Ctrl+Shift+Z)` : 'Nothing to redo'}
        aria-label="Redo"
        data-testid="redo-btn"
      >
        <Redo2 className="w-4 h-4" />
      </button>
    </div>
  )
}

/** Project name (used for the saved file name) and an unsaved-changes dot. */
const ProjectTitle: React.FC = () => {
  const projectName = useAppStore((s) => s.projectName)
  const setProjectName = useAppStore((s) => s.setProjectName)
  const saved = useAppStore((s) => s.savedProjectFingerprint)
  const fingerprint = useAppStore((s) => currentProjectFingerprint(s))
  const hasContent = useAppStore((s) => !!s.inputImage || s.colorSliders.length > 0)
  const dirty = hasUnsavedChanges(saved, fingerprint, hasContent)

  return (
    <div className="flex items-center gap-1 min-w-[4.5rem] w-40 shrink">
      <input
        value={projectName}
        onChange={(e) => setProjectName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === 'Escape') (e.target as HTMLInputElement).blur()
        }}
        placeholder="Untitled project"
        aria-label="Project name"
        className="flex-1 min-w-0 w-full px-1.5 py-1 rounded text-xs bg-transparent border border-transparent hover:border-gray-700 focus:border-gray-600 focus:bg-gray-800 text-gray-100 placeholder:text-gray-500 outline-none truncate"
        data-testid="project-name-input"
      />
      {dirty && (
        <span
          className="w-2 h-2 rounded-full bg-amber-500 flex-shrink-0"
          title="Unsaved changes — File › Save project (Ctrl+S)"
          role="img"
          aria-label="Unsaved changes"
          data-testid="project-dirty-indicator"
        />
      )}
    </div>
  )
}

const LOSS_W = 220
const LOSS_H = 48

/** The running job's progress; click for iteration, loss curve and timing. */
const JobProgress: React.FC<{ elapsedEta: { elapsed: number; eta: number; stalled: boolean } | null }> = ({ elapsedEta }) => {
  const currentJob = useAppStore((s) => s.currentJob)
  const lossHistory = useAppStore((s) => s.lossHistory)
  const lossJobId = useAppStore((s) => s.lossJobId)
  const connection = useAppStore((s) => s.serverConnection)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!currentJob) return null
  const phase = currentJob.status === 'paused' ? 'Paused' : currentJob.phase || 'Optimizing'
  const points = lossJobId === currentJob.job_id ? lossHistory : []
  const bestLoss = points.length ? Math.min(...points.map((p) => p.loss)) : null
  const disconnected = connection !== 'ok'
  // "Stalled" only means something while the server is reachable; a lost
  // connection gets its own, louder state (and the banner under the bar).
  const stalled = !disconnected && currentJob.status === 'running' && !!elapsedEta?.stalled
  const timing = currentJob.status === 'running' && elapsedEta
    ? `${formatDuration(elapsedEta.elapsed)}${!elapsedEta.stalled && elapsedEta.eta > 0 ? ` · ${formatDuration(elapsedEta.eta)} left` : ''}`
    : null
  const barColor = disconnected ? '#6b7280' : currentJob.status === 'paused' ? '#eab308' : '#3b82f6'

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={`flex items-center gap-2 px-3 py-1 rounded whitespace-nowrap ${
          connection === 'lost' ? 'bg-red-600/15 border border-red-600/40 hover:bg-red-600/25' : 'bg-gray-800 hover:bg-gray-700'
        }`}
        title="Show progress details"
        data-testid="job-progress"
        data-connection={connection}
      >
        {connection === 'lost' ? (
          <WifiOff className="w-3.5 h-3.5 text-red-500" />
        ) : connection === 'reconnecting' ? (
          <Loader2 className="w-3.5 h-3.5 text-amber-500 animate-spin" />
        ) : currentJob.status === 'paused' ? (
          <Pause className="w-3.5 h-3.5 text-yellow-500" />
        ) : (
          <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />
        )}
        <span className={`text-xs ${connection === 'lost' ? 'text-red-400 font-medium' : connection === 'reconnecting' ? 'text-amber-500' : 'text-gray-200'}`} data-testid="job-phase">
          {connection === 'lost' ? 'Disconnected' : connection === 'reconnecting' ? 'Reconnecting…' : phase}
        </span>
        <div className="w-16 xl:w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${currentJob.progress}%`, backgroundColor: barColor }}
          />
        </div>
        <span className="text-xs text-gray-300 tabular-nums">{currentJob.progress.toFixed(1)}%</span>
        <span className="hidden 2xl:inline text-xs text-gray-400 tabular-nums" data-testid="job-iteration-count">
          ({currentJob.iteration}/{currentJob.total_iterations})
        </span>
        {currentJob.loss !== null && (
          <span className="hidden 2xl:inline text-xs text-gray-400 tabular-nums" title="Lower is closer to the input image">
            Loss: {currentJob.loss.toFixed(4)}
          </span>
        )}
        {timing && !disconnected && (
          <span className="text-xs text-gray-400 tabular-nums" data-testid="job-elapsed-eta">
            {timing}
          </span>
        )}
        {stalled && (
          <span
            className="flex items-center gap-1 text-xs text-amber-500"
            title="No progress has arrived for over 30 seconds. Some steps are slow, but if this lasts, check the server's console."
            data-testid="job-stalled"
          >
            <AlertTriangle className="w-3 h-3" />
            No progress
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-64 z-50 p-3 rounded border border-gray-700 bg-gray-800 shadow-lg text-xs text-gray-300 space-y-2" data-testid="job-progress-details">
          <div className="flex justify-between"><span className="text-gray-400">Phase</span><span className="text-gray-100">{phase}</span></div>
          <div className="flex justify-between"><span className="text-gray-400">Iteration</span><span className="text-gray-100 tabular-nums">{currentJob.iteration} / {currentJob.total_iterations}</span></div>
          <div className="flex justify-between" title="Lower is closer to the input image">
            <span className="text-gray-400">Loss</span>
            <span className="text-gray-100 tabular-nums" data-testid="job-loss">{currentJob.loss !== null ? currentJob.loss.toFixed(4) : '—'}</span>
          </div>
          {bestLoss !== null && (
            <div className="flex justify-between"><span className="text-gray-400">Lowest so far</span><span className="text-gray-100 tabular-nums">{bestLoss.toFixed(4)}</span></div>
          )}
          {timing && <div className="flex justify-between"><span className="text-gray-400">Time</span><span className="text-gray-100 tabular-nums">{timing}</span></div>}
          {disconnected && (
            <div className="flex justify-between"><span className="text-gray-400">Server</span><span className={connection === 'lost' ? 'text-red-400' : 'text-amber-500'}>{connection === 'lost' ? 'not reachable' : 'reconnecting…'}</span></div>
          )}
          <div>
            <div className="text-[11px] text-gray-400 mb-1">Loss over the run</div>
            {points.length >= 2 ? (
              <svg width={LOSS_W} height={LOSS_H} className="block rounded bg-gray-900" data-testid="loss-sparkline">
                <path d={sparklinePath(points, LOSS_W, LOSS_H - 4)} transform="translate(0,2)" fill="none" stroke="#3b82f6" strokeWidth="1.5" />
              </svg>
            ) : (
              <p className="text-[11px] text-gray-500">Appears after a few progress updates.</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/** Done / out of date, and what pruning did — one status area. */
const ResultStatus: React.FC = () => {
  const currentJob = useAppStore((s) => s.currentJob)
  const settings = useAppStore((s) => s.settings)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const runInputsByJob = useAppStore((s) => s.runInputsByJob)
  const pruningJob = useAppStore((s) => s.pruningJob)
  const pruningBaseline = useAppStore((s) => s.pruningBaseline)
  const colorSliders = useAppStore((s) => s.colorSliders)

  const reasons = useMemo(
    () => (currentJob ? staleReasons(runInputsByJob[currentJob.job_id], settings as unknown as Record<string, unknown>, activeFilaments.map((f) => f.uuid)) : []),
    [currentJob, runInputsByJob, settings, activeFilaments],
  )
  const pruneSummary = useMemo(() => {
    if (!pruningBaseline) return null
    return describePruningChange(pruningBaseline, resultCounts(buildPrintPlan(colorSliders, [], settings)))
  }, [pruningBaseline, colorSliders, settings])

  const stale = reasons.length > 0
  const pruned = pruningJob?.status === 'completed'
  const pruneCancelled = pruningJob?.status === 'cancelled'

  return (
    <div
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded border ${stale ? 'bg-amber-500/15 border-amber-500/40' : 'bg-emerald-600/15 border-emerald-600/40'}`}
      data-testid="result-status"
    >
      {stale ? (
        <span
          className="flex items-center gap-1.5 text-xs font-medium text-amber-500"
          title={`Changed since this run: ${reasons.map(settingName).join(', ')}. Run again to update the result.`}
          data-testid="job-stale-badge"
        >
          <AlertTriangle className="w-3.5 h-3.5" />
          Out of date
        </span>
      ) : (
        <span className="flex items-center gap-1.5" data-testid="job-done-badge">
          <Check className="w-3.5 h-3.5 text-emerald-500" />
          <span className="text-xs font-medium text-emerald-600">Done</span>
        </span>
      )}
      {pruned && (
        <span className="flex items-center gap-1 text-xs text-purple-500" title={pruneSummary ? `Pruned: ${pruneSummary}` : 'Pruned'} data-testid="pruning-done">
          <span aria-hidden className="text-gray-500">·</span>
          <Scissors className="w-3 h-3" />
          Pruned
        </span>
      )}
      {pruneCancelled && (
        <span className="text-xs text-yellow-600" data-testid="pruning-cancelled">
          <span aria-hidden className="text-gray-500">· </span>
          Pruning cancelled (partial result kept)
        </span>
      )}
    </div>
  )
}

export const TopBar: React.FC = () => {
  const setSettingsModalOpen = useAppStore((s) => s.setSettingsModalOpen)
  const setTutorialOpen = useAppStore((s) => s.setTutorialOpen)
  const theme = useAppStore((s) => s.theme)
  const toggleTheme = useAppStore((s) => s.toggleTheme)
  const currentJob = useAppStore((s) => s.currentJob)
  const startOptimization = useAppStore((s) => s.startOptimization)
  const pauseOptimization = useAppStore((s) => s.pauseOptimization)
  const resumeOptimization = useAppStore((s) => s.resumeOptimization)
  const cancelOptimization = useAppStore((s) => s.cancelOptimization)
  const setPruningModalOpen = useAppStore((s) => s.setPruningModalOpen)
  const pruningJob = useAppStore((s) => s.pruningJob)
  const pausePruning = useAppStore((s) => s.pausePruning)
  const resumePruning = useAppStore((s) => s.resumePruning)
  const cancelPruning = useAppStore((s) => s.cancelPruning)
  const activeFilaments = useAppStore((s) => s.activeFilaments)
  const inputImage = useAppStore((s) => s.inputImage)
  const initStatus = useAppStore((s) => s.initState.status)
  const settings = useAppStore((s) => s.settings)
  const runInputsByJob = useAppStore((s) => s.runInputsByJob)
  const pushToast = useAppStore((s) => s.pushToast)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const serverLost = useAppStore((s) => s.serverConnection === 'lost')

  const trackerRef = useRef<ProgressTracker | null>(null)
  const runButtonRef = useRef<HTMLButtonElement>(null)
  const [runFlash, flashRun] = useFlash()
  const [elapsedEta, setElapsedEta] = useState<{ elapsed: number; eta: number; stalled: boolean } | null>(null)
  const [startError, setStartError] = useState<string | null>(null)
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null)
  // Read from /api/system/version (pyproject.toml), never a hardcoded literal.
  const [currentVersion, setCurrentVersion] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/system/version')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { if (data?.version) setCurrentVersion(data.version) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    // Delayed so this non-urgent GitHub call doesn't compete with the page's
    // own startup requests.
    const timer = setTimeout(() => {
      fetch('/api/system/update-check')
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => { if (data?.update_available) setUpdateInfo(data) })
        .catch(() => {})
    }, 4000)
    return () => clearTimeout(timer)
  }, [])

  // The "Run" workflow step points at the Run button.
  useEffect(
    () =>
      onUiCommand('focus-run', () => {
        runButtonRef.current?.focus()
        flashRun()
      }),
    [flashRun],
  )

  const hasResult = currentJob?.status === 'completed'
  const resultStale = hasResult && staleReasons(
    runInputsByJob[currentJob.job_id],
    settings as unknown as Record<string, unknown>,
    activeFilaments.map((f) => f.uuid),
  ).length > 0
  const previewBuilding = initStatus === 'initializing'
  const canRun = activeFilaments.length > 0 && !!inputImage && !previewBuilding
  // Run is disabled while the auto-preview is still being built: both are GPU
  // jobs, and starting a run in the middle used to make them compete.
  const runDisabledReason = !inputImage
    ? 'Upload an input image first'
    : activeFilaments.length === 0
      ? 'Add at least one active filament first'
      : previewBuilding
        ? 'Preparing the preview…'
        : ''
  const canPrune = hasResult

  const jobId = currentJob?.job_id
  const isActive = currentJob?.status === 'running' || currentJob?.status === 'paused' || currentJob?.status === 'pending'

  // The websocket pushes a fresh `currentJob` object on every progress
  // update — far more often than once a second, and at irregular intervals
  // (a burst of iterations can land within milliseconds of each other).
  // `latestProgressRef` decouples "how often we hear about progress" from
  // "how often we sample it": only the ref updates on every message, while
  // the tracker itself is fed from a steady 1s interval below.
  const latestProgressRef = useRef(0)
  useEffect(() => {
    latestProgressRef.current = (currentJob?.progress || 0) / 100
  }, [currentJob?.progress])

  useEffect(() => {
    if (!isActive || !jobId) {
      trackerRef.current = null
      setElapsedEta(null)
      return
    }
    // Regression: this used to depend on `currentJob` itself, so the
    // interval was torn down and rebuilt on every single progress message —
    // `createProgressTracker`'s EMA assumes evenly-spaced ~1s samples to
    // smooth over, and instead got fed samples at whatever irregular
    // cadence the websocket happened to deliver at, which is what made the
    // "Xs left" estimate visibly jump around every iteration. Keying this
    // effect on the stable (isActive, jobId) pair instead means the
    // interval — and its 1s cadence — survives for the whole run; only
    // `latestProgressRef` above tracks each new message.
    trackerRef.current = createProgressTracker()
    const tick = () => {
      const tracker = trackerRef.current
      if (!tracker) return
      const state = tracker.update(latestProgressRef.current, Date.now())
      setElapsedEta({ elapsed: state.elapsed, eta: state.eta, stalled: state.stalled })
    }
    tick()
    const interval = setInterval(tick, 1000)
    return () => clearInterval(interval)
  }, [isActive, jobId])

  // A "can't start" message is about the state at click time; once that's
  // fixed it shouldn't linger next to an enabled Run button.
  useEffect(() => {
    setStartError(null)
  }, [canRun, currentJob?.status])

  const handleStart = async () => {
    setStartError(null)
    if (currentJob?.status !== 'paused' && !canRun) {
      setStartError(runDisabledReason)
      return
    }
    try {
      if (currentJob?.status === 'paused') {
        await resumeOptimization(currentJob.job_id)
        return
      }
      const { slidersEditedByHand, colorSliders } = useAppStore.getState()
      if (slidersEditedByHand && colorSliders.some((s) => s.enabled && s.filament_uuid)) {
        const ok = await requestConfirm({
          title: 'Replace your color layers?',
          message: 'Running the optimizer replaces the color layers you set up by hand with its own result.\nYou can get yours back with Undo.',
          confirmLabel: 'Run anyway',
        })
        if (!ok) return
      }
      await startOptimization()
    } catch (e) {
      console.error('Failed to start:', e)
      setStartError(e instanceof Error ? e.message : 'Failed to start optimization')
    }
  }

  const handlePause = async () => {
    if (currentJob?.status !== 'running') return
    try {
      await pauseOptimization(currentJob.job_id)
    } catch (e) {
      pushToast(`Failed to pause: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleCancel = async () => {
    if (!currentJob) return
    try {
      await cancelOptimization(currentJob.job_id)
    } catch (e) {
      // With the server gone there is nothing to cancel any more; stop
      // showing a run that can't be followed instead of an error.
      if (useAppStore.getState().serverConnection === 'lost') {
        useAppStore.getState().setCurrentJob(null)
        pushToast('The server can\'t be reached, so the run was cleared from this page.', 'info')
        return
      }
      pushToast(`Failed to cancel: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const withToast = (action: string, fn: () => Promise<void>) => async () => {
    try {
      await fn()
    } catch (e) {
      pushToast(`Failed to ${action}: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  // Run is the thing to do next unless there's an up-to-date result.
  const runIsPrimary = !hasResult || resultStale
  const flashRing = runFlash ? 'ring-2 ring-cyan-500 ring-offset-1 ring-offset-gray-900' : ''

  return (
    <div
      className="flex items-center justify-between gap-3 px-3 bg-gray-900 border-b border-gray-700 flex-shrink-0"
      style={{ height: 'var(--topbar-height)' }}
      data-testid="top-bar"
    >
      <div className="flex items-center gap-3 min-w-0">
        <h1 className="text-sm font-bold text-gray-100 flex-shrink-0">AutoForge</h1>
        {/* Hidden during a run below very wide screens: the progress readout needs the room. */}
        {currentVersion && (
          <span className={`${isActive ? 'hidden 2xl:inline' : ''} text-xs text-gray-500 flex-shrink-0`} data-testid="app-version">v{currentVersion}</span>
        )}
        <FileMenu />
        <ProjectTitle />
        <HistoryControls />
        <WorkflowSteps />
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        {isActive && currentJob && <JobProgress elapsedEta={elapsedEta} />}

        {hasResult && !isActive && <ResultStatus />}

        {(pruningJob?.status === 'running' || pruningJob?.status === 'pending' || pruningJob?.status === 'paused') && (
          <div className="flex items-center gap-2 px-3 py-1 bg-gray-800 rounded" data-testid="pruning-indicator">
            {pruningJob.status === 'paused' ? (
              <Pause className="w-3.5 h-3.5 text-yellow-500" />
            ) : (
              <Loader2 className="w-3.5 h-3.5 text-purple-400 animate-spin" />
            )}
            <div className="w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${pruningJob.progress}%`, backgroundColor: pruningJob.status === 'paused' ? '#eab308' : '#a855f7' }}
              />
            </div>
            <span className="text-xs text-gray-200" data-testid="pruning-phase">
              {pruningJob.phase ? `${pruningJob.phase} ` : 'Pruning '}
              {pruningJob.progress.toFixed(1)}%
            </span>
            {pruningJob.status === 'running' ? (
              <button onClick={withToast('pause pruning', () => pausePruning(pruningJob.job_id))} className="p-1 text-yellow-500 hover:text-yellow-400" title="Pause pruning" aria-label="Pause pruning" data-testid="prune-pause-btn">
                <Pause className="w-3.5 h-3.5" />
              </button>
            ) : pruningJob.status === 'paused' ? (
              <button onClick={withToast('resume pruning', () => resumePruning(pruningJob.job_id))} className="p-1 text-blue-400 hover:text-blue-300" title="Resume pruning" aria-label="Resume pruning" data-testid="prune-resume-btn">
                <Play className="w-3.5 h-3.5" />
              </button>
            ) : null}
            <button onClick={withToast('cancel pruning', () => cancelPruning(pruningJob.job_id))} className="p-1 text-red-400 hover:text-red-300" title="Cancel pruning" aria-label="Cancel pruning" data-testid="prune-cancel-btn">
              <Square className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {startError ? (
          <span className="text-xs text-red-500 max-w-72 truncate" title={startError} data-testid="start-error">
            {startError}
          </span>
        ) : (
          !isActive && !canRun && (
            <span className="text-xs text-gray-400" data-testid="run-disabled-reason">
              {runDisabledReason}
            </span>
          )
        )}

        {currentJob?.status === 'running' ? (
          <button
            onClick={handlePause}
            disabled={serverLost}
            title={serverLost ? "The server can't be reached" : undefined}
            className={outlineButton('border-yellow-600 text-yellow-500 hover:bg-yellow-600/15')}
            data-testid="top-pause-btn"
          >
            <Pause className="w-3.5 h-3.5" />
            Pause
          </button>
        ) : currentJob?.status === 'paused' ? (
          <button ref={runButtonRef} onClick={handleStart} className={`${primaryButton} ${flashRing}`} data-testid="top-resume-btn">
            <Play className="w-3.5 h-3.5" />
            Resume
          </button>
        ) : (
          <button
            ref={runButtonRef}
            onClick={handleStart}
            disabled={!canRun || currentJob?.status === 'pending'}
            title={canRun ? (resultStale ? 'Settings changed since this result — run again to update it' : hasResult ? 'Run the optimizer again' : 'Let the optimizer pick colors and heights') : runDisabledReason}
            className={`${runIsPrimary ? primaryButton : outlineButton('border-gray-600 text-gray-200 hover:bg-gray-800')} ${flashRing}`}
            data-testid="top-start-btn"
            data-primary={runIsPrimary}
          >
            <Play className="w-3.5 h-3.5" />
            {hasResult ? 'Run again' : 'Run'}
          </button>
        )}

        {isActive && (
          <button
            onClick={handleCancel}
            title={serverLost ? 'Stop showing this run (the server that ran it is gone)' : undefined}
            className={outlineButton('border-red-600 text-red-500 hover:bg-red-600/15')}
            data-testid="top-cancel-btn"
          >
            <Square className="w-3.5 h-3.5" />
            Cancel
          </button>
        )}

        {/* Nothing to prune while a run is going (it's disabled then), and
            the bar needs the room for the progress readout. */}
        {!isActive && (
        <button
          onClick={() => setPruningModalOpen(true)}
          disabled={!canPrune}
          title={canPrune ? 'Reduce colors, swaps and layers of this result' : 'Run an optimization first'}
          className={outlineButton('border-purple-600 text-purple-500 hover:bg-purple-600/15')}
          data-testid="top-pruning-btn"
        >
          <Scissors className="w-3.5 h-3.5" />
          Pruning
        </button>
        )}

        {updateInfo && (
          <a
            href={updateInfo.release_url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 px-2 py-1.5 text-xs text-emerald-500 hover:bg-gray-800 rounded"
            title={`AutoForge ${updateInfo.latest_version} is available (current: ${updateInfo.current_version})`}
            data-testid="update-available-badge"
          >
            <ArrowUpCircle className="w-3.5 h-3.5" />
            Update available
          </a>
        )}

        <button
          onClick={toggleTheme}
          className={iconButton}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          data-testid="theme-toggle-btn"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <button
          onClick={() => setTutorialOpen(true)}
          className={iconButton}
          title="How AutoForge works — the getting-started tour"
          aria-label="Open the tutorial"
          data-testid="tutorial-open-btn"
        >
          <HelpCircle className="w-4 h-4" />
        </button>

        <button
          onClick={() => setSettingsModalOpen(true)}
          className={iconButton}
          title="Settings"
          aria-label="Settings"
          data-testid="settings-button"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      <PruningModal />
    </div>
  )
}
