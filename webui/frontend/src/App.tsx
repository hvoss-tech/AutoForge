import React, { useEffect, useRef } from 'react'
import { ChevronsRight, History, X } from 'lucide-react'
import { TopBar } from './components/TopBar'
import { FilamentLibrary } from './components/FilamentLibrary'
import { InputImagePanel } from './components/InputImagePanel'
import { Preview3DPanel } from './components/Preview3DPanel'
import { ColorCore } from './components/ColorCore'
import { BottomPanel } from './components/BottomPanel'
import { HistoryDrawer } from './components/HistoryDrawer'
import { ConfirmDialog } from './components/ConfirmDialog'
import { StatusBar } from './components/StatusBar'
import { SettingsModal } from './components/SettingsModal'
import { NewFilamentModal } from './components/NewFilamentModal'
import { ImportModal } from './components/ImportModal'
import { TutorialModal } from './components/TutorialModal'
import { ToastContainer } from './components/ToastContainer'
import { ResizeHandle } from './components/ui/resize-handle'
import { useAppStore } from './store/appStore'
import { useJobWebSocket } from './hooks/useJobWebSocket'
import { useAutoPreviewInit } from './hooks/useAutoPreviewInit'
import { useElementHeight, usePersistentState } from './hooks/usePersistentState'
import { historyShortcut } from './lib/history'
import { clampSize } from './lib/layout'
import { onUiCommand } from './lib/uiEvents'

const SIDEBAR_DEFAULT = 330
const SIDEBAR_MIN = 240
const SIDEBAR_MAX = 560
const BOTTOM_DEFAULT = 240
// Tab bar, overview strip, column header and two rows stay visible.
const BOTTOM_MIN = 140
const VIEWER_MIN = 180

/** Shown after a reload picked the previous session back up. */
const SessionBanner: React.FC = () => {
  const dismiss = useAppStore((s) => s.dismissSessionRestored)
  const startNewProject = useAppStore((s) => s.startNewProject)
  const requestConfirm = useAppStore((s) => s.requestConfirm)
  const hasResult = useAppStore((s) => s.currentJob?.status === 'completed')
  const hasLayers = useAppStore((s) => s.colorSliders.length > 0)
  const restored = ['the image', ...(hasLayers ? ['color layers'] : []), ...(hasResult ? ['result'] : [])]
  const restoredText = restored.length > 1 ? `${restored.slice(0, -1).join(', ')} and ${restored[restored.length - 1]}` : restored[0]

  const handleNew = async () => {
    const ok = await requestConfirm({
      title: 'Start a new project?',
      message: 'The image, color layers and result are cleared. Settings and active filaments stay.\nYou can get everything back with Undo.',
      confirmLabel: 'Start new project',
    })
    if (ok) startNewProject()
  }

  return (
    <div className="flex items-center gap-2 mx-2 mt-2 px-3 py-1.5 rounded bg-blue-600/10 border border-blue-600/30 text-xs text-gray-200" data-testid="session-restored-banner">
      <History className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
      <span className="flex-1">Picked up where you left off: {restoredText} from your last session.</span>
      <button onClick={handleNew} className="px-2 py-0.5 rounded border border-gray-600 hover:bg-gray-700" data-testid="start-new-project-btn">
        Start new project
      </button>
      <button onClick={dismiss} className="p-0.5 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-700" aria-label="Dismiss" data-testid="session-restored-dismiss">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

const App: React.FC = () => {
  const loadProjectState = useAppStore((s) => s.loadProjectState)
  const loadActiveFilaments = useAppStore((s) => s.loadActiveFilaments)
  const loadCurrentJob = useAppStore((s) => s.loadCurrentJob)
  const restoreSessionImage = useAppStore((s) => s.restoreSessionImage)
  const currentJob = useAppStore((s) => s.currentJob)
  const sessionRestored = useAppStore((s) => s.sessionRestored)

  const [sidebarWidth, setSidebarWidth] = usePersistentState<number>('autoforge-sidebar-width', SIDEBAR_DEFAULT)
  const [sidebarCollapsed, setSidebarCollapsed] = usePersistentState<boolean>('autoforge-sidebar-collapsed', false)
  const [bottomHeight, setBottomHeight] = usePersistentState<number>('autoforge-bottom-panel-height', BOTTOM_DEFAULT)
  const mainRef = useRef<HTMLElement>(null)
  const mainHeight = useElementHeight(mainRef)
  // Up to 70% of the column, and never squeezing the image/3D row below its minimum.
  const bottomMax = mainHeight ? Math.max(BOTTOM_MIN, Math.min(mainHeight * 0.7, mainHeight - VIEWER_MIN)) : 10000
  const effectiveBottom = clampSize(bottomHeight, BOTTOM_MIN, bottomMax)

  // Connect to optimization WebSocket when a job is active
  const activeJobId = currentJob && ['pending', 'running', 'paused'].includes(currentJob.status)
    ? currentJob.job_id
    : null
  useJobWebSocket(activeJobId)
  useAutoPreviewInit()

  useEffect(() => onUiCommand('focus-library', () => setSidebarCollapsed(false)), [setSidebarCollapsed])

  useEffect(() => {
    const loadInitial = async () => {
      // loadProjectState/loadActiveFilaments/loadCurrentJob each swallow their
      // own fetch errors internally (falling back to defaults/empty state) so
      // they never reject — a try/catch around them can never see a backend
      // that isn't up yet. Probe a real endpoint directly instead, so a
      // frontend that loads before the backend is accepting connections
      // (e.g. a Docker Compose stack without a healthcheck dependency)
      // actually retries instead of silently settling into "no project".
      for (let attempt = 0; attempt < 20; attempt++) {
        try {
          const response = await fetch('/api/project/state')
          if (response.ok) break
        } catch {
          // Backend not reachable yet – retry
        }
        await new Promise(r => setTimeout(r, 1000))
      }
      await loadProjectState()
      await loadActiveFilaments()
      await loadCurrentJob()
      // After the job is known: whether the image needs a new preview depends on it.
      await restoreSessionImage()
    }
    loadInitial()
  }, [loadProjectState, loadActiveFilaments, loadCurrentJob, restoreSessionImage])

  // Last-resort safety net: an uncaught exception or unhandled promise
  // rejection anywhere in the app used to just vanish into the browser
  // console — the page kept running in whatever broken state caused it,
  // with nothing telling the user something failed.
  useEffect(() => {
    const pushToast = useAppStore.getState().pushToast
    const onError = (e: ErrorEvent) => {
      pushToast(`Unexpected error: ${e.message}`)
    }
    const onRejection = (e: PromiseRejectionEvent) => {
      const reason = e.reason instanceof Error ? e.reason.message : String(e.reason)
      pushToast(`Unexpected error: ${reason}`)
    }
    window.addEventListener('error', onError)
    window.addEventListener('unhandledrejection', onRejection)
    return () => {
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onRejection)
    }
  }, [])

  // Keyboard shortcuts: undo/redo, and Ctrl+S to save the project
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 's') {
        e.preventDefault()
        useAppStore.getState().saveProjectToFile()
        return
      }
      const action = historyShortcut(e)
      if (!action) return
      e.preventDefault()
      if (action === 'undo') useAppStore.getState().undo()
      else useAppStore.getState().redo()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div data-testid="app" style={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh', backgroundColor: 'var(--bg-main)', color: 'var(--text-primary)' }}>
      <TopBar />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {sidebarCollapsed ? (
          <aside className="flex-shrink-0 flex flex-col items-center py-2 w-9" style={{ backgroundColor: 'var(--bg-sidebar)', borderRight: '1px solid var(--border)' }} data-testid="sidebar-collapsed">
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="p-1.5 rounded text-gray-300 hover:text-gray-100 hover:bg-gray-700"
              title="Show the filament library"
              aria-label="Show the filament library"
              data-testid="sidebar-expand-btn"
            >
              <ChevronsRight className="w-4 h-4" />
            </button>
            <span className="mt-2 text-[11px] text-gray-400 [writing-mode:vertical-rl] rotate-180 select-none">Filaments</span>
          </aside>
        ) : (
          <>
            <aside
              style={{ width: clampSize(sidebarWidth, SIDEBAR_MIN, SIDEBAR_MAX), backgroundColor: 'var(--bg-sidebar)', overflow: 'hidden', flexShrink: 0 }}
              data-testid="sidebar"
            >
              <FilamentLibrary onCollapse={() => setSidebarCollapsed(true)} />
            </aside>
            <div style={{ borderRight: '1px solid var(--border)' }} className="flex">
              <ResizeHandle
                direction="column"
                value={clampSize(sidebarWidth, SIDEBAR_MIN, SIDEBAR_MAX)}
                min={SIDEBAR_MIN}
                max={SIDEBAR_MAX}
                onChange={setSidebarWidth}
                onReset={() => setSidebarWidth(SIDEBAR_DEFAULT)}
                label="Filament library width"
                data-testid="sidebar-resizer"
              />
            </div>
          </>
        )}

        <main ref={mainRef} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          {sessionRestored && <SessionBanner />}
          <div style={{ flex: 1, display: 'flex', gap: 8, padding: '8px 8px 0', minHeight: 0 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <InputImagePanel />
            </div>
            <div style={{ width: 88, flexShrink: 0, display: 'flex' }}>
              <ColorCore />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <Preview3DPanel />
            </div>
          </div>

          <ResizeHandle
            direction="row"
            value={effectiveBottom}
            min={BOTTOM_MIN}
            max={bottomMax}
            onChange={setBottomHeight}
            onReset={() => setBottomHeight(BOTTOM_DEFAULT)}
            invert
            label="Color layers panel height"
            data-testid="bottom-panel-resizer"
          />

          <div style={{ height: effectiveBottom, flexShrink: 0, minWidth: 0, overflow: 'hidden', borderTop: '1px solid var(--border)' }} data-testid="bottom-panel-container">
            <BottomPanel />
          </div>
        </main>
      </div>

      <StatusBar />

      <SettingsModal />
      <NewFilamentModal />
      <ImportModal />
      <HistoryDrawer />
      <ConfirmDialog />
      <TutorialModal />
      <ToastContainer />
    </div>
  )
}

export default App
