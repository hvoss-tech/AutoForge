import { useEffect, useRef, useCallback } from 'react'
import type { JobStatus } from '../types'
import { useAppStore } from '../store/appStore'

const MAX_RECONNECT_ATTEMPTS = 5
const BASE_RECONNECT_DELAY = 500 // ms
// Once the WS gives up reconnecting, fall back to plain status polling (the
// same thing pruning already does at 1 Hz) rather than leaving the progress
// bar frozen with no error for the rest of a long run — a laptop sleep or a
// ~20s network blip should not silently stop updates.
const POLL_FALLBACK_INTERVAL = 2000 // ms

export function useJobWebSocket(jobId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const mountedRef = useRef(true)
  const setCurrentJob = useAppStore((s) => s.setCurrentJob)

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const startPolling = useCallback(
    (polledJobId: string) => {
      stopPolling()
      pollTimerRef.current = setInterval(async () => {
        try {
          const response = await fetch(`/api/optimize/status/${polledJobId}`)
          if (!mountedRef.current) return
          if (!response.ok) return
          const data: JobStatus = await response.json()
          setCurrentJob(data)
          if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            stopPolling()
          }
        } catch {
          // Transient network error — keep polling, same as before.
        }
      }, POLL_FALLBACK_INTERVAL)
    },
    [setCurrentJob, stopPolling]
  )

  const connect = useCallback(() => {
    const currentJobId = jobId
    if (!currentJobId) return

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current.close()
      wsRef.current = null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/optimize/${currentJobId}`)
    wsRef.current = ws
    let intentionalClose = false
    // onerror and onclose both fire for a failed connection (a browser
    // always closes after erroring); routing both to the same handler used
    // to consume the reconnect-attempt budget twice per actual failure and
    // could double-schedule a reconnect. Guard so only the first of the
    // pair does anything.
    let closeHandled = false

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
      stopPolling()
    }

    ws.onmessage = (event) => {
      try {
        const data: JobStatus = JSON.parse(event.data)
        // A backend restart mid-run sends this instead of a real status;
        // storing it verbatim put a value outside the JobStatus union into
        // currentJob (and from there into undo snapshots) with nothing in
        // the UI explaining the vanished run.
        if ((data.status as string) === 'not_found') {
          intentionalClose = true
          stopPolling()
          setCurrentJob(null)
          useAppStore.getState().pushToast(
            'Lost track of this job (the server may have restarted).',
            'warning'
          )
          return
        }
        // The in-panel "Optimization failed" banner (Preview3DPanel) only
        // helps if that panel happens to be visible — it's fully covered
        // by the Settings/Pruning modals, which is exactly where a user
        // who just clicked Run is likely looking. A toast surfaces the
        // same failure (including an OOM's message) regardless of what's
        // currently on screen.
        const previousStatus = useAppStore.getState().currentJob?.status
        if (data.status === 'failed' && previousStatus !== 'failed') {
          // Only the summary line(s) — friendly_error_message() (backend)
          // puts the full raw exception after a blank line, which belongs
          // in Preview3DPanel's collapsible "Show details", not stretching
          // a toast across the screen.
          const summary = data.error?.split('\n\n')[0]
          useAppStore.getState().pushToast(
            summary ? `Optimization failed: ${summary}` : 'Optimization failed.'
          )
        }
        setCurrentJob(data)
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
          intentionalClose = true
          stopPolling()
          ws.close()
        }
      } catch {
        // ignore non-JSON messages
      }
    }

    const handleClose = () => {
      if (closeHandled) return
      closeHandled = true
      wsRef.current = null
      if (!mountedRef.current) return
      if (intentionalClose) return
      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          Math.pow(2, reconnectAttemptsRef.current) * BASE_RECONNECT_DELAY,
          10000
        )
        reconnectAttemptsRef.current++
        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) connect()
        }, delay)
      } else {
        // Reconnects exhausted: don't leave the progress bar frozen with no
        // feedback for the rest of a long run — poll instead.
        startPolling(currentJobId)
      }
    }

    ws.onclose = handleClose
    ws.onerror = handleClose
  }, [jobId, setCurrentJob, startPolling, stopPolling])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      stopPolling()
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.onmessage = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [jobId, connect, stopPolling])

  return wsRef.current
}

export default useJobWebSocket
