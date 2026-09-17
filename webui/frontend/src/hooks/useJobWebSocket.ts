import { useEffect, useRef, useCallback } from 'react'
import type { JobStatus } from '../types'
import { useAppStore } from '../store/appStore'

const MAX_RECONNECT_ATTEMPTS = 5
const BASE_RECONNECT_DELAY = 500 // ms

export function useJobWebSocket(jobId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)
  const setCurrentJob = useAppStore((s) => s.setCurrentJob)

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

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const data: JobStatus = JSON.parse(event.data)
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
          ws.close()
        }
      } catch {
        // ignore non-JSON messages
      }
    }

    const handleClose = () => {
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
      }
    }

    ws.onclose = handleClose
    ws.onerror = handleClose
  }, [jobId, setCurrentJob])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.onmessage = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [jobId, connect])

  return wsRef.current
}

export default useJobWebSocket
