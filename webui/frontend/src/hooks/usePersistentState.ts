import { useCallback, useEffect, useState } from 'react'

/** useState that remembers its value in localStorage (a per-browser
 * convenience; falls back to the default when storage is unavailable). */
export function usePersistentState<T extends string | number | boolean>(key: string, fallback: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      if (raw === null) return fallback
      if (typeof fallback === 'number') {
        const n = Number(raw)
        return (Number.isFinite(n) ? n : fallback) as T
      }
      if (typeof fallback === 'boolean') return (raw === '1') as T
      return raw as T
    } catch {
      return fallback
    }
  })
  const set = useCallback(
    (next: T) => {
      setValue(next)
      try {
        localStorage.setItem(key, typeof next === 'boolean' ? (next ? '1' : '0') : String(next))
      } catch {
        // not remembered — fine
      }
    },
    [key],
  )
  return [value, set]
}

/** Height of an element, kept up to date. */
export function useElementHeight(ref: React.RefObject<HTMLElement>): number {
  const [height, setHeight] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setHeight(el.clientHeight)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return height
}

/** Width of an element, kept up to date. */
export function useElementWidth(ref: React.RefObject<HTMLElement>): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setWidth(el.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return width
}

/** Briefly true after `trigger()` — for pointing at the panel a click refers to. */
export function useFlash(ms = 1200): [boolean, () => void] {
  const [on, setOn] = useState(false)
  useEffect(() => {
    if (!on) return
    const t = setTimeout(() => setOn(false), ms)
    return () => clearTimeout(t)
  }, [on, ms])
  return [on, useCallback(() => setOn(true), [])]
}
