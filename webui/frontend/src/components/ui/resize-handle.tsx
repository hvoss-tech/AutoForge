import React from 'react'
import { clampSize } from '../../lib/layout'

interface ResizeHandleProps {
  /** 'row' sits between stacked panes and drags up/down; 'column' sits
   * between side-by-side panes and drags left/right. */
  direction: 'row' | 'column'
  value: number
  min: number
  max: number
  onChange: (value: number) => void
  onReset: () => void
  /** Dragging towards the start (up/left) grows the pane — for a pane below
   * or to the right of the handle. */
  invert?: boolean
  label: string
  'data-testid'?: string
}

const KEY_STEP = 16

/** Draggable divider between two panes. Also works from the keyboard (arrow
 * keys, Shift for bigger steps, Home/End) and resets on double-click. */
export const ResizeHandle: React.FC<ResizeHandleProps> = ({ direction, value, min, max, onChange, onReset, invert, label, ...rest }) => {
  const drag = React.useRef<{ start: number; value: number } | null>(null)
  const [dragging, setDragging] = React.useState(false)
  const sign = invert ? -1 : 1
  const coord = (e: React.PointerEvent) => (direction === 'row' ? e.clientY : e.clientX)

  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? KEY_STEP * 4 : KEY_STEP
    const growKeys = direction === 'row' ? ['ArrowDown'] : ['ArrowRight']
    const shrinkKeys = direction === 'row' ? ['ArrowUp'] : ['ArrowLeft']
    let next: number | null = null
    if (growKeys.includes(e.key)) next = value + sign * step
    else if (shrinkKeys.includes(e.key)) next = value - sign * step
    else if (e.key === 'Home') next = min
    else if (e.key === 'End') next = max
    if (next === null) return
    e.preventDefault()
    onChange(clampSize(next, min, max))
  }

  return (
    <div
      role="separator"
      aria-orientation={direction === 'row' ? 'horizontal' : 'vertical'}
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={Math.round(min)}
      aria-valuemax={Math.round(max)}
      tabIndex={0}
      title={`${label} — drag to resize, double-click to reset`}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.preventDefault()
        ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
        drag.current = { start: coord(e), value }
        setDragging(true)
      }}
      onPointerMove={(e) => {
        if (!drag.current) return
        onChange(clampSize(drag.current.value + sign * (coord(e) - drag.current.start), min, max))
      }}
      onPointerUp={() => {
        drag.current = null
        setDragging(false)
      }}
      onPointerCancel={() => {
        drag.current = null
        setDragging(false)
      }}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
      className={`group relative flex-shrink-0 flex items-center justify-center outline-none ${
        direction === 'row' ? 'h-2 w-full cursor-row-resize' : 'w-2 h-full cursor-col-resize'
      }`}
      {...rest}
    >
      <div
        className={`rounded-full transition-colors group-hover:bg-cyan-500 group-focus-visible:bg-cyan-500 ${dragging ? 'bg-cyan-500' : 'bg-gray-600'} ${
          direction === 'row' ? 'h-1 w-10' : 'w-1 h-10'
        }`}
      />
    </div>
  )
}
