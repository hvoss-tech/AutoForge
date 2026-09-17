import React from 'react'
import { parseNumberDraft, settleNumberDraft } from '../../lib/numberDraft'

type NativeProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type' | 'min' | 'max'>

export interface NumberInputProps extends NativeProps {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
  integer?: boolean
}

/** A number field you can actually type into: the text is kept as typed
 * while focused, valid in-range values are committed as you go, and on blur
 * the field settles on a clamped value (or reverts if it isn't a number). */
export const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ value, onValueChange, min, max, integer, onBlur, onFocus, ...rest }, ref) => {
    const rules = { min, max, integer }
    const [draft, setDraft] = React.useState(String(value))
    const focused = React.useRef(false)

    // Follow outside changes (undo, loading a project, another control) —
    // but not while the user is mid-edit, unless the value really differs
    // from what they've typed.
    React.useEffect(() => {
      if (!focused.current || parseNumberDraft(draft, rules) !== value) setDraft(String(value))
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [value])

    return (
      <input
        {...rest}
        ref={ref}
        type="number"
        min={min}
        max={max}
        value={draft}
        onFocus={(e) => {
          focused.current = true
          onFocus?.(e)
        }}
        onChange={(e) => {
          setDraft(e.target.value)
          const n = parseNumberDraft(e.target.value, rules)
          if (n !== null && n !== value) onValueChange(n)
        }}
        onBlur={(e) => {
          focused.current = false
          const settled = settleNumberDraft(e.target.value, value, rules)
          setDraft(String(settled))
          if (settled !== value) onValueChange(settled)
          onBlur?.(e)
        }}
      />
    )
  },
)
NumberInput.displayName = 'NumberInput'
