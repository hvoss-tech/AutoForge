import React, { useEffect, useRef } from 'react'
import { Frame, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import {
  BORDER_HANGERS,
  BORDER_SHAPES,
  BORDER_WIDTH_RANGE,
  HANGER_DIAMETER_RANGE,
  borderActive,
  describeBorder,
  formatPrintSize,
  maxCornerRadius,
  normalizeBorder,
  type BorderSettings,
  type BorderShape,
} from '../lib/border'
import type { BorderPreview } from '../hooks/useBorderPreview'
import { NumberInput } from './ui/number-input'
import { Slider } from './ui/slider'

/** Small outline of each border shape around a picture. */
const ShapeIcon: React.FC<{ shape: BorderShape }> = ({ shape }) => (
  <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} aria-hidden>
    <rect x="8" y="7" width="8" height="10" fill="currentColor" opacity={0.35} stroke="none" />
    {shape === 'rectangle' && <rect x="4" y="3" width="16" height="18" rx="2.5" />}
    {shape === 'oval' && <ellipse cx="12" cy="12" rx="8.5" ry="10" />}
    {shape === 'circle' && <circle cx="12" cy="12" r="10" />}
    {shape === 'outline' && <rect x="5.5" y="4.5" width="13" height="15" rx="3.5" strokeDasharray="2 1.5" />}
    {shape === 'none' && <path d="M5 19 19 5" />}
  </svg>
)

const SliderField: React.FC<{
  label: string
  value: number
  min: number
  max: number
  /** Typing may go past the slider's handy range, up to the real limit. */
  inputMax?: number
  step: number
  onChange: (value: number) => void
  testId: string
  disabled?: boolean
  title?: string
}> = ({ label, value, min, max, inputMax, step, onChange, testId, disabled, title }) => (
  <label className={`block ${disabled ? 'opacity-50' : ''}`} title={title}>
    <div className="flex items-center justify-between mb-1">
      <span className="text-[11px] text-gray-300">{label}</span>
      <span className="flex items-center gap-1">
        <NumberInput
          value={value}
          min={min}
          max={inputMax ?? max}
          step={step}
          disabled={disabled}
          onValueChange={onChange}
          className="w-14 px-1 py-0.5 text-[11px] bg-gray-900 border border-gray-600 rounded text-right"
          data-testid={`${testId}-input`}
        />
        <span className="text-[11px] text-gray-500">mm</span>
      </span>
    </div>
    <Slider
      value={[Math.min(value, max)]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={([v]) => onChange(v)}
      aria-label={label}
      data-testid={`${testId}-slider`}
    />
  </label>
)

/** The 3D view's border picker: a button in the panel header that opens a
 * small panel of shape, width, corner and hanging-hole options. Every change
 * shows in the 3D view at once and goes into the exported STL. */
export const BorderControls: React.FC<{
  preview: BorderPreview
  baseColor: string
  available: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
}> = ({ preview, baseColor, available, open, onOpenChange: setOpen }) => {
  const settings = useAppStore((s) => s.settings)
  const setSettings = useAppStore((s) => s.setSettings)
  const ref = useRef<HTMLDivElement>(null)
  const border = normalizeBorder(settings.border)
  const active = borderActive(border)

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, setOpen])

  const update = (changes: Partial<BorderSettings>) => {
    const current = normalizeBorder(useAppStore.getState().settings.border)
    setSettings({ ...useAppStore.getState().settings, border: { ...current, ...changes } })
  }

  const radiusMax = maxCornerRadius(border.width)
  const printSize = formatPrintSize(preview.sizeMm)
  const pictureSize = formatPrintSize(preview.modelSizeMm)
  const summary = describeBorder(border)

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-1 px-2 py-0.5 text-[11px] rounded border ${active ? 'border-cyan-600 bg-cyan-900/40 text-cyan-200' : 'border-gray-600 text-gray-300 hover:bg-gray-700'}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={summary ? `Border: ${summary}` : 'Add a border around the model'}
        data-testid="border-btn"
      >
        <Frame className="w-3.5 h-3.5" />
        {active ? `Border: ${BORDER_SHAPES.find((s) => s.value === border.shape)?.label}` : 'Border'}
        {preview.loading && <Loader2 className="w-3 h-3 animate-spin" />}
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-1 w-72 bg-gray-800 border border-gray-600 rounded shadow-lg z-30 p-3 space-y-3"
          role="dialog"
          aria-label="Border"
          data-testid="border-panel"
        >
          <div>
            <div className="text-[11px] text-gray-400 mb-1.5">Shape</div>
            <div className="grid grid-cols-5 gap-1" role="radiogroup" aria-label="Border shape">
              {BORDER_SHAPES.map((s) => (
                <button
                  key={s.value}
                  role="radio"
                  aria-checked={border.shape === s.value}
                  onClick={() => update({ shape: s.value })}
                  title={s.hint}
                  className={`flex flex-col items-center gap-0.5 py-1 rounded border text-[10px] ${border.shape === s.value ? 'border-cyan-500 bg-cyan-900/40 text-cyan-100' : 'border-gray-700 text-gray-300 hover:bg-gray-700'}`}
                  data-testid={`border-shape-${s.value}`}
                >
                  <ShapeIcon shape={s.value} />
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {active && (
            <>
              <SliderField
                label="Width"
                value={border.width}
                min={BORDER_WIDTH_RANGE.min}
                max={30}
                inputMax={BORDER_WIDTH_RANGE.max}
                step={0.5}
                onChange={(width) => update({ width })}
                testId="border-width"
                title={border.shape === 'oval' || border.shape === 'circle' ? 'The least space between the picture and the edge' : 'Space added around the picture'}
              />
              {border.shape === 'rectangle' && (
                <SliderField
                  label="Corner radius"
                  value={Math.min(border.corner_radius, radiusMax)}
                  min={0}
                  max={radiusMax}
                  step={0.5}
                  onChange={(corner_radius) => update({ corner_radius })}
                  testId="border-radius"
                  title={`Up to ${radiusMax} mm at this width, so the corner never cuts into the picture`}
                />
              )}
              <div>
                <div className="text-[11px] text-gray-400 mb-1.5">Hanging hole</div>
                <div className="grid grid-cols-3 gap-1" role="radiogroup" aria-label="Hanging hole">
                  {BORDER_HANGERS.map((h) => (
                    <button
                      key={h.value}
                      role="radio"
                      aria-checked={border.hanger === h.value}
                      onClick={() => update({ hanger: h.value })}
                      title={h.hint}
                      className={`py-1 rounded border text-[11px] ${border.hanger === h.value ? 'border-cyan-500 bg-cyan-900/40 text-cyan-100' : 'border-gray-700 text-gray-300 hover:bg-gray-700'}`}
                      data-testid={`border-hanger-${h.value}`}
                    >
                      {h.label}
                    </button>
                  ))}
                </div>
              </div>
              {border.hanger !== 'none' && (
                <SliderField
                  label="Hole diameter"
                  value={border.hanger_diameter}
                  min={HANGER_DIAMETER_RANGE.min}
                  max={12}
                  inputMax={HANGER_DIAMETER_RANGE.max}
                  step={0.5}
                  onChange={(hanger_diameter) => update({ hanger_diameter })}
                  testId="border-hole"
                  title="A tab grows out of the border when it is too narrow for the hole"
                />
              )}
            </>
          )}

          <div className="pt-2 border-t border-gray-700 text-[11px] text-gray-400 space-y-1" data-testid="border-info">
            {!available ? (
              <p>Shown on the 3D model once there is one. You can pick a border now.</p>
            ) : preview.error ? (
              <p className="text-red-400" data-testid="border-error">{preview.error}</p>
            ) : (
              printSize && (
                <p data-testid="border-size">
                  Print size <span className="text-gray-200">{printSize}</span>
                  {active && pictureSize && <> (picture {pictureSize})</>}
                </p>
              )
            )}
            <p className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm border border-gray-500" style={{ backgroundColor: baseColor }} />
              Printed with the base, in the base filament. Part of the exported STL; no new run needed.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
