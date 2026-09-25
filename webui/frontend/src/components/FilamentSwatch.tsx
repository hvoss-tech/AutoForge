import React from 'react'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import { contrastTextColor, normalizeHex } from '../lib/color'
import type { Filament } from '../types'

interface FilamentSwatchProps {
  filament: Filament
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  onClick?: () => void
  onDragStart?: (e: React.DragEvent) => void
  selected?: boolean
  /** Print the TD value inside the swatch. Off where a TD column already
   * shows it next to the swatch. */
  showTd?: boolean
}

const sizeMap = {
  sm: { swatch: 'w-8 h-4', text: 'text-[10px]', label: 'text-[10px]' },
  md: { swatch: 'w-12 h-6', text: 'text-xs', label: 'text-xs' },
  lg: { swatch: 'w-16 h-8', text: 'text-sm', label: 'text-sm' },
}

export const FilamentSwatch: React.FC<FilamentSwatchProps> = ({
  filament,
  size = 'md',
  showLabel = true,
  onClick,
  onDragStart,
  selected = false,
  showTd = true,
}) => {
  const sizeStyle = sizeMap[size]
  const textColor = contrastTextColor(filament.color)
  const label = filament.name || normalizeHex(filament.color)

  const widthRem = parseInt(sizeStyle.swatch.match(/w-(\d+)/)?.[1] ?? '12', 10) * 4
  const heightRem = parseInt(sizeStyle.swatch.match(/h-(\d+)/)?.[1] ?? '6', 10) * 4

  const swatch = (
    <div
      className={`relative flex items-center justify-center rounded cursor-default select-none ${
        selected ? 'ring-2 ring-primary ring-offset-1 ring-offset-background' : ''
      }`}
      style={{
        backgroundColor: filament.color,
        width: `${widthRem}px`,
        height: `${heightRem}px`,
        borderRadius: '4px',
        boxShadow: 'inset 0 0 0 1px rgba(128,128,128,0.35)',
        overflow: 'hidden',
        flexShrink: 0,
      }}
      draggable={!!onDragStart}
      onDragStart={onDragStart}
      onClick={onClick}
    >
      {showTd && filament.td !== undefined && (
        <span
          className={`font-mono font-bold ${sizeStyle.text}`}
          style={{ color: textColor }}
        >
          {filament.td}
        </span>
      )}
    </div>
  )

  if (!showLabel) {
    // Relies on the single TooltipProvider mounted once at the app root
    // (App.tsx): a provider per swatch here tore its portal down
    // independently from React's own unmount of the row whenever the
    // filament list re-rendered (search/sort/brand collapse), racing
    // Radix's Presence cleanup against React's commit and throwing
    // "insertBefore"/"removeChild" NotFoundErrors.
    return (
      <Tooltip>
        <TooltipTrigger asChild>{swatch}</TooltipTrigger>
        <TooltipContent side="top" className="text-xs">
          <div>{label}</div>
          <div>TD: {filament.td?.toFixed(1)}</div>
          <div>{normalizeHex(filament.color)}</div>
        </TooltipContent>
      </Tooltip>
    )
  }

  return (
    <div className="flex items-center gap-2">
      {swatch}
      {showLabel && (
        <div className="flex flex-col min-w-0">
          <span className={`truncate ${sizeStyle.label} text-foreground`}>
            {label}
          </span>
          {filament.brand && (
            <span className="text-[10px] text-muted-foreground truncate">
              {filament.brand} · {filament.filament_type}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
