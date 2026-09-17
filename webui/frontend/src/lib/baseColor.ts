/** Which color the base/background slab actually is.
 *
 * `settings.background_color` is only half the answer. With
 * `auto_background_color` on (the default), the pipeline replaces it with
 * whichever active filament is closest to the image's dominant color, and
 * that choice lives in the pipeline result — the frontend learns it from
 * `GET /api/sliders/base` (`resolvedBase`). Reading settings alone therefore
 * showed a base color the print does not use: the swap instructions and the
 * print plan named one color while the model was built with another.
 */

export interface BaseSettings {
  background_color: string
  background_height: number
  layer_height: number
  auto_background_color: boolean
}

export interface ResolvedBaseInfo {
  color: string
  height_mm: number
  layers: number
  filament_uuid: string
  auto: boolean
}

export const DEFAULT_BASE_COLOR = '#000000'

export function effectiveBaseColor(
  settings: Pick<BaseSettings, 'background_color'>,
  resolvedBase: Pick<ResolvedBaseInfo, 'color'> | null | undefined,
): string {
  return resolvedBase?.color || settings.background_color || DEFAULT_BASE_COLOR
}

/** The library filament the base color belongs to, if any ('' otherwise). */
export function effectiveBaseFilamentUuid(
  resolvedBase: Pick<ResolvedBaseInfo, 'filament_uuid'> | null | undefined,
): string {
  return resolvedBase?.filament_uuid ?? ''
}

/** Base thickness in whole layers. Derived from the height and the layer
 * height rather than read back from anywhere, so it stays right when either
 * one changes. */
export function baseLayerCount(settings: Pick<BaseSettings, 'background_height' | 'layer_height'>): number {
  const layerHeight = settings.layer_height || 0.04
  return Math.round((settings.background_height || 0) / layerHeight)
}

/** Whether the base color is still being picked automatically. */
export function baseIsAuto(
  settings: Pick<BaseSettings, 'auto_background_color'>,
  resolvedBase: Pick<ResolvedBaseInfo, 'auto'> | null | undefined,
): boolean {
  return resolvedBase ? resolvedBase.auto : !!settings.auto_background_color
}

/** `settings` with the base color the print really uses, for anything that
 * reads `background_color` out of settings (the print plan, the exported
 * swap instructions). */
export function withEffectiveBaseColor<T extends Pick<BaseSettings, 'background_color'>>(
  settings: T,
  resolvedBase: Pick<ResolvedBaseInfo, 'color'> | null | undefined,
): T {
  const color = effectiveBaseColor(settings, resolvedBase)
  return color === settings.background_color ? settings : { ...settings, background_color: color }
}
