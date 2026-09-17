/** Guidance for the "STL size" field in the status bar.
 *
 * The field was labelled just "Size", which read as the size of anything —
 * the image, the preview, the print bed. It sets one thing: the longest side
 * of the exported STL.
 *
 * Above LARGE_STL_SIZE_MM the exported mesh gets big enough to cost real
 * VRAM, and the detail bought by going larger is only visible on prints past
 * roughly 350 mm. Because the model comes from an image rather than from a
 * measured object, the same export scales to about twice this size in a
 * slicer before the layer steps start to show — so the practical advice is
 * to export at a sensible size and scale up there.
 */
export const LARGE_STL_SIZE_MM = 200

/** Size past which scaling in the slicer is the better move. */
export const NOTICEABLE_DETAIL_SIZE_MM = 350

export const STL_SIZE_HINT =
  'Length of the longest side of the printed model. It only scales the STL — the optimizer works from the image, ' +
  'so the same result can be printed at roughly double this size before the layer steps become noticeable. ' +
  `Above ${LARGE_STL_SIZE_MM} mm the mesh uses a lot of VRAM, and the extra detail is only visible on prints ` +
  `larger than about ${NOTICEABLE_DETAIL_SIZE_MM} mm.`

/** The warning to show next to the field, or null when the size is fine. */
export function largeStlSizeWarning(size: number): string | null {
  if (!Number.isFinite(size) || size <= LARGE_STL_SIZE_MM) return null
  return (
    `${size} mm builds a large mesh and can use a lot of VRAM. The extra detail only shows on prints above ` +
    `~${NOTICEABLE_DETAIL_SIZE_MM} mm — most people export at ${LARGE_STL_SIZE_MM} mm or less and scale the model ` +
    'up in the slicer instead.'
  )
}
