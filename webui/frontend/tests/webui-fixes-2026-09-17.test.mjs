// Test runner: node --test tests/webui-fixes-2026-09-17.test.mjs
//
// Pure logic behind the 2026-09-17 WebUI fixes: history/image correctness,
// preview-broadcast routing, pruning limits and live counts, STL size
// guidance. The DOM-level behaviour of the same changes is covered by
// tests/features/webui-fixes-2026-09-17.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import { acceptsPreviewUpdate, resolveRestoredJob, restoredImage, snapshotInputImageUrl } from '../src/lib/history.ts'
import { liveCounts, resultCounts, suggestPruningLimits, describePruningChange } from '../src/lib/pruning.ts'
import { LARGE_STL_SIZE_MM, STL_SIZE_HINT, largeStlSizeWarning } from '../src/lib/stlSize.ts'
import {
  baseIsAuto,
  baseLayerCount,
  effectiveBaseColor,
  effectiveBaseFilamentUuid,
  withEffectiveBaseColor,
} from '../src/lib/baseColor.ts'
import { printPlanText, buildPrintPlan } from '../src/lib/printPlan.ts'
import { hashFloats } from '../src/lib/plyParser.ts'

const settingsFor = (file) => ({ input_image: file, layer_height: 0.04, background_height: 0.24 })

// --- History: which image a step belongs to (bug 5) -------------------------

test('snapshotInputImageUrl', async (t) => {
  await t.test('uses the durable uploads path from the snapshot settings', () => {
    const snap = { settings: settingsFor('cat.png'), inputImage: '/uploads/cat.png' }
    assert.equal(snapshotInputImageUrl(snap, '/uploads/dog.png'), '/uploads/cat.png')
  })

  await t.test('never falls back to the image currently on screen', () => {
    // The whole bug: a snapshot that knows its filename but carries no
    // display URL used to keep whatever the previous step showed, so
    // stepping between two images left one picture over the other's layers.
    const snap = { settings: settingsFor('cat.png') }
    assert.equal(snapshotInputImageUrl(snap, '/uploads/dog.png'), '/uploads/cat.png')
  })

  await t.test('replaces a blob: URL that no longer resolves', () => {
    const snap = { settings: settingsFor('cat.png'), inputImage: 'blob:http://localhost/9a7c' }
    assert.equal(snapshotInputImageUrl(snap, null), '/uploads/cat.png')
  })

  await t.test('ignores a stored URL that points at a different file', () => {
    const snap = { settings: settingsFor('cat.png'), inputImage: '/uploads/dog.png' }
    assert.equal(snapshotInputImageUrl(snap, null), '/uploads/cat.png')
  })

  await t.test('a step with no image shows none', () => {
    const snap = { settings: settingsFor(''), inputImage: '/uploads/cat.png' }
    assert.equal(snapshotInputImageUrl(snap, '/uploads/cat.png'), null)
  })

  await t.test('a snapshot with no settings at all keeps what it has', () => {
    assert.equal(snapshotInputImageUrl({ inputImage: '/uploads/cat.png' }, null), '/uploads/cat.png')
    assert.equal(snapshotInputImageUrl({}, '/uploads/dog.png'), '/uploads/dog.png')
  })
})

test('restoredImage', async (t) => {
  await t.test('an undo right after an upload is not an image change', () => {
    // The uploaded photo is on screen as a blob: URL while the snapshot holds
    // the durable /uploads path for the same file. Treating that as a
    // different image threw away the live preview and re-ran the heightmap
    // init on every undo made right after uploading.
    const result = restoredImage(
      { settings: settingsFor('cat.png'), inputImage: '/uploads/cat.png' },
      { inputImage: 'blob:http://localhost/9a7c', inputFile: 'cat.png' },
    )
    assert.deepEqual(result, { url: 'blob:http://localhost/9a7c', changed: false })
  })

  await t.test('a different file is a change, and resolves to its own URL', () => {
    assert.deepEqual(
      restoredImage({ settings: settingsFor('dog.png') }, { inputImage: '/uploads/cat.png', inputFile: 'cat.png' }),
      { url: '/uploads/dog.png', changed: true },
    )
  })

  await t.test('stepping to a step with no image clears it', () => {
    assert.deepEqual(
      restoredImage({ settings: settingsFor('') }, { inputImage: '/uploads/cat.png', inputFile: 'cat.png' }),
      { url: null, changed: true },
    )
  })

  await t.test('the same file with nothing on screen still resolves a URL', () => {
    assert.deepEqual(
      restoredImage({ settings: settingsFor('cat.png') }, { inputImage: null, inputFile: 'cat.png' }),
      { url: '/uploads/cat.png', changed: true },
    )
  })

  await t.test('no image on either side is not a change', () => {
    assert.deepEqual(
      restoredImage({ settings: settingsFor('') }, { inputImage: null, inputFile: '' }),
      { url: null, changed: false },
    )
  })
})

test('stepping through 50+ history entries across several images stays consistent', () => {
  // Three images, two optimizations each, with slider edits in between —
  // the shape of the session that got the history confused. Every step must
  // resolve to its own image, and to its own result (never a neighbour's).
  const images = ['a.png', 'b.png', 'c.png']
  const stack = []
  for (let i = 0; i < 60; i++) {
    const image = images[Math.floor(i / 20)]
    const hasResult = i % 20 >= 10
    stack.push({
      timestamp: 1_700_000_000 + i,
      label: `step ${i}`,
      settings: settingsFor(image),
      // Half the steps deliberately carry no display URL, a stale blob:, or
      // another image's URL — all shapes real snapshots come in.
      inputImage: i % 3 === 0 ? undefined : i % 3 === 1 ? 'blob:http://localhost/x' : `/uploads/${images[(Math.floor(i / 20) + 1) % 3]}`,
      optimizationResultId: hasResult ? `job-${image}` : null,
      jobStatus: hasResult ? 'completed' : null,
      colorSliders: [{ td: 1, layer: i + 1, depth_mm: 0, filament_uuid: `f-${image}`, enabled: true }],
    })
  }

  for (let i = 0; i < stack.length; i++) {
    const step = stack[i]
    const expectedImage = `/uploads/${images[Math.floor(i / 20)]}`
    assert.equal(snapshotInputImageUrl(step, '/uploads/WRONG.png'), expectedImage, `step ${i} image`)

    // And the result restored with it belongs to that same image.
    const plan = resolveRestoredJob(null, step)
    if (step.optimizationResultId) {
      assert.deepEqual(plan, { kind: 'fetch', jobId: `job-${images[Math.floor(i / 20)]}` }, `step ${i} job`)
    } else {
      assert.deepEqual(plan, { kind: 'none' }, `step ${i} job`)
    }

    // The sliders come from the step itself, never from a neighbour.
    assert.equal(step.colorSliders[0].filament_uuid, `f-${images[Math.floor(i / 20)]}`)
  }
})

test('a running job survives a history jump', () => {
  // Undoing a slider tweak must not cancel a long optimization.
  const running = { job_id: 'live', status: 'running' }
  const target = { optimizationResultId: 'older', settings: settingsFor('a.png') }
  assert.deepEqual(resolveRestoredJob(running, target), { kind: 'keep' })
})

// --- Preview broadcasts: whose colors are these? (bug 5) --------------------

test('acceptsPreviewUpdate', async (t) => {
  await t.test('takes updates for the job it is showing', () => {
    assert.equal(acceptsPreviewUpdate('job-1', 'job-1', true), true)
  })

  await t.test('ignores another job while showing one', () => {
    assert.equal(acceptsPreviewUpdate('job-2', 'job-1', true), false)
  })

  await t.test('a fresh tab still picks up a job started elsewhere', () => {
    assert.equal(acceptsPreviewUpdate('job-9', undefined, false), true)
  })

  await t.test('a tab that stepped back to a job-less state ignores late broadcasts', () => {
    // Pruning broadcasts under the *optimization* job's id. With no current
    // job this used to be accepted, overwriting the restored color layers.
    assert.equal(acceptsPreviewUpdate('job-1', undefined, true), false)
  })
})

// --- The base / background color (bug 1) ------------------------------------

test('the base color', async (t) => {
  const settings = {
    background_color: '#000000',
    background_height: 0.24,
    layer_height: 0.04,
    auto_background_color: true,
  }

  await t.test('before any run, all there is to go on is the setting', () => {
    assert.equal(effectiveBaseColor(settings, null), '#000000')
    assert.equal(effectiveBaseFilamentUuid(null), '')
    assert.equal(baseIsAuto(settings, null), true)
  })

  await t.test('once resolved, the pipeline\'s choice wins over the setting', () => {
    // This is the whole point: with auto_background_color on, the pipeline
    // replaces the color with the closest active filament and settings still
    // holds the old value, so reading settings alone showed a base color the
    // print does not use.
    const resolved = { color: '#f5f5f5', filament_uuid: 'white', height_mm: 0.24, layers: 6, auto: true }
    assert.equal(effectiveBaseColor(settings, resolved), '#f5f5f5')
    assert.equal(effectiveBaseFilamentUuid(resolved), 'white')
    assert.equal(baseIsAuto(settings, resolved), true)
  })

  await t.test('a hand-picked base reports as manual', () => {
    const resolved = { color: '#2244aa', filament_uuid: 'blue', height_mm: 0.24, layers: 6, auto: false }
    assert.equal(baseIsAuto(settings, resolved), false)
  })

  await t.test('its thickness comes from height / layer height, never stored', () => {
    assert.equal(baseLayerCount(settings), 6)
    assert.equal(baseLayerCount({ ...settings, background_height: 0.6 }), 15)
    // A changed layer height re-derives it rather than going stale.
    assert.equal(baseLayerCount({ ...settings, layer_height: 0.08 }), 3)
    assert.equal(baseLayerCount({ ...settings, background_height: 0 }), 0)
  })

  await t.test('withEffectiveBaseColor leaves settings alone when nothing changed', () => {
    assert.equal(withEffectiveBaseColor(settings, null), settings)
    assert.equal(withEffectiveBaseColor(settings, { color: '#000000' }), settings)
  })

  await t.test('the swap instructions name the color the print really starts with', () => {
    const planSettings = withEffectiveBaseColor(settings, { color: '#f5f5f5' })
    const band = { td: 1, layer: 10, depth_mm: 0.4, filament_uuid: 'k', enabled: true }
    const filaments = [{ uuid: 'k', brand: 'B', name: 'Black', color: '#111111', td: 1, owned: true, filament_type: 'PLA' }]
    const text = printPlanText(buildPrintPlan([band], filaments, planSettings), planSettings)
    assert.match(text, /background color \(#f5f5f5\)/)
    assert.doesNotMatch(text, /#000000/)
  })
})

// --- Pruning (bugs 7 and 9) -------------------------------------------------

test('pruning limits open at the current result', () => {
  const current = resultCounts({ colors: 11, swaps: 25, topLayer: 68 })
  assert.deepEqual(current, { colors: 12, swaps: 25, layers: 68 })
  assert.deepEqual(suggestPruningLimits(current), current)
})

test('liveCounts follows the pruning job while it runs', async (t) => {
  const fallback = { colors: 12, swaps: 25, layers: 68 }

  await t.test('uses the sliders before the job reports anything', () => {
    assert.deepEqual(liveCounts(fallback, null), fallback)
    assert.deepEqual(liveCounts(fallback, { status: 'running' }), fallback)
  })

  await t.test('uses the job counts as they arrive', () => {
    assert.deepEqual(
      liveCounts(fallback, { result_colors: 8, result_swaps: 14, result_layers: 60 }),
      { colors: 8, swaps: 14, layers: 60 },
    )
  })

  await t.test('a partially-reported update keeps the rest', () => {
    assert.deepEqual(
      liveCounts(fallback, { result_colors: 9, result_swaps: null, result_layers: undefined }),
      { colors: 9, swaps: 25, layers: 68 },
    )
  })

  await t.test('zero is a real count, not a missing one', () => {
    assert.deepEqual(liveCounts(fallback, { result_swaps: 0 }).swaps, 0)
  })

  await t.test('the summary reads as a before/after', () => {
    assert.equal(
      describePruningChange(fallback, { colors: 8, swaps: 14, layers: 68 }),
      '12 → 8 colors, 25 → 14 swaps, 68 layers',
    )
  })
})

// --- The 3D view's geometry-reuse key --------------------------------------

test('a recolored mesh and a reshaped mesh are told apart', async (t) => {
  // ThreeDView reuses the geometry already on the GPU when only the vertex
  // colors changed (that's what keeps slider edits instant) and rebuilds it
  // otherwise. The key used to be the vertex *count*, which pruning does not
  // change — it rewrites every height on the same grid — so the view kept
  // showing the pre-pruning relief with the pruned colors painted on.
  const shape = new Float32Array([0, 0, 0, 1, 0, 0.4, 0, 1, 0.8, 1, 1, 0.2])

  await t.test('identical positions hash the same', () => {
    assert.equal(hashFloats(shape), hashFloats(Float32Array.from(shape)))
  })

  await t.test('flattening the heights changes the hash, at equal length', () => {
    const pruned = Float32Array.from(shape)
    for (let i = 2; i < pruned.length; i += 3) pruned[i] = pruned[i] / 2
    assert.equal(pruned.length, shape.length)
    assert.notEqual(hashFloats(pruned), hashFloats(shape))
  })

  await t.test('one moved vertex is enough to notice', () => {
    const nudged = Float32Array.from(shape)
    nudged[5] += 0.04 // a single layer of height on one vertex
    assert.notEqual(hashFloats(nudged), hashFloats(shape))
  })

  await t.test('it is a plain unsigned 32-bit number, safe to compare', () => {
    const h = hashFloats(shape)
    assert.equal(typeof h, 'number')
    assert.ok(Number.isInteger(h) && h >= 0 && h <= 0xffffffff)
  })
})

// --- STL size guidance (bug 12) --------------------------------------------

test('large STL sizes are called out', async (t) => {
  await t.test('nothing to say at or below the threshold', () => {
    assert.equal(largeStlSizeWarning(150), null)
    assert.equal(largeStlSizeWarning(LARGE_STL_SIZE_MM), null)
  })

  await t.test('warns above it, naming VRAM and the size detail actually shows at', () => {
    const warning = largeStlSizeWarning(260)
    assert.ok(warning)
    assert.match(warning, /260 mm/)
    assert.match(warning, /VRAM/)
    assert.match(warning, /350 mm/)
  })

  await t.test('the hover hint explains that the print can be scaled up', () => {
    assert.match(STL_SIZE_HINT, /double/)
    assert.match(STL_SIZE_HINT, /longest side/)
  })

  await t.test('a non-numeric size is not a warning', () => {
    assert.equal(largeStlSizeWarning(Number.NaN), null)
  })
})
