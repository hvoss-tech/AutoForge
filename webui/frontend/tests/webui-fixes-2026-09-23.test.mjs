// Test runner: node --test tests/webui-fixes-2026-09-23.test.mjs
//
// Pure logic behind the 2026-09-23 WebUI bug review. The DOM-level behavior
// of the same changes (ColorCore drag runaway, EditFilamentModal auto-save
// re-arm, band-drop rejection feedback, FlatForge export) is covered by
// tests/features/webui-fixes-2026-09-23.spec.ts.

import test from 'node:test'
import assert from 'node:assert/strict'
import { acceptsPreviewUpdate, jobBelongsToImage } from '../src/lib/history.ts'

// --- acceptsPreviewUpdate: the pre-run sentinel must always get through
// when there's no current job (bug 7) ---------------------------------------

test('acceptsPreviewUpdate accepts the pre-run sentinel with no current job', async (t) => {
  await t.test('accepted the first time, before any job has ever been tracked', () => {
    assert.equal(acceptsPreviewUpdate('__init__', undefined, false), true)
  })

  await t.test('regression: still accepted once this tab has tracked a job before', () => {
    // hasTrackedAnyJob is a module flag that, before this fix, stayed
    // permanently true after the first job this tab ever saw — even one
    // since completed, cancelled by a new project, or replaced by a fresh
    // upload. A slider-assignment preview made before ever clicking Run
    // (which broadcasts under the '__init__' sentinel) must still reach a
    // tab in that state, or every render after the first job silently
    // stopped updating the panel.
    assert.equal(acceptsPreviewUpdate('__init__', undefined, true), true)
  })

  await t.test('a real job id (not the sentinel) is still gated as before', () => {
    assert.equal(acceptsPreviewUpdate('job-1', undefined, true), false)
  })

  await t.test('the sentinel is ignored once a real job is being shown', () => {
    // A stale pre-run broadcast arriving after Run has started must not
    // clobber the live job's preview.
    assert.equal(acceptsPreviewUpdate('__init__', 'job-1', true), false)
  })
})

// --- jobBelongsToImage: a restored/latest job must match the open image
// (bug 5) ---------------------------------------------------------------

test('jobBelongsToImage', async (t) => {
  await t.test('the same file belongs', () => {
    assert.equal(jobBelongsToImage('cat.png', 'cat.png'), true)
  })

  await t.test('a different file does not belong', () => {
    // Regression: /api/optimize/latest has no notion of "this session" — a
    // completed job from a different image than the one open now used to
    // be restored anyway, showing an unrelated PLY under the current photo
    // with its STL/PLY export enabled.
    assert.equal(jobBelongsToImage('dog.png', 'cat.png'), false)
  })

  await t.test('an older job with no recorded image is treated as belonging', () => {
    // Jobs recorded before the backend carried input_image have nothing to
    // compare against; "we don't know" must not silently refuse to ever
    // restore them (that would be a behavior change with no bug behind it).
    assert.equal(jobBelongsToImage(null, 'cat.png'), true)
    assert.equal(jobBelongsToImage(undefined, 'cat.png'), true)
  })

  await t.test('a project with no image of its own does not inherit an old job\'s image', () => {
    // Regression: this used to return true, on the theory that "no current
    // image" meant "we haven't checked yet" (like the legacy-job case
    // above). But the caller (loadCurrentJob) only runs after the current
    // project's settings have already loaded, so a null here means the
    // project genuinely has no image — e.g. right after "Start new
    // project", then an F5 reload before anything was uploaded again. That
    // used to restore the previous project's finished output (an old
    // job's image and PLY) into a fresh project that should show nothing.
    assert.equal(jobBelongsToImage('cat.png', null), false)
    assert.equal(jobBelongsToImage('cat.png', undefined), false)
  })

  await t.test('a legacy job with no recorded image still matches a fresh project with none either', () => {
    assert.equal(jobBelongsToImage(null, null), true)
  })
})
