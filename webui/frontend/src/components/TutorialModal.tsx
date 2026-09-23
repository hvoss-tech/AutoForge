import React from 'react'
import { BookOpen, Check, ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { LARGE_STL_SIZE_MM, NOTICEABLE_DETAIL_SIZE_MM } from '../lib/stlSize'
import { DEFAULT_FOCUS_STRENGTH, MAX_FOCUS_STRENGTH, MIN_FOCUS_STRENGTH } from '../lib/focusMask'

interface Step {
  id: string
  title: string
  body: React.ReactNode
}

/** What AutoForge does, and the order to do it in.
 *
 * Written for someone who has just opened the app for the first time and has
 * never heard of transmission distance or Gumbel softmax: every step says
 * what to click, what it produces, and why it matters for the print. */
export const TUTORIAL_STEPS: Step[] = [
  {
    id: 'what',
    title: 'What AutoForge makes',
    body: (
      <>
        <p>
          AutoForge turns a picture into a <strong>flat 3D print built from stacked layers of coloured filament</strong>.
          There is no multi-material printer involved: the model is printed in one piece, and you swap the filament by
          hand at a handful of layers along the way.
        </p>
        <p>
          The picture appears because thin plastic is slightly see-through. A light colour laid over a dark one only
          half hides it, so stacking a few filaments at different heights mixes far more shades than you have spools.
          How see-through a filament is, is its <strong>TD</strong> (transmission distance) — the higher the number, the
          more the layers underneath show through.
        </p>
        <p>You get an STL to print, and a list of which filament to load at which layer.</p>
      </>
    ),
  },
  {
    id: 'setup',
    title: '1 · Your image and your filaments',
    body: (
      <>
        <p>
          <strong>Drop an image</strong> into the left panel (or click it to browse). Pictures with clear shapes and
          good contrast work best; very busy photos turn muddy.
        </p>
        <p>
          In the <strong>Filament Library</strong> on the left, click <strong>+</strong> on each filament you actually
          own to make it <em>active</em>. The optimizer only ever uses active filaments — it cannot pick a colour you
          haven't given it. Four to eight well-spread colours (including a dark and a light one) is a good starting set.
        </p>
        <p>
          As soon as there's an image and one active filament, the right panel builds a quick 3D preview of the shape.
          That preview is just the heights — the colours come next.
        </p>
      </>
    ),
  },
  {
    id: 'focus',
    title: 'Optional · Mark what matters most',
    body: (
      <>
        <p>
          The optimizer spreads its effort over the whole picture. If some parts matter more than the rest — a face,
          the eyes, some lettering — you can tell it so by painting <strong>focus areas</strong>.
        </p>
        <p>
          Click <strong>Focus</strong> in the image panel and paint over those parts with the brush. Use{' '}
          <strong>Erase</strong> (or press <strong>E</strong>) to take paint away, <strong>[</strong> and{' '}
          <strong>]</strong> to change the brush size, scroll to zoom in for detail and right-drag to move around.
          Press <strong>Done</strong> when you're finished — the focus areas are saved with the project and used from
          the next run.
        </p>
        <p>
          <strong>How much they count</strong> is up to you: the <em>Painted areas count … more</em> slider under the
          picture goes from {MIN_FOCUS_STRENGTH}× to {MAX_FOCUS_STRENGTH}× (default {DEFAULT_FOCUS_STRENGTH}×) and
          applies to all painted areas at once. The rest of the picture still counts, just less — so if the focus areas
          still come out off, raise it; if the background suffers too much, lower it.
        </p>
        <p>
          Not sure where to paint? After a run, switch the image panel to <strong>Differences</strong>: it lights up
          where the print strays from the picture, brightest where it's furthest off. Paint the bright parts you care
          about, and run again.
        </p>
        <p>Keep it to the few areas that really matter: painting everything is the same as painting nothing.</p>
      </>
    ),
  },
  {
    id: 'run',
    title: '2 · Run the optimizer',
    body: (
      <>
        <p>
          Press <strong>Run</strong>. AutoForge works out, at the same time, how tall every pixel should be and which
          filament each layer should use, checking the result against your photo thousands of times and keeping the
          best one. It takes from under a minute to several minutes depending on the image and your GPU.
        </p>
        <p>
          You can watch it converge in the preview, and <strong>Pause</strong> or <strong>Cancel</strong> at any point —
          the best result so far is kept either way.
        </p>
        <p>
          When it finishes, the <strong>Color layers</strong> panel at the bottom fills in with the stack it chose: one
          row per band of layers, bottom row first. The <strong>Base</strong> row is the solid slab everything is
          printed on; its thickness is fixed (change it with <em>Base</em> in the status bar), but you can swap which
          filament it uses.
        </p>
      </>
    ),
  },
  {
    id: 'prune',
    title: '3 · Prune it into something printable',
    body: (
      <>
        <p>
          A raw result often wants a dozen colours and thirty filament swaps. <strong>Pruning</strong> cuts that down
          while keeping the picture as close as it can — this is the step that turns a nice render into a print you'd
          actually stand at the machine for.
        </p>
        <p>
          Open <strong>Pruning</strong>, and you'll see what the result currently costs: colours (including the base),
          swaps, and layers. The limits start at exactly those numbers. Lower whichever one you care about and press
          <strong> Start pruning</strong>; the counts update live as it works.
        </p>
        <p>
          <strong>Run pruning more than once.</strong> Each pass is a greedy search that starts from the result it is
          given, so a second and third pass keep finding improvements — and stepping the limits down gradually
          (say 12 colours, then 9, then 7) gives a visibly better print than demanding 7 in one jump. Tick{' '}
          <strong>Keep pruning until it stops improving</strong> and it will repeat on its own until a pass no longer
          helps; you can stop it whenever you like, and the best result so far is kept either way.
        </p>
      </>
    ),
  },
  {
    id: 'adjust',
    title: '4 · Adjust by hand (optional)',
    body: (
      <>
        <p>
          Every row in <strong>Color layers</strong> can be changed: click the swatch to use a different filament, drag
          the slider or type a number to move where the band ends, and use the grip to reorder bands. Drag a filament
          straight from the library onto a row to assign it.
        </p>
        <p>
          The 2D and 3D previews re-render as you go, so you can see what a change costs before committing. Scrolling
          over the list scrolls the list — sliders only move when you drag them.
        </p>
        <p>
          Everything is undoable (<strong>Ctrl+Z</strong>), and <strong>History</strong> in the top bar lets you jump
          back to any earlier state, including finished results.
        </p>
      </>
    ),
  },
  {
    id: 'export',
    title: '5 · Export and print',
    body: (
      <>
        <p>
          <strong>File ▸ Export</strong> gives you the STL, the swap instructions, the preview image and a HueForge
          project file — or all of them as one zip.
        </p>
        <p>
          Print it at <strong>100% infill</strong> with the layer height shown in the status bar, and pause at each
          layer in the instructions to change filament.
        </p>
        <p>
          <strong>About STL size.</strong> It sets the longest side of the exported model. Above {LARGE_STL_SIZE_MM} mm
          the mesh gets large and can use a lot of VRAM, and the extra detail only becomes visible on prints bigger
          than roughly {NOTICEABLE_DETAIL_SIZE_MM}&nbsp;mm. In practice a model made at one size can be scaled to about
          double in your slicer before the layer steps start to show — so export at a sensible size and scale up there
          rather than here.
        </p>
      </>
    ),
  },
]

export const TutorialModal: React.FC = () => {
  const open = useAppStore((s) => s.tutorialOpen)
  const setOpen = useAppStore((s) => s.setTutorialOpen)
  const [index, setIndex] = React.useState(0)

  // Reopening from the ? button starts at the beginning again.
  React.useEffect(() => {
    if (open) setIndex(0)
  }, [open])

  const step = TUTORIAL_STEPS[index]
  const last = index === TUTORIAL_STEPS.length - 1

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-2xl p-0 gap-0 bg-gray-900 border-gray-700" data-testid="tutorial-modal">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold text-gray-100">
            <BookOpen className="w-4 h-4" /> Getting started with AutoForge
          </DialogTitle>
          <button
            onClick={() => setOpen(false)}
            className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-800"
            aria-label="Close the tutorial"
            data-testid="tutorial-close-x"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <DialogDescription className="sr-only">
          A short tour of AutoForge: what it makes, and how to go from an image to a printable model.
        </DialogDescription>

        <div className="px-4 pt-3">
          <ol className="flex items-center gap-1" data-testid="tutorial-steps" aria-label="Tutorial steps">
            {TUTORIAL_STEPS.map((s, i) => (
              <li key={s.id} className="flex-1">
                <button
                  onClick={() => setIndex(i)}
                  aria-current={i === index ? 'step' : undefined}
                  aria-label={s.title}
                  className={`w-full h-1 rounded-full ${i === index ? 'bg-cyan-500' : i < index ? 'bg-cyan-800' : 'bg-gray-700'}`}
                  data-testid={`tutorial-step-dot-${i}`}
                  data-active={i === index || undefined}
                />
              </li>
            ))}
          </ol>
        </div>

        <div className="px-4 py-4 min-h-64" data-testid="tutorial-body" data-step={step.id}>
          <h3 className="text-sm font-semibold text-gray-100 mb-2" data-testid="tutorial-title">{step.title}</h3>
          <div className="space-y-2 text-xs leading-relaxed text-gray-300 [&_strong]:text-gray-100">{step.body}</div>
        </div>

        <div className="flex items-center justify-between gap-2 px-4 py-3 border-t border-gray-700">
          <span className="text-[11px] text-gray-500 tabular-nums" data-testid="tutorial-progress">
            {index + 1} of {TUTORIAL_STEPS.length}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setIndex((i) => Math.max(0, i - 1))}
              disabled={index === 0}
              className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100 disabled:opacity-40 disabled:hover:bg-gray-700"
              data-testid="tutorial-back-btn"
            >
              <ChevronLeft className="w-3.5 h-3.5" /> Back
            </button>
            {last ? (
              <button
                onClick={() => setOpen(false)}
                className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-cyan-600 hover:bg-cyan-500 text-white"
                data-testid="tutorial-done-btn"
              >
                <Check className="w-3.5 h-3.5" /> Start using AutoForge
              </button>
            ) : (
              <button
                onClick={() => setIndex((i) => Math.min(TUTORIAL_STEPS.length - 1, i + 1))}
                className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-cyan-600 hover:bg-cyan-500 text-white"
                data-testid="tutorial-next-btn"
              >
                Next <ChevronRight className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
