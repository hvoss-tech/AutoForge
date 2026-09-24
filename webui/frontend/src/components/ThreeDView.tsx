import React, { useEffect, useRef, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import * as THREE from 'three'
import { cn } from '../lib/utils'
import { parseColoredMeshAsync } from '../lib/plyWorkerClient'
import { colorsNewerThan, currentColorSeq, subscribeVertexColors, writeTopVertexColors } from '../lib/meshColors'
import type { StackSegment } from '../lib/colorStack'

interface ThreeDViewProps {
  stlUrl?: string | null
  coloredPlyUrl?: string | null
  /** Rendered instead of a fetched mesh when there's no optimization result
   * yet — a simple stacked-slab preview of the color sliders' filament
   * stack, so the panel shows something meaningful before the first run. */
  stackSegments?: StackSegment[]
  className?: string
}

export type CameraView = 'top' | 'angled'

// Meshes are centered and scaled to ~4 units (centerAndScaleGeom); the
// heightmap lies in the XY plane with height along +Z.
const VIEW_POSITIONS: Record<CameraView, [number, number, number]> = {
  top: [0, 0, 6.5],
  angled: [0, -4.2, 4.6],
}

/** Moves the camera when the toolbar asks for a view. The request carries a
 * counter so clicking the same view again (after orbiting away) still resets. */
const CameraRig: React.FC<{ request: { view: CameraView; n: number }; controls: React.RefObject<OrbitControlsImpl> }> = ({ request, controls }) => {
  const camera = useThree((s) => s.camera)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    camera.position.set(...VIEW_POSITIONS[request.view])
    camera.up.set(0, 1, 0)
    camera.lookAt(0, 0, 0)
    controls.current?.target.set(0, 0, 0)
    controls.current?.update()
    invalidate()
  }, [request, camera, controls, invalidate])
  return null
}

const ColorStackPreview: React.FC<{ segments: StackSegment[] }> = ({ segments }) => {
  if (segments.length === 0) return null
  const maxLayer = segments[segments.length - 1].layerIndex
  const footprint = 3
  const totalHeight = 3
  const bandHeight = totalHeight / maxLayer

  return (
    <group position={[0, -totalHeight / 2, 0]}>
      {segments.map((seg) => (
        <mesh key={seg.layerIndex} position={[0, (seg.layerIndex - 0.5) * bandHeight, 0]}>
          <boxGeometry args={[footprint, bandHeight * 1.02, footprint]} />
          <meshStandardMaterial color={seg.color} roughness={0.7} metalness={0.1} toneMapped={false} />
        </mesh>
      ))}
    </group>
  )
}

function centerAndScaleGeom(geom: THREE.BufferGeometry): THREE.BufferGeometry {
  geom.computeBoundingBox()
  const box = geom.boundingBox
  if (!box || !isFinite(box.max.x - box.min.x)) return geom
  const center = new THREE.Vector3()
  box.getCenter(center)
  geom.translate(-center.x, -center.y, -center.z)
  const size = new THREE.Vector3()
  box.getSize(size)
  const maxDim = Math.max(size.x, size.y, size.z, 0.001)
  const targetSize = 4
  const scale = targetSize / maxDim
  if (isFinite(scale) && scale > 0) {
    geom.scale(scale, scale, scale)
  }
  return geom
}

const ColoredMesh: React.FC<{ plyUrl: string }> = ({ plyUrl }) => {
  const [mesh, setMesh] = useState<{ geom: THREE.BufferGeometry; colors: boolean } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const invalidate = useThree((s) => s.invalidate)
  // Tracks the geometry currently on screen, keyed by the job the PLY came
  // from (the URL without its `?v=` cache-busting suffix) and a hash of its
  // vertex positions. A slider/color-core edit re-renders the *same* job's
  // mesh with the per-pixel height solution untouched — only vertex colors
  // differ — so when the positions are identical we can skip rebuilding the
  // geometry entirely and just overwrite its color attribute in place. That
  // avoids the two most expensive steps on every debounced edit:
  // recomputing vertex normals over every face, and re-uploading
  // position/index buffers to the GPU (only the color buffer needs a fresh
  // upload).
  //
  // The key used to be the vertex *count*, which is not an identity at all:
  // pruning rewrites every height on the same grid, so the count matched and
  // this kept showing the pre-pruning relief with the pruned colors painted
  // onto it — the pruned shape (fewer layers, spikes removed) never appeared
  // until something else forced a rebuild.
  const currentRef = useRef<{ geom: THREE.BufferGeometry; jobKey: string; positionsHash: number } | null>(null)

  // Live slider edits: recolor the geometry on screen in place (see
  // lib/meshColors) instead of waiting for a whole new PLY.
  const applyColors = (geom: THREE.BufferGeometry, rgb: Uint8Array): boolean => {
    const colorAttr = geom.getAttribute('color') as THREE.BufferAttribute | undefined
    if (!colorAttr || !writeTopVertexColors(colorAttr.array as Float32Array, rgb)) return false
    colorAttr.needsUpdate = true
    invalidate()
    // For timing live edits (tests/jobs/live-recolor.spec.ts, devtools).
    performance.mark('autoforge:mesh-recolored')
    return true
  }
  useEffect(
    () =>
      subscribeVertexColors((update) => {
        const current = currentRef.current
        if (!current || current.jobKey !== update.meshKey) return false
        return applyColors(current.geom, update.rgb)
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [invalidate],
  )

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    // Deliberately NOT clearing `mesh` here — every slider/color-core edit
    // bumps the preview version and therefore this URL, and clearing the
    // mesh up front blanked the whole 3D view (showing "Loading 3D
    // model...") for the round trip on every single edit. Keep rendering
    // whatever mesh is already there and swap it only once the new one has
    // actually finished loading, so there's never a blank frame.
    setError(null)

    const jobKey = plyUrl.split('?')[0]
    // Colors published while this file downloads are newer than the file.
    const loadStartSeq = currentColorSeq()

    fetch(plyUrl)
      .then(r => {
        // A 404/500 response body is JSON, not PLY bytes — reading it as
        // an arrayBuffer and handing it to the PLY parser produced the
        // confusing "Invalid PLY: no end_header" error instead of a
        // message that actually explains what went wrong.
        if (!r.ok) throw new Error(r.status === 404 ? 'No 3D model available for this result' : `Failed to fetch model (HTTP ${r.status})`)
        return r.arrayBuffer()
      })
      // Parsing runs in a worker: on the main thread it blocked every
      // interaction on the page for the whole parse, which is what made the
      // 3D panel feel like it hung on each update.
      .then(buffer => parseColoredMeshAsync(buffer))
      .then(({ positions, colors, indices, positionsHash }) => {
        if (cancelled) return

        const vertexCount = positions.length / 3
        if (vertexCount === 0 || indices.length === 0) {
          throw new Error('Empty PLY mesh')
        }

        const current = currentRef.current
        if (current && current.jobKey === jobKey && current.positionsHash === positionsHash && colors) {
          const colorAttr = current.geom.getAttribute('color') as THREE.BufferAttribute | undefined
          if (colorAttr && colorAttr.array.length === colors.length) {
            ;(colorAttr.array as Float32Array).set(colors)
            colorAttr.needsUpdate = true
          } else {
            current.geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
          }
          const newer = colorsNewerThan(jobKey, loadStartSeq)
          if (newer) applyColors(current.geom, newer.rgb)
          setMesh({ geom: current.geom, colors: true })
          setLoading(false)
          invalidate()
          return
        }

        const geom = new THREE.BufferGeometry()
        geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
        geom.setIndex(new THREE.BufferAttribute(indices, 1))
        geom.computeVertexNormals()

        const centered = centerAndScaleGeom(geom)

        if (colors) {
          centered.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
        }

        // React Three Fiber won't auto-dispose the geometry being replaced
        // here — it's passed as a plain `geometry` prop, not a nested
        // <bufferGeometry> child, so its GPU buffers would otherwise leak
        // on every new job/result loaded into this view.
        if (current && current.geom !== centered) {
          current.geom.dispose()
        }

        currentRef.current = { geom: centered, jobKey, positionsHash }
        performance.mark('autoforge:mesh-loaded')
        const newer = colors ? colorsNewerThan(jobKey, loadStartSeq) : null
        if (newer) applyColors(centered, newer.rgb)
        setMesh({ geom: centered, colors: !!colors })
        setLoading(false)
        invalidate()
      })
      .catch(err => {
        if (!cancelled) {
          console.error('Failed to load 3D model:', err)
          setError(err.message)
          setLoading(false)
        }
      })

    return () => { cancelled = true }
  }, [plyUrl, invalidate])

  useEffect(() => () => currentRef.current?.geom.dispose(), [])

  // The full-panel messages below only apply to the very first load —
  // once a mesh has successfully rendered once, later fetches (slider
  // edits, live progress updates) update in place; a stale mesh on screen
  // beats a blank one while the new one loads, and beats replacing good
  // content with an error banner over a transient refresh failure.
  if (!mesh) {
    if (loading) {
      return (
        <Html center>
          <div className="text-xs text-muted-foreground">Loading 3D model...</div>
        </Html>
      )
    }

    if (error) {
      return (
        <Html center>
          <div className="text-xs text-red-400">Failed to load model: {error}</div>
        </Html>
      )
    }

    return (
      <Html center>
        <div className="text-sm text-muted-foreground">
          Upload an image and run optimization to see the 3D preview
        </div>
      </Html>
    )
  }

  return (
    <mesh geometry={mesh.geom}>
      <meshStandardMaterial
        roughness={0.7}
        metalness={0.1}
        side={THREE.DoubleSide}
        vertexColors={mesh.colors}
        toneMapped={false}
      />
    </mesh>
  )
}

/** Canvas background follows the app theme (it stayed near-black in light mode). */
function useThreeDTheme() {
  const [clear, setClear] = useState(() => readClear())
  useEffect(() => {
    const observer = new MutationObserver(() => setClear(readClear()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])
  return { clear }
}

function readClear(): string {
  return document.documentElement.getAttribute('data-theme') === 'light' ? '#dde3ea' : '#0f1c2a'
}

const ClearColor: React.FC<{ color: string }> = ({ color }) => {
  const gl = useThree((s) => s.gl)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    gl.setClearColor(new THREE.Color(color))
    invalidate()
  }, [gl, color, invalidate])
  return null
}

const CanvasInvalidator: React.FC = () => {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    invalidate()
  }, [invalidate])
  return null
}

export const ThreeDView: React.FC<ThreeDViewProps> = ({
  stlUrl,
  coloredPlyUrl,
  stackSegments,
  className,
}) => {
  const theme = useThreeDTheme()
  const isModel = !!(coloredPlyUrl || stlUrl)
  // Looking straight down shows the picture the way it will look printed;
  // the old fixed oblique camera made results look small and skewed.
  const [viewRequest, setViewRequest] = useState<{ view: CameraView; n: number }>({ view: isModel ? 'top' : 'angled', n: 0 })
  const controlsRef = useRef<OrbitControlsImpl>(null)
  const requestView = (view: CameraView) => setViewRequest((r) => ({ view, n: r.n + 1 }))
  // Switching from the color-stack preview to a model (or back) resets to
  // that content's natural view instead of keeping the previous camera.
  useEffect(() => {
    requestView(isModel ? 'top' : 'angled')
  }, [isModel])

  return (
    <div className={cn('relative h-full w-full overflow-hidden rounded-lg', className)}>
      {isModel && (
        <div className="absolute top-2 right-2 z-10 flex gap-1" data-testid="view-controls">
          <button onClick={() => requestView('top')} className="px-2 py-1 text-xs rounded bg-gray-900/80 text-gray-200 hover:bg-gray-800 border border-gray-700" data-testid="view-top-btn" title="Look straight down, like the finished print on a wall">
            Top
          </button>
          <button onClick={() => requestView('angled')} className="px-2 py-1 text-xs rounded bg-gray-900/80 text-gray-200 hover:bg-gray-800 border border-gray-700" data-testid="view-angled-btn" title="Tilted view to see the relief">
            Angled
          </button>
          <button onClick={() => requestView(viewRequest.view)} className="px-2 py-1 text-xs rounded bg-gray-900/80 text-gray-200 hover:bg-gray-800 border border-gray-700" data-testid="view-reset-btn" title="Undo zoom and panning">
            Fit
          </button>
        </div>
      )}
      <Canvas
        frameloop="demand"
        dpr={[1, 1.5]}
        camera={{ position: VIEW_POSITIONS[isModel ? 'top' : 'angled'], fov: 45, near: 0.1, far: 1000 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'low-power', toneMapping: THREE.NoToneMapping }}
        onCreated={(state: any) => {
          state.gl.setClearColor(new THREE.Color(theme.clear))
        }}
      >
        <ClearColor color={theme.clear} />
        <CameraRig request={viewRequest} controls={controlsRef} />
        <CanvasInvalidator />
        {/* Three's MeshStandardMaterial (like every physically-based
            material it ships) runs a Lambertian BRDF that divides diffuse
            reflectance by π — a directional/ambient light of intensity 1
            hitting a surface head-on only returns ~32% of the albedo. The
            composited 2D preview PNG has no such falloff (it's the raw
            Beer-Lambert compositing result), so without compensating here
            the lit 3D mesh reads as measurably darker than the 2D preview
            for the exact same colors. Driving the (direction-independent)
            ambient light close to π restores that ~1:1 albedo baseline on
            every face regardless of orientation; the hemisphere/directional
            lights stay low so there's still enough per-face shading to read
            the model's relief, without any face going noticeably darker
            than its true color. */}
        <ambientLight intensity={3.05} />
        <hemisphereLight args={['#ffffff', '#334155', 0.25]} />
        <directionalLight position={[5, 10, 5]} intensity={0.35} />
        <directionalLight position={[-5, -5, -5]} intensity={0.15} />
        {coloredPlyUrl ? (
          <ColoredMesh plyUrl={coloredPlyUrl} />
        ) : stlUrl ? (
          <ColoredMesh plyUrl={stlUrl} />
        ) : stackSegments ? (
          <ColorStackPreview segments={stackSegments} />
        ) : null}
        <OrbitControls
          ref={controlsRef}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={0.5}
          maxDistance={50}
          enableDamping={true}
          dampingFactor={0.05}
          onStart={() => { /* invalidate via CanvasInvalidator */ }}
          onChange={() => { /* invalidate via CanvasInvalidator */ }}
          onEnd={() => { /* invalidate via CanvasInvalidator */ }}
        />
      </Canvas>
    </div>
  )
}