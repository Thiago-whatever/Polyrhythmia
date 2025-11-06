// src/components/visualizer/Waveform.jsx
import PropTypes from 'prop-types'
import { useMemo } from 'react'
import { Box } from '@mui/material'

/**
 * Waveform (SVG)
 * - Muestra barras verticales (amplitud 0..1) y un playhead.
 * - Props:
 *   data: number[] (amplitudes 0..1)
 *   width, height: número (px)
 *   progress: número 0..1 (posición del playhead)
 *   rounded: bool (bordes redondeados)
 *   onBarClick: (index)=>void (opcional)
 *   colors: { start, end, bg } gradiente y fondo
 */
export default function Waveform({
  data = [],
  width = 960,
  height = 220,
  progress = 0,
  rounded = true,
  onBarClick,
  colors = {
    start: '#C2185B',
    end: '#3F51B5',
    bg: 'rgba(0,0,0,0.04)',
  },
}) {
  const N = data.length || 32
  const bars = data.length ? data : Array.from({ length: N }, (_, i) => 0.25 + 0.65 * Math.random())

  const { barWidth, gap, radius } = useMemo(() => {
    const gapPx = 4
    const w = Math.max(2, (width - (N + 1) * gapPx) / N)
    return { barWidth: w, gap: gapPx, radius: rounded ? 4 : 0 }
  }, [N, rounded, width])

  const playX = useMemo(() => {
    const total = N * (barWidth + gap) + gap
    return Math.min(width - 2, gap + progress * (total - gap))
  }, [N, barWidth, gap, progress, width])

  return (
    <Box
      sx={{
        width,
        height,
        borderRadius: 3,
        background: colors.bg,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <svg width={width} height={height} role="img" aria-label="Visualización del patrón rítmico">
        {/* Gradiente */}
        <defs>
          <linearGradient id="wf-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={colors.start} />
            <stop offset="100%" stopColor={colors.end} />
          </linearGradient>
        </defs>

        {/* Barras */}
        {bars.map((v, i) => {
          const x = gap + i * (barWidth + gap)
          const h = Math.max(4, v * (height - 24))
          const y = (height - h) / 2
          const onClick = onBarClick ? () => onBarClick(i) : undefined
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barWidth}
              height={h}
              rx={radius}
              ry={radius}
              fill="url(#wf-grad)"
              opacity={0.9}
              style={{ cursor: onBarClick ? 'pointer' : 'default' }}
              onClick={onClick}
            />
          )
        })}

        {/* Playhead */}
        <line
          x1={playX}
          y1={0}
          x2={playX}
          y2={height}
          stroke="white"
          strokeWidth="2"
          strokeOpacity="0.85"
        />
      </svg>
    </Box>
  )
}

Waveform.propTypes = {
  data: PropTypes.arrayOf(PropTypes.number),
  width: PropTypes.number,
  height: PropTypes.number,
  progress: PropTypes.number,
  rounded: PropTypes.bool,
  onBarClick: PropTypes.func,
  colors: PropTypes.shape({
    start: PropTypes.string,
    end: PropTypes.string,
    bg: PropTypes.string,
  }),
}
