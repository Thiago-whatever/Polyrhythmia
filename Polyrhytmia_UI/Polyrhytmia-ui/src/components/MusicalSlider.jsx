// src/components/inputs/MusicalSlider.jsx
import PropTypes from 'prop-types'
import { Box, Slider as MuiSlider, Typography, Stack, Tooltip } from '@mui/material'
import MusicNoteRoundedIcon from '@mui/icons-material/MusicNoteRounded'
import GraphicEqRoundedIcon from '@mui/icons-material/GraphicEqRounded'
import SpeedRoundedIcon from '@mui/icons-material/SpeedRounded'

/**
 * MusicalSlider
 * - Estilo musical con iconos a los lados y ticks.
 * - Props:
 *   label: string
 *   value: number
 *   min, max, step: number
 *   onChange: (e, val)=>void
 *   iconLeft, iconRight: ReactNode (opcionales)
 *   unit: string (p. ej. 'BPM')
 *   marks: boolean | [{ value, label }]
 */
export default function MusicalSlider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  iconLeft = <MusicNoteRoundedIcon />,
  iconRight = <GraphicEqRoundedIcon />,
  unit,
  marks = false,
}) {
  return (
    <Box
      sx={{
        width: '100%',
        p: 2,
        borderRadius: 3,
        background:
          'linear-gradient(90deg, rgba(123,31,162,0.06) 0%, rgba(63,81,181,0.06) 100%)',
      }}
    >
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
          {label}
        </Typography>
        <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
          {value}{unit ? ` ${unit}` : ''}
        </Typography>
      </Stack>

      <Stack direction="row" spacing={1.5} alignItems="center">
        <Tooltip title="Menos">
          <Box sx={{ color: 'text.secondary' }}>{iconLeft}</Box>
        </Tooltip>

        <MuiSlider
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          marks={marks}
          valueLabelDisplay="auto"
          sx={{
            flex: 1,
            '& .MuiSlider-rail': { opacity: 0.3 },
            '& .MuiSlider-track': {
              border: 'none',
              background:
                'linear-gradient(90deg, #C2185B 0%, #7C4DFF 50%, #3F51B5 100%)',
            },
            '& .MuiSlider-thumb': {
              width: 18,
              height: 18,
              boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
              '&:hover, &.Mui-focusVisible': {
                boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
              },
            },
          }}
        />

        <Tooltip title="Más">
          <Box sx={{ color: 'text.secondary' }}>{iconRight}</Box>
        </Tooltip>
      </Stack>
    </Box>
  )
}

MusicalSlider.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number.isRequired,
  min: PropTypes.number.isRequired,
  max: PropTypes.number.isRequired,
  step: PropTypes.number,
  onChange: PropTypes.func.isRequired,
  iconLeft: PropTypes.node,
  iconRight: PropTypes.node,
  unit: PropTypes.string,
  marks: PropTypes.oneOfType([PropTypes.bool, PropTypes.array]),
}

/** Ejemplos de iconos:
 *  - Tempo/BPM: iconLeft={<SpeedRoundedIcon />} iconRight={<SpeedRoundedIcon />}
 *  - Complejidad: iconLeft={<MusicNoteRoundedIcon />} iconRight={<GraphicEqRoundedIcon />}
 */
export const Icons = {
  Music: <MusicNoteRoundedIcon />,
  Spectrum: <GraphicEqRoundedIcon />,
  Speed: <SpeedRoundedIcon />,
}
