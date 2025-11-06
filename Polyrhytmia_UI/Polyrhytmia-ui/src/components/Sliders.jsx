// src/components/inputs/Slider.jsx
import PropTypes from 'prop-types'
import { Box, Slider as MuiSlider, Typography } from '@mui/material'

/**
 * Slider - Componente reutilizable basado en Material UI
 * Props:
 * - label: string
 * - value: number
 * - min: number
 * - max: number
 * - step: number
 * - onChange: function
 */
export default function Slider({ label, value, min, max, step, onChange }) {
  return (
    <Box sx={{ width: '100%', mb: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        {label}: {value}
      </Typography>
      <MuiSlider
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        valueLabelDisplay="auto"
      />
    </Box>
  )
}

Slider.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.number.isRequired,
  min: PropTypes.number.isRequired,
  max: PropTypes.number.isRequired,
  step: PropTypes.number,
  onChange: PropTypes.func.isRequired,
}
