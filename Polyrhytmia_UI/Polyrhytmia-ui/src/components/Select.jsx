// src/components/inputs/Select.jsx
import PropTypes from 'prop-types'
import { FormControl, InputLabel, Select as MuiSelect, MenuItem } from '@mui/material'

/**
 * Select - Componente reutilizable basado en Material UI
 * Props:
 * - label: string
 * - value: string | number
 * - options: [{ value, label }]
 * - onChange: function
 * - fullWidth: bool
 */
export default function Select({ label, value, options, onChange, fullWidth = true }) {
  return (
    <FormControl fullWidth={fullWidth} size="small">
      <InputLabel>{label}</InputLabel>
      <MuiSelect
        value={value}
        label={label}
        onChange={onChange}
      >
        {options.map((opt) => (
          <MenuItem key={opt.value} value={opt.value}>
            {opt.label}
          </MenuItem>
        ))}
      </MuiSelect>
    </FormControl>
  )
}

Select.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  options: PropTypes.arrayOf(
    PropTypes.shape({
      value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      label: PropTypes.string.isRequired,
    })
  ).isRequired,
  onChange: PropTypes.func.isRequired,
  fullWidth: PropTypes.bool,
}
