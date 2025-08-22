// src/components/feedback/Toast.jsx
import PropTypes from 'prop-types'
import { Snackbar, Alert, Slide } from '@mui/material'

function SlideDown(props) {
  return <Slide {...props} direction="down" />
}

/**
 * Toast - Notificación superior estilo 'toast'
 * Props:
 * - open: boolean
 * - message: string | ReactNode
 * - severity: 'success' | 'info' | 'warning' | 'error'
 * - onClose: function
 * - autoHideDuration: ms (default 3000)
 * - anchorOrigin: { vertical, horizontal } (default top/center)
 */
export default function Toast({
  open,
  message,
  severity = 'success',
  onClose,
  autoHideDuration = 3000,
  anchorOrigin = { vertical: 'top', horizontal: 'center' },
}) {
  return (
    <Snackbar
      open={open}
      onClose={onClose}
      autoHideDuration={autoHideDuration}
      anchorOrigin={anchorOrigin}
      TransitionComponent={SlideDown}
      sx={{ mt: 1 }} // pequeño margen desde el borde superior
    >
      <Alert
        onClose={onClose}
        severity={severity}
        variant="filled"
        elevation={6}
        sx={{
          borderRadius: 3,
          minWidth: 320,
        }}
      >
        {message}
      </Alert>
    </Snackbar>
  )
}

Toast.propTypes = {
  open: PropTypes.bool.isRequired,
  message: PropTypes.oneOfType([PropTypes.string, PropTypes.node]).isRequired,
  severity: PropTypes.oneOf(['success', 'info', 'warning', 'error']),
  onClose: PropTypes.func.isRequired,
  autoHideDuration: PropTypes.number,
  anchorOrigin: PropTypes.shape({
    vertical: PropTypes.oneOf(['top', 'bottom']),
    horizontal: PropTypes.oneOf(['left', 'center', 'right']),
  }),
}
