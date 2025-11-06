import { useEffect, useCallback } from 'react'
import PropTypes from 'prop-types'
import { Button, Stack, Paper, Tooltip } from '@mui/material'
import PlayArrowRoundedIcon from '@mui/icons-material/PlayArrowRounded'
import PauseRoundedIcon from '@mui/icons-material/PauseRounded'
import DownloadRoundedIcon from '@mui/icons-material/DownloadRounded'

/**
 * Controls (Transport Bar)
 * - Play / Pause del loop
 * - Exportar a MIDI
 * - Atajos: Space (play/pause), Ctrl+E (export)
 */
export default function Controls({
  isPlaying = false,
  isGenerating = false,
  isExporting = false,
  disabled = false,
  onPlay = () => {},
  onPause = () => {},
  onExport = () => {},
}) {
  // Toggle play/pause
  const togglePlay = useCallback(() => {
    if (disabled || isGenerating) return
    if (isPlaying) onPause()
    else onPlay()
  }, [disabled, isGenerating, isPlaying, onPause, onPlay])

  // Keyboard shortcuts: Space = play/pause, Ctrl+E = export
  useEffect(() => {
    const handler = (e) => {
      // Evitar conflicto con inputs/textarea
      const tag = (e.target?.tagName || '').toLowerCase()
      const editing = tag === 'input' || tag === 'textarea' || e.target?.isContentEditable
      if (editing) return

      // Space → play/pause
      if (e.code === 'Space') {
        e.preventDefault()
        togglePlay()
      }
      // Ctrl/Cmd + E → export
      if ((e.ctrlKey || e.metaKey) && (e.key === 'e' || e.key === 'E')) {
        e.preventDefault()
        if (!disabled && !isExporting) onExport()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [disabled, isExporting, onExport, togglePlay])

  const playLabel = isPlaying ? 'Pausar (Espacio)' : (isGenerating ? 'Generando…' : 'Reproducir (Espacio)')
  const exportLabel = isExporting ? 'Exportando…' : 'Exportar MIDI (Ctrl+E)'

  return (
    <Paper
      elevation={3}
      sx={{
        px: 2,
        py: 1.5,
        borderRadius: 3,
        backdropFilter: 'blur(6px)',
        background:
          'linear-gradient(90deg, rgba(123,31,162,0.10) 0%, rgba(63,81,181,0.10) 100%)',
      }}
      role="region"
      aria-label="Controles de transporte"
    >
      <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="center">
        <Tooltip title={playLabel} arrow placement="top">
          <span>
            <Button
              variant="contained"
              color={isPlaying ? 'secondary' : 'primary'}
              size="large"
              disableElevation
              onClick={togglePlay}
              disabled={disabled || isGenerating}
              startIcon={isPlaying ? <PauseRoundedIcon /> : <PlayArrowRoundedIcon />}
              aria-label={isPlaying ? 'Pausar' : 'Reproducir'}
            >
              {isPlaying ? 'Pausar' : 'Reproducir'}
            </Button>
          </span>
        </Tooltip>

        <Tooltip title={exportLabel} arrow placement="top">
          <span>
            <Button
              variant="outlined"
              color="primary"
              size="large"
              onClick={onExport}
              disabled={disabled || isExporting || isGenerating}
              startIcon={<DownloadRoundedIcon />}
              aria-label="Exportar a MIDI"
            >
              {isExporting ? 'Exportando…' : 'Exportar'}
            </Button>
          </span>
        </Tooltip>
      </Stack>
    </Paper>
  )
}

Controls.propTypes = {
  isPlaying: PropTypes.bool,
  isGenerating: PropTypes.bool,
  isExporting: PropTypes.bool,
  disabled: PropTypes.bool,
  onPlay: PropTypes.func,
  onPause: PropTypes.func,
  onExport: PropTypes.func,
}
