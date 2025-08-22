import { useState } from 'react'
import { Container, Box, Button } from '@mui/material'
import Toast from '../components/Toast'
import Controls from '../components/Controls'
import Select from '../components/Select'
import MusicalSlider, { Icons as SliderIcons } from '../components/MusicalSlider'
import Waveform from '../components/WaveForm'

export default function MainPage() {
  const [toast, setToast] = useState({
    open: false,
    message: '',
    severity: 'success',
  })

  const showToast = (message, severity = 'success') =>
    setToast({ open: true, message, severity })

  const handleToastClose = (_, reason) => {
    if (reason === 'clickaway') return
    setToast((t) => ({ ...t, open: false }))
  }

  // Ejemplos de callbacks reales
  const [isPlaying, setIsPlaying] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

    const [bpm, setBpm] = useState(120)
    const [complexity, setComplexity] = useState(5)
    const [progress, setProgress] = useState(0) // 0..1 para el playhead
    const [pattern, setPattern] = useState([])  // amplitudes 0..1 (opcional)

    const [instrument, setInstrument] = useState('drums')

    const instrumentOptions = [
    { value: 'drums', label: 'Batería' },
    { value: 'piano', label: 'Piano' },
    { value: 'bass', label: 'Bajo' },
    { value: 'guitar', label: 'Guitarra' },
    ]

  const handlePlay = async () => {
    try {
      setIsGenerating(true)
      // TODO: generar patrón vía API
      // await generatePattern(params)
      setIsPlaying(true)
      showToast('¡Terminado! El nuevo loop está listo para que lo escuches.', 'success')
    } catch (e) {
      showToast('Ocurrió un error al generar el loop.', 'error')
    } finally {
      setIsGenerating(false)
    }
  }

  const handlePause = () => {
    setIsPlaying(false)
    showToast('Reproducción pausada', 'info')
  }

  const handleExport = async () => {
    try {
      setIsExporting(true)
      // TODO: exportar a MIDI
      // await exportMidi(currentPattern)
      showToast('MIDI exportado correctamente.', 'success')
    } catch (e) {
      showToast('No se pudo exportar el MIDI.', 'error')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Demo de disparo manual del toast (puedes quitarlo) */}
      <Box sx={{ mb: 2 }}>
        <Button onClick={() => showToast('Ejemplo de notificación', 'info')}>
          Probar Toast
        </Button>
      </Box>

      <Box sx={{ mb: 3 }}>
        <Select
            label="Instrumento"
            value={instrument}
            options={instrumentOptions}
            onChange={(e) => setInstrument(e.target.value)}
        />
        </Box>

        <Box sx={{ mb: 3 }}>
            <MusicalSlider
                label="Tempo (BPM)"
                value={bpm}
                min={40}
                max={240}
                step={1}
                unit="BPM"
                onChange={(_, v) => setBpm(v)}
                iconLeft={SliderIcons.Speed}
                iconRight={SliderIcons.Speed}
                marks={[{ value: 60, label: '60' }, { value: 120, label: '120' }, { value: 180, label: '180' }]}
            />
            </Box>

            <Box sx={{ mb: 3 }}>
            <MusicalSlider
                label="Complejidad rítmica"
                value={complexity}
                min={1}
                max={10}
                step={1}
                onChange={(_, v) => setComplexity(v)}
                iconLeft={SliderIcons.Music}
                iconRight={SliderIcons.Spectrum}
                marks
            />
            </Box>

            <Box sx={{ my: 4, display: 'flex', justifyContent: 'center' }}>
            <Waveform
                data={pattern}          // si está vacío, se autogenera una forma amigable
                width={920}
                height={220}
                progress={progress}     // actualiza 0..1 durante reproducción
                rounded
                onBarClick={(i) => console.log('bar', i)}
            />
            </Box>

      {/* Aquí irían tu panel de parámetros y visualizador */}

      <Box sx={{ mt: 6 }}>
        <Controls
          isPlaying={isPlaying}
          isGenerating={isGenerating}
          isExporting={isExporting}
          onPlay={handlePlay}
          onPause={handlePause}
          onExport={handleExport}
        />
      </Box>

      {/* Toast global */}
      <Toast
        open={toast.open}
        message={toast.message}
        severity={toast.severity}
        onClose={handleToastClose}
        autoHideDuration={3200}
      />
    </Container>
  )
}
