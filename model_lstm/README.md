# 🥁 Polyrhythmia LSTM Model

This project trains and evaluates an **improved LSTM model for drum pattern generation**, using tokenized sequences of 16-step × 9-instrument bars.

---

## 📁 Project Structure

```plaintext
model_lstm/
│
├── configs/
│   ├── dataset_2.yaml
│   ├── training_2.yaml
│
├── data/
│   └── processed/
│       └── dataset_2/
│           ├── train.npz
│           ├── validation.npz
│           ├── test.npz
│           ├── vocab_topN.json
│
├── models/
│   ├── checkpoints_improved/
│   │   └── best_2.h5
│   └── final/
│       ├── best_2.h5
│       ├── best_2_ls010_do035_vocab128.h5
│
├── outputs/
│   ├── samples_bars.jsonl
│   └── samples_bars.mid
│
├── runs/
│   ├── improved/
│   │   └── YYYYMMDD_HHMMSS/ (logs + hparams.json)
│   ├── improved_ls010_do035_vocab128/
│   │   └── YYYYMMDD_HHMMSS/
│
├── notebooks/
│   └── (Jupyter notebooks for inference and audio playback)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── quantize_and_tokenize.py
│   ├── infer/
│   │   └── sample_loops.py
│   ├── modeling/
│   │   └── model_lstm_2.py
│   ├── train/
│   │   └── train_lstm_2.py
│   ├── metrics/
│   │   └── rythm_metrics.py
│   └── utils/
│       └── export_to_midi.py
```

---

## 🧠 Workflow Overview

### 1. Data Processing

* Each bar is tokenized as a **16-step × 9-instrument bitmask**.
* Data is split into `train`, `validation`, and `test`.
* A `vocab_topN.json` vocabulary file is built with **cap = 128** tokens to reduce long-tail noise.
* Final datasets are stored in `data/processed/dataset_2/`.

---

### 2. Model Training

* LSTM model with token embeddings, positional embeddings, and style projection.
* Two stacked LSTM layers (384 units each), with configurable dropout.
* Custom loss with **label smoothing** and **perplexity** as an evaluation metric.

```bash
python -m src.train.train_lstm_2
```

* Checkpoints are stored in `models/checkpoints_improved/`.
* Best models are saved in `models/final/`.
* Logs and hyperparameters are stored under `runs/`.

---

### 3. Inference & Sampling

Generate new drum loops from a trained model:

```bash
python -m src.infer.sample_loops \
  --model_path "models/final/best_2_ls010_do035_vocab128.h5" \
  --vocab_path "data/processed/dataset_2/vocab_topN.json" \
  --num_bars 32 \
  --temperature 0.9 \
  --top_k 10 \
  --out_json "outputs/samples_bars.jsonl"
```

---

### 4. MIDI Export

Convert generated sequences to MIDI:

```bash
python -m src.utils.export_to_midi \
  --bars_jsonl "outputs/samples_bars.jsonl" \
  --vocab_path "data/processed/dataset_2/vocab_topN.json" \
  --out_midi "outputs/samples_bars.mid" \
  --bpm 100 \
  --velocity 95 \
  --note_len 0.06
```

For playback inside notebooks, the project uses `pretty_midi` and `pyFluidSynth` with a **SoundFont (.sf2)** such as `FluidR3_GM.sf2`.

---

## 📊 Model Training Results

| Model                              | LS   | Dropout | Vocab Cap | Train Acc | Val Acc | Test Acc | Test Perplexity |
| ---------------------------------- | ---- | ------- | --------- | --------- | ------- | -------- | --------------- |
| Baseline (best_2.h5)               | 0.05 | 0.30    | 166       | ~0.67     | ~0.55   | 0.5491   | 4.7762          |
| Tuned (ls=0.1, do=0.35, vocab=128) | 0.10 | 0.35    | 128       | ~0.67     | ~0.55   | 0.5559   | 4.8042          |
| Extended training (final)          | 0.10 | 0.35    | 128       | 0.6749    | 0.5538  | 0.5559   | 4.8042          |

✅ Label smoothing = 0.1 improved calibration.
✅ Higher dropout increased robustness.
✅ Top-N vocabulary (128) reduced noise from rare tokens.
✅ ReduceLROnPlateau + EarlyStopping improved convergence stability.

---

## 🎧 Notebook Playback

Inside the `notebooks/` directory, you can create a Jupyter notebook to:

* Load generated bars
* Convert them to PrettyMIDI
* Play audio directly inside the notebook

```python
import json, pretty_midi, IPython.display as ipd
from src.utils.export_to_midi import load_vocab, export_bars_to_midi

VOCAB_PATH = "../data/processed/dataset_2/vocab_topN.json"
SAMPLES_JSONL = "../outputs/samples_bars.jsonl"
TEMP_MIDI = "temp.mid"

# Load bars
with open(SAMPLES_JSONL, "r", encoding="utf-8") as f:
    bars = [json.loads(line)["tokens"] for line in f]

_, id2bitmask, unk_id = load_vocab(VOCAB_PATH)

# Export to temporary MIDI
export_bars_to_midi(bars[:8], id2bitmask, TEMP_MIDI, bpm=100)

# Playback
midi_data = pretty_midi.PrettyMIDI(TEMP_MIDI)
audio_data = midi_data.fluidsynth(sf2_path="C:/tools/fluidsynth/soundfonts/FluidR3_GM.sf2")
ipd.Audio(audio_data, rate=44100)
```

---

## 📝 Installation Notes

### Main Dependencies

* Python 3.10
* TensorFlow 2.x
* pretty_midi
* pyfluidsynth
* numpy, argparse, IPython

### Installing FluidSynth on Windows

1. Install FluidSynth manually and place DLLs in:

   ```
   C:\tools\fluidsynth\bin
   ```
2. Download a SoundFont (e.g., `FluidR3_GM.sf2`) and place it in:

   ```
   C:\tools\fluidsynth\soundfonts\
   ```
3. Verify installation:

   ```bash
   python -c "import fluidsynth; print('Fluidsynth OK')"
   ```

---

## 🚀 Next Steps

* Fine-tune hyperparameters (label smoothing 0.1–0.15, dropout 0.35–0.4).
* Increase `swap_prob` (0.08–0.10) for data augmentation.
* Experiment with creative sampling: `temp=1.0`, `top_k=20`.
* Implement qualitative metrics (note density, per-instrument F1, exact bar match %).

---

## ⚡ Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/model_lstm.git
cd model_lstm
```

### 2. Create and Activate a Conda Environment

It's recommended to use **Python 3.10** and a virtual environment to avoid dependency conflicts.

```bash
conda create -n poly-lstm python=3.10
conda activate poly-lstm
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can generate one from your environment with:

```bash
pip freeze > requirements.txt
```

Minimum required packages include:

```
tensorflow>=2.10
numpy
pretty_midi
pyfluidsynth
IPython
```

### 4. Install FluidSynth (Windows)

FluidSynth is required for audio playback of generated MIDI files inside notebooks.

1. Download the **FluidSynth Windows binary** from:
   👉 [https://github.com/FluidSynth/fluidsynth/releases](https://github.com/FluidSynth/fluidsynth/releases)

2. Extract it to:

   ```
   C:\tools\fluidsynth\
   ```

   Ensure the `bin` folder contains `fluidsynth.dll`.

3. Download a General MIDI **SoundFont (.sf2)** — e.g., [FluidR3_GM.sf2](https://member.keymusician.com/Member/FluidR3_GM/index.html)

   Place it in:

   ```
   C:\tools\fluidsynth\soundfonts\
   ```

4. Test installation:

   ```bash
   python -c "import fluidsynth; print('Fluidsynth OK')"
   ```

If the DLL is not found, add this to your environment variables or edit `fluidsynth.py` to point to your `C:\tools\fluidsynth\bin` directory.

---

### 5. Launch Jupyter Notebook (Optional)

To visualize and play generated loops:

```bash
conda install jupyter -y
jupyter notebook
```

Open any notebook under `notebooks/` and run the inference and audio playback cells.


---

**Author:**
LSTM Polyrhythm Generation Project — Polyrythmia — 2025
Santiago Flores Larrondo
📌 Developed in an Anaconda environment (Windows 10, TensorFlow 2.x)

