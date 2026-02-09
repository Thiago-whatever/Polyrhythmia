# 🥁 Polyrhythmia

> **Polyrhythmia** is an AI system that generates complex rhythmic patterns (polyrhythms) across multiple genres, tempos, and time signatures — combining deep learning (LSTM-based models) with an interactive web interface for real-time exploration.

---

## 🎵 Overview

**Polyrhythmia** combines two main components:

1. **Generative Core (AI Model)** – A deep recurrent network (LSTM) trained on the Groove MIDI Dataset to learn multi-instrument drum patterns and generate new, stylistically coherent sequences.  
2. **User Interface (UI)** – An interactive web app that allows musicians and researchers to:
   - Choose genres, tempos, and style embeddings  
   - Generate rhythmic loops  
   - Play and visualize the resulting polyrhythms directly in the browser  

Together, they form a complete AI-assisted rhythm creation environment — bridging data-driven generative music and human creativity.

---

## 🧠 System Architecture

<p align="center">
   <img width="706" height="342" alt="image" src="https://github.com/user-attachments/assets/ea442dfe-9a68-4526-b453-8fd14af5fcb8" />
</p>

| Component | Description |
|------------|-------------|
| **model_lstm/** | Contains Python notebooks and training scripts for the LSTM-based rhythm generator. |
| **midi_experiments/** | Includes data processing pipelines, MIDI tokenization utilities, and experimental setups. |
| **Polyrhythmia_UI/** | React-based front-end for interactive rhythm generation, playback, and export. |
| **music21_experiments/** | Exploratory scripts for analyzing rhythmic patterns and metric relationships using the `music21` library. |

---

## ⚙️ Features

- 🎚️ **Genre control** — Generate loops in styles like jazz, samba, bossa, hip-hop, or afro-cuban.
- 🔁 **30-second rhythmic continuity** — Models trained for long-term rhythmic consistency.
- 🧩 **Polyrhythmic texture generation** — Multi-instrument simultaneous activations with probabilistic diversity.
- 🎛️ **Dynamic temperature sampling** — Top-k, nucleus (top-p), and seed variation for each sample.
- 💻 **Browser-based UI** — Visualizes generated patterns and allows exporting `.mid` or `.wav` loops.
- 🔉 **High-fidelity audio synthesis** — Uses the FluidR3_GM SoundFont via PrettyMIDI + FluidSynth backend.
- 📈 **Metrics-driven validation** — Controls density and polyphony deviations to prevent over-activation noise.

---

## 🧩 Core Model (LSTM Generator)

The LSTM-based model is trained to predict instrument activations per time-step across 9 drum channels:

```

[KICK, SNARE, HH_CLOSED, HH_OPEN, TOM_LOW, TOM_MID, TOM_HIGH, CRASH, RIDE]

````

### Architecture Summary

| Layer | Output Shape | Parameters |
|-------|---------------|------------|
| Input (tokens, pos, style) | (None, 16, 176) | — |
| LSTM (1) | (None, 16, 384) | 861,696 |
| LSTM (2) | (None, 16, 384) | 1,181,184 |
| Dense + Softmax | (None, 16, 128) | 49,280 |
| **Total params** | — | **~2.1M** |

### Key Details

- **Loss:** Custom Sparse Categorical Cross-Entropy with Label Smoothing (`SparseCELS`)  
- **Conditioning:** Genre/style embedding vector (`one-hot` over 6 genres)  
- **Sequence Length:** 16 steps per bar, up to 30 seconds generated  
- **Sampling Controls:** Randomized `top_k`, `temperature`, and `top_p` per bar  

---

## 🧪 Experiments

All training and generation notebooks are in `model_lstm/` and `midi_experiments/`.

| Notebook | Purpose |
|-----------|----------|
| `01_preprocess_data.ipynb` | Converts Groove MIDI Dataset to token + bitmask format |
| `02_train_model_v1.ipynb` | Baseline training with fixed embeddings |
| `03_train_model_v2.ipynb` | Adds label smoothing and regularization |
| `04_gen_with_soundfont_model2.ipynb` | SoundFont-based sample generation |
| `05_gen_with_soundfont_model3.ipynb` | Extended generation with diversity and post-processing |
| `utils_debug_metrics.ipynb` | Density, polyphony, and diagnostic visualizations |

---

## 🖥️ Polyrhythmia UI

<p align="center">
   <img width="710" height="399" alt="image" src="https://github.com/user-attachments/assets/6f52fcc5-de52-416b-9e29-0450a47a02c8" />
</p>

The **UI component** (in `/Polyrhythmia_UI/Polyrhythmia-ui`) provides a front-end built with **React** and **Web Audio API** for interactive rhythmic exploration.

### Features
- 🎛️ Select genre, tempo, and complexity level  
- 🎶 Preview generated loops directly in browser  
- ⬇️ Export to `.wav` or `.mid`  
- 🪄 Trigger model regeneration through backend API  

### Tech Stack
| Layer | Technology |
|--------|-------------|
| Front-end | React, Tone.js, Web Audio API |
| Backend | Python (Flask/FastAPI wrapper for model inference) |
| Model Integration | TensorFlow / Keras |
| Deployment | Local or cloud container (Docker-ready) |

---

## 🔉 Example Outputs

| Genre | Duration | Dens_L1 | K_L1 | Sample |
|--------|-----------|----------|------|--------|
| Jazz | 30.2 s | 0.074 | 0.073 | 🎧 `jazz_sample01.wav` |
| Samba | 30.4 s | 0.091 | 0.085 | 🎧 `samba_sample03.wav` |
| Hip-Hop | 29.9 s | 0.101 | 0.094 | 🎧 `hiphop_sample02.wav` |

Each genre is generated with a **unique seed and sampling configuration** to ensure diversity.

---

## 📊 Evaluation Metrics

| Metric | Meaning | Target |
|---------|----------|--------|
| `density_L1` | Absolute deviation from expected activation density | < 0.25 |
| `K_L1` | Polyphony deviation | < 0.15 |
| `duration` | Approx. 30 seconds | ✓ |
| `variety` | Seed-controlled stylistic randomness | High |

---

## 🧱 Installation

```bash
# Clone repository
git clone https://github.com/Thiago-whatever/Polyrhythmia.git
cd Polyrhythmia

# Create environment
conda create -n poly-lstm python=3.11
conda activate poly-lstm

# Install dependencies
pip install -r requirements.txt

# Run backend inference server
python src/server/run_inference.py

# Run UI (React)
cd Polyrhythmia_UI/Polyrhythmia-ui
npm install
npm start
````

---

## 🚀 Usage

1. Launch the backend server (`Flask` or `FastAPI` model endpoint).
2. Start the React front-end.
3. Choose a genre and tempo.
4. Generate and listen to 30-second loops.
5. Export audio or MIDI for DAW integration.

---

## 🧭 Roadmap

* [ ] Add Transformer-based continuation for long rhythmic structures
* [ ] Integrate real-time model inference into the UI
* [ ] Support multi-track visualization (per instrument)
* [ ] Evaluate with groove and syncopation metrics
* [ ] Optional fine-tuning with Latin rhythm datasets

---

## 🧾 Citation

> Flores-Larrondo, S. (2025).
> *Polyrhythmia: Generative LSTM Models for Rhythmic Sequence Synthesis.*
> ITAM – Departamento de Computación.

---

## 💡 Acknowledgments

* Google Magenta – Groove MIDI Dataset
* TensorFlow / Keras
* PrettyMIDI, PyFluidSynth, SoundFile
* `music21` Library for rhythmic analysis
* Advisors and collaborators for guidance

---

## 📬 Contact

* **Author:** Santiago Flores-Larrondo
* **Affiliation:** ITAM
* **Email:** [[s.flores.l@outlook.com](mailto:s.flores.l@outlook.com)]
* **GitHub:** [@Thiago-whatever]((https://github.com/Thiago-whatever))

---

*“Where rhythm meets intelligence.”*


