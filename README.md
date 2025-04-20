# Environment‑Sound Classification with Deep Learning

> **Using Deep Learning to Distinguish Environments from their Sounds**

This project tackles the problem of recognising environments from raw audio alone. It investigates how far convolutional neural networks—fed with Google AudioSet’s VGGish embeddings—can tell apart soundscapes such as urban traffic, birdsong or indoor speech, and what training‑set size is needed to do so reliably. Alongside the models it ships an interactive Tkinter + UMAP tool that lets you drop up to 500 pre‑processed .npy clips and immediately see where each new recording sits among the learned classes.

---
## Key Features
| Module | Tech | Purpose |
|--------|------|---------|
| **Approach 1 – 2D‑CNN on Spectrogram‑like Images** | Keras · TensorFlow | Treats each 128×N embedding as an 8‑bit image; heavy data‑augmentation |
| **Approach 2 – 1D‑CNN on Raw Embeddings** | Keras · TensorFlow | Faster, memory‑light training on the full 22 k balanced AudioSet sample |
| **Approach 3 – 1D‑CNN (150 k samples)** | ↑ as above | Scales to 150 k balanced + unbalanced samples; best validation accuracy |
| **GUI** | Tkinter · UMAP | Upload ≤ 500 `.npy` files, pick a model, interactive 2‑D scatter with tooltips |
| **Notebooks** | Colab‑ready IPYNB | Reproduce training for each approach |

---
## Getting Started
### Install dependencies
> Tested with **Python 3.10** and **TensorFlow 2.15 (CPU / GPU)**
```bash
pip install tensorflow keras numpy matplotlib pandas umap-learn scikit-learn pillow
```

### 2  Prepare data
Google’s AudioSet embeddings are released as TFRecords. Run one of the notebooks in `notebooks/` to:
1. **Parse** TFRecords → `.npy` arrays (shape =`[num_frames, 128]`).
2. **(opt.)** Down‑sample or balance classes.
3. Save to `data/balanced/` or `data/unbalanced/`.

---
## Results
| Model | Training set | Val. Acc |
|-------|--------------|---------:|
| **Model 1** | 500 spectrogram‑images | 0.35 |
| **Model 2** | 22 k balanced embeddings | 0.41 |
| **Model 3** | 150 k balanced + unbalanced | **0.43** |

Model 3 also doubles the number of discernible clusters on UMAP (4 versus 2), giving clearer separation of human, music, animal and environmental sounds.

---
## Limitations & Future Work
* **Imbalanced classes:** unbalanced AudioSet still dominates some labels; consider focal loss or class weights.
* **Real‑time pipeline:** current GUI expects offline VGGish embeddings—wrap the VGGish model for live microphone streaming.
* **Unsupervised variant:** experiment with VAE / Contrastive Audio Embedding for better cluster‑quality without labels.

---
## License
Released under the **MIT License** – see `LICENSE`.

