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
## Research Questions & Scope
This work set out to answer three academic questions distilled from the dissertation:
1. **Can a deep‑learning model accurately label diverse soundscapes?** We trained CNNs that predict up to **527** AudioSet labels grouped into 20 custom classes.
2. **Is it better to treat audio embeddings as images or as raw vectors?** Approach 1 converts each 128‑D embedding sequence into a greyscale bitmap; Approaches 2‑3 feed the 128‑D vectors directly into 1‑D (and experimental 2‑D) convolutions.
3. **How much data is _enough_?** We compared balanced subsets of **500**, **22 176** and **150 000** clips, tracking validation accuracy and cluster separation.

## Dataset
* **Source** – Google **AudioSet** TFRecords: >2 M 10‑second clips pre‑embedded with *VGGish*  
* **Splits used**  
  * *Balanced train* 22 176 clips (all classes equally represented)  
  * *Unbalanced train* 127  824‑clip sample (Approach 3)  
  * *Eval* 20 383 clips (held‑out during training)  
* Each TFExample is parsed into **[T×128]** float32 matrices (≈ 96 frames per 10 s).

## Methodology at a Glance
| Approach | Input | Net type | Train size | Notes |
|----------|-------|----------|-----------:|-------|
| **1** | 128×T images | 2‑D CNN | 500 | heavy data‑aug, GPU‑light |
| **2** | raw embeddings | 1‑D CNN + BN | 22 k | best trade‑off size/accuracy |
| **3** | raw embeddings | 1‑D CNN (deeper) | 150 k | >2× clusters resolved on UMAP |

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

