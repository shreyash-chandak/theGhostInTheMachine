This `README.md` is designed to be the professional landing page for your research project. It synthesizes your technical implementations, your "messy research" insights (like the Gemma reasoning loss), and the final adversarial findings.

---

# The Detective and The Turing Test: Adversarial Stylometry

### Precog Research Group Recruitment 2026 | NLP Theme

This repository contains the complete implementation of the Precog 2026 NLP recruitment tasks. The project explores the evolving boundary between human-written Victorian literature (Jane Austen and Charles Dickens) and AI-generated mimicry. It moves from baseline statistical detection to deep learning interpretability and adversarial bypass techniques.

---

## 🚀 Project Overview

The core of this project is a **Multi-Tiered Detection System** designed to identify three classes of text:

1. **Human**: Original prose from Austen and Dickens.
2. **AI Neutral**: Standard LLM-generated text on Victorian topics.
3. **AI Mimicked**: LLM-generated text explicitly prompted to mimic Victorian stylometry.

---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.9+
* Google Colab (recommended for GPU-accelerated Tier C) or a local machine with an NVIDIA GPU.

### Dependencies

```bash
pip install pandas numpy xgboost scikit-learn torch spacy textstat nltk gensim transformers peft accelerate datasets captum joblib
python -m spacy download en_core_web_sm

```

### Resource Requirements

* **GloVe Embeddings**: The scripts automatically download `glove-wiki-gigaword-100` via the `gensim` API.
* **Google Drive**: If running on Colab, ensure your `MODEL_DIR` path is set to save/load `.pkl`, `.pt`, and LoRA adapters.

---

## 📂 Tasks Completed

### Task 0: Generation of Dataset

* **Process**: Shifted from Gemini 1.5 Flash (due to rate limits) to **Gemma 3 27B**.
* **Insights**: Overcame "Reasoning Loss" in Gemma by implementing **Topic-Cycling** and concise prompting to reduce duplicates from ~180 to <5.
* **Sanitization**: Removed all direct speech (dialogue) from human texts to ensure the models focused on prose rhythm rather than Victorian punctuation cues.

### Task 1: The Detective - Statistical

* Extracted features: TTR, Hapax Legomena, POS Ratios, Tree Depth, and Flesch-Kincaid.
* **The Complexity Paradox**: Discovered that AI text often yields higher complexity scores (TTR/Tree Depth) than human text because LLMs generate high-density individual paragraphs, whereas human prose flows with varied coherence.

### Task 2: Multi-Tiered Classification

* **Tier A (XGBoost)**: Baseline detection using Task 1 features.
* **Tier B (FFNN)**: Neural Network using weighted GloVe embeddings.
* **Tier C (Transformer)**: DistilBERT fine-tuned with **LoRA** (Low-Rank Adaptation).
* **Binary Track**: Implemented a parallel "Shadow Binary" classifier to compare 3-way forensic accuracy against standard Human vs. AI detection.

### Task 3: The Smoking Gun

* **Interpretability**: Used **Captum (Layer Integrated Gradients)** to generate word-level saliency maps.
* **Findings**: Identified "AI-isms" (*tapestry, delve, testament*) as high-attribution features for the Transformer.
* **Error Analysis**: Documented cases where the "over-correcting" of AI mimicry (using too many archaic words) actually helped the detector flag the text as synthetic.

### Task 4: The Turing Test

* **The Humanizer**: An adversarial pipeline that replaces AI-isms with synonyms and injects "bursty" sentence rhythms.
* **Genetic Algorithm**: Implemented a GA where the fitness function is the "Human" probability score. Successfully evolved AI paragraphs to achieve **>90% Human confidence**.

---

## 📖 Execution Instructions

Each task is contained in a standalone Jupyter Notebook. Follow this sequence:

1. **`task1_statistical_detective.ipynb`**:
* Load your raw CSV.
* Run the feature extraction cell (Note: takes ~10-15 mins for 1500 rows).
* Generates `task1_features.csv`.


2. **`task2_multi_tiered_detective.ipynb`**:
* Loads features from Task 1.
* Trains and saves all three models to your `MODEL_DIR`.


3. **`task3_smoking_gun.ipynb`**:
* Loads the Tier C model.
* Run the attribution cells to visualize "AI-isms" in red/green.


4. **`task4_adversarial_mirror.ipynb`**:
* Contains the Genetic Algorithm.
* Requires an active connection to an LLM API (or manual input) for the "Mutation" step.



---

## 📈 Results Summary

| Model | 3-Way Accuracy | Binary F1-Score |
| --- | --- | --- |
| Tier A (XGBoost) | ~XX% | ~XX% |
| Tier B (FFNN) | ~XX% | ~XX% |
| Tier C (DistilBERT) | ~XX% | ~XX% |

---

## 📜 Citations

* OpenAI: *New AI classifier for indicating AI-written text*.
* Kaggle: *Human vs AI Text Dataset*.
* Nature: *Transformers in Stylometry (2025)*.
* *Full list of citations available in the LaTeX report.*

---

**Author**: Shreyash Chandak

**Affiliation**: IIIT Hyderabad

**Contact**: [Your Email/Contact Info]