
# The Ghost In The Machine

### Precog Recruitment 2026 | NLP Task

This repository contains the complete implementation of the Precog 2026 NLP recruitment task. The project explores the evolving boundary between human-written Victorian literature (Jane Austen and Charles Dickens), AI-generated mimicry and regular AI-generated text.

---

## Project Overview

The core of this project is a Detection System designed to identify three classes of text:

1. **Human**: Original prose from Austen and Dickens.
2. **AI Neutral**: Standard LLM-generated text on Victorian topics.
3. **AI Mimicked**: LLM-generated text explicitly prompted to mimic Victorian stylometry.

---

## Tasks Completed

### Task 0: The Library Of Babel

### Task 1: The Fingerprint

### Task 2: The Multi-Tiered Detective

### Task 3: The Smoking Gun

### Task 4: The Turing Test

---

## Details & Execution Instructions

Each task is contained in a standalone Jupyter Notebook.

1. **`task0.ipynb`**:
* Gemini API Key must be uploaded to colab secrets as `GEMINI_API_KEY`
* This file generates the dataset (~3000 samples). 
* I accidentally cleared the cell outputs after downloading the dataset, that is why the cell outputs are empty.
* Upload `prompt.py` to the session storage before running the AI generated paragraphs generation cells.
* It takes ~2 hours to generate (scraping ~2000 human samples, generating 1000 AI generated samples using repeated gemini API calls)
* The generated datasets can be found in `/datasets`
* `/datasets/complete_dataset.csv` is the one used for the task here on.

2. **`task1.ipynb`**:
* To run this, all the files in `datasets/` need to uploaded to colab session storage.
* In case there are errors in the `pip install` or `import` cells, restart the session and run again.
* In case a line of code is loading a file from Google drive, comment the line out and use `pd.read_csv(filename)`. Similarly in case of saving to Google Drive, use `<name>.to_csv()` instead.


3. **`task2_tertiary.ipynb`** and **`task2_binary.ipynb`**:
* GPU recommended
* The tertiary model classifies into Human/AI_Neutral/AI_Mimicked. The binary model does Human/AI
* To run these, `/datasets/complete_dataset.csv` and `results/task1_results/task1_results.csv` needs to be uploaded to colab session storage. Additionally, `external_datasets/your_dataset_5000.csv` needs to be uploaded for an external test (not needed as per task requirements)
* In case there are errors in the `pip install` or `import` cells, restart the session and run again.
* In case a line of code is loading a file from Google drive, comment the line out and use `pd.read_csv(filename)`. Similarly in case of saving to Google Drive, use `<name>.to_csv()` instead.

4. **`task3_tertiary.ipynb`** and **`task3_binary.ipynb`**:
* GPU recommended
* Since the task document asks to analyse the best performing model, I made two notebooks since I found ffnn to be better for binary classification and distilbert to be better for three way.
* `/datasets/complete_dataset.csv` has to be uploaded to colab session storage.
* Both scripts require the model from task2 to be uploaded: `/models/tierC_transformer_tertiary/` and `models/tierB_embedding_nn_binary.pt` respectively. To upload the folder for tierC, you might need to make a folder on colab first and then upload all the files inside that.
* `results/task2_results/task2_forensic_results_tertiary.csv` as well for `task3_tertiary.ipynb`.
* In case there are errors in the `pip install` or `import` cells, restart the session and run again.
* In case a line of code is loading a file from Google drive, comment the line out and use `pd.read_csv(filename)`. Similarly in case of saving to Google Drive, use `<name>.to_csv()` instead.

5. **`task4.ipynb`**:
* GPU recommended
* Gemini API Key must be uploaded to colab secrets as `GEMINI_API_KEY`
* `models/tierB_embedding_nn_binary.pt` needs to be uploaded to colab session storage.
* In case there are errors in the `pip install` or `import` cells, restart the session and run again.
* In case a line of code is loading a file from Google drive, comment the line out and use `pd.read_csv(filename)`. Similarly in case of saving to Google Drive, use `<name>.to_csv()` instead.
---
