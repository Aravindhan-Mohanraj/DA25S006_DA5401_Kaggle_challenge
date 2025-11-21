# 🧮 DA5401 – Kaggle Metric Learning Challenge  
### *DA25S006 Aravindhan Mohanraj*

## 📘 Overview
This project tackles the **DA5401 Kaggle Challenge**, where the task is to predict a **fitness score (0–10)** between:

- an **AI Evaluation Metric Definition** (given as an embedding), and  
- a **Prompt–Response (PR) text pair**  

The score represents how well the response matches the evaluation metric, as judged by an LLM.

The notebook implements a **transformer-based regression model** using MPNet to learn this semantic similarity.

---

## 📂 Dataset Files

| File | Description |
|------|-------------|
| `metric_names.json` | List of metric names used to label embeddings |
| `metric_name_embeddings.npy` | 768-dim embeddings for each metric |
| `train_data.json` | Training samples with prompts, responses, metric names & judge scores |
| `test_data.json` | Test set without scores |
| `sample_submission.csv` | Kaggle submission format |

The dataset includes multiple languages: Tamil, Hindi, Bengali, Assamese, Bodo, Sindhi, English.

---

## 🚀 What the Notebook Does

### **1. Load and Prepare Data**
- Loads metric embeddings and matches them with PR-pair records.
- Loads the multilingual training text:
  - `prompt`
  - `system_prompt`
  - `expected_response`
- Loads judge scores from `train_data.json`.
- Tokenizes and vectorizes the text using **HuggingFace MPNet**.
- Combines text embeddings with metric embeddings.

---

### **2. Build the Regression Model**
- Uses **MPNet** as the text encoder.
- Adds a **fully connected regression head**.
- Loss function: **MSELoss**
- Optimizer: **Adam**
- Handles:
  - Device placement (GPU if available)
  - Dataloaders
  - Training epochs
  - Gradient updates
  - Validation loop

---

### **3. Train the Model**
- Batches training data
- Encodes text and metric embeddings
- Trains the MPNet + MLP model
- Displays training progress and loss values
- Saves the trained model

---

### **4. Run Inference on Test Data**
- Loads the saved model
- Loads and tokenizes test prompts
- Recomputes MPNet embeddings
- Concatenates embeddings with metric vectors
- Performs **forward pass** to generate predicted scores
- Output:

```
submission_mpnet_5Folds.csv
```

## 📈 Final Outputs

The notebook generates two submission files:

1. **`submission_mpnet_5Folds.csv`**  
   Raw model predictions (float scores)

File matches the Kaggle competition format.

---

## 🛠️ Requirements

- Python ≥ 3.9
- Libraries:
  - `torch`
  - `transformers`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `json`

---

## ▶️ Running the Notebook

```bash
jupyter notebook Kaggle_Assignment.ipynb
```

Ensure dataset files are placed in the same directory or update the paths accordingly.

---

## 🧠 Summary

This notebook implements a complete **transformer-based metric learning pipeline**:

- Text encoding using MPNet  
- Metric-text embedding fusion  
- Regression model prediction  
- Kaggle-ready submission generation  

The workflow aligns with the competition goal: learning a similarity function between metric definitions and PR text pairs.
