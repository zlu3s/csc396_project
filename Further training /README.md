## Sentiment Analysis Fine-Tuning Experiments

This repository contains Python scripts for conducting and evaluating sequence classification experiments using a pre-trained RoBERTa model. The experiments explore various fine-tuning strategies (Full, LoRA, Layer-Freezing) and the impact of data augmentation through transfer learning from a secondary domain.

---

### Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [File Descriptions](#file-descriptions)
3.  [Experiment Setup](#experiment-setup)
4.  [Running the Experiments](#running-the-experiments)
5.  [Evaluation and Results](#evaluation-and-results)

---

### 1. Prerequisites

Before running the scripts, ensure you have the necessary libraries and data setup:

* **Python Environment:** Python 3.8+
* **Libraries:** Install required libraries using `pip`:
    ```bash
    pip install torch pandas numpy matplotlib scikit-learn transformers peft seaborn tqdm
    ```
* **Model and Data:**
    * **Pre-trained Model:** Fine-tune the pre-trained RoBERTa with the code in '../Roberta Transformer'
    * **Datasets:** Ensure the data files are structured as follows:
        ```
        data/
        ├── perc_train.csv      # Primary (Poem) training set
        ├── perc_test.csv       # Primary (Poem) test set
        ├── 500songs_train.csv  # Secondary (Lyrics) training set
        └── 500songs_test.csv   # Secondary (Lyrics) test set
        ```

### 2. File Descriptions

| Filename | Description | Purpose |
| :--- | :--- | :--- |
| `training_experiments.py` | **Fine-Tuning Strategies** | Performs comparative training using four distinct fine-tuning methods (`cls_head`, `last_layers`, `full`, `lora`) on the primary poem dataset (`perc_train.csv`). |
| `experiment_transfer_learning.py` | **Data Mixing Experiment (LoRA)** | Investigates the effect of mixing poem data (100%) with varying ratios (0.0 to 1.0) of lyrics data. Evaluates the resulting model on both poem and lyrics test sets. |
| `test.py` | **Model Inference & Evaluation** | A generic script for evaluating a trained model, specifically designed to handle **long input texts** using a sliding window technique. It generates a classification report and a confusion matrix heatmap. |
| `test_dumb.py` | **Dumb Baseline Evaluation** | Calculates baseline metrics by simulating a simple classifier that always predicts the same, most frequent class (class 1: 'joy'). Useful for establishing a lower performance bound. |
| `plot.py` | **Result Visualization** | Generates bar plots from hardcoded performance scores to visually compare the effectiveness of the four fine-tuning strategies (`cls_head`, `last_layers`, `Full`, `Lora`) on the poem and lyrics test sets. |

### 3. Experiment Setup

The core experiments revolve around two main objectives:

#### A. Fine-Tuning Method Comparison (`training_experiments.py`)

This script compares four parameter-efficient fine-tuning (PEFT) and standard approaches:

1.  **`cls_head`**: Only the final classification layer is trained.
2.  **`last_layers`**: The classification layer and the last two transformer encoder layers are trained.
3.  **`full`**: All model parameters are fine-tuned.
4.  **`lora`**: LoRA (Low-Rank Adaptation) is applied to the `query` and `value` attention matrices. 

#### B. Data Transfer Learning (`experiment_transfer_learning.py`)

This script tests data augmentation by creating a mixed training set:

$$D_{mixed} = D_{poem}^{train} \cup D_{lyrics}^{train, sampled}$$

The sampling fraction for $D_{lyrics}$ is controlled by the `MIX_RATIOS` array, which includes `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. All models in this experiment use the **LoRA** fine-tuning method.

### 4. Running the Experiments

#### Step 1: Run Fine-Tuning Strategy Comparison

Execute the script to train four models using different methods on the poem dataset.

```bash
python training_experiments.py
````

  * **Output:** Trained models will be saved to `./results_model/{method}/`, and a combined metrics plot will be saved in each subdirectory.

#### Step 2: Run Transfer Learning Experiment

Execute the script to train six models, each with a different ratio of mixed lyrics data.

```bash
python experiment_transfer_learning.py
```

  * **Output:** Trained models will be saved to `./transfer_models/ratio_{ratio}/`. Final results are compiled in `experiment_results_dual_eval.csv`, and a plot is saved as `dual_evaluation_plot.png`.

### 5\. Evaluation and Results

#### Model Evaluation (`test_model.py`)

To evaluate any saved model (e.g., the best model from the `full` method), update the configuration in `test_model.py`. This script implements a sliding window strategy for long documents.

1.  Open `test_model.py`.
2.  Change `MODEL_DIR` to the path of the saved model (e.g., `./results_model/full`).
3.  Change `CSV_PATH` to your target test set (e.g., `datasets/perc_test.csv` or `datasets/500songs_test.csv`).
4.  Run the script:
    ```bash
    python test.py
    ```

#### Baseline Evaluation (`test_dumb.py`)

To establish the random baseline:

```bash
python test_dumb.py
```

  * **Output:** The classification report and confusion matrix for the dumb predictor will be printed and saved.

#### Plotting Results (`plot.py`)

To visualize the hardcoded results from the Fine-Tuning Strategy Comparison:

```bash
python plot.py
```

  * **Output:** Two bar plots, `poem_performance.png` and `lyrics_performance.png`, will be generated.
