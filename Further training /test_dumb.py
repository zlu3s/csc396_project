import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIR = "../Roberta Transformer/roberta-base-sentiments"
CSV_PATH = "data/500songs_test.csv"
DATA_COLUMN_NAME = "lyrics"

class TestTextDataset(Dataset):
    """Dataset for loading text and labels."""
    def __init__(self, csv_path):
        print(f"Loading Dataset: {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=[DATA_COLUMN_NAME]).reset_index(drop=True)
            self.texts = df[DATA_COLUMN_NAME].tolist()
            if "label" not in df.columns:
                print("Couldn't find 'label'. Will use 'None'")
                self.labels = [None] * len(self.texts)
            else:
                self.labels = df["label"].tolist()
            self.df = df
            print(f"Finished loading dataset, total samples: {len(self.texts)}")
        except FileNotFoundError:
            print(f"Error: couldn't find dataset: {csv_path}")
            raise
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

dataset = TestTextDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("-" * 30)


def predict_long_text(text):
    """Simulates a 'dumb' classifier that always predicts label 1 ('joy')."""
    return 1

predictions = []
labels = []

for text, label in tqdm(loader, desc="Predicting"):
    text = text[0]
    if label[0] is not None:
        label_val = int(label[0])
        labels.append(label_val)
    else:
        label_val = None 

    pred = predict_long_text(text)
    predictions.append(pred)

print("Finished.")
print("-" * 30)

# -------------------------------------------------------
# 6. Metric
# -------------------------------------------------------

if labels and len(predictions) == len(labels):

    from sklearn.metrics import classification_report, confusion_matrix
    
    LABEL_NAMES = {
        0: "sad",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())] 
    
    ALL_LABELS = list(LABEL_NAMES.keys()) 

    # Classification Report
    report = classification_report(
        labels, 
        predictions, 
        labels=ALL_LABELS,
        target_names=TARGET_NAMES, 
        zero_division=0
    )
    print("\n**Classification Report**")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions, labels=ALL_LABELS) 
    cm_df = pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES)
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    
    plt.title('Confusion Matrix (Dumb Baseline)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    base_name = os.path.basename(CSV_PATH)
    file_name_core = os.path.splitext(base_name)[0]
    new_file_name = f"DUMB_BASELINE_{file_name_core}_confusion_matrix.png"
  
    plt.savefig(new_file_name, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix heatmap saved as: {new_file_name}")

    # Simple Accuracy calculation
    correct = sum([1 if p == y else 0 for p, y in zip(predictions, labels)])
    acc = correct / len(labels)

    print("\n**Summary**")
    print(f"Average accuracy: {acc:.4f}")
    print(f"Number of samples: {len(labels)}")
    print("-" * 30)
else:
    print("Missing labels or a mismatch between the number of predictions and the number of labels.")
    print("-" * 30)
    