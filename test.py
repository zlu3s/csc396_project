'''
model test
'''


import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn.functional import softmax
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Initialization
# -------------------------------------------------------
MODEL_DIR = "./results_model/lora"
CSV_PATH = "datasets/perc_test.csv"
DATA_COLUMN_NAME = "Poem"
MAX_LEN = 256                             # chunk length
STRIDE = 128                              # stride window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"**Testing**")
print(f"Model Directory: {MODEL_DIR}")
print(f"Test Dataset Directory: {CSV_PATH}")
print(f"Device: {DEVICE}")
print("-" * 30)

# -------------------------------------------------------
# 2. Load Model and tokenizer
# -------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
print("Finished loading Tokenizer..")

model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print("Finished loading Model")
print("-" * 30)

# -------------------------------------------------------
# 3. Load Dataset
# -------------------------------------------------------
class TestTextDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Loading Dataset: {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
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
# Keep batch_size = 1
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("-" * 30)

# -------------------------------------------------------
# 4. Striding Tokenizer
# -------------------------------------------------------
def sliding_window_encode(text, tokenizer, max_len=256, stride=128):
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    chunks = []
    start = 0
    while start < len(tokens):
        # [CLS][SEP]
        end = start + max_len - 2 
        chunk_tokens = tokens[start:end]

        encoded = tokenizer.prepare_for_model(
            chunk_tokens,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        chunks.append(encoded)
        start += stride

    return chunks
# -------------------------------------------------------
# 5. Model Inference
# -------------------------------------------------------
def predict_long_text(text):
    chunks = sliding_window_encode(text, tokenizer, MAX_LEN, STRIDE)
    all_logits = []

    with torch.no_grad():
        for ch in chunks:
            new_ch = {}
            for k, v in ch.items():
                if v.dim() == 1:
                    new_ch[k] = v.unsqueeze(0).to(DEVICE)
                else:
                    new_ch[k] = v.to(DEVICE)
            
            outputs = model(**new_ch)
            all_logits.append(outputs.logits.cpu())

    # Softmax all chuncks
    all_probs = [softmax(logits, dim=-1) for logits in all_logits]
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    pred = avg_probs.argmax(dim=-1).item()
    
    return pred

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
    TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())] # ['sad', 'joy', 'love', 'anger', 'fear', 'surprise']
    
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
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    base_name = os.path.basename(CVS_PATH)
    file_name_core = os.path.splitext(base_name)[0]
    new_file_name = f"{MODEL_DIR}_{file_name_core}_confusion_matrix.png"
  
    plt.savefig(new_file_name, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix heatmap saved as: {new_file_name}")

    # Simple Accuracy calculation (keeping it)
    correct = sum([1 if p == y else 0 for p, y in zip(predictions, labels)])
    acc = correct / len(labels)

    print("\n**Summary**")
    print(f"Average accuracy: {acc:.4f}")
    print(f"Number of samples: {len(labels)}")
    print("-" * 30)
else:
    print("Missing labels or a mismatch between the number of predictions and the number of labels.")
    print("-" * 30)


# -------------------------------------------------------
# 7. Save Results
# -------------------------------------------------------
'''
output_path = "perc_test_with_predictions.csv"
dataset.df["prediction"] = predictions

if "label" not in dataset.df.columns and labels:
    pass 
elif not dataset.df.empty and len(dataset.df) == len(labels):
    dataset.df["label"] = labels 

dataset.df.to_csv(output_path, index=False)
print(f"Prediction saved to: {output_path}")
'''