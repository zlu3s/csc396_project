import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model

# ============================================================
#                   CONFIGURATIONS
# ============================================================
MODEL_DIR = "../Roberta Transformer/roberta-base-sentiments"   # Path to the base model
PERC_TRAIN = "data/perc_train.csv"     # Poem training set
PERC_TEST  = "data/perc_test.csv"      # Poem test set
SONGS_TRAIN = "data/500songs_train.csv" # Lyrics training set
SONGS_TEST  = "data/500songs_test.csv"  # Lyrics test set

# Mixing ratios: 0.0 means 100% poem data, 1.0 means 100% poem + 100% lyrics data
MIX_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {DEVICE}")

# ============================================================
#                   DATA UTILITIES
# ============================================================
def load_and_standardize(csv_path):
    """Reads CSV and standardizes column names to 'text' and 'label'."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Warning] File not found: {csv_path}")
        return pd.DataFrame(columns=['text', 'label'])

    # Standardize column names
    if 'Poem' in df.columns:
        df = df.rename(columns={'Poem': 'text'})
    elif 'lyrics' in df.columns:
        df = df.rename(columns={'lyrics': 'text'})
    
    # Keep only necessary columns and drop NaNs
    if 'text' in df.columns and 'label' in df.columns:
        return df[['text', 'label']].dropna()
    else:
        print(f"[Error] Columns missing in {csv_path}. Available: {df.columns}")
        return pd.DataFrame(columns=['text', 'label'])

def create_mixed_train_set(ratio):
    """Builds a mixed training set: 100% Poem + Ratio * Lyrics."""
    # 1. Load base data (Poems)
    df_perc = load_and_standardize(PERC_TRAIN)
    
    # 2. Load and sample Lyrics data
    df_songs = load_and_standardize(SONGS_TRAIN)
    df_songs_sample = pd.DataFrame()
    
    if ratio > 0 and not df_songs.empty:
        # Use random_state for reproducibility
        df_songs_sample = df_songs.sample(frac=ratio, random_state=42)
    
    # 3. Concatenate
    combined_df = pd.concat([df_perc, df_songs_sample], ignore_index=True)
    
    # 4. Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   [Dataset Info] Ratio {ratio}: Poems={len(df_perc)} + Lyrics={len(df_songs_sample)} -> Total={len(combined_df)}")
    return combined_df

# ============================================================
#                   Dataset & DataLoader
# ============================================================
class TextDataset(Dataset):
    """Generic text dataset for PyTorch."""
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch, tokenizer):
    """Collate function for tokenizing a batch of texts."""
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)

# ============================================================
#                   EVALUATION FUNCTION (Weighted F1)
# ============================================================
def evaluate(model, loader, desc="Evaluating"):
    """Evaluates the model and returns the weighted F1 score."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, mask, labels in tqdm(loader, desc=desc, leave=False):
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Key metric is weighted F1 score
    return f1_score(all_labels, all_preds, average='weighted')

# ============================================================
#                   MAIN EXPERIMENT LOGIC
# ============================================================
def run_experiment():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    
    # 1. Pre-load test sets (only once)
    print("\n--- Loading Test Sets ---")
    df_test_perc = load_and_standardize(PERC_TEST)
    df_test_songs = load_and_standardize(SONGS_TEST)
    
    test_loader_perc = DataLoader(TextDataset(df_test_perc), batch_size=BATCH_SIZE, 
                                  collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader_songs = DataLoader(TextDataset(df_test_songs), batch_size=BATCH_SIZE, 
                                   collate_fn=lambda b: collate_fn(b, tokenizer))
    
    results = {
        "ratio": [],
        "f1_perc": [],
        "f1_songs": []
    }

    # 2. Loop through different mixing ratios
    for ratio in MIX_RATIOS:
        print(f"\n{'='*40}")
        print(f"Starting Training with Lyrics Ratio: {ratio}")
        print(f"{'='*40}")
        
        # Prepare the mixed training set for the current ratio
        train_df = create_mixed_train_set(ratio)
        train_loader = DataLoader(TextDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, tokenizer))
        
        # Initialize the model with LoRA (reset for each ratio)
        base_model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
        peft_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"], task_type="SEQ_CLS"
        )
        model = get_peft_model(base_model, peft_config)
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        # Training loop
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"[Ratio {ratio}] Epoch {epoch+1}/{EPOCHS}")
            
            for input_ids, mask, labels in pbar:
                input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        # Evaluate on both test sets after training
        print(f"   -> Evaluating on PERC test set...")
        f1_perc = evaluate(model, test_loader_perc, desc="Test PERC")
        
        print(f"   -> Evaluating on SONGS test set...")
        f1_songs = evaluate(model, test_loader_songs, desc="Test SONGS")
        
        print(f"   [Result] Ratio {ratio} | F1 (Perc): {f1_perc:.4f} | F1 (Songs): {f1_songs:.4f}")
        
        # Record results
        results["ratio"].append(ratio)
        results["f1_perc"].append(f1_perc)
        results["f1_songs"].append(f1_songs)
        
        # Save the model for this ratio
        save_path = f"./transfer_models/ratio_{ratio}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)

    # 3. Final results and plotting
    print("\n\n================ FINAL RESULTS ================")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("experiment_results_dual_eval.csv", index=False)
    
    # Plotting F1 scores vs. ratio
    plt.figure(figsize=(10, 6))
    # Corrected dictionary keys for plotting
    plt.plot(results["ratio"], results["f1_perc"], 'o-', label='Poem Test Set', linewidth=2)
    plt.plot(results["ratio"], results["f1_songs"], 's--', label='Lyrics Test Set', linewidth=2)
    

    plt.xlabel("Ratio of Lyrics Data Added")
    plt.ylabel("Weighted Avg F1 Score")
    plt.title("Transfer Learning Performance by Data Mixing Ratio (LoRA)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("dual_evaluation_plot.png")
    print("Plot saved to 'dual_evaluation_plot.png'")

if __name__ == "__main__":
    run_experiment()
    