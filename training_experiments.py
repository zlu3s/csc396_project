import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score

from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ============================================================
#                   CONFIGURATIONS
# ============================================================
MODEL_DIR = "./roberta-base-sentiments"
TRAIN_CSV = "datasets/perc_train.csv"
TEST_CSV  = "datasets/perc_test.csv"

TEXT_COL = "Poem"
LABEL_COL = "label"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

METHODS = ["cls_head", "last_layers", "full", "lora"]


# ============================================================
#                   DATASET DEFINITIONS
# ============================================================
class TextDataset(Dataset):
    """Dataset for training and evaluation."""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[TEXT_COL])
        self.texts = df[TEXT_COL].tolist()
        self.labels = df[LABEL_COL].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================================
#                   COLLATE FUNCTION
# ============================================================
def collate_fn(batch, tokenizer):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return enc["input_ids"], enc["attention_mask"], labels


# ============================================================
#              ACCURACY EVALUATION FUNCTION
# ============================================================
def evaluate_accuracy(model, loader):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return f1

# ============================================================
#               PLOTTING FUNCTION
# ============================================================
def plot_history_combined(train_losses, val_f1s, method_name, save_dir):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx() 

    color_f1 = 'tab:red'
    ax2.set_ylabel('F1 Score', color=color_f1) 
    ax2.plot(epochs, val_f1s, 'r-s', label='Validation F1', color=color_f1)
    ax2.tick_params(axis='y', labelcolor=color_f1)

    plt.title(f'{method_name} - Training Metrics Over Epochs')
    
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')
    
    fig.tight_layout()
    plot_path = os.path.join(save_dir, f"{method_name}_metrics_combined.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved → {plot_path}")

# ============================================================
#               TRAINING LOOP (USED BY ALL METHODS)
# ============================================================
def train(model, tokenizer, train_loader, test_loader, lr, save_path, method_name):
    """Universal training loop for all 4 methods."""
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    print(f"\n========== Training: {method_name} ==========")
    
    train_losses = []
    val_f1s = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[{method_name}] Epoch {epoch+1}/{EPOCHS}")

        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        f1 = evaluate_accuracy(model, test_loader)
        
        train_losses.append(avg_loss)
        val_f1s.append(f1)
        
        print(f"[{method_name}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1-score: {f1:.4f}")

    plot_history_combined(train_losses, val_f1s, method_name, save_path)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved → {save_path}")


# ============================================================
#                       MAIN LOGIC
# ============================================================
def run_all_experiments():

    # Load tokenizer once
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)

    # Load datasets
    train_data = TextDataset(TRAIN_CSV)
    test_data  = TextDataset(TEST_CSV)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    for method in METHODS:
        print("\n" + "=" * 60)
        print(f"Running experiment: {method}")
        print("=" * 60)

        # Load base model fresh each time
        model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)

        # --------------------------------------------------
        # Method A: classifier head only
        # --------------------------------------------------
        if method == "cls_head":
            lr = 3e-5
            for p in model.roberta.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True

        # --------------------------------------------------
        # Method B: last 2 transformer layers
        # --------------------------------------------------
        elif method == "last_layers":
            lr = 1e-5
            N = 2
            total_layers = 12
            target = list(range(total_layers - N, total_layers))

            for name, p in model.named_parameters():
                p.requires_grad = False

            for name, p in model.roberta.named_parameters():
                if any(f"layer.{i}" in name for i in target):
                    p.requires_grad = True

            for p in model.classifier.parameters():
                p.requires_grad = True

        # --------------------------------------------------
        # Method C: full finetuning
        # --------------------------------------------------
        elif method == "full":
            lr = 2e-5
            for p in model.parameters():
                p.requires_grad = True

        # --------------------------------------------------
        # Method D: LoRA
        # --------------------------------------------------
        elif method == "lora":
            lr = 1e-4
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_cfg)
            model.to(DEVICE)
            model.print_trainable_parameters()

        # --------------------------------------------------
        # Run training
        # --------------------------------------------------
        save_dir = f"./results_model/{method}"
        os.makedirs(save_dir, exist_ok=True)

        train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            save_path=save_dir,
            method_name=method,
        )


# ============================================================
#                          ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_all_experiments()