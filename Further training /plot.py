import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.3)

blue = "#2E5AAC"
red = "#D1495B"
baseline_color = "#2A2A2A"

methods = ["cls_head", "last_layers", "Full", "Lora"]

# Weighted F1 scores for the Poem dataset
poem_scores = [0.46, 0.44, 0.60, 0.57]
poem_baseline = 0.37
poem_dumb = 0.17

# Weighted F1 scores for the Lyrics dataset
lyrics_scores = [0.13, 0.14, 0.19, 0.17]
lyrics_baseline = 0.17
lyrics_dumb = 0.14

# --- Plot 1: Poem Dataset Performance ---
plt.figure(figsize=(8,5))
sns.barplot(x=methods, y=poem_scores, color=blue)
plt.axhline(poem_baseline, color=baseline_color, linestyle="--", linewidth=2.8, label="Baseline")
plt.axhline(poem_dumb, color=baseline_color, linestyle=":", linewidth=2.8, label="Dumb Baseline")
plt.title("Poem Dataset Performance")
plt.ylabel("Weighted Avg F1-score")
plt.legend()
plt.tight_layout()
plt.savefig("poem_performance.png", dpi=300)
plt.show()

# --- Plot 2: Lyrics Dataset Performance ---
plt.figure(figsize=(8,5))
sns.barplot(x=methods, y=lyrics_scores, color=red)
plt.axhline(lyrics_baseline, color=baseline_color, linestyle="--", linewidth=2.8, label="Baseline")
plt.axhline(lyrics_dumb, color=baseline_color, linestyle=":", linewidth=2.8, label="Dumb Baseline")
plt.title("Lyrics Dataset Performance")
plt.ylabel("Weighted Avg F1-score")
plt.legend()
plt.tight_layout()
plt.savefig("lyrics_performance.png", dpi=300)
plt.show()