# Emotion Classification in Creative Text: Domain Adaptation and Fine-Tuning Strategies for RoBERTa on Poetry and Song Lyrics
## Repository structure
```text
.
├── Roberta Transformer/       # Stage 1: Fine-tune RoBERTa on general text   
├── Further training/          # Stage 2: Further training experiments  
├── t5_model_implement/        # An experiment of data augmentation with fine-tuned T5 model  
├── Final_Report.pdf
├── Slides.pdf
└── README.md  

```
### Labels Used
```python
labels = ['sad', 'joy', 'love', 'anger', 'fear', 'surprise']
```
Where
```python
0 = sad
1 = joy
2 = love
3 = anger
4 = fear
5 = surprise
```

## Dataset Citations
### PERC Dataset
Ponnarassery-, Sreeja (2017), “Poem Emotion Recognition Corpus (PERC)”, Mendeley Data, V1, doi: 10.17632/n9vbc8g9cx.1
https://data.mendeley.com/datasets/n9vbc8g9cx/1

### dair-ai Dataset
https://huggingface.co/datasets/dair-ai/emotion

### 500 songs Dataset
https://github.com/LLM-HITCS25S/LyricsEmotionAttribution
