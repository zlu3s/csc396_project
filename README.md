# CSC 396 Final Project: Emotion Classification in Creative Text: Domain Adaptation and Fine-Tuning Strategies for RoBERTa on Poetry and Song Lyrics
## Model info

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

### Training Structure
Datasets with columns:
```python
Dataset({
    features: ['text', 'label'],
    num_rows: n
})
```
Where text is either individual lines of poems or song lyrics, and label is dtype int64

## How to Use
### 1) Import the base sentiment model
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

name = './roberta-base-sentiments'
tokenizer = RobertaTokenizer.from_pretrained(name)
model = RobertaForSequenceClassification.from_pretrained(name)
```
### 2) Fine tune towards either poems or songs
Example
```python
from transformers import TrainingArguments, Trainer

trainging_args = TrainingArguments( ... )
trainer = Trainer( ... )
trainer.Train()
```
### 3) Test on new songs or poems
```python
from transformers import Trainer

def tokenize(dataset):
    return tokenizer(dataset['text'], truncation=True, padding='max_length')
test1 = songs.map(
    tokenize, batched=True,
)
test1 = test1.cast_column("label", Value("int64"))

output = trainer.predict(dataset)
```

## Dataset Citations
### PERC Dataset
Ponnarassery-, Sreeja (2017), “Poem Emotion Recognition Corpus (PERC)”, Mendeley Data, V1, doi: 10.17632/n9vbc8g9cx.1

### dair-ai Dataset
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}

### 500 songs Dataset
https://github.com/LLM-HITCS25S/LyricsEmotionAttribution
